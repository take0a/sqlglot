from __future__ import annotations

import typing as t

from sqlglot import expressions as exp
from sqlglot.errors import UnsupportedError
from sqlglot.helper import find_new_name, name_sequence, seq_get


if t.TYPE_CHECKING:
    from sqlglot._typing import E
    from sqlglot.generator import Generator


def preprocess(
    transforms: t.List[t.Callable[[exp.Expression], exp.Expression]],
    generator: t.Optional[t.Callable[[Generator, exp.Expression], str]] = None,
) -> t.Callable[[Generator, exp.Expression], str]:
    """
    Creates a new transform by chaining a sequence of transformations and converts the resulting
    expression to SQL, using either the "_sql" method corresponding to the resulting expression,
    or the appropriate `Generator.TRANSFORMS` function (when applicable -- see below).
    一連の変換を連鎖させることで新しい変換を作成し、結果の式に対応する "_sql" メソッド、または適切な 
    `Generator.TRANSFORMS` 関数 (該当する場合 - 以下を参照) を使用して、結果の式を SQL に変換します。

    Args:
        transforms: sequence of transform functions. These will be called in order.
            変換関数のシーケンス。これらは順番に呼び出されます。

    Returns:
        Function that can be used as a generator transform.
        ジェネレータ変換として使用できる関数。
    """

    def _to_sql(self, expression: exp.Expression) -> str:
        expression_type = type(expression)

        try:
            expression = transforms[0](expression)
            for transform in transforms[1:]:
                expression = transform(expression)
        except UnsupportedError as unsupported_error:
            self.unsupported(str(unsupported_error))

        if generator:
            return generator(self, expression)

        _sql_handler = getattr(self, expression.key + "_sql", None)
        if _sql_handler:
            return _sql_handler(expression)

        transforms_handler = self.TRANSFORMS.get(type(expression))
        if transforms_handler:
            if expression_type is type(expression):
                if isinstance(expression, exp.Func):
                    return self.function_fallback_sql(expression)

                # Ensures we don't enter an infinite loop. This can happen when the original expression
                # has the same type as the final expression and there's no _sql method available for it,
                # because then it'd re-enter _to_sql.
                # 無限ループに陥らないようにします。これは、元の式と最終式が同じ型で、_sqlメソッドが利用できない
                # 場合に発生する可能性があります。その場合、_to_sqlメソッドが再び実行されるためです。
                raise ValueError(
                    f"Expression type {expression.__class__.__name__} requires a _sql method in order to be transformed."
                )

            return transforms_handler(self, expression)

        raise ValueError(f"Unsupported expression type {expression.__class__.__name__}.")

    return _to_sql


def unnest_generate_date_array_using_recursive_cte(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Select):
        count = 0
        recursive_ctes = []

        for unnest in expression.find_all(exp.Unnest):
            if (
                not isinstance(unnest.parent, (exp.From, exp.Join))
                or len(unnest.expressions) != 1
                or not isinstance(unnest.expressions[0], exp.GenerateDateArray)
            ):
                continue

            generate_date_array = unnest.expressions[0]
            start = generate_date_array.args.get("start")
            end = generate_date_array.args.get("end")
            step = generate_date_array.args.get("step")

            if not start or not end or not isinstance(step, exp.Interval):
                continue

            alias = unnest.args.get("alias")
            column_name = alias.columns[0] if isinstance(alias, exp.TableAlias) else "date_value"

            start = exp.cast(start, "date")
            date_add = exp.func(
                "date_add", column_name, exp.Literal.number(step.name), step.args.get("unit")
            )
            cast_date_add = exp.cast(date_add, "date")

            cte_name = "_generated_dates" + (f"_{count}" if count else "")

            base_query = exp.select(start.as_(column_name))
            recursive_query = (
                exp.select(cast_date_add)
                .from_(cte_name)
                .where(cast_date_add <= exp.cast(end, "date"))
            )
            cte_query = base_query.union(recursive_query, distinct=False)

            generate_dates_query = exp.select(column_name).from_(cte_name)
            unnest.replace(generate_dates_query.subquery(cte_name))

            recursive_ctes.append(
                exp.alias_(exp.CTE(this=cte_query), cte_name, table=[column_name])
            )
            count += 1

        if recursive_ctes:
            with_expression = expression.args.get("with_") or exp.With()
            with_expression.set("recursive", True)
            with_expression.set("expressions", [*recursive_ctes, *with_expression.expressions])
            expression.set("with_", with_expression)

    return expression


def unnest_generate_series(expression: exp.Expression) -> exp.Expression:
    """Unnests GENERATE_SERIES or SEQUENCE table references.
    GENERATE_SERIES または SEQUENCE テーブル参照をネスト解除します。"""
    this = expression.this
    if isinstance(expression, exp.Table) and isinstance(this, exp.GenerateSeries):
        unnest = exp.Unnest(expressions=[this])
        if expression.alias:
            return exp.alias_(unnest, alias="_u", table=[expression.alias], copy=False)

        return unnest

    return expression


def eliminate_distinct_on(expression: exp.Expression) -> exp.Expression:
    """
    Convert SELECT DISTINCT ON statements to a subquery with a window function.
    SELECT DISTINCT ON ステートメントをウィンドウ関数を使用してサブクエリに変換します。

    This is useful for dialects that don't support SELECT DISTINCT ON but support window functions.
    これは、SELECT DISTINCT ON をサポートしていないがウィンドウ関数をサポートしている方言に役立ちます。

    Args:
        expression: the expression that will be transformed.
            変換される式。

    Returns:
        The transformed expression.
        変形された式。
    """
    if (
        isinstance(expression, exp.Select)
        and expression.args.get("distinct")
        and isinstance(expression.args["distinct"].args.get("on"), exp.Tuple)
    ):
        row_number_window_alias = find_new_name(expression.named_selects, "_row_number")

        distinct_cols = expression.args["distinct"].pop().args["on"].expressions
        window = exp.Window(this=exp.RowNumber(), partition_by=distinct_cols)

        order = expression.args.get("order")
        if order:
            window.set("order", order.pop())
        else:
            window.set("order", exp.Order(expressions=[c.copy() for c in distinct_cols]))

        window = exp.alias_(window, row_number_window_alias)
        expression.select(window, copy=False)

        # We add aliases to the projections so that we can safely reference them in the outer query
        new_selects = []
        taken_names = {row_number_window_alias}
        for select in expression.selects[:-1]:
            if select.is_star:
                new_selects = [exp.Star()]
                break

            if not isinstance(select, exp.Alias):
                alias = find_new_name(taken_names, select.output_name or "_col")
                quoted = select.this.args.get("quoted") if isinstance(select, exp.Column) else None
                select = select.replace(exp.alias_(select, alias, quoted=quoted))

            taken_names.add(select.output_name)
            new_selects.append(select.args["alias"])

        return (
            exp.select(*new_selects, copy=False)
            .from_(expression.subquery("_t", copy=False), copy=False)
            .where(exp.column(row_number_window_alias).eq(1), copy=False)
        )

    return expression


def eliminate_qualify(expression: exp.Expression) -> exp.Expression:
    """
    Convert SELECT statements that contain the QUALIFY clause into subqueries, filtered equivalently.
    QUALIFY 句を含む SELECT 文を、同等にフィルタリングされたサブクエリに変換します。

    The idea behind this transformation can be seen in Snowflake's documentation for QUALIFY:
    この変換の背後にある考え方は、Snowflake の QUALIFY に関するドキュメントに記載されています。
    https://docs.snowflake.com/en/sql-reference/constructs/qualify

    Some dialects don't support window functions in the WHERE clause, so we need to include them as
    projections in the subquery, in order to refer to them in the outer filter using aliases. Also,
    if a column is referenced in the QUALIFY clause but is not selected, we need to include it too,
    otherwise we won't be able to refer to it in the outer query's WHERE clause. Finally, if a
    newly aliased projection is referenced in the QUALIFY clause, it will be replaced by the
    corresponding expression to avoid creating invalid column references.
    一部の方言ではWHERE句でウィンドウ関数をサポートしていないため、外部フィルターでエイリアスを使用して
    ウィンドウ関数を参照するには、サブクエリにウィンドウ関数を射影として含める必要があります。
    また、QUALIFY句で参照されている列が選択されていない場合は、その列も含める必要があります。
    そうしないと、外部クエリのWHERE句でその列を参照できなくなります。最後に、新たにエイリアスが付けられた
    射影がQUALIFY句で参照されている場合、無効な列参照の作成を防ぐため、対応する式に置き換えられます。
    """
    if isinstance(expression, exp.Select) and expression.args.get("qualify"):
        taken = set(expression.named_selects)
        for select in expression.selects:
            if not select.alias_or_name:
                alias = find_new_name(taken, "_c")
                select.replace(exp.alias_(select, alias))
                taken.add(alias)

        def _select_alias_or_name(select: exp.Expression) -> str | exp.Column:
            alias_or_name = select.alias_or_name
            identifier = select.args.get("alias") or select.this
            if isinstance(identifier, exp.Identifier):
                return exp.column(alias_or_name, quoted=identifier.args.get("quoted"))
            return alias_or_name

        outer_selects = exp.select(*list(map(_select_alias_or_name, expression.selects)))
        qualify_filters = expression.args["qualify"].pop().this
        expression_by_alias = {
            select.alias: select.this
            for select in expression.selects
            if isinstance(select, exp.Alias)
        }

        select_candidates = exp.Window if expression.is_star else (exp.Window, exp.Column)
        for select_candidate in list(qualify_filters.find_all(select_candidates)):
            if isinstance(select_candidate, exp.Window):
                if expression_by_alias:
                    for column in select_candidate.find_all(exp.Column):
                        expr = expression_by_alias.get(column.name)
                        if expr:
                            column.replace(expr)

                alias = find_new_name(expression.named_selects, "_w")
                expression.select(exp.alias_(select_candidate, alias), copy=False)
                column = exp.column(alias)

                if isinstance(select_candidate.parent, exp.Qualify):
                    qualify_filters = column
                else:
                    select_candidate.replace(column)
            elif select_candidate.name not in expression.named_selects:
                expression.select(select_candidate.copy(), copy=False)

        return outer_selects.from_(expression.subquery(alias="_t", copy=False), copy=False).where(
            qualify_filters, copy=False
        )

    return expression


def remove_precision_parameterized_types(expression: exp.Expression) -> exp.Expression:
    """
    Some dialects only allow the precision for parameterized types to be defined in the DDL and not in
    other expressions. This transforms removes the precision from parameterized types in expressions.
    一部の方言では、パラメータ化された型の精度はDDLでのみ定義でき、他の式では定義できません。
    この変換により、式内のパラメータ化された型から精度が削除されます。
    """
    for node in expression.find_all(exp.DataType):
        node.set(
            "expressions", [e for e in node.expressions if not isinstance(e, exp.DataTypeParam)]
        )

    return expression


def unqualify_unnest(expression: exp.Expression) -> exp.Expression:
    """Remove references to unnest table aliases, added by the optimizer's qualify_columns step.
    オプティマイザーの qualified_columns ステップによって追加された、ネスト解除テーブル エイリアスへの参照を削除します。"""
    from sqlglot.optimizer.scope import find_all_in_scope

    if isinstance(expression, exp.Select):
        unnest_aliases = {
            unnest.alias
            for unnest in find_all_in_scope(expression, exp.Unnest)
            if isinstance(unnest.parent, (exp.From, exp.Join))
        }
        if unnest_aliases:
            for column in expression.find_all(exp.Column):
                leftmost_part = column.parts[0]
                if leftmost_part.arg_key != "this" and leftmost_part.this in unnest_aliases:
                    leftmost_part.pop()

    return expression


def unnest_to_explode(
    expression: exp.Expression,
    unnest_using_arrays_zip: bool = True,
) -> exp.Expression:
    """Convert cross join unnest into lateral view explode.
    cross join を lateral view explode にアンネスト変換します"""

    def _unnest_zip_exprs(
        u: exp.Unnest, unnest_exprs: t.List[exp.Expression], has_multi_expr: bool
    ) -> t.List[exp.Expression]:
        if has_multi_expr:
            if not unnest_using_arrays_zip:
                raise UnsupportedError("Cannot transpile UNNEST with multiple input arrays")

            # Use INLINE(ARRAYS_ZIP(...)) for multiple expressions
            zip_exprs: t.List[exp.Expression] = [
                exp.Anonymous(this="ARRAYS_ZIP", expressions=unnest_exprs)
            ]
            u.set("expressions", zip_exprs)
            return zip_exprs
        return unnest_exprs

    def _udtf_type(u: exp.Unnest, has_multi_expr: bool) -> t.Type[exp.Func]:
        if u.args.get("offset"):
            return exp.Posexplode
        return exp.Inline if has_multi_expr else exp.Explode

    if isinstance(expression, exp.Select):
        from_ = expression.args.get("from_")

        if from_ and isinstance(from_.this, exp.Unnest):
            unnest = from_.this
            alias = unnest.args.get("alias")
            exprs = unnest.expressions
            has_multi_expr = len(exprs) > 1
            this, *_ = _unnest_zip_exprs(unnest, exprs, has_multi_expr)

            columns = alias.columns if alias else []
            offset = unnest.args.get("offset")
            if offset:
                columns.insert(
                    0, offset if isinstance(offset, exp.Identifier) else exp.to_identifier("pos")
                )

            unnest.replace(
                exp.Table(
                    this=_udtf_type(unnest, has_multi_expr)(this=this),
                    alias=exp.TableAlias(this=alias.this, columns=columns) if alias else None,
                )
            )

        joins = expression.args.get("joins") or []
        for join in list(joins):
            join_expr = join.this

            is_lateral = isinstance(join_expr, exp.Lateral)

            unnest = join_expr.this if is_lateral else join_expr

            if isinstance(unnest, exp.Unnest):
                if is_lateral:
                    alias = join_expr.args.get("alias")
                else:
                    alias = unnest.args.get("alias")
                exprs = unnest.expressions
                # The number of unnest.expressions will be changed by _unnest_zip_exprs, we need to record it here
                # unnest.expressionsの数は_unnest_zip_exprsによって変更されるので、ここで記録する必要があります。
                has_multi_expr = len(exprs) > 1
                exprs = _unnest_zip_exprs(unnest, exprs, has_multi_expr)

                joins.remove(join)

                alias_cols = alias.columns if alias else []

                # # Handle UNNEST to LATERAL VIEW EXPLODE: Exception is raised when there are 0 or > 2 aliases
                # Spark LATERAL VIEW EXPLODE requires single alias for array/struct and two for Map type column unlike unnest in trino/presto which can take an arbitrary amount.
                # Refs: https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-lateral-view.html

                if not has_multi_expr and len(alias_cols) not in (1, 2):
                    raise UnsupportedError(
                        "CROSS JOIN UNNEST to LATERAL VIEW EXPLODE transformation requires explicit column aliases"
                    )

                offset = unnest.args.get("offset")
                if offset:
                    alias_cols.insert(
                        0,
                        offset if isinstance(offset, exp.Identifier) else exp.to_identifier("pos"),
                    )

                for e, column in zip(exprs, alias_cols):
                    expression.append(
                        "laterals",
                        exp.Lateral(
                            this=_udtf_type(unnest, has_multi_expr)(this=e),
                            view=True,
                            alias=exp.TableAlias(
                                this=alias.this,  # type: ignore
                                columns=alias_cols,
                            ),
                        ),
                    )

    return expression


def explode_projection_to_unnest(
    index_offset: int = 0,
) -> t.Callable[[exp.Expression], exp.Expression]:
    """Convert explode/posexplode projections into unnests."""

    def _explode_projection_to_unnest(expression: exp.Expression) -> exp.Expression:
        if isinstance(expression, exp.Select):
            from sqlglot.optimizer.scope import Scope

            taken_select_names = set(expression.named_selects)
            taken_source_names = {name for name, _ in Scope(expression).references}

            def new_name(names: t.Set[str], name: str) -> str:
                name = find_new_name(names, name)
                names.add(name)
                return name

            arrays: t.List[exp.Condition] = []
            series_alias = new_name(taken_select_names, "pos")
            series = exp.alias_(
                exp.Unnest(
                    expressions=[exp.GenerateSeries(start=exp.Literal.number(index_offset))]
                ),
                new_name(taken_source_names, "_u"),
                table=[series_alias],
            )

            # we use list here because expression.selects is mutated inside the loop
            # ここでリストを使用するのは、expression.selectsがループ内で変更されるためです。
            for select in list(expression.selects):
                explode = select.find(exp.Explode)

                if explode:
                    pos_alias = ""
                    explode_alias = ""

                    if isinstance(select, exp.Alias):
                        explode_alias = select.args["alias"]
                        alias = select
                    elif isinstance(select, exp.Aliases):
                        pos_alias = select.aliases[0]
                        explode_alias = select.aliases[1]
                        alias = select.replace(exp.alias_(select.this, "", copy=False))
                    else:
                        alias = select.replace(exp.alias_(select, ""))
                        explode = alias.find(exp.Explode)
                        assert explode

                    is_posexplode = isinstance(explode, exp.Posexplode)
                    explode_arg = explode.this

                    if isinstance(explode, exp.ExplodeOuter):
                        bracket = explode_arg[0]
                        bracket.set("safe", True)
                        bracket.set("offset", True)
                        explode_arg = exp.func(
                            "IF",
                            exp.func(
                                "ARRAY_SIZE", exp.func("COALESCE", explode_arg, exp.Array())
                            ).eq(0),
                            exp.array(bracket, copy=False),
                            explode_arg,
                        )

                    # This ensures that we won't use [POS]EXPLODE's argument as a new selection
                    # これにより、[POS]EXPLODEの引数を新しい選択範囲として使用しないことが保証されます。
                    if isinstance(explode_arg, exp.Column):
                        taken_select_names.add(explode_arg.output_name)

                    unnest_source_alias = new_name(taken_source_names, "_u")

                    if not explode_alias:
                        explode_alias = new_name(taken_select_names, "col")

                        if is_posexplode:
                            pos_alias = new_name(taken_select_names, "pos")

                    if not pos_alias:
                        pos_alias = new_name(taken_select_names, "pos")

                    alias.set("alias", exp.to_identifier(explode_alias))

                    series_table_alias = series.args["alias"].this
                    column = exp.If(
                        this=exp.column(series_alias, table=series_table_alias).eq(
                            exp.column(pos_alias, table=unnest_source_alias)
                        ),
                        true=exp.column(explode_alias, table=unnest_source_alias),
                    )

                    explode.replace(column)

                    if is_posexplode:
                        expressions = expression.expressions
                        expressions.insert(
                            expressions.index(alias) + 1,
                            exp.If(
                                this=exp.column(series_alias, table=series_table_alias).eq(
                                    exp.column(pos_alias, table=unnest_source_alias)
                                ),
                                true=exp.column(pos_alias, table=unnest_source_alias),
                            ).as_(pos_alias),
                        )
                        expression.set("expressions", expressions)

                    if not arrays:
                        if expression.args.get("from_"):
                            expression.join(series, copy=False, join_type="CROSS")
                        else:
                            expression.from_(series, copy=False)

                    size: exp.Condition = exp.ArraySize(this=explode_arg.copy())
                    arrays.append(size)

                    # trino doesn't support left join unnest with on conditions
                    # if it did, this would be much simpler
                    # trinoは条件付きで左結合のunnestをサポートしていません。
                    # もしサポートしていたら、もっとシンプルになります。
                    expression.join(
                        exp.alias_(
                            exp.Unnest(
                                expressions=[explode_arg.copy()],
                                offset=exp.to_identifier(pos_alias),
                            ),
                            unnest_source_alias,
                            table=[explode_alias],
                        ),
                        join_type="CROSS",
                        copy=False,
                    )

                    if index_offset != 1:
                        size = size - 1

                    expression.where(
                        exp.column(series_alias, table=series_table_alias)
                        .eq(exp.column(pos_alias, table=unnest_source_alias))
                        .or_(
                            (exp.column(series_alias, table=series_table_alias) > size).and_(
                                exp.column(pos_alias, table=unnest_source_alias).eq(size)
                            )
                        ),
                        copy=False,
                    )

            if arrays:
                end: exp.Condition = exp.Greatest(this=arrays[0], expressions=arrays[1:])

                if index_offset != 1:
                    end = end - (1 - index_offset)
                series.expressions[0].set("end", end)

        return expression

    return _explode_projection_to_unnest


def add_within_group_for_percentiles(expression: exp.Expression) -> exp.Expression:
    """Transforms percentiles by adding a WITHIN GROUP clause to them.
    WITHIN GROUP 句を追加してパーセンタイルを変換します。"""
    if (
        isinstance(expression, exp.PERCENTILES)
        and not isinstance(expression.parent, exp.WithinGroup)
        and expression.expression
    ):
        column = expression.this.pop()
        expression.set("this", expression.expression.pop())
        order = exp.Order(expressions=[exp.Ordered(this=column)])
        expression = exp.WithinGroup(this=expression, expression=order)

    return expression


def remove_within_group_for_percentiles(expression: exp.Expression) -> exp.Expression:
    """Transforms percentiles by getting rid of their corresponding WITHIN GROUP clause.
    対応する WITHIN GROUP 句を削除してパーセンタイルを変換します。"""
    if (
        isinstance(expression, exp.WithinGroup)
        and isinstance(expression.this, exp.PERCENTILES)
        and isinstance(expression.expression, exp.Order)
    ):
        quantile = expression.this.this
        input_value = t.cast(exp.Ordered, expression.find(exp.Ordered)).this
        return expression.replace(exp.ApproxQuantile(this=input_value, quantile=quantile))

    return expression


def add_recursive_cte_column_names(expression: exp.Expression) -> exp.Expression:
    """Uses projection output names in recursive CTE definitions to define the CTEs' columns.
    再帰 CTE 定義で投影出力名を使用して、CTE の列を定義します。"""
    if isinstance(expression, exp.With) and expression.recursive:
        next_name = name_sequence("_c_")

        for cte in expression.expressions:
            if not cte.args["alias"].columns:
                query = cte.this
                if isinstance(query, exp.SetOperation):
                    query = query.this

                cte.args["alias"].set(
                    "columns",
                    [exp.to_identifier(s.alias_or_name or next_name()) for s in query.selects],
                )

    return expression


def epoch_cast_to_ts(expression: exp.Expression) -> exp.Expression:
    """Replace 'epoch' in casts by the equivalent date literal.
    キャスト内の 'epoch' を同等の日付リテラルに置き換えます。"""
    if (
        isinstance(expression, (exp.Cast, exp.TryCast))
        and expression.name.lower() == "epoch"
        and expression.to.this in exp.DataType.TEMPORAL_TYPES
    ):
        expression.this.replace(exp.Literal.string("1970-01-01 00:00:00"))

    return expression


def eliminate_semi_and_anti_joins(expression: exp.Expression) -> exp.Expression:
    """Convert SEMI and ANTI joins into equivalent forms that use EXIST instead.
    SEMI 結合と ANTI 結合を、代わりに EXIST を使用する同等の形式に変換します。"""
    if isinstance(expression, exp.Select):
        for join in expression.args.get("joins") or []:
            on = join.args.get("on")
            if on and join.kind in ("SEMI", "ANTI"):
                subquery = exp.select("1").from_(join.this).where(on)
                exists = exp.Exists(this=subquery)
                if join.kind == "ANTI":
                    exists = exists.not_(copy=False)

                join.pop()
                expression.where(exists, copy=False)

    return expression


def eliminate_full_outer_join(expression: exp.Expression) -> exp.Expression:
    """
    Converts a query with a FULL OUTER join to a union of identical queries that
    use LEFT/RIGHT OUTER joins instead. This transformation currently only works
    for queries that have a single FULL OUTER join.
    FULL OUTER結合を含むクエリを、LEFT/RIGHT OUTER結合を使用する同一クエリの結合に変換します。
    この変換は現在、単一のFULL OUTER結合を含むクエリに対してのみ機能します。
    """
    if isinstance(expression, exp.Select):
        full_outer_joins = [
            (index, join)
            for index, join in enumerate(expression.args.get("joins") or [])
            if join.side == "FULL"
        ]

        if len(full_outer_joins) == 1:
            expression_copy = expression.copy()
            expression.set("limit", None)
            index, full_outer_join = full_outer_joins[0]

            tables = (expression.args["from_"].alias_or_name, full_outer_join.alias_or_name)
            join_conditions = full_outer_join.args.get("on") or exp.and_(
                *[
                    exp.column(col, tables[0]).eq(exp.column(col, tables[1]))
                    for col in full_outer_join.args.get("using")
                ]
            )

            full_outer_join.set("side", "left")
            anti_join_clause = (
                exp.select("1").from_(expression.args["from_"]).where(join_conditions)
            )
            expression_copy.args["joins"][index].set("side", "right")
            expression_copy = expression_copy.where(exp.Exists(this=anti_join_clause).not_())
            expression_copy.set("with_", None)  # remove CTEs from RIGHT side
            expression.set("order", None)  # remove order by from LEFT side

            return exp.union(expression, expression_copy, copy=False, distinct=False)

    return expression


def move_ctes_to_top_level(expression: E) -> E:
    """
    Some dialects (e.g. Hive, T-SQL, Spark prior to version 3) only allow CTEs to be
    defined at the top-level, so for example queries like:
    一部の方言 (例: Hive、T-SQL、バージョン 3 より前の Spark) では、CTE を最上位レベルでのみ定義できます。
    たとえば、次のようなクエリになります。

        SELECT * FROM (WITH t(c) AS (SELECT 1) SELECT * FROM t) AS subq

    are invalid in those dialects. This transformation can be used to ensure all CTEs are
    moved to the top level so that the final SQL code is valid from a syntax standpoint.
    はこれらの方言では無効です。
    この変換により、すべてのCTEが最上位レベルに移動され、最終的なSQLコードが構文の観点から有効になります。

    TODO: handle name clashes whilst moving CTEs (it can get quite tricky & costly).
    CTE を移動する際にハンドル名の衝突が発生します (非常に扱いにくくなり、コストもかかる場合があります)。
    """
    top_level_with = expression.args.get("with_")
    for inner_with in expression.find_all(exp.With):
        if inner_with.parent is expression:
            continue

        if not top_level_with:
            top_level_with = inner_with.pop()
            expression.set("with_", top_level_with)
        else:
            if inner_with.recursive:
                top_level_with.set("recursive", True)

            parent_cte = inner_with.find_ancestor(exp.CTE)
            inner_with.pop()

            if parent_cte:
                i = top_level_with.expressions.index(parent_cte)
                top_level_with.expressions[i:i] = inner_with.expressions
                top_level_with.set("expressions", top_level_with.expressions)
            else:
                top_level_with.set(
                    "expressions", top_level_with.expressions + inner_with.expressions
                )

    return expression


def ensure_bools(expression: exp.Expression) -> exp.Expression:
    """Converts numeric values used in conditions into explicit boolean expressions.
    条件で使用される数値を明示的なブール式に変換します。"""
    from sqlglot.optimizer.canonicalize import ensure_bools

    def _ensure_bool(node: exp.Expression) -> None:
        if (
            node.is_number
            or (
                not isinstance(node, exp.SubqueryPredicate)
                and node.is_type(exp.DataType.Type.UNKNOWN, *exp.DataType.NUMERIC_TYPES)
            )
            or (isinstance(node, exp.Column) and not node.type)
        ):
            node.replace(node.neq(0))

    for node in expression.walk():
        ensure_bools(node, _ensure_bool)

    return expression


def unqualify_columns(expression: exp.Expression) -> exp.Expression:
    for column in expression.find_all(exp.Column):
        # We only wanna pop off the table, db, catalog args
        # テーブル、データベース、カタログ引数をポップしたいだけです
        for part in column.parts[:-1]:
            part.pop()

    return expression


def remove_unique_constraints(expression: exp.Expression) -> exp.Expression:
    assert isinstance(expression, exp.Create)
    for constraint in expression.find_all(exp.UniqueColumnConstraint):
        if constraint.parent:
            constraint.parent.pop()

    return expression


def ctas_with_tmp_tables_to_create_tmp_view(
    expression: exp.Expression,
    tmp_storage_provider: t.Callable[[exp.Expression], exp.Expression] = lambda e: e,
) -> exp.Expression:
    assert isinstance(expression, exp.Create)
    properties = expression.args.get("properties")
    temporary = any(
        isinstance(prop, exp.TemporaryProperty)
        for prop in (properties.expressions if properties else [])
    )

    # CTAS with temp tables map to CREATE TEMPORARY VIEW
    # 一時テーブルを含む CTAS は CREATE TEMPORARY VIEW にマップされます
    if expression.kind == "TABLE" and temporary:
        if expression.expression:
            return exp.Create(
                kind="TEMPORARY VIEW",
                this=expression.this,
                expression=expression.expression,
            )
        return tmp_storage_provider(expression)

    return expression


def move_schema_columns_to_partitioned_by(expression: exp.Expression) -> exp.Expression:
    """
    In Hive, the PARTITIONED BY property acts as an extension of a table's schema. When the
    PARTITIONED BY value is an array of column names, they are transformed into a schema.
    The corresponding columns are removed from the create statement.
    Hiveでは、PARTITIONED BYプロパティはテーブルスキーマの拡張として機能します。
    PARTITIONED BY値が列名の配列である場合、それらはスキーマに変換されます。
    対応する列はcreateステートメントから削除されます。
    """
    assert isinstance(expression, exp.Create)
    has_schema = isinstance(expression.this, exp.Schema)
    is_partitionable = expression.kind in {"TABLE", "VIEW"}

    if has_schema and is_partitionable:
        prop = expression.find(exp.PartitionedByProperty)
        if prop and prop.this and not isinstance(prop.this, exp.Schema):
            schema = expression.this
            columns = {v.name.upper() for v in prop.this.expressions}
            partitions = [col for col in schema.expressions if col.name.upper() in columns]
            schema.set("expressions", [e for e in schema.expressions if e not in partitions])
            prop.replace(exp.PartitionedByProperty(this=exp.Schema(expressions=partitions)))
            expression.set("this", schema)

    return expression


def move_partitioned_by_to_schema_columns(expression: exp.Expression) -> exp.Expression:
    """
    Spark 3 supports both "HIVEFORMAT" and "DATASOURCE" formats for CREATE TABLE.
    Spark 3 は、CREATE TABLE の「HIVEFORMAT」と「DATASOURCE」の両方の形式をサポートしています。

    Currently, SQLGlot uses the DATASOURCE format for Spark 3.
    現在、SQLGlot は Spark 3 に DATASOURCE 形式を使用しています。
    """
    assert isinstance(expression, exp.Create)
    prop = expression.find(exp.PartitionedByProperty)
    if (
        prop
        and prop.this
        and isinstance(prop.this, exp.Schema)
        and all(isinstance(e, exp.ColumnDef) and e.kind for e in prop.this.expressions)
    ):
        prop_this = exp.Tuple(
            expressions=[exp.to_identifier(e.this) for e in prop.this.expressions]
        )
        schema = expression.this
        for e in prop.this.expressions:
            schema.append("expressions", e)
        prop.set("this", prop_this)

    return expression


def struct_kv_to_alias(expression: exp.Expression) -> exp.Expression:
    """Converts struct arguments to aliases, e.g. STRUCT(1 AS y).
    構造体の引数をエイリアスに変換します。例: STRUCT(1 AS y)。"""
    if isinstance(expression, exp.Struct):
        expression.set(
            "expressions",
            [
                exp.alias_(e.expression, e.this) if isinstance(e, exp.PropertyEQ) else e
                for e in expression.expressions
            ],
        )

    return expression


def eliminate_join_marks(expression: exp.Expression) -> exp.Expression:
    """https://docs.oracle.com/cd/B19306_01/server.102/b14200/queries006.htm#sthref3178

    1. You cannot specify the (+) operator in a query block that also contains FROM clause join syntax.
    1. FROM 句の結合構文を含むクエリブロックでは、(+) 演算子を指定できません。

    2. The (+) operator can appear only in the WHERE clause or, in the context of left-correlation (that is, when specifying the TABLE clause) in the FROM clause, and can be applied only to a column of a table or view.
    2. (+) 演算子は、WHERE 句内、または FROM 句の左相関コンテキスト（つまり、TABLE 句を指定する場合）内でのみ使用でき、テーブルまたはビューの列にのみ適用できます。

    The (+) operator does not produce an outer join if you specify one table in the outer query and the other table in an inner query.
    外部クエ​​リで 1 つのテーブルを指定し、内部クエリで別のテーブルを指定した場合、(+) 演算子は外部結合を生成しません。

    You cannot use the (+) operator to outer-join a table to itself, although self joins are valid.
    自己結合は有効ですが、(+) 演算子を使用してテーブルを自身に外部結合することはできません。

    The (+) operator can be applied only to a column, not to an arbitrary expression. However, an arbitrary expression can contain one or more columns marked with the (+) operator.
    (+) 演算子は列にのみ適用でき、任意の式には適用できません。ただし、任意の式には、(+) 演算子でマークされた 1 つ以上の列を含めることができます。

    A WHERE condition containing the (+) operator cannot be combined with another condition using the OR logical operator.
    (+) 演算子を含む WHERE 条件は、OR 論理演算子を使用して他の条件と組み合わせることはできません。

    A WHERE condition cannot use the IN comparison condition to compare a column marked with the (+) operator with an expression.
    WHERE 条件では、IN 比較条件を使用して、(+) 演算子でマークされた列と式を比較することはできません。

    A WHERE condition cannot compare any column marked with the (+) operator with a subquery.
    WHERE 条件では、(+) 演算子でマークされた列をサブクエリと比較することはできません。

    -- example with WHERE
    SELECT d.department_name, sum(e.salary) as total_salary
    FROM departments d, employees e
    WHERE e.department_id(+) = d.department_id
    group by department_name

    -- example of left correlation in select
    SELECT d.department_name, (
        SELECT SUM(e.salary)
            FROM employees e
            WHERE e.department_id(+) = d.department_id) AS total_salary
    FROM departments d;

    -- example of left correlation in from
    SELECT d.department_name, t.total_salary
    FROM departments d, (
            SELECT SUM(e.salary) AS total_salary
            FROM employees e
            WHERE e.department_id(+) = d.department_id
        ) t
    """

    from sqlglot.optimizer.scope import traverse_scope
    from sqlglot.optimizer.normalize import normalize, normalized
    from collections import defaultdict

    # we go in reverse to check the main query for left correlation
    # 左相関のメインクエリを逆にチェックします
    for scope in reversed(traverse_scope(expression)):
        query = scope.expression

        where = query.args.get("where")
        joins = query.args.get("joins", [])

        # knockout: we do not support left correlation (see point 2)
        # ノックアウト: 左相関はサポートしていません（ポイント2を参照）
        assert not scope.is_correlated_subquery, "Correlated queries are not supported"

        # nothing to do - we check it here after knockout above
        # 何もする必要はありません - 上記のノックアウト後にここで確認します
        if not where or not any(c.args.get("join_mark") for c in where.find_all(exp.Column)):
            continue

        # make sure we have AND of ORs to have clear join terms
        # 明確な結合条件を持つようにANDまたはORを使用する
        where = normalize(where.this)
        assert normalized(where), "Cannot normalize JOIN predicates"

        joins_ons = defaultdict(list)  # dict of {name: list of join AND conditions}
        for cond in [where] if not isinstance(where, exp.And) else where.flatten():
            join_cols = [col for col in cond.find_all(exp.Column) if col.args.get("join_mark")]

            left_join_table = set(col.table for col in join_cols)
            if not left_join_table:
                continue

            assert not (
                len(left_join_table) > 1
            ), "Cannot combine JOIN predicates from different tables"

            for col in join_cols:
                col.set("join_mark", False)

            joins_ons[left_join_table.pop()].append(cond)

        old_joins = {join.alias_or_name: join for join in joins}
        new_joins = {}
        query_from = query.args["from_"]

        for table, predicates in joins_ons.items():
            join_what = old_joins.get(table, query_from).this.copy()
            new_joins[join_what.alias_or_name] = exp.Join(
                this=join_what, on=exp.and_(*predicates), kind="LEFT"
            )

            for p in predicates:
                while isinstance(p.parent, exp.Paren):
                    p.parent.replace(p)

                parent = p.parent
                p.pop()
                if isinstance(parent, exp.Binary):
                    parent.replace(parent.right if parent.left is None else parent.left)
                elif isinstance(parent, exp.Where):
                    parent.pop()

        if query_from.alias_or_name in new_joins:
            only_old_joins = old_joins.keys() - new_joins.keys()
            assert (
                len(only_old_joins) >= 1
            ), "Cannot determine which table to use in the new FROM clause"

            new_from_name = list(only_old_joins)[0]
            query.set("from_", exp.From(this=old_joins[new_from_name].this))

        if new_joins:
            for n, j in old_joins.items():  # preserve any other joins
                if n not in new_joins and n != query.args["from_"].name:
                    if not j.kind:
                        j.set("kind", "CROSS")
                    new_joins[n] = j
            query.set("joins", list(new_joins.values()))

    return expression


def any_to_exists(expression: exp.Expression) -> exp.Expression:
    """
    Transform ANY operator to Spark's EXISTS
    ANY演算子をSparkのEXISTSに変換する

    For example,
        - Postgres: SELECT * FROM tbl WHERE 5 > ANY(tbl.col)
        - Spark: SELECT * FROM tbl WHERE EXISTS(tbl.col, x -> x < 5)

    Both ANY and EXISTS accept queries but currently only array expressions are supported for this
    transformation
    ANYとEXISTSはどちらもクエリを受け入れますが、現在この変換では配列式のみがサポートされています。
    """
    if isinstance(expression, exp.Select):
        for any_expr in expression.find_all(exp.Any):
            this = any_expr.this
            if isinstance(this, exp.Query) or isinstance(any_expr.parent, (exp.Like, exp.ILike)):
                continue

            binop = any_expr.parent
            if isinstance(binop, exp.Binary):
                lambda_arg = exp.to_identifier("x")
                any_expr.replace(lambda_arg)
                lambda_expr = exp.Lambda(this=binop.copy(), expressions=[lambda_arg])
                binop.replace(exp.Exists(this=this.unnest(), expression=lambda_expr))

    return expression


def eliminate_window_clause(expression: exp.Expression) -> exp.Expression:
    """Eliminates the `WINDOW` query clause by inling each named window.
    名前付きウィンドウをそれぞれインライン化することで、`WINDOW` クエリ句を削除します。"""
    if isinstance(expression, exp.Select) and expression.args.get("windows"):
        from sqlglot.optimizer.scope import find_all_in_scope

        windows = expression.args["windows"]
        expression.set("windows", None)

        window_expression: t.Dict[str, exp.Expression] = {}

        def _inline_inherited_window(window: exp.Expression) -> None:
            inherited_window = window_expression.get(window.alias.lower())
            if not inherited_window:
                return

            window.set("alias", None)
            for key in ("partition_by", "order", "spec"):
                arg = inherited_window.args.get(key)
                if arg:
                    window.set(key, arg.copy())

        for window in windows:
            _inline_inherited_window(window)
            window_expression[window.name.lower()] = window

        for window in find_all_in_scope(expression, exp.Window):
            _inline_inherited_window(window)

    return expression


def inherit_struct_field_names(expression: exp.Expression) -> exp.Expression:
    """
    Inherit field names from the first struct in an array.
    配列の最初の構造体からフィールド名を継承します。

    BigQuery supports implicitly inheriting names from the first STRUCT in an array:
    BigQuery は、配列の最初の STRUCT から名前を暗黙的に継承することをサポートしています。

    Example:
        ARRAY[
          STRUCT('Alice' AS name, 85 AS score),  -- defines names
          STRUCT('Bob', 92),                     -- inherits names
          STRUCT('Diana', 95)                    -- inherits names
        ]

    This transformation makes the field names explicit on all structs by adding
    PropertyEQ nodes, in order to facilitate transpilation to other dialects.
    この変換により、PropertyEQ ノードを追加することですべての構造体のフィールド名が
    明示的になり、他の方言へのトランスパイルが容易になります。

    Args:
        expression: The expression tree to transform
            変換する式ツリー

    Returns:
        The modified expression with field names inherited in all structs
        すべての構造体でフィールド名が継承されている変更された式
    """
    if (
        isinstance(expression, exp.Array)
        and expression.args.get("struct_name_inheritance")
        and isinstance(first_item := seq_get(expression.expressions, 0), exp.Struct)
        and all(isinstance(fld, exp.PropertyEQ) for fld in first_item.expressions)
    ):
        field_names = [fld.this for fld in first_item.expressions]

        # Apply field names to subsequent structs that don't have them
        # フィールド名を持たない後続の構造体にフィールド名を適用する
        for struct in expression.expressions[1:]:
            if not isinstance(struct, exp.Struct) or len(struct.expressions) != len(field_names):
                continue

            # Convert unnamed expressions to PropertyEQ with inherited names
            # 名前のない式を継承された名前を持つ PropertyEQ に変換する
            new_expressions = []
            for i, expr in enumerate(struct.expressions):
                if not isinstance(expr, exp.PropertyEQ):
                    # Create PropertyEQ: field_name := value
                    new_expressions.append(
                        exp.PropertyEQ(
                            this=exp.Identifier(this=field_names[i].copy()), expression=expr
                        )
                    )
                else:
                    new_expressions.append(expr)

            struct.set("expressions", new_expressions)

    return expression
