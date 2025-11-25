from __future__ import annotations

import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get

logger = logging.getLogger("sqlglot")

TRAVERSABLES = (exp.Query, exp.DDL, exp.DML)


class ScopeType(Enum):
    ROOT = auto()
    SUBQUERY = auto()
    DERIVED_TABLE = auto()
    CTE = auto()
    UNION = auto()
    UDTF = auto()


class Scope:
    """
    Selection scope.
    選択範囲。

    Attributes:
        expression (exp.Select|exp.SetOperation): Root expression of this scope
            このスコープのルートの式
        sources (dict[str, exp.Table|Scope]): Mapping of source name to either
            a Table expression or another Scope instance. For example:
            ソース名をテーブル式または別のスコープインスタンスにマッピングします。例:
                SELECT * FROM x                     {"x": Table(this="x")}
                SELECT * FROM x AS y                {"y": Table(this="x")}
                SELECT * FROM (SELECT ...) AS y     {"y": Scope(...)}
        lateral_sources (dict[str, exp.Table|Scope]): Sources from laterals
            For example:
                SELECT c FROM x LATERAL VIEW EXPLODE (a) AS c;
            The LATERAL VIEW EXPLODE gets x as a source.
        cte_sources (dict[str, Scope]): Sources from CTES
        outer_columns (list[str]): If this is a derived table or CTE, and the outer query
            defines a column list for the alias of this scope, this is that list of columns.
            これが派生テーブルまたは CTE であり、外部クエリがこのスコープのエイリアスの列リストを
            定義している場合、これはその列のリストになります。
            For example:
                SELECT * FROM (SELECT ...) AS y(col1, col2)
            The inner query would have `["col1", "col2"]` for its `outer_columns`
            内部クエリの`outer_columns`には`["col1", "col2"]`が含まれます
        parent (Scope): Parent scope
            親スコープ
        scope_type (ScopeType): Type of this scope, relative to it's parent
            このスコープの親に対する相対的なタイプ
        subquery_scopes (list[Scope]): List of all child scopes for subqueries
            サブクエリのすべての子スコープのリスト
        cte_scopes (list[Scope]): List of all child scopes for CTEs
            CTE のすべての子スコープのリスト
        derived_table_scopes (list[Scope]): List of all child scopes for derived_tables
            派生テーブルのすべての子スコープのリスト
        udtf_scopes (list[Scope]): List of all child scopes for user defined tabular functions
            ユーザー定義の表形式関数のすべての子スコープのリスト
        table_scopes (list[Scope]): derived_table_scopes + udtf_scopes, in the order that they're defined
            定義された順序で、derived_table_scopes + udtf_scopes
        union_scopes (list[Scope, Scope]): If this Scope is for a Union expression, this will be
            a list of the left and right child scopes.
            このスコープが Union 式用である場合、これは左と右の子スコープのリストになります。
    """

    def __init__(
        self,
        expression,
        sources=None,
        outer_columns=None,
        parent=None,
        scope_type=ScopeType.ROOT,
        lateral_sources=None,
        cte_sources=None,
        can_be_correlated=None,
    ):
        self.expression = expression
        self.sources = sources or {}
        self.lateral_sources = lateral_sources or {}
        self.cte_sources = cte_sources or {}
        self.sources.update(self.lateral_sources)
        self.sources.update(self.cte_sources)
        self.outer_columns = outer_columns or []
        self.parent = parent
        self.scope_type = scope_type
        self.subquery_scopes = []
        self.derived_table_scopes = []
        self.table_scopes = []
        self.cte_scopes = []
        self.union_scopes = []
        self.udtf_scopes = []
        self.can_be_correlated = can_be_correlated
        self.clear_cache()

    def clear_cache(self):
        self._collected = False
        self._raw_columns = None
        self._table_columns = None
        self._stars = None
        self._derived_tables = None
        self._udtfs = None
        self._tables = None
        self._ctes = None
        self._subqueries = None
        self._selected_sources = None
        self._columns = None
        self._external_columns = None
        self._join_hints = None
        self._pivots = None
        self._references = None
        self._semi_anti_join_tables = None

    def branch(
        self, expression, scope_type, sources=None, cte_sources=None, lateral_sources=None, **kwargs
    ):
        """Branch from the current scope to a new, inner scope
        現在のスコープから新しい内部スコープに分岐する"""
        return Scope(
            expression=expression.unnest(),
            sources=sources.copy() if sources else None,
            parent=self,
            scope_type=scope_type,
            cte_sources={**self.cte_sources, **(cte_sources or {})},
            lateral_sources=lateral_sources.copy() if lateral_sources else None,
            can_be_correlated=self.can_be_correlated
            or scope_type in (ScopeType.SUBQUERY, ScopeType.UDTF),
            **kwargs,
        )

    def _collect(self):
        self._tables = []
        self._ctes = []
        self._subqueries = []
        self._derived_tables = []
        self._udtfs = []
        self._raw_columns = []
        self._table_columns = []
        self._stars = []
        self._join_hints = []
        self._semi_anti_join_tables = set()

        for node in self.walk(bfs=False):
            if node is self.expression:
                continue

            if isinstance(node, exp.Dot) and node.is_star:
                self._stars.append(node)
            elif isinstance(node, exp.Column) and not isinstance(node, exp.Pseudocolumn):
                if isinstance(node.this, exp.Star):
                    self._stars.append(node)
                else:
                    self._raw_columns.append(node)
            elif isinstance(node, exp.Table) and not isinstance(node.parent, exp.JoinHint):
                parent = node.parent
                if isinstance(parent, exp.Join) and parent.is_semi_or_anti_join:
                    self._semi_anti_join_tables.add(node.alias_or_name)

                self._tables.append(node)
            elif isinstance(node, exp.JoinHint):
                self._join_hints.append(node)
            elif isinstance(node, exp.UDTF):
                self._udtfs.append(node)
            elif isinstance(node, exp.CTE):
                self._ctes.append(node)
            elif _is_derived_table(node) and _is_from_or_join(node):
                self._derived_tables.append(node)
            elif isinstance(node, exp.UNWRAPPED_QUERIES):
                self._subqueries.append(node)
            elif isinstance(node, exp.TableColumn):
                self._table_columns.append(node)

        self._collected = True

    def _ensure_collected(self):
        if not self._collected:
            self._collect()

    def walk(self, bfs=True, prune=None):
        return walk_in_scope(self.expression, bfs=bfs, prune=None)

    def find(self, *expression_types, bfs=True):
        return find_in_scope(self.expression, expression_types, bfs=bfs)

    def find_all(self, *expression_types, bfs=True):
        return find_all_in_scope(self.expression, expression_types, bfs=bfs)

    def replace(self, old, new):
        """
        Replace `old` with `new`.
        `old` を `new` に置き換えます。

        This can be used instead of `exp.Expression.replace` to ensure the `Scope` is kept up-to-date.
        `exp.Expression.replace` の代わりにこれを使用することで、`Scope` を最新の状態に維持できます。

        Args:
            old (exp.Expression): old node
            new (exp.Expression): new node
        """
        old.replace(new)
        self.clear_cache()

    @property
    def tables(self):
        """
        List of tables in this scope.
        このスコープ内のテーブルのリスト。

        Returns:
            list[exp.Table]: tables
        """
        self._ensure_collected()
        return self._tables

    @property
    def ctes(self):
        """
        List of CTEs in this scope.
        このスコープ内の CTE のリスト。

        Returns:
            list[exp.CTE]: ctes
        """
        self._ensure_collected()
        return self._ctes

    @property
    def derived_tables(self):
        """
        List of derived tables in this scope.
        このスコープ内の派生テーブルのリスト。

        For example:
            SELECT * FROM (SELECT ...) <- that's a derived table

        Returns:
            list[exp.Subquery]: derived tables
        """
        self._ensure_collected()
        return self._derived_tables

    @property
    def udtfs(self):
        """
        List of "User Defined Tabular Functions" in this scope.
        このスコープ内の「ユーザー定義の表形式関数」のリスト。

        Returns:
            list[exp.UDTF]: UDTFs
        """
        self._ensure_collected()
        return self._udtfs

    @property
    def subqueries(self):
        """
        List of subqueries in this scope.
        このスコープ内のサブクエリのリスト。

        For example:
            SELECT * FROM x WHERE a IN (SELECT ...) <- that's a subquery

        Returns:
            list[exp.Select | exp.SetOperation]: subqueries
        """
        self._ensure_collected()
        return self._subqueries

    @property
    def stars(self) -> t.List[exp.Column | exp.Dot]:
        """
        List of star expressions (columns or dots) in this scope.
        このスコープ内のスター式 (列またはドット) のリスト。
        """
        self._ensure_collected()
        return self._stars

    @property
    def columns(self):
        """
        List of columns in this scope.
        このスコープ内の列のリスト。

        Returns:
            list[exp.Column]: Column instances in this scope, plus any
                Columns that reference this scope from correlated subqueries.
                このスコープ内の列インスタンスと、相関サブクエリからこのスコープを参照するすべての列。
        """
        if self._columns is None:
            self._ensure_collected()
            columns = self._raw_columns

            external_columns = [
                column
                for scope in itertools.chain(
                    self.subquery_scopes,
                    self.udtf_scopes,
                    (dts for dts in self.derived_table_scopes if dts.can_be_correlated),
                )
                for column in scope.external_columns
            ]

            named_selects = set(self.expression.named_selects)

            self._columns = []
            for column in columns + external_columns:
                ancestor = column.find_ancestor(
                    exp.Select,
                    exp.Qualify,
                    exp.Order,
                    exp.Having,
                    exp.Hint,
                    exp.Table,
                    exp.Star,
                    exp.Distinct,
                )
                if (
                    not ancestor
                    or column.table
                    or isinstance(ancestor, exp.Select)
                    or (isinstance(ancestor, exp.Table) and not isinstance(ancestor.this, exp.Func))
                    or (
                        isinstance(ancestor, (exp.Order, exp.Distinct))
                        and (
                            isinstance(ancestor.parent, (exp.Window, exp.WithinGroup))
                            or not isinstance(ancestor.parent, exp.Select)
                            or column.name not in named_selects
                        )
                    )
                    or (isinstance(ancestor, exp.Star) and not column.arg_key == "except_")
                ):
                    self._columns.append(column)

        return self._columns

    @property
    def table_columns(self):
        if self._table_columns is None:
            self._ensure_collected()

        return self._table_columns

    @property
    def selected_sources(self):
        """
        Mapping of nodes and sources that are actually selected from in this scope.
        このスコープ内で実際に選択されるノードとソースのマッピング。

        That is, all tables in a schema are selectable at any point. But a
        table only becomes a selected source if it's included in a FROM or JOIN clause.
        つまり、スキーマ内のすべてのテーブルはいつでも選択可能です。ただし、
        テーブルが選択されるソースとなるのは、FROM 句または JOIN 句に含まれている場合のみです。

        Returns:
            dict[str, (exp.Table|exp.Select, exp.Table|Scope)]: selected sources and nodes
        """
        if self._selected_sources is None:
            result = {}

            for name, node in self.references:
                if name in self._semi_anti_join_tables:
                    # The RHS table of SEMI/ANTI joins shouldn't be collected as a
                    # selected source
                    # SEMI/ANTI結合のRHSテーブルは、選択されたソースとして収集されるべきではない。
                    continue

                if name in result:
                    raise OptimizeError(f"Alias already used: {name}")
                if name in self.sources:
                    result[name] = (node, self.sources[name])

            self._selected_sources = result
        return self._selected_sources

    @property
    def references(self) -> t.List[t.Tuple[str, exp.Expression]]:
        if self._references is None:
            self._references = []

            for table in self.tables:
                self._references.append((table.alias_or_name, table))
            for expression in itertools.chain(self.derived_tables, self.udtfs):
                self._references.append(
                    (
                        _get_source_alias(expression),
                        expression if expression.args.get("pivots") else expression.unnest(),
                    )
                )

        return self._references

    @property
    def external_columns(self):
        """
        Columns that appear to reference sources in outer scopes.
        外部スコープ内のソースを参照するように見える列。

        Returns:
            list[exp.Column]: Column instances that don't reference
                sources in the current scope.
                現在のスコープ内のソースを参照しない列インスタンス。
        """
        if self._external_columns is None:
            if isinstance(self.expression, exp.SetOperation):
                left, right = self.union_scopes
                self._external_columns = left.external_columns + right.external_columns
            else:
                self._external_columns = [
                    c
                    for c in self.columns
                    if c.table not in self.selected_sources
                    and c.table not in self.semi_or_anti_join_tables
                ]

        return self._external_columns

    @property
    def unqualified_columns(self):
        """
        Unqualified columns in the current scope.
        現在のスコープ内の修飾されていない列。

        Returns:
             list[exp.Column]: Unqualified columns
        """
        return [c for c in self.columns if not c.table]

    @property
    def join_hints(self):
        """
        Hints that exist in the scope that reference tables
        テーブルを参照するスコープ内に存在するヒント

        Returns:
            list[exp.JoinHint]: Join hints that are referenced within the scope
            list[exp.JoinHint]: スコープ内で参照される結合ヒント
        """
        if self._join_hints is None:
            return []
        return self._join_hints

    @property
    def pivots(self):
        if not self._pivots:
            self._pivots = [
                pivot for _, node in self.references for pivot in node.args.get("pivots") or []
            ]

        return self._pivots

    @property
    def semi_or_anti_join_tables(self):
        return self._semi_anti_join_tables or set()

    def source_columns(self, source_name):
        """
        Get all columns in the current scope for a particular source.
        特定のソースの現在のスコープ内のすべての列を取得します。

        Args:
            source_name (str): Name of the source
        Returns:
            list[exp.Column]: Column instances that reference `source_name`
            list[exp.Column]: `source_name` を参照する列インスタンス
        """
        return [column for column in self.columns if column.table == source_name]

    @property
    def is_subquery(self):
        """Determine if this scope is a subquery
        このスコープがサブクエリであるかどうかを判定する"""
        return self.scope_type == ScopeType.SUBQUERY

    @property
    def is_derived_table(self):
        """Determine if this scope is a derived table
        このスコープが派生テーブルであるかどうかを判別します"""
        return self.scope_type == ScopeType.DERIVED_TABLE

    @property
    def is_union(self):
        """Determine if this scope is a union
        このスコープがユニオンであるかどうかを判定する"""
        return self.scope_type == ScopeType.UNION

    @property
    def is_cte(self):
        """Determine if this scope is a common table expression
        このスコープが共通テーブル式であるかどうかを判断します"""
        return self.scope_type == ScopeType.CTE

    @property
    def is_root(self):
        """Determine if this is the root scope
        これがルートスコープであるかどうかを判断する"""
        return self.scope_type == ScopeType.ROOT

    @property
    def is_udtf(self):
        """Determine if this scope is a UDTF (User Defined Table Function)
        このスコープがUDTF（ユーザー定義テーブル関数）であるかどうかを判断します"""
        return self.scope_type == ScopeType.UDTF

    @property
    def is_correlated_subquery(self):
        """Determine if this scope is a correlated subquery
        このスコープが相関サブクエリであるかどうかを判断します"""
        return bool(self.can_be_correlated and self.external_columns)

    def rename_source(self, old_name, new_name):
        """Rename a source in this scope
        このスコープ内のソースの名前を変更する"""
        columns = self.sources.pop(old_name or "", [])
        self.sources[new_name] = columns

    def add_source(self, name, source):
        """Add a source to this scope
        このスコープにソースを追加する"""
        self.sources[name] = source
        self.clear_cache()

    def remove_source(self, name):
        """Remove a source from this scope
        このスコープからソースを削除する"""
        self.sources.pop(name, None)
        self.clear_cache()

    def __repr__(self):
        return f"Scope<{self.expression.sql()}>"

    def traverse(self):
        """
        Traverse the scope tree from this node.
        このノードからスコープ ツリーをトラバースします。

        Yields:
            Scope: scope instances in depth-first-search post-order
            Scope: 深さ優先探索のスコープインスタンスを後順で探索する
        """
        stack = [self]
        result = []
        while stack:
            scope = stack.pop()
            result.append(scope)
            stack.extend(
                itertools.chain(
                    scope.cte_scopes,
                    scope.union_scopes,
                    scope.table_scopes,
                    scope.subquery_scopes,
                )
            )

        yield from reversed(result)

    def ref_count(self):
        """
        Count the number of times each scope in this tree is referenced.
        このツリー内の各スコープが参照される回数をカウントします。

        Returns:
            dict[int, int]: Mapping of Scope instance ID to reference count
            dict[int, int]: スコープインスタンスIDと参照カウントのマッピング
        """
        scope_ref_count = defaultdict(lambda: 0)

        for scope in self.traverse():
            for _, source in scope.selected_sources.values():
                scope_ref_count[id(source)] += 1

            for name in scope._semi_anti_join_tables:
                # semi/anti join sources are not actually selected but we still need to
                # increment their ref count to avoid them being optimized away
                # セミ/アンチ結合ソースは実際には選択されていませんが、
                # 最適化によって除去されないように参照カウントを増やす必要があります。
                if name in scope.sources:
                    scope_ref_count[id(scope.sources[name])] += 1

        return scope_ref_count


def traverse_scope(expression: exp.Expression) -> t.List[Scope]:
    """
    Traverse an expression by its "scopes".
    式を「スコープ」でトラバースします。

    "Scope" represents the current context of a Select statement.
    「スコープ」はSelect文の現在のコンテキストを表します。

    This is helpful for optimizing queries, where we need more information than
    the expression tree itself. For example, we might care about the source
    names within a subquery. Returns a list because a generator could result in
    incomplete properties which is confusing.
    これは、式ツリー自体よりも多くの情報が必要なクエリの最適化に役立ちます。例えば、
    サブクエリ内のソース名に注目したい場合などです。ジェネレーターは不完全な
    プロパティを生成する可能性があり、混乱を招く可能性があるため、リストを返します。

    Examples:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT a FROM x) AS y")
        >>> scopes = traverse_scope(expression)
        >>> scopes[0].expression.sql(), list(scopes[0].sources)
        ('SELECT a FROM x', ['x'])
        >>> scopes[1].expression.sql(), list(scopes[1].sources)
        ('SELECT a FROM (SELECT a FROM x) AS y', ['y'])

    Args:
        expression: Expression to traverse

    Returns:
        A list of the created scope instances
        作成されたスコープインスタンスのリスト
    """
    if isinstance(expression, TRAVERSABLES):
        return list(_traverse_scope(Scope(expression)))
    return []


def build_scope(expression: exp.Expression) -> t.Optional[Scope]:
    """
    Build a scope tree.
    スコープ ツリーを構築します。

    Args:
        expression: Expression to build the scope tree for.
            スコープ ツリーを構築する式。

    Returns:
        The root scope
        ルートスコープ
    """
    return seq_get(traverse_scope(expression), -1)


def _traverse_scope(scope):
    expression = scope.expression

    if isinstance(expression, exp.Select):
        yield from _traverse_select(scope)
    elif isinstance(expression, exp.SetOperation):
        yield from _traverse_ctes(scope)
        yield from _traverse_union(scope)
        return
    elif isinstance(expression, exp.Subquery):
        if scope.is_root:
            yield from _traverse_select(scope)
        else:
            yield from _traverse_subqueries(scope)
    elif isinstance(expression, exp.Table):
        yield from _traverse_tables(scope)
    elif isinstance(expression, exp.UDTF):
        yield from _traverse_udtfs(scope)
    elif isinstance(expression, exp.DDL):
        if isinstance(expression.expression, exp.Query):
            yield from _traverse_ctes(scope)
            yield from _traverse_scope(Scope(expression.expression, cte_sources=scope.cte_sources))
        return
    elif isinstance(expression, exp.DML):
        yield from _traverse_ctes(scope)
        for query in find_all_in_scope(expression, exp.Query):
            # This check ensures we don't yield the CTE/nested queries twice
            if not isinstance(query.parent, (exp.CTE, exp.Subquery)):
                yield from _traverse_scope(Scope(query, cte_sources=scope.cte_sources))
        return
    else:
        logger.warning("Cannot traverse scope %s with type '%s'", expression, type(expression))
        return

    yield scope


def _traverse_select(scope):
    yield from _traverse_ctes(scope)
    yield from _traverse_tables(scope)
    yield from _traverse_subqueries(scope)


def _traverse_union(scope):
    prev_scope = None
    union_scope_stack = [scope]
    expression_stack = [scope.expression.right, scope.expression.left]

    while expression_stack:
        expression = expression_stack.pop()
        union_scope = union_scope_stack[-1]

        new_scope = union_scope.branch(
            expression,
            outer_columns=union_scope.outer_columns,
            scope_type=ScopeType.UNION,
        )

        if isinstance(expression, exp.SetOperation):
            yield from _traverse_ctes(new_scope)

            union_scope_stack.append(new_scope)
            expression_stack.extend([expression.right, expression.left])
            continue

        for scope in _traverse_scope(new_scope):
            yield scope

        if prev_scope:
            union_scope_stack.pop()
            union_scope.union_scopes = [prev_scope, scope]
            prev_scope = union_scope

            yield union_scope
        else:
            prev_scope = scope


def _traverse_ctes(scope):
    sources = {}

    for cte in scope.ctes:
        cte_name = cte.alias

        # if the scope is a recursive cte, it must be in the form of base_case UNION recursive.
        # thus the recursive scope is the first section of the union.
        # スコープが再帰 CTE の場合、base_case UNION recursive の形式にする必要があります。
        # したがって、再帰スコープはユニオンの最初のセクションになります。
        with_ = scope.expression.args.get("with_")
        if with_ and with_.recursive:
            union = cte.this

            if isinstance(union, exp.SetOperation):
                sources[cte_name] = scope.branch(union.this, scope_type=ScopeType.CTE)

        child_scope = None

        for child_scope in _traverse_scope(
            scope.branch(
                cte.this,
                cte_sources=sources,
                outer_columns=cte.alias_column_names,
                scope_type=ScopeType.CTE,
            )
        ):
            yield child_scope

        # append the final child_scope yielded
        # 最後に得られたchild_scopeを追加する
        if child_scope:
            sources[cte_name] = child_scope
            scope.cte_scopes.append(child_scope)

    scope.sources.update(sources)
    scope.cte_sources.update(sources)


def _is_derived_table(expression: exp.Subquery) -> bool:
    """
    We represent (tbl1 JOIN tbl2) as a Subquery, but it's not really a "derived table",
    as it doesn't introduce a new scope. If an alias is present, it shadows all names
    under the Subquery, so that's one exception to this rule.
    (tbl1 JOIN tbl2) はサブクエリとして表現されていますが、新しいスコープを導入しないため、
    実際には「派生テーブル」ではありません。エイリアスが存在する場合、サブクエリの下位にある
    すべての名前がエイリアスによって隠蔽されるため、これはこのルールの例外となります。
    """
    return isinstance(expression, exp.Subquery) and bool(
        expression.alias or isinstance(expression.this, exp.UNWRAPPED_QUERIES)
    )


def _is_from_or_join(expression: exp.Expression) -> bool:
    """
    Determine if `expression` is the FROM or JOIN clause of a SELECT statement.
    `expression` が SELECT ステートメントの FROM 句または JOIN 句であるかどうかを判断します。
    """
    parent = expression.parent

    # Subqueries can be arbitrarily nested
    # サブクエリは任意にネストできる
    while isinstance(parent, exp.Subquery):
        parent = parent.parent

    return isinstance(parent, (exp.From, exp.Join))


def _traverse_tables(scope):
    sources = {}

    # Traverse FROMs, JOINs, and LATERALs in the order they are defined
    # FROM、JOIN、LATERALを定義された順序で走査します。
    expressions = []
    from_ = scope.expression.args.get("from_")
    if from_:
        expressions.append(from_.this)

    for join in scope.expression.args.get("joins") or []:
        expressions.append(join.this)

    if isinstance(scope.expression, exp.Table):
        expressions.append(scope.expression)

    expressions.extend(scope.expression.args.get("laterals") or [])

    for expression in expressions:
        if isinstance(expression, exp.Final):
            expression = expression.this
        if isinstance(expression, exp.Table):
            table_name = expression.name
            source_name = expression.alias_or_name

            if table_name in scope.sources and not expression.db:
                # This is a reference to a parent source (e.g. a CTE), not an actual table, unless
                # it is pivoted, because then we get back a new table and hence a new source.
                # これは、ピボットされない限り、実際のテーブルではなく、親ソース (CTE など) への参照です。
                # ピボットされると、新しいテーブル、つまり新しいソースが返されるためです。
                pivots = expression.args.get("pivots")
                if pivots:
                    sources[pivots[0].alias] = expression
                else:
                    sources[source_name] = scope.sources[table_name]
            elif source_name in sources:
                sources[find_new_name(sources, table_name)] = expression
            else:
                sources[source_name] = expression

            # Make sure to not include the joins twice
            # 結合を2回含めないようにしてください
            if expression is not scope.expression:
                expressions.extend(join.this for join in expression.args.get("joins") or [])

            continue

        if not isinstance(expression, exp.DerivedTable):
            continue

        if isinstance(expression, exp.UDTF):
            lateral_sources = sources
            scope_type = ScopeType.UDTF
            scopes = scope.udtf_scopes
        elif _is_derived_table(expression):
            lateral_sources = None
            scope_type = ScopeType.DERIVED_TABLE
            scopes = scope.derived_table_scopes
            expressions.extend(join.this for join in expression.args.get("joins") or [])
        else:
            # Makes sure we check for possible sources in nested table constructs
            # ネストされたテーブル構造内の可能なソースを確認する
            expressions.append(expression.this)
            expressions.extend(join.this for join in expression.args.get("joins") or [])
            continue

        child_scope = None

        for child_scope in _traverse_scope(
            scope.branch(
                expression,
                lateral_sources=lateral_sources,
                outer_columns=expression.alias_column_names,
                scope_type=scope_type,
            )
        ):
            yield child_scope

            # Tables without aliases will be set as ""
            # This shouldn't be a problem once qualify_columns runs, as it adds aliases on everything.
            # Until then, this means that only a single, unaliased derived table is allowed (rather,
            # the latest one wins.
            # エイリアスのないテーブルは "" として設定されます。
            # 一度 qualified_columns が実行されると、すべてのテーブルにエイリアスが追加されるため、
            # これは問題になりません。それまでは、エイリアスのない派生テーブルは 1 つだけ許可されます
            #  (最新のテーブルが優先されます)。
            sources[_get_source_alias(expression)] = child_scope

        # append the final child_scope yielded
        # 最後に得られたchild_scopeを追加する
        if child_scope:
            scopes.append(child_scope)
            scope.table_scopes.append(child_scope)

    scope.sources.update(sources)


def _traverse_subqueries(scope):
    for subquery in scope.subqueries:
        top = None
        for child_scope in _traverse_scope(scope.branch(subquery, scope_type=ScopeType.SUBQUERY)):
            yield child_scope
            top = child_scope
        scope.subquery_scopes.append(top)


def _traverse_udtfs(scope):
    if isinstance(scope.expression, exp.Unnest):
        expressions = scope.expression.expressions
    elif isinstance(scope.expression, exp.Lateral):
        expressions = [scope.expression.this]
    else:
        expressions = []

    sources = {}
    for expression in expressions:
        if _is_derived_table(expression):
            top = None
            for child_scope in _traverse_scope(
                scope.branch(
                    expression,
                    scope_type=ScopeType.SUBQUERY,
                    outer_columns=expression.alias_column_names,
                )
            ):
                yield child_scope
                top = child_scope
                sources[_get_source_alias(expression)] = child_scope

            scope.subquery_scopes.append(top)

    scope.sources.update(sources)


def walk_in_scope(expression, bfs=True, prune=None):
    """
    Returns a generator object which visits all nodes in the syntrax tree, stopping at
    nodes that start child scopes.
    構文ツリー内のすべてのノードを訪問し、子スコープを開始するノードで停止するジェネレーター 
    オブジェクトを返します。

    Args:
        expression (exp.Expression):
        bfs (bool): if set to True the BFS traversal order will be applied,
            otherwise the DFS traversal will be used instead.
            True に設定すると BFS トラバーサル順序が適用され、それ以外の場合は代わりに 
            DFS トラバーサルが使用されます。
        prune ((node, parent, arg_key) -> bool): callable that returns True if
            the generator should stop traversing this branch of the tree.
            ジェネレーターがツリーのこのブランチのトラバースを停止する必要がある場合に 
            True を返す呼び出し可能オブジェクト。

    Yields:
        tuple[exp.Expression, Optional[exp.Expression], str]: node, parent, arg key
        tuple[exp.Expression, Optional[exp.Expression], str]: ノード、親、引数キー
    """
    # We'll use this variable to pass state into the dfs generator.
    # Whenever we set it to True, we exclude a subtree from traversal.
    # この変数を使って、DFSジェネレータに状態を渡します。
    # Trueに設定すると、サブツリーがトラバーサルから除外されます。
    crossed_scope_boundary = False

    for node in expression.walk(
        bfs=bfs, prune=lambda n: crossed_scope_boundary or (prune and prune(n))
    ):
        crossed_scope_boundary = False

        yield node

        if node is expression:
            continue

        if (
            isinstance(node, exp.CTE)
            or (
                isinstance(node.parent, (exp.From, exp.Join, exp.Subquery))
                and _is_derived_table(node)
            )
            or (isinstance(node.parent, exp.UDTF) and isinstance(node, exp.Query))
            or isinstance(node, exp.UNWRAPPED_QUERIES)
        ):
            crossed_scope_boundary = True

            if isinstance(node, (exp.Subquery, exp.UDTF)):
                # The following args are not actually in the inner scope, so we should visit them
                # 次の引数は実際には内部スコープにないので、それらを参照する必要があります。
                for key in ("joins", "laterals", "pivots"):
                    for arg in node.args.get(key) or []:
                        yield from walk_in_scope(arg, bfs=bfs)


def find_all_in_scope(expression, expression_types, bfs=True):
    """
    Returns a generator object which visits all nodes in this scope and only yields those that
    match at least one of the specified expression types.
    このスコープ内のすべてのノードを参照し、指定された式タイプの少なくとも1つに一致するノードのみを
    生成するジェネレーターオブジェクトを返します。

    This does NOT traverse into subscopes.
    これはサブスコープを走査しません。

    Args:
        expression (exp.Expression):
        expression_types (tuple[type]|type): the expression type(s) to match.
        bfs (bool): True to use breadth-first search, False to use depth-first.
            幅優先探索を使用する場合は True、深さ優先探索を使用する場合は False。

    Yields:
        exp.Expression: nodes
    """
    for expression in walk_in_scope(expression, bfs=bfs):
        if isinstance(expression, tuple(ensure_collection(expression_types))):
            yield expression


def find_in_scope(expression, expression_types, bfs=True):
    """
    Returns the first node in this scope which matches at least one of the specified types.
    このスコープ内で、指定された型の少なくとも1つに一致する最初のノードを返します。

    This does NOT traverse into subscopes.
    これはサブスコープを走査しません。

    Args:
        expression (exp.Expression):
        expression_types (tuple[type]|type): the expression type(s) to match.
        bfs (bool): True to use breadth-first search, False to use depth-first.
            幅優先探索を使用する場合は True、深さ優先探索を使用する場合は False。

    Returns:
        exp.Expression: the node which matches the criteria or None if no node matching
        the criteria was found.
        exp.Expression: 条件に一致するノード。条件に一致するノードが見つからなかった場合は None。
    """
    return next(find_all_in_scope(expression, expression_types, bfs=bfs), None)


def _get_source_alias(expression):
    alias_arg = expression.args.get("alias")
    alias_name = expression.alias

    if not alias_name and isinstance(alias_arg, exp.TableAlias) and len(alias_arg.columns) == 1:
        alias_name = alias_arg.columns[0].name

    return alias_name
