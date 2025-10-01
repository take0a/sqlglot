from __future__ import annotations

import inspect
import typing as t

from sqlglot import Schema, exp
from sqlglot.dialects.dialect import DialectType
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.canonicalize import canonicalize
from sqlglot.optimizer.eliminate_ctes import eliminate_ctes
from sqlglot.optimizer.eliminate_joins import eliminate_joins
from sqlglot.optimizer.eliminate_subqueries import eliminate_subqueries
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.optimize_joins import optimize_joins
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.qualify_columns import quote_identifiers
from sqlglot.optimizer.simplify import simplify
from sqlglot.optimizer.unnest_subqueries import unnest_subqueries
from sqlglot.schema import ensure_schema

RULES = (
    qualify,
    pushdown_projections,
    normalize,
    unnest_subqueries,
    pushdown_predicates,
    optimize_joins,
    eliminate_subqueries,
    merge_subqueries,
    eliminate_joins,
    eliminate_ctes,
    quote_identifiers,
    annotate_types,
    canonicalize,
    simplify,
)


def optimize(
    expression: str | exp.Expression,
    schema: t.Optional[dict | Schema] = None,
    db: t.Optional[str | exp.Identifier] = None,
    catalog: t.Optional[str | exp.Identifier] = None,
    dialect: DialectType = None,
    rules: t.Sequence[t.Callable] = RULES,
    **kwargs,
) -> exp.Expression:
    """
    Rewrite a sqlglot AST into an optimized form.
    sqlglot AST を最適化された形式に書き換えます。

    Args:
        expression: expression to optimize 最適化する式
        schema: database schema.
            This can either be an instance of `sqlglot.optimizer.Schema` or a mapping in one of
            the following forms:
            データベーススキーマ。これは `sqlglot.optimizer.Schema` のインスタンス、
            または次のいずれかの形式のマッピングになります。
                1. {table: {col: type}}
                2. {db: {table: {col: type}}}
                3. {catalog: {db: {table: {col: type}}}}
            If no schema is provided then the default schema defined at `sqlgot.schema` will be used
            スキーマが指定されていない場合は、`sqlgot.schema` で定義されたデフォルトのスキーマが使用されます。
        db: specify the default database, as might be set by a `USE DATABASE db` statement
            `USE DATABASE db` ステートメントで設定されるデフォルトのデータベースを指定します。
        catalog: specify the default catalog, as might be set by a `USE CATALOG c` statement
            `USE CATALOG c` 文で設定されるデフォルトのカタログを指定します。
        dialect: The dialect to parse the sql string. SQL 文字列を解析する方言。
        rules: sequence of optimizer rules to use.
            Many of the rules require tables and columns to be qualified.
            Do not remove `qualify` from the sequence of rules unless you know what you're doing!
            使用するオプティマイザルールのシーケンス。
            多くのルールでは、テーブルと列を修飾する必要があります。
            何をしているのかよく理解していない限り、ルールのシーケンスから `qualify` を削除しないでください。
        **kwargs: If a rule has a keyword argument with a same name in **kwargs, it will be passed in.
            ルールに **kwargs と同じ名前のキーワード引数がある場合は、それが渡されます。

    Returns:
        The optimized expression. 最適化された式
    """
    schema = ensure_schema(schema, dialect=dialect)
    possible_kwargs = {
        "db": db,
        "catalog": catalog,
        "schema": schema,
        "dialect": dialect,
        "isolate_tables": True,  # needed for other optimizations to perform well
        "quote_identifiers": False,
        **kwargs,
    }

    optimized = exp.maybe_parse(expression, dialect=dialect, copy=True)
    for rule in rules:
        # Find any additional rule parameters, beyond `expression`
        rule_params = inspect.getfullargspec(rule).args
        rule_kwargs = {
            param: possible_kwargs[param] for param in rule_params if param in possible_kwargs
        }
        optimized = rule(optimized, **rule_kwargs)

    return optimized
