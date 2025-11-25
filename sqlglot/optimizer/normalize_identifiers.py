from __future__ import annotations

import typing as t

from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, DialectType

if t.TYPE_CHECKING:
    from sqlglot._typing import E


@t.overload
def normalize_identifiers(
    expression: E, dialect: DialectType = None, store_original_column_identifiers: bool = False
) -> E: ...


@t.overload
def normalize_identifiers(
    expression: str, dialect: DialectType = None, store_original_column_identifiers: bool = False
) -> exp.Identifier: ...


def normalize_identifiers(expression, dialect=None, store_original_column_identifiers=False):
    """
    Normalize identifiers by converting them to either lower or upper case,
    ensuring the semantics are preserved in each case (e.g. by respecting
    case-sensitivity).
    識別子を小文字または大文字に変換することで正規化し、それぞれのケースで意味が
    保持されるようにします（例：大文字と小文字の区別を尊重する）。

    This transformation reflects how identifiers would be resolved by the engine corresponding
    to each SQL dialect, and plays a very important role in the standardization of the AST.
    この変換は、各SQL方言に対応するエンジンが識別子をどのように解決するかを反映しており、
    ASTの標準化において非常に重要な役割を果たします。

    It's possible to make this a no-op by adding a special comment next to the
    identifier of interest:
    対象の識別子の横に特別なコメントを追加することで、この処理をno-opにすることも可能です。

        SELECT a /* sqlglot.meta case_sensitive */ FROM table

    In this example, the identifier `a` will not be normalized.
    この例では、識別子 `a` は正規化されません。

    Note:
        Some dialects (e.g. DuckDB) treat all identifiers as case-insensitive even
        when they're quoted, so in these cases all identifiers are normalized.
        一部の方言 (DuckDB など) では、引用符で囲まれている場合でもすべての識別子の大文字と
        小文字を区別しないものとして扱うため、このような場合にはすべての識別子が正規化されます。

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one('SELECT Bar.A AS A FROM "Foo".Bar')
        >>> normalize_identifiers(expression).sql()
        'SELECT bar.a AS a FROM "Foo".bar'
        >>> normalize_identifiers("foo", dialect="snowflake").sql(dialect="snowflake")
        'FOO'

    Args:
        expression: The expression to transform.
            変換する式。
        dialect: The dialect to use in order to decide how to normalize identifiers.
            識別子を正規化する方法を決定するために使用する方言。
        store_original_column_identifiers: Whether to store the original column identifiers in
            the meta data of the expression in case we want to undo the normalization at a later point.
            後で正規化を元に戻す場合に備えて、式のメタデータに元の列識別子を保存するかどうか。

    Returns:
        The transformed expression.
        変形された式。
    """
    dialect = Dialect.get_or_raise(dialect)

    if isinstance(expression, str):
        expression = exp.parse_identifier(expression, dialect=dialect)

    for node in expression.walk(prune=lambda n: n.meta.get("case_sensitive")):
        if not node.meta.get("case_sensitive"):
            if store_original_column_identifiers and isinstance(node, exp.Column):
                # TODO: This does not handle non-column cases, e.g PARSE_JSON(...).key
                parent = node
                while parent and isinstance(parent.parent, exp.Dot):
                    parent = parent.parent

                node.meta["dot_parts"] = [p.name for p in parent.parts]

            dialect.normalize_identifier(node)

    return expression
