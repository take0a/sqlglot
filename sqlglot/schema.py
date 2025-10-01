from __future__ import annotations

import abc
import typing as t

from sqlglot import expressions as exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.errors import SchemaError
from sqlglot.helper import dict_depth, first
from sqlglot.trie import TrieResult, in_trie, new_trie

if t.TYPE_CHECKING:
    from sqlglot.dialects.dialect import DialectType

    ColumnMapping = t.Union[t.Dict, str, t.List]


class Schema(abc.ABC):
    """Abstract base class for database schemas
    データベーススキーマの抽象基本クラス"""

    dialect: DialectType

    @abc.abstractmethod
    def add_table(
        self,
        table: exp.Table | str,
        column_mapping: t.Optional[ColumnMapping] = None,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
        match_depth: bool = True,
    ) -> None:
        """
        Register or update a table. Some implementing classes may require column information to also be provided.
        The added table must have the necessary number of qualifiers in its path to match the schema's nesting level.
        テーブルを登録または更新します。実装クラスによっては、列情報も提供する必要がある場合があります。
        追加するテーブルには、スキーマのネストレベルに一致するように、パスに必要な数の修飾子が必要です。

        Args:
            table: the `Table` expression instance or string representing the table.
                テーブルを表す `Table` 式インスタンスまたは文字列。
            column_mapping: a column mapping that describes the structure of the table.
                テーブルの構造を記述する列マッピング。
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
                `table` が文字列の場合に解析するために使用される SQL 方言。
            normalize: whether to normalize identifiers according to the dialect of interest.
                関心のある方言に応じて識別子を正規化するかどうか。
            match_depth: whether to enforce that the table must match the schema's depth or not.
                テーブルがスキーマの深さと一致することを強制するかどうか。
        """

    @abc.abstractmethod
    def column_names(
        self,
        table: exp.Table | str,
        only_visible: bool = False,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> t.Sequence[str]:
        """
        Get the column names for a table.
        テーブルの列名を取得する

        Args:
            table: the `Table` expression instance.
                `Table` 式インスタンス。
            only_visible: whether to include invisible columns.
                非表示の列を含めるかどうか。
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
                `table` が文字列の場合に解析するために使用される SQL 方言。
            normalize: whether to normalize identifiers according to the dialect of interest.
                関心のある方言に応じて識別子を正規化するかどうか。

        Returns:
            The sequence of column names.
            列名の順序。
        """

    @abc.abstractmethod
    def get_column_type(
        self,
        table: exp.Table | str,
        column: exp.Column | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.DataType:
        """
        Get the `sqlglot.exp.DataType` type of a column in the schema.
        スキーマ内の列の `sqlglot.exp.DataType` タイプを取得します。

        Args:
            table: the source table. 元のテーブル。
            column: the target column. 対象のカラム。
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
                `table` が文字列の場合に解析するために使用される SQL 方言。
            normalize: whether to normalize identifiers according to the dialect of interest.
                関心のある方言に応じて識別子を正規化するかどうか。

        Returns:
            The resulting column type.
            結果の列タイプ。
        """

    def has_column(
        self,
        table: exp.Table | str,
        column: exp.Column | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> bool:
        """
        Returns whether `column` appears in `table`'s schema.
        `column` が `table` のスキーマに現れるかどうかを返します。

        Args:
            table: the source table. 元のテーブル。
            column: the target column. 対象のカラム。
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
                `table` が文字列の場合に解析するために使用される SQL 方言。
            normalize: whether to normalize identifiers according to the dialect of interest.
                関心のある方言に応じて識別子を正規化するかどうか。

        Returns:
            True if the column appears in the schema, False otherwise.
            列がスキーマに表示される場合は True、それ以外の場合は False。
        """
        name = column if isinstance(column, str) else column.name
        return name in self.column_names(table, dialect=dialect, normalize=normalize)

    @property
    @abc.abstractmethod
    def supported_table_args(self) -> t.Tuple[str, ...]:
        """
        Table arguments this schema support, e.g. `("this", "db", "catalog")`
        このスキーマがサポートするテーブル引数、例: `("this", "db", "catalog")`
        """

    @property
    def empty(self) -> bool:
        """Returns whether the schema is empty.
        スキーマが空かどうかを返します。"""
        return True


class AbstractMappingSchema:
    def __init__(
        self,
        mapping: t.Optional[t.Dict] = None,
    ) -> None:
        self.mapping = mapping or {}
        self.mapping_trie = new_trie(
            tuple(reversed(t)) for t in flatten_schema(self.mapping, depth=self.depth())
        )
        self._supported_table_args: t.Tuple[str, ...] = tuple()

    @property
    def empty(self) -> bool:
        return not self.mapping

    def depth(self) -> int:
        return dict_depth(self.mapping)

    @property
    def supported_table_args(self) -> t.Tuple[str, ...]:
        if not self._supported_table_args and self.mapping:
            depth = self.depth()

            if not depth:  # None
                self._supported_table_args = tuple()
            elif 1 <= depth <= 3:
                self._supported_table_args = exp.TABLE_PARTS[:depth]
            else:
                raise SchemaError(f"Invalid mapping shape. Depth: {depth}")

        return self._supported_table_args

    def table_parts(self, table: exp.Table) -> t.List[str]:
        return [part.name for part in reversed(table.parts)]

    def find(
        self, table: exp.Table, raise_on_missing: bool = True, ensure_data_types: bool = False
    ) -> t.Optional[t.Any]:
        """
        Returns the schema of a given table.
        指定されたテーブルのスキーマを返します。

        Args:
            table: the target table. 対象のテーブル
            raise_on_missing: whether to raise in case the schema is not found.
                スキーマが見つからない場合に例外を発生するかどうか。
            ensure_data_types: whether to convert `str` types to their `DataType` equivalents.
                `str` 型を `DataType` の同等の型に変換するかどうか。

        Returns:
            The schema of the target table. 対象のテーブルのスキーマ。
        """
        parts = self.table_parts(table)[0 : len(self.supported_table_args)]
        value, trie = in_trie(self.mapping_trie, parts)

        if value == TrieResult.FAILED:
            return None

        if value == TrieResult.PREFIX:
            possibilities = flatten_schema(trie)

            if len(possibilities) == 1:
                parts.extend(possibilities[0])
            else:
                message = ", ".join(".".join(parts) for parts in possibilities)
                if raise_on_missing:
                    raise SchemaError(f"Ambiguous mapping for {table}: {message}.")
                return None

        return self.nested_get(parts, raise_on_missing=raise_on_missing)

    def nested_get(
        self, parts: t.Sequence[str], d: t.Optional[t.Dict] = None, raise_on_missing=True
    ) -> t.Optional[t.Any]:
        return nested_get(
            d or self.mapping,
            *zip(self.supported_table_args, reversed(parts)),
            raise_on_missing=raise_on_missing,
        )


class MappingSchema(AbstractMappingSchema, Schema):
    """
    Schema based on a nested mapping.
    ネストされたマッピングに基づくスキーマ。

    Args:
        schema: Mapping in one of the following forms:
            次のいずれかの形式でマッピングします。
            1. {table: {col: type}}
            2. {db: {table: {col: type}}}
            3. {catalog: {db: {table: {col: type}}}}
            4. None - Tables will be added later
        visible: Optional mapping of which columns in the schema are visible. If not provided, all columns
            are assumed to be visible. The nesting should mirror that of the schema:
            スキーマ内のどの列が表示されるかを示すオプションのマッピング。指定しない場合は、
            すべての列が表示されるものとみなされます。ネストはスキーマのネストを反映する必要があります。
            1. {table: set(*cols)}}
            2. {db: {table: set(*cols)}}}
            3. {catalog: {db: {table: set(*cols)}}}}
        dialect: The dialect to be used for custom type mappings & parsing string arguments.
        normalize: Whether to normalize identifier names according to the given dialect or not.
    """

    def __init__(
        self,
        schema: t.Optional[t.Dict] = None,
        visible: t.Optional[t.Dict] = None,
        dialect: DialectType = None,
        normalize: bool = True,
    ) -> None:
        self.dialect = dialect
        self.visible = {} if visible is None else visible
        self.normalize = normalize
        self._type_mapping_cache: t.Dict[str, exp.DataType] = {}
        self._depth = 0
        schema = {} if schema is None else schema

        super().__init__(self._normalize(schema) if self.normalize else schema)

    @classmethod
    def from_mapping_schema(cls, mapping_schema: MappingSchema) -> MappingSchema:
        return MappingSchema(
            schema=mapping_schema.mapping,
            visible=mapping_schema.visible,
            dialect=mapping_schema.dialect,
            normalize=mapping_schema.normalize,
        )

    def find(
        self, table: exp.Table, raise_on_missing: bool = True, ensure_data_types: bool = False
    ) -> t.Optional[t.Any]:
        schema = super().find(
            table, raise_on_missing=raise_on_missing, ensure_data_types=ensure_data_types
        )
        if ensure_data_types and isinstance(schema, dict):
            schema = {
                col: self._to_data_type(dtype) if isinstance(dtype, str) else dtype
                for col, dtype in schema.items()
            }

        return schema

    def copy(self, **kwargs) -> MappingSchema:
        return MappingSchema(
            **{  # type: ignore
                "schema": self.mapping.copy(),
                "visible": self.visible.copy(),
                "dialect": self.dialect,
                "normalize": self.normalize,
                **kwargs,
            }
        )

    def add_table(
        self,
        table: exp.Table | str,
        column_mapping: t.Optional[ColumnMapping] = None,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
        match_depth: bool = True,
    ) -> None:
        """
        Register or update a table. Updates are only performed if a new column mapping is provided.
        The added table must have the necessary number of qualifiers in its path to match the schema's nesting level.
        テーブルを登録または更新します。更新は、新しい列マッピングが指定されている場合にのみ実行されます。
        追加されたテーブルのパスには、スキーマのネストレベルに一致するように必要な数の修飾子が含まれている必要があります。

        Args:
            table: the `Table` expression instance or string representing the table.
                テーブルを表す `Table` 式インスタンスまたは文字列。
            column_mapping: a column mapping that describes the structure of the table.
                テーブルの構造を記述する列マッピング。
            dialect: the SQL dialect that will be used to parse `table` if it's a string.
                `table` が文字列の場合に解析するために使用される SQL 方言。
            normalize: whether to normalize identifiers according to the dialect of interest.
                関心のある方言に応じて識別子を正規化するかどうか。
            match_depth: 
                テーブルがスキーマの深さと一致することを強制するかどうか。
        """
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)

        if match_depth and not self.empty and len(normalized_table.parts) != self.depth():
            raise SchemaError(
                f"Table {normalized_table.sql(dialect=self.dialect)} must match the "
                f"schema's nesting level: {self.depth()}."
            )

        normalized_column_mapping = {
            self._normalize_name(key, dialect=dialect, normalize=normalize): value
            for key, value in ensure_column_mapping(column_mapping).items()
        }

        schema = self.find(normalized_table, raise_on_missing=False)
        if schema and not normalized_column_mapping:
            return

        parts = self.table_parts(normalized_table)

        nested_set(self.mapping, tuple(reversed(parts)), normalized_column_mapping)
        new_trie([parts], self.mapping_trie)

    def column_names(
        self,
        table: exp.Table | str,
        only_visible: bool = False,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> t.List[str]:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)

        schema = self.find(normalized_table)
        if schema is None:
            return []

        if not only_visible or not self.visible:
            return list(schema)

        visible = self.nested_get(self.table_parts(normalized_table), self.visible) or []
        return [col for col in schema if col in visible]

    def get_column_type(
        self,
        table: exp.Table | str,
        column: exp.Column | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.DataType:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)

        normalized_column_name = self._normalize_name(
            column if isinstance(column, str) else column.this, dialect=dialect, normalize=normalize
        )

        table_schema = self.find(normalized_table, raise_on_missing=False)
        if table_schema:
            column_type = table_schema.get(normalized_column_name)

            if isinstance(column_type, exp.DataType):
                return column_type
            elif isinstance(column_type, str):
                return self._to_data_type(column_type, dialect=dialect)

        return exp.DataType.build("unknown")

    def has_column(
        self,
        table: exp.Table | str,
        column: exp.Column | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> bool:
        normalized_table = self._normalize_table(table, dialect=dialect, normalize=normalize)

        normalized_column_name = self._normalize_name(
            column if isinstance(column, str) else column.this, dialect=dialect, normalize=normalize
        )

        table_schema = self.find(normalized_table, raise_on_missing=False)
        return normalized_column_name in table_schema if table_schema else False

    def _normalize(self, schema: t.Dict) -> t.Dict:
        """
        Normalizes all identifiers in the schema.
        スキーマ内のすべての識別子を正規化します。

        Args:
            schema: the schema to normalize. 正規化するスキーマ。

        Returns:
            The normalized schema mapping. 正規化されたスキーマ マッピング。
        """
        normalized_mapping: t.Dict = {}
        flattened_schema = flatten_schema(schema)
        error_msg = "Table {} must match the schema's nesting level: {}."

        for keys in flattened_schema:
            columns = nested_get(schema, *zip(keys, keys))

            if not isinstance(columns, dict):
                raise SchemaError(error_msg.format(".".join(keys[:-1]), len(flattened_schema[0])))
            if not columns:
                raise SchemaError(f"Table {'.'.join(keys[:-1])} must have at least one column")
            if isinstance(first(columns.values()), dict):
                raise SchemaError(
                    error_msg.format(
                        ".".join(keys + flatten_schema(columns)[0]), len(flattened_schema[0])
                    ),
                )

            normalized_keys = [self._normalize_name(key, is_table=True) for key in keys]
            for column_name, column_type in columns.items():
                nested_set(
                    normalized_mapping,
                    normalized_keys + [self._normalize_name(column_name)],
                    column_type,
                )

        return normalized_mapping

    def _normalize_table(
        self,
        table: exp.Table | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.Table:
        dialect = dialect or self.dialect
        normalize = self.normalize if normalize is None else normalize

        normalized_table = exp.maybe_parse(table, into=exp.Table, dialect=dialect, copy=normalize)

        if normalize:
            for part in normalized_table.parts:
                if isinstance(part, exp.Identifier):
                    part.replace(
                        normalize_name(part, dialect=dialect, is_table=True, normalize=normalize)
                    )

        return normalized_table

    def _normalize_name(
        self,
        name: str | exp.Identifier,
        dialect: DialectType = None,
        is_table: bool = False,
        normalize: t.Optional[bool] = None,
    ) -> str:
        return normalize_name(
            name,
            dialect=dialect or self.dialect,
            is_table=is_table,
            normalize=self.normalize if normalize is None else normalize,
        ).name

    def depth(self) -> int:
        if not self.empty and not self._depth:
            # The columns themselves are a mapping, but we don't want to include those
            self._depth = super().depth() - 1
        return self._depth

    def _to_data_type(self, schema_type: str, dialect: DialectType = None) -> exp.DataType:
        """
        Convert a type represented as a string to the corresponding `sqlglot.exp.DataType` object.
        文字列として表される型を対応する `sqlglot.exp.DataType` オブジェクトに変換します。

        Args:
            schema_type: the type we want to convert. 変換したい型。
            dialect: the SQL dialect that will be used to parse `schema_type`, if needed.
                必要に応じて、`schema_type` を解析するために使用される SQL 方言。

        Returns:
            The resulting expression type. 結果の式の型。
        """
        if schema_type not in self._type_mapping_cache:
            dialect = dialect or self.dialect
            udt = Dialect.get_or_raise(dialect).SUPPORTS_USER_DEFINED_TYPES

            try:
                expression = exp.DataType.build(schema_type, dialect=dialect, udt=udt)
                self._type_mapping_cache[schema_type] = expression
            except AttributeError:
                in_dialect = f" in dialect {dialect}" if dialect else ""
                raise SchemaError(f"Failed to build type '{schema_type}'{in_dialect}.")

        return self._type_mapping_cache[schema_type]


def normalize_name(
    identifier: str | exp.Identifier,
    dialect: DialectType = None,
    is_table: bool = False,
    normalize: t.Optional[bool] = True,
) -> exp.Identifier:
    if isinstance(identifier, str):
        identifier = exp.parse_identifier(identifier, dialect=dialect)

    if not normalize:
        return identifier

    # this is used for normalize_identifier, bigquery has special rules pertaining tables
    identifier.meta["is_table"] = is_table
    return Dialect.get_or_raise(dialect).normalize_identifier(identifier)


def ensure_schema(schema: Schema | t.Optional[t.Dict], **kwargs: t.Any) -> Schema:
    if isinstance(schema, Schema):
        return schema

    return MappingSchema(schema, **kwargs)


def ensure_column_mapping(mapping: t.Optional[ColumnMapping]) -> t.Dict:
    if mapping is None:
        return {}
    elif isinstance(mapping, dict):
        return mapping
    elif isinstance(mapping, str):
        col_name_type_strs = [x.strip() for x in mapping.split(",")]
        return {
            name_type_str.split(":")[0].strip(): name_type_str.split(":")[1].strip()
            for name_type_str in col_name_type_strs
        }
    elif isinstance(mapping, list):
        return {x.strip(): None for x in mapping}

    raise ValueError(f"Invalid mapping provided: {type(mapping)}")


def flatten_schema(
    schema: t.Dict, depth: t.Optional[int] = None, keys: t.Optional[t.List[str]] = None
) -> t.List[t.List[str]]:
    tables = []
    keys = keys or []
    depth = dict_depth(schema) - 1 if depth is None else depth

    for k, v in schema.items():
        if depth == 1 or not isinstance(v, dict):
            tables.append(keys + [k])
        elif depth >= 2:
            tables.extend(flatten_schema(v, depth - 1, keys + [k]))

    return tables


def nested_get(
    d: t.Dict, *path: t.Tuple[str, str], raise_on_missing: bool = True
) -> t.Optional[t.Any]:
    """
    Get a value for a nested dictionary.
    ネストされた辞書の値を取得します。

    Args:
        d: the dictionary to search. 検索する辞書。
        *path: tuples of (name, key), where:
            `key` is the key in the dictionary to get.
            `name` is a string to use in the error if `key` isn't found.
             (名前、キー) のタプル。ここで:
            `key` は取得する辞書のキーです。
            `name` は `key` が見つからない場合にエラーで使用する文字列です。

    Returns:
        The value or None if it doesn't exist.
        値、または存在しない場合は None。
    """
    for name, key in path:
        d = d.get(key)  # type: ignore
        if d is None:
            if raise_on_missing:
                name = "table" if name == "this" else name
                raise ValueError(f"Unknown {name}: {key}")
            return None

    return d


def nested_set(d: t.Dict, keys: t.Sequence[str], value: t.Any) -> t.Dict:
    """
    In-place set a value for a nested dictionary
    ネストされた辞書の値をその場で設定する

    Example:
        >>> nested_set({}, ["top_key", "second_key"], "value")
        {'top_key': {'second_key': 'value'}}

        >>> nested_set({"top_key": {"third_key": "third_value"}}, ["top_key", "second_key"], "value")
        {'top_key': {'third_key': 'third_value', 'second_key': 'value'}}

    Args:
        d: dictionary to update. 更新する辞書。
        keys: the keys that makeup the path to `value`.
            `value` へのパスを構成するキー。
        value: the value to set in the dictionary for the given key path.
            指定されたキーパスの辞書に設定する値。

    Returns:
        The (possibly) updated dictionary. （おそらく）更新された辞書。
    """
    if not keys:
        return d

    if len(keys) == 1:
        d[keys[0]] = value
        return d

    subd = d
    for key in keys[:-1]:
        if key not in subd:
            subd = subd.setdefault(key, {})
        else:
            subd = subd[key]

    subd[keys[-1]] = value
    return d
