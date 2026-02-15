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

    @property
    def dialect(self) -> t.Optional[Dialect]:
        """
        Returns None by default. Subclasses that require dialect-specific
        behavior should override this property.
        """
        return None

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

    def get_udf_type(
        self,
        udf: exp.Anonymous | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.DataType:
        """
        Get the return type of a UDF.

        Args:
            udf: the UDF expression or string.
            dialect: the SQL dialect for parsing string arguments.
            normalize: whether to normalize identifiers.

        Returns:
            The return type as a DataType, or UNKNOWN if not found.
        """
        return exp.DataType.build("unknown")

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
        udf_mapping: t.Optional[t.Dict] = None,
    ) -> None:
        self.mapping = mapping or {}
        self.mapping_trie = new_trie(
            tuple(reversed(t)) for t in flatten_schema(self.mapping, depth=self.depth())
        )

        self.udf_mapping = udf_mapping or {}
        self.udf_trie = new_trie(
            tuple(reversed(t)) for t in flatten_schema(self.udf_mapping, depth=self.udf_depth())
        )

        self._supported_table_args: t.Tuple[str, ...] = tuple()

    @property
    def empty(self) -> bool:
        return not self.mapping

    def depth(self) -> int:
        return dict_depth(self.mapping)

    def udf_depth(self) -> int:
        return dict_depth(self.udf_mapping)

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
        return [p.name for p in reversed(table.parts)]

    def udf_parts(self, udf: exp.Anonymous) -> t.List[str]:
        # a.b.c(...) is represented as Dot(Dot(a, b), Anonymous(c, ...))
        parent = udf.parent
        parts = [p.name for p in parent.flatten()] if isinstance(parent, exp.Dot) else [udf.name]
        return list(reversed(parts))[0 : self.udf_depth()]

    def _find_in_trie(
        self,
        parts: t.List[str],
        trie: t.Dict,
        raise_on_missing: bool,
    ) -> t.Optional[t.List[str]]:
        value, trie = in_trie(trie, parts)

        if value == TrieResult.FAILED:
            return None

        if value == TrieResult.PREFIX:
            possibilities = flatten_schema(trie)

            if len(possibilities) == 1:
                parts.extend(possibilities[0])
            else:
                if raise_on_missing:
                    joined_parts = ".".join(parts)
                    message = ", ".join(".".join(p) for p in possibilities)
                    raise SchemaError(f"Ambiguous mapping for {joined_parts}: {message}.")

                return None

        return parts

    def find(
        self, table: exp.Table, raise_on_missing: bool = True, ensure_data_types: bool = False
    ) -> t.Optional[t.Any]:
        """
        Returns the schema of a given table.

        Args:
            table: the target table.
            raise_on_missing: whether to raise in case the schema is not found.
            ensure_data_types: whether to convert `str` types to their `DataType` equivalents.

        Returns:
            The schema of the target table.
        """
        parts = self.table_parts(table)[0 : len(self.supported_table_args)]
        resolved_parts = self._find_in_trie(parts, self.mapping_trie, raise_on_missing)

        if resolved_parts is None:
            return None

        return self.nested_get(resolved_parts, raise_on_missing=raise_on_missing)

    def find_udf(self, udf: exp.Anonymous, raise_on_missing: bool = False) -> t.Optional[t.Any]:
        """
        Returns the return type of a given UDF.

        Args:
            udf: the target UDF expression.
            raise_on_missing: whether to raise if the UDF is not found.

        Returns:
            The return type of the UDF, or None if not found.
        """
        parts = self.udf_parts(udf)
        resolved_parts = self._find_in_trie(parts, self.udf_trie, raise_on_missing)

        if resolved_parts is None:
            return None

        return nested_get(
            self.udf_mapping,
            *zip(resolved_parts, reversed(resolved_parts)),
            raise_on_missing=raise_on_missing,
        )

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
        udf_mapping: t.Optional[t.Dict] = None,
    ) -> None:
        self.visible = {} if visible is None else visible
        self.normalize = normalize
        self._dialect = Dialect.get_or_raise(dialect)
        self._type_mapping_cache: t.Dict[str, exp.DataType] = {}
        self._normalized_table_cache: t.Dict[t.Tuple[exp.Table, DialectType, bool], exp.Table] = {}
        self._normalized_name_cache: t.Dict[t.Tuple[str, DialectType, bool, bool], str] = {}
        self._depth = 0
        schema = {} if schema is None else schema
        udf_mapping = {} if udf_mapping is None else udf_mapping

        super().__init__(
            self._normalize(schema) if self.normalize else schema,
            self._normalize_udfs(udf_mapping) if self.normalize else udf_mapping,
        )

    @property
    def dialect(self) -> Dialect:
        """Returns the dialect for this mapping schema."""
        return self._dialect

    @classmethod
    def from_mapping_schema(cls, mapping_schema: MappingSchema) -> MappingSchema:
        return MappingSchema(
            schema=mapping_schema.mapping,
            visible=mapping_schema.visible,
            dialect=mapping_schema.dialect,
            normalize=mapping_schema.normalize,
            udf_mapping=mapping_schema.udf_mapping,
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
                "udf_mapping": self.udf_mapping.copy(),
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

    def get_udf_type(
        self,
        udf: exp.Anonymous | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.DataType:
        """
        Get the return type of a UDF.

        Args:
            udf: the UDF expression or string (e.g., "db.my_func()").
            dialect: the SQL dialect for parsing string arguments.
            normalize: whether to normalize identifiers.

        Returns:
            The return type as a DataType, or UNKNOWN if not found.
        """
        parts = self._normalize_udf(udf, dialect=dialect, normalize=normalize)
        resolved_parts = self._find_in_trie(parts, self.udf_trie, raise_on_missing=False)

        if resolved_parts is None:
            return exp.DataType.build("unknown")

        udf_type = nested_get(
            self.udf_mapping,
            *zip(resolved_parts, reversed(resolved_parts)),
            raise_on_missing=False,
        )

        if isinstance(udf_type, exp.DataType):
            return udf_type
        elif isinstance(udf_type, str):
            return self._to_data_type(udf_type, dialect=dialect)

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

    def _normalize_udfs(self, udfs: t.Dict) -> t.Dict:
        """
        Normalizes all identifiers in the UDF mapping.

        Args:
            udfs: the UDF mapping to normalize.

        Returns:
            The normalized UDF mapping.
        """
        normalized_mapping: t.Dict = {}

        for keys in flatten_schema(udfs, depth=dict_depth(udfs)):
            udf_type = nested_get(udfs, *zip(keys, keys))
            normalized_keys = [self._normalize_name(key, is_table=True) for key in keys]
            nested_set(normalized_mapping, normalized_keys, udf_type)

        return normalized_mapping

    def _normalize_udf(
        self,
        udf: exp.Anonymous | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> t.List[str]:
        """
        Extract and normalize UDF parts for lookup.

        Args:
            udf: the UDF expression or qualified string (e.g., "db.my_func()").
            dialect: the SQL dialect for parsing.
            normalize: whether to normalize identifiers.

        Returns:
            A list of normalized UDF parts (reversed for trie lookup).
        """
        dialect = dialect or self.dialect
        normalize = self.normalize if normalize is None else normalize

        if isinstance(udf, str):
            parsed: exp.Expression = exp.maybe_parse(udf, dialect=dialect)

            if isinstance(parsed, exp.Anonymous):
                udf = parsed
            elif isinstance(parsed, exp.Dot) and isinstance(parsed.expression, exp.Anonymous):
                udf = parsed.expression
            else:
                raise SchemaError(f"Unable to parse UDF from: {udf!r}")
        parts = self.udf_parts(udf)

        if normalize:
            parts = [self._normalize_name(part, dialect=dialect, is_table=True) for part in parts]

        return parts

    def _normalize_table(
        self,
        table: exp.Table | str,
        dialect: DialectType = None,
        normalize: t.Optional[bool] = None,
    ) -> exp.Table:
        dialect = dialect or self.dialect
        normalize = self.normalize if normalize is None else normalize

        # Cache normalized tables by object id for exp.Table inputs
        # This is effective when the same Table object is looked up multiple times
        if isinstance(table, exp.Table) and (
            cached := self._normalized_table_cache.get((table, dialect, normalize))
        ):
            return cached

        normalized_table = exp.maybe_parse(table, into=exp.Table, dialect=dialect, copy=normalize)

        if normalize:
            for part in normalized_table.parts:
                if isinstance(part, exp.Identifier):
                    part.replace(
                        normalize_name(part, dialect=dialect, is_table=True, normalize=normalize)
                    )

        self._normalized_table_cache[(t.cast(exp.Table, table), dialect, normalize)] = (
            normalized_table
        )
        return normalized_table

    def _normalize_name(
        self,
        name: str | exp.Identifier,
        dialect: DialectType = None,
        is_table: bool = False,
        normalize: t.Optional[bool] = None,
    ) -> str:
        normalize = self.normalize if normalize is None else normalize

        dialect = dialect or self.dialect
        name_str = name if isinstance(name, str) else name.name
        cache_key = (name_str, dialect, is_table, normalize)

        if cached := self._normalized_name_cache.get(cache_key):
            return cached

        result = normalize_name(
            name,
            dialect=dialect,
            is_table=is_table,
            normalize=normalize,
        ).name

        self._normalized_name_cache[cache_key] = result
        return result

    def depth(self) -> int:
        if not self.empty and not self._depth:
            # The columns themselves are a mapping, but we don't want to include those
            self._depth = super().depth() - 1
        return self._depth

    def _to_data_type(self, schema_type: str, dialect: DialectType = None) -> exp.DataType:
        """
        Convert a type represented as a string to the corresponding `sqlglot.exp.DataType` object.

        Args:
            schema_type: the type we want to convert.
            dialect: the SQL dialect that will be used to parse `schema_type`, if needed.

        Returns:
            The resulting expression type.
        """
        if schema_type not in self._type_mapping_cache:
            dialect = Dialect.get_or_raise(dialect) if dialect else self.dialect
            udt = dialect.SUPPORTS_USER_DEFINED_TYPES

            try:
                expression = exp.DataType.build(schema_type, dialect=dialect, udt=udt)
                expression.transform(dialect.normalize_identifier, copy=False)
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
