from __future__ import annotations


from sqlglot import exp, transforms
from sqlglot.dialects.dialect import NormalizationStrategy
from sqlglot.dialects.tsql import TSQL
from sqlglot.tokens import TokenType


def _cap_data_type_precision(expression: exp.DataType, max_precision: int = 6) -> exp.DataType:
    """
    Cap the precision of to a maximum of `max_precision` digits.
    If no precision is specified, default to `max_precision`.
    精度を最大 `max_precision` 桁に制限します。
    精度が指定されていない場合は、デフォルトで `max_precision` になります。
    """

    precision_param = expression.find(exp.DataTypeParam)

    if precision_param and precision_param.this.is_int:
        current_precision = precision_param.this.to_py()
        target_precision = min(current_precision, max_precision)
    else:
        target_precision = max_precision

    return exp.DataType(
        this=expression.this,
        expressions=[exp.DataTypeParam(this=exp.Literal.number(target_precision))],
    )


def _add_default_precision_to_varchar(expression: exp.Expression) -> exp.Expression:
    """Transform function to add VARCHAR(MAX) or CHAR(MAX) for cross-dialect conversion.
    方言間の変換のために VARCHAR(MAX) または CHAR(MAX) を追加する変換関数。"""
    if (
        isinstance(expression, exp.Create)
        and expression.kind == "TABLE"
        and isinstance(expression.this, exp.Schema)
    ):
        for column in expression.this.expressions:
            if isinstance(column, exp.ColumnDef):
                column_type = column.kind
                if (
                    isinstance(column_type, exp.DataType)
                    and column_type.this in (exp.DataType.Type.VARCHAR, exp.DataType.Type.CHAR)
                    and not column_type.expressions
                ):
                    # For transpilation, VARCHAR/CHAR without precision becomes VARCHAR(MAX)/CHAR(MAX)
                    # トランスパイルの場合、精度のないVARCHAR/CHARはVARCHAR(MAX)/CHAR(MAX)になります。
                    column_type.set("expressions", [exp.var("MAX")])

    return expression


class Fabric(TSQL):
    """
    Microsoft Fabric Data Warehouse dialect that inherits from T-SQL.
    T-SQL を継承した Microsoft Fabric データ ウェアハウス方言です。

    Microsoft Fabric is a cloud-based analytics platform that provides a unified
    data warehouse experience. While it shares much of T-SQL's syntax, it has
    specific differences and limitations that this dialect addresses.
    Microsoft Fabric は、統合されたデータ ウェアハウス エクスペリエンスを提供する
    クラウドベースの分析プラットフォームです。T-SQL の構文と多くの部分を共有していますが、
    この方言が対処する特定の相違点や制限事項があります。


    Key differences from T-SQL:
    - Case-sensitive identifiers (unlike T-SQL which is case-insensitive)
    - Limited data type support with mappings to supported alternatives
    - Temporal types (DATETIME2, DATETIMEOFFSET, TIME) limited to 6 digits precision
    - Certain legacy types (MONEY, SMALLMONEY, etc.) are not supported
    - Unicode types (NCHAR, NVARCHAR) are mapped to non-unicode equivalents
    T-SQLとの主な違い：
    - 大文字と小文字を区別する識別子（大文字と小文字を区別しないT-SQLとは異なります）
    - サポートされている代替データ型へのマッピングによる、サポートされるデータ型が限定されています
    - 時間型（DATETIME2、DATETIMEOFFSET、TIME）は6桁の精度に制限されています
    - 一部のレガシー型（MONEY、SMALLMONEYなど）はサポートされていません
    - Unicode型（NCHAR、NVARCHAR）は、Unicode以外の同等のデータ型にマッピングされます

    References:
    - Data Types: https://learn.microsoft.com/en-us/fabric/data-warehouse/data-types
    - T-SQL Surface Area: https://learn.microsoft.com/en-us/fabric/data-warehouse/tsql-surface-area
    """

    # Fabric is case-sensitive unlike T-SQL which is case-insensitive
    # Fabricは大文字と小文字を区別しますが、T-SQLは大文字と小文字を区別しません。
    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_SENSITIVE

    class Tokenizer(TSQL.Tokenizer):
        # Override T-SQL tokenizer to handle TIMESTAMP differently
        # In T-SQL, TIMESTAMP is a synonym for ROWVERSION, but in Fabric we want it to be a datetime type
        # Also add UTINYINT keyword mapping since T-SQL doesn't have it
        # T-SQLトークナイザーをオーバーライドして、TIMESTAMPを別の方法で処理します。
        # T-SQLではTIMESTAMPはROWVERSIONの同義語ですが、Fabricではdatetime型として使用します。
        # T-SQLにはUTINYINTキーワードマッピングがないため、UTINYINTキーワードマッピングも追加します。
        KEYWORDS = {
            **TSQL.Tokenizer.KEYWORDS,
            "TIMESTAMP": TokenType.TIMESTAMP,
            "UTINYINT": TokenType.UTINYINT,
        }

    class Parser(TSQL.Parser):
        def _parse_create(self) -> exp.Create | exp.Command:
            create = super()._parse_create()

            if isinstance(create, exp.Create):
                # Transform VARCHAR/CHAR without precision to VARCHAR(1)/CHAR(1)
                # 精度のないVARCHAR/CHARをVARCHAR(1)/CHAR(1)に変換する
                if create.kind == "TABLE" and isinstance(create.this, exp.Schema):
                    for column in create.this.expressions:
                        if isinstance(column, exp.ColumnDef):
                            column_type = column.kind
                            if (
                                isinstance(column_type, exp.DataType)
                                and column_type.this
                                in (exp.DataType.Type.VARCHAR, exp.DataType.Type.CHAR)
                                and not column_type.expressions
                            ):
                                # Add default precision of 1 to VARCHAR/CHAR without precision
                                # When n isn't specified in a data definition or variable declaration statement, the default length is 1.
                                # 精度のないVARCHAR/CHARにデフォルトの精度1を追加します。
                                # データ定義または変数宣言ステートメントでnが指定されていない場合、デフォルトの長さは1になります。
                                # https://learn.microsoft.com/en-us/sql/t-sql/data-types/char-and-varchar-transact-sql?view=sql-server-ver17#remarks
                                column_type.set("expressions", [exp.Literal.number("1")])

            return create

    class Generator(TSQL.Generator):
        # Fabric-specific type mappings - override T-SQL types that aren't supported
        # Fabric 固有の型マッピング - サポートされていない T-SQL 型をオーバーライドします
        # Reference: https://learn.microsoft.com/en-us/fabric/data-warehouse/data-types
        TYPE_MAPPING = {
            **TSQL.Generator.TYPE_MAPPING,
            exp.DataType.Type.DATETIME: "DATETIME2",
            exp.DataType.Type.DECIMAL: "DECIMAL",
            exp.DataType.Type.IMAGE: "VARBINARY",
            exp.DataType.Type.INT: "INT",
            exp.DataType.Type.JSON: "VARCHAR",
            exp.DataType.Type.MONEY: "DECIMAL",
            exp.DataType.Type.NCHAR: "CHAR",
            exp.DataType.Type.NVARCHAR: "VARCHAR",
            exp.DataType.Type.ROWVERSION: "ROWVERSION",
            exp.DataType.Type.SMALLDATETIME: "DATETIME2",
            exp.DataType.Type.SMALLMONEY: "DECIMAL",
            exp.DataType.Type.TIMESTAMP: "DATETIME2",
            exp.DataType.Type.TIMESTAMPNTZ: "DATETIME2",
            exp.DataType.Type.TIMESTAMPTZ: "DATETIME2",
            exp.DataType.Type.TINYINT: "SMALLINT",
            exp.DataType.Type.UTINYINT: "SMALLINT",
            exp.DataType.Type.UUID: "UNIQUEIDENTIFIER",
            exp.DataType.Type.XML: "VARCHAR",
        }

        TRANSFORMS = {
            **TSQL.Generator.TRANSFORMS,
            exp.Create: transforms.preprocess([_add_default_precision_to_varchar]),
        }

        def datatype_sql(self, expression: exp.DataType) -> str:
            # Check if this is a temporal type that needs precision handling. Fabric limits temporal
            # types to max 6 digits precision. When no precision is specified, we default to 6 digits.
            # 精度処理が必要な時間型かどうかを確認してください。Fabric では時間型の精度は最大 6 桁に制限されています。
            # 精度が指定されていない場合は、デフォルトで 6 桁になります。
            if (
                expression.is_type(*exp.DataType.TEMPORAL_TYPES)
                and expression.this != exp.DataType.Type.DATE
            ):
                # Create a new expression with the capped precision
                # 精度上限付きの新しい式を作成する
                expression = _cap_data_type_precision(expression)

            return super().datatype_sql(expression)

        def cast_sql(self, expression: exp.Cast, safe_prefix: str | None = None) -> str:
            # Cast to DATETIMEOFFSET if inside an AT TIME ZONE expression
            # AT TIME ZONE式内にある場合はDATETIMEOFFSETにキャストする
            # https://learn.microsoft.com/en-us/sql/t-sql/data-types/datetimeoffset-transact-sql#microsoft-fabric-support
            if expression.is_type(exp.DataType.Type.TIMESTAMPTZ):
                at_time_zone = expression.find_ancestor(exp.AtTimeZone, exp.Select)

                # Return normal cast, if the expression is not in an AT TIME ZONE context
                # 式がAT TIME ZONEコンテキストにない場合は、通常のキャストを返します。
                if not isinstance(at_time_zone, exp.AtTimeZone):
                    return super().cast_sql(expression, safe_prefix)

                # Get the precision from the original TIMESTAMPTZ cast and cap it to 6
                # オリジナルのTIMESTAMPTZキャストから精度を取得し、6に制限します
                capped_data_type = _cap_data_type_precision(expression.to, max_precision=6)
                precision = capped_data_type.find(exp.DataTypeParam)
                precision_value = (
                    precision.this.to_py() if precision and precision.this.is_int else 6
                )

                # Do the cast explicitly to bypass sqlglot's default handling
                # sqlglotのデフォルトの処理を回避するために明示的にキャストを実行します
                datetimeoffset = f"CAST({expression.this} AS DATETIMEOFFSET({precision_value}))"

                return self.sql(datetimeoffset)

            return super().cast_sql(expression, safe_prefix)

        def attimezone_sql(self, expression: exp.AtTimeZone) -> str:
            # Wrap the AT TIME ZONE expression in a cast to DATETIME2 if it contains a TIMESTAMPTZ
            # AT TIME ZONE式にTIMESTAMPTZが含まれている場合は、DATETIME2へのキャストで囲みます。
            ## https://learn.microsoft.com/en-us/sql/t-sql/data-types/datetimeoffset-transact-sql#microsoft-fabric-support
            timestamptz_cast = expression.find(exp.Cast)
            if timestamptz_cast and timestamptz_cast.to.is_type(exp.DataType.Type.TIMESTAMPTZ):
                # Get the precision from the original TIMESTAMPTZ cast and cap it to 6
                # オリジナルのTIMESTAMPTZキャストから精度を取得し、6に制限します
                data_type = timestamptz_cast.to
                capped_data_type = _cap_data_type_precision(data_type, max_precision=6)
                precision_param = capped_data_type.find(exp.DataTypeParam)
                precision = precision_param.this.to_py() if precision_param else 6

                # Generate the AT TIME ZONE expression (which will handle the inner cast conversion)
                # AT TIME ZONE式を生成します（内部キャスト変換を処理します）
                at_time_zone_sql = super().attimezone_sql(expression)

                # Wrap it in an outer cast to DATETIME2
                # DATETIME2への外側のキャストでラップします
                return f"CAST({at_time_zone_sql} AS DATETIME2({precision}))"

            return super().attimezone_sql(expression)

        def unixtotime_sql(self, expression: exp.UnixToTime) -> str:
            scale = expression.args.get("scale")
            timestamp = expression.this

            if scale not in (None, exp.UnixToTime.SECONDS):
                self.unsupported(f"UnixToTime scale {scale} is not supported by Fabric")
                return ""

            # Convert unix timestamp (seconds) to microseconds and round to avoid decimals
            # Unixタイムスタンプ（秒）をマイクロ秒に変換し、小数点以下を避けるために丸めます。
            microseconds = timestamp * exp.Literal.number("1e6")
            rounded = exp.func("round", microseconds, 0)
            rounded_ms_as_bigint = exp.cast(rounded, exp.DataType.Type.BIGINT)

            # Create the base datetime as '1970-01-01' cast to DATETIME2(6)
            # '1970-01-01'をDATETIME2にキャストしてベース日時を作成する(6)
            epoch_start = exp.cast("'1970-01-01'", "datetime2(6)", dialect="fabric")

            dateadd = exp.DateAdd(
                this=epoch_start,
                expression=rounded_ms_as_bigint,
                unit=exp.Literal.string("MICROSECONDS"),
            )
            return self.sql(dateadd)
