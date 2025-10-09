![SQLGlot logo](sqlglot.png)

SQLGlot は、依存性のない SQL パーサー、トランスパイラー、オプティマイザー、そしてエンジンです。SQL のフォーマットや、[DuckDB](https://duckdb.org/)、[Presto](https://prestodb.io/) / [Trino](https://trino.io/)、[Spark](https://spark.apache.org/) / [Databricks](https://www.databricks.com/)、[Snowflake](https://www.snowflake.com/en/)、[BigQuery](https://cloud.google.com/bigquery/) といった [30 種類の方言](https://github.com/tobymao/sqlglot/blob/main/sqlglot/dialects/__init__.py) 間の変換に使用できます。SQLGlot は、多様な SQL 入力を読み取り、対象となる方言で構文的および意味的に正しい SQL を出力することを目的としています。

これは非常に包括的な汎用SQLパーサーであり、堅牢な[テストスイート](https://github.com/tobymao/sqlglot/blob/main/tests/)を備えています。また、純粋にPythonで記述されているにもかかわらず、非常に[パフォーマンス](#ベンチマーク)に優れています。

パーサーは簡単に[カスタマイズ](#カスタム方言)したり、クエリを[分析](#メタデータ)したり、式ツリーを走査したり、プログラムでSQLを[構築](#ビルドと変更-sql)したりできます。

SQLGlotは、括弧の不一致や予約語の誤った使用など、さまざまな[構文エラー](#パーサーエラー)を検出できます。これらのエラーはハイライト表示され、方言の非互換性については、設定に応じて警告または例外を出力できます。

SQLGlot の詳細については、API [ドキュメント](https://sqlglot.com/) と式ツリー [入門書](https://github.com/tobymao/sqlglot/blob/main/posts/ast_primer.md) をご覧ください。

SQLGlot への貢献は大歓迎です。まずは [貢献ガイド](https://github.com/tobymao/sqlglot/blob/main/CONTRIBUTING.md) と [オンボーディングドキュメント](https://github.com/tobymao/sqlglot/blob/main/posts/onboarding.md) をお読みください。

## Table of Contents

* [Install](#install)
* [Versioning](#versioning)
* [Get in Touch](#get-in-touch)
* [FAQ](#faq)
* [Examples](#examples)
   * [Formatting and Transpiling](#formatting-and-transpiling)
   * [Metadata](#metadata)
   * [Parser Errors](#parser-errors)
   * [Unsupported Errors](#unsupported-errors)
   * [Build and Modify SQL](#build-and-modify-sql)
   * [SQL Optimizer](#sql-optimizer)
   * [AST Introspection](#ast-introspection)
   * [AST Diff](#ast-diff)
   * [Custom Dialects](#custom-dialects)
   * [SQL Execution](#sql-execution)
* [Used By](#used-by)
* [Documentation](#documentation)
* [Run Tests and Lint](#run-tests-and-lint)
* [Benchmarks](#benchmarks)
* [Optional Dependencies](#optional-dependencies)
* [Supported Dialects](#supported-dialects)

## Install

From PyPI:

```bash
pip3 install "sqlglot[rs]"

# Without Rust tokenizer (slower):
# pip3 install sqlglot
```

または、ローカルチェックアウトをご利用ください:

```
# Optionally prefix with UV=1 to use uv for the installation
make install
```

開発要件（オプション）:

```
# Optionally prefix with UV=1 to use uv for the installation
make install-dev
```

## Versioning

バージョン番号が `MAJOR`.`MINOR`.`PATCH` の場合、SQLGlot は以下のバージョン管理戦略を使用します。

- `PATCH` バージョンは、後方互換性のある修正または機能追加があった場合に増加します。
- `MINOR` バージョンは、後方互換性のない修正または機能追加があった場合に増加します。
- `MAJOR` バージョンは、後方互換性のない重要な修正または機能追加があった場合に増加します。

## Get in Touch

皆様からのご意見をお待ちしております。コミュニティ[Slackチャンネル](https://tobikodata.com/slack)にぜひご参加ください！

## FAQ

有効なはずのSQLを解析しようとしましたが、失敗しました。なぜでしょうか？

* 多くの場合、このような問題は解析中に「ソース」方言が省略されていることが原因で発生します。例えば、Spark SQLで記述されたSQLクエリを正しく解析するには、`parse_one(sql, dialect="spark")` （または `read="spark"`）と記述します。方言が指定されていない場合、`parse_one` は「SQLGlot方言」に従ってクエリを解析しようとします。SQLGlot方言は、サポートされているすべての方言のスーパーセットとなるように設計されています。方言を指定しても問題が解決しない場合は、問題を報告してください。

SQLを出力しようとしましたが、正しい方言ではありません！

* 解析と同様に、SQLを生成する場合もターゲット方言を指定する必要があります。指定しない場合は、デフォルトでSQLGlot方言が使用されます。例えば、Spark SQL から DuckDB にクエリをトランスパイルするには、`parse_one(sql, dialect="spark").sql(dialect="duckdb")` を実行します（または、`transpile(sql, read="spark", write="duckdb")` を実行します）。

sqlglot.dataframe はどうなりましたか？

* PySpark データフレーム API は、v24 で [SQLFrame](https://github.com/eakmanrq/sqlframe) というスタンドアロンライブラリに移動されました。これにより、SQL を生成するだけでなく、クエリを実行できるようになりました。

## Examples

### Formatting and Transpiling

ある方言から別の方言へ簡単に翻訳できます。例えば、日付/時刻関数は方言によって異なるため、扱いが難しい場合があります。

```python
import sqlglot
sqlglot.transpile("SELECT EPOCH_MS(1618088028295)", read="duckdb", write="hive")[0]
```

```sql
'SELECT FROM_UNIXTIME(1618088028295 / POW(10, 3))'
```

SQLGlot はカスタム時間形式を翻訳することもできます。

```python
import sqlglot
sqlglot.transpile("SELECT STRFTIME(x, '%y-%-m-%S')", read="duckdb", write="hive")[0]
```

```sql
"SELECT DATE_FORMAT(x, 'yy-M-ss')"
```

識別子の区切り文字とデータ型も翻訳できます。

```python
import sqlglot

# Spark SQL requires backticks (`) for delimited identifiers and uses `FLOAT` over `REAL`
sql = """WITH baz AS (SELECT a, c FROM foo WHERE a = 1) SELECT f.a, b.b, baz.c, CAST("b"."a" AS REAL) d FROM foo f JOIN bar b ON f.a = b.a LEFT JOIN baz ON f.a = baz.a"""

# Translates the query into Spark SQL, formats it, and delimits all of its identifiers
print(sqlglot.transpile(sql, write="spark", identify=True, pretty=True)[0])
```

```sql
WITH `baz` AS (
  SELECT
    `a`,
    `c`
  FROM `foo`
  WHERE
    `a` = 1
)
SELECT
  `f`.`a`,
  `b`.`b`,
  `baz`.`c`,
  CAST(`b`.`a` AS FLOAT) AS `d`
FROM `foo` AS `f`
JOIN `bar` AS `b`
  ON `f`.`a` = `b`.`a`
LEFT JOIN `baz`
  ON `f`.`a` = `baz`.`a`
```

コメントもベストエフォート方式で保存されます。

```python
sql = """
/* multi
   line
   comment
*/
SELECT
  tbl.cola /* comment 1 */ + tbl.colb /* comment 2 */,
  CAST(x AS SIGNED), # comment 3
  y               -- comment 4
FROM
  bar /* comment 5 */,
  tbl #          comment 6
"""

# Note: MySQL-specific comments (`#`) are converted into standard syntax
print(sqlglot.transpile(sql, read='mysql', pretty=True)[0])
```

```sql
/* multi
   line
   comment
*/
SELECT
  tbl.cola /* comment 1 */ + tbl.colb /* comment 2 */,
  CAST(x AS INT), /* comment 3 */
  y /* comment 4 */
FROM bar /* comment 5 */, tbl /*          comment 6 */
```


### Metadata

式ヘルパーを使用して SQL を調べ、クエリ内の列やテーブルを検索するなどの操作を行うことができます。

```python
from sqlglot import parse_one, exp

# print all column references (a and b)
for column in parse_one("SELECT a, b + 1 AS c FROM d").find_all(exp.Column):
    print(column.alias_or_name)

# find all projections in select statements (a and c)
for select in parse_one("SELECT a, b + 1 AS c FROM d").find_all(exp.Select):
    for projection in select.expressions:
        print(projection.alias_or_name)

# find all tables (x, y, z)
for table in parse_one("SELECT * FROM x JOIN y JOIN z").find_all(exp.Table):
    print(table.name)
```

SQLGlot の内部について詳しくは、[ast 入門](https://github.com/tobymao/sqlglot/blob/main/posts/ast_primer.md) をお読みください。

### Parser Errors

パーサーが構文エラーを検出すると、`ParseError` を発生させます。

```python
import sqlglot
sqlglot.transpile("SELECT foo FROM (SELECT baz FROM t")
```

```
sqlglot.errors.ParseError: Expecting ). Line 1, Col: 34.
  SELECT foo FROM (SELECT baz FROM t
                                   ~
```

構造化された構文エラーはプログラムでの使用のためにアクセス可能です。

```python
import sqlglot.errors
try:
    sqlglot.transpile("SELECT foo FROM (SELECT baz FROM t")
except sqlglot.errors.ParseError as e:
    print(e.errors)
```

```python
[{
  'description': 'Expecting )',
  'line': 1,
  'col': 34,
  'start_context': 'SELECT foo FROM (SELECT baz FROM ',
  'highlight': 't',
  'end_context': '',
  'into_expression': None
}]
```

### Unsupported Errors

特定の方言間では、一部のクエリを翻訳できない場合があります。このような場合、SQLGlot は警告を発し、デフォルトでベストエフォート型の翻訳を実行します。

```python
import sqlglot
sqlglot.transpile("SELECT APPROX_DISTINCT(a, 0.1) FROM foo", read="presto", write="hive")
```

```sql
APPROX_COUNT_DISTINCT does not support accuracy
'SELECT APPROX_COUNT_DISTINCT(a) FROM foo'
```

この動作は、[`unsupported_level`](https://github.com/tobymao/sqlglot/blob/b0e8dc96ba179edb1776647b5bde4e704238b44d/sqlglot/errors.py#L9) 属性を設定することで変更できます。例えば、代わりに例外を発生させるには、`RAISE` または `IMMEDIATE` のいずれかに設定できます。

```python
import sqlglot
sqlglot.transpile("SELECT APPROX_DISTINCT(a, 0.1) FROM foo", read="presto", write="hive", unsupported_level=sqlglot.ErrorLevel.RAISE)
```

```
sqlglot.errors.UnsupportedError: APPROX_COUNT_DISTINCT does not support accuracy
```

クエリの中には、参照されているテーブルのスキーマなど、正確なトランスパイルを行うために追加情報を必要とするものがあります。これは、特定の変換が型に依存し、そのセマンティクスを理解するために型推論が必要となるためです。`qualify` および `annotate_types` オプティマイザ [ルール](https://github.com/tobymao/sqlglot/tree/main/sqlglot/optimizer) はこの点に役立ちますが、オーバーヘッドと複雑さが増大するため、デフォルトでは使用されません。

トランスパイルは一般的に難しい問題であるため、SQLGlot では「増分」アプローチで解決しています。そのため、現在一部の入力をサポートしていない方言ペアが存在する可能性がありますが、これは今後改善される見込みです。十分に文書化されテストされた問題やプルリクエスト（PR）は大変歓迎いたします。ガイダンスが必要な場合は、お気軽に[お問い合わせください](#get-in-touch)！

### Build and Modify SQL

SQLGlot は SQL 式の段階的な構築をサポートします。

```python
from sqlglot import select, condition

where = condition("x=1").and_("y=1")
select("*").from_("y").where(where).sql()
```

```sql
'SELECT * FROM y WHERE x = 1 AND y = 1'
```

解析されたツリーを変更することは可能です:

```python
from sqlglot import parse_one
parse_one("SELECT x FROM y").from_("z").sql()
```

```sql
'SELECT x FROM z'
```

解析された式は、ツリー内の各ノードにマッピング関数を適用することで再帰的に変換することもできます。

```python
from sqlglot import exp, parse_one

expression_tree = parse_one("SELECT a FROM x")

def transformer(node):
    if isinstance(node, exp.Column) and node.name == "a":
        return parse_one("FUN(a)")
    return node

transformed_tree = expression_tree.transform(transformer)
transformed_tree.sql()
```

```sql
'SELECT FUN(a) FROM x'
```

### SQL Optimizer

SQLGlotはクエリを「最適化された」形式に書き換えることができます。様々な[テクニック](https://github.com/tobymao/sqlglot/blob/main/sqlglot/optimizer/optimizer.py)を用いて、新しい正規表現ASTを作成します。このASTは、クエリを標準化したり、実際のエンジンを実装するための基盤を提供したりするために使用できます。例えば、

```python
import sqlglot
from sqlglot.optimizer import optimize

print(
    optimize(
        sqlglot.parse_one("""
            SELECT A OR (B OR (C AND D))
            FROM x
            WHERE Z = date '2021-01-01' + INTERVAL '1' month OR 1 = 0
        """),
        schema={"x": {"A": "INT", "B": "INT", "C": "INT", "D": "INT", "Z": "STRING"}}
    ).sql(pretty=True)
)
```

```sql
SELECT
  (
    "x"."a" <> 0 OR "x"."b" <> 0 OR "x"."c" <> 0
  )
  AND (
    "x"."a" <> 0 OR "x"."b" <> 0 OR "x"."d" <> 0
  ) AS "_col_0"
FROM "x" AS "x"
WHERE
  CAST("x"."z" AS DATE) = CAST('2021-02-01' AS DATE)
```

### AST Introspection

`repr` を呼び出すと、解析された SQL の最後のバージョンを確認できます。

```python
from sqlglot import parse_one
print(repr(parse_one("SELECT a + 1 AS z")))
```

```python
Select(
  expressions=[
    Alias(
      this=Add(
        this=Column(
          this=Identifier(this=a, quoted=False)),
        expression=Literal(this=1, is_string=False)),
      alias=Identifier(this=z, quoted=False))])
```

### AST Diff

SQLGlot は、2 つの式の意味の違いを計算し、ソース式をターゲット式に変換するために必要な一連のアクションの形式で変更を出力します。

```python
from sqlglot import diff, parse_one
diff(parse_one("SELECT a + b, c, d"), parse_one("SELECT c, a - b, d"))
```

```python
[
  Remove(expression=Add(
    this=Column(
      this=Identifier(this=a, quoted=False)),
    expression=Column(
      this=Identifier(this=b, quoted=False)))),
  Insert(expression=Sub(
    this=Column(
      this=Identifier(this=a, quoted=False)),
    expression=Column(
      this=Identifier(this=b, quoted=False)))),
  Keep(
    source=Column(this=Identifier(this=a, quoted=False)),
    target=Column(this=Identifier(this=a, quoted=False))),
  ...
]
```

参照: [SQL のセマンティック ディフ](https://github.com/tobymao/sqlglot/blob/main/posts/sql_diff.md)。

### Custom Dialects

[方言](https://github.com/tobymao/sqlglot/tree/main/sqlglot/dialects)は、`Dialect` をサブクラス化することで追加できます。

```python
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class Custom(Dialect):
    class Tokenizer(Tokenizer):
        QUOTES = ["'", '"']
        IDENTIFIERS = ["`"]

        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "INT64": TokenType.BIGINT,
            "FLOAT64": TokenType.DOUBLE,
        }

    class Generator(Generator):
        TRANSFORMS = {exp.Array: lambda self, e: f"[{self.expressions(e)}]"}

        TYPE_MAPPING = {
            exp.DataType.Type.TINYINT: "INT64",
            exp.DataType.Type.SMALLINT: "INT64",
            exp.DataType.Type.INT: "INT64",
            exp.DataType.Type.BIGINT: "INT64",
            exp.DataType.Type.DECIMAL: "NUMERIC",
            exp.DataType.Type.FLOAT: "FLOAT64",
            exp.DataType.Type.DOUBLE: "FLOAT64",
            exp.DataType.Type.BOOLEAN: "BOOL",
            exp.DataType.Type.TEXT: "STRING",
        }

print(Dialect["custom"])
```

```
<class '__main__.Custom'>
```

### SQL Execution

SQLGlotは、テーブルがPython辞書として表現されているSQLクエリを解釈できます。このエンジンは高速ではありませんが、単体テストやPythonオブジェクト間でSQLをネイティブに実行する際に役立ちます。さらに、この基盤は[Arrow](https://arrow.apache.org/docs/index.html)や[Pandas](https://pandas.pydata.org/)などの高速コンピューティングカーネルと容易に統合できます。

以下の例は、集計と結合を含むクエリの実行を示しています。

```python
from sqlglot.executor import execute

tables = {
    "sushi": [
        {"id": 1, "price": 1.0},
        {"id": 2, "price": 2.0},
        {"id": 3, "price": 3.0},
    ],
    "order_items": [
        {"sushi_id": 1, "order_id": 1},
        {"sushi_id": 1, "order_id": 1},
        {"sushi_id": 2, "order_id": 1},
        {"sushi_id": 3, "order_id": 2},
    ],
    "orders": [
        {"id": 1, "user_id": 1},
        {"id": 2, "user_id": 2},
    ],
}

execute(
    """
    SELECT
      o.user_id,
      SUM(s.price) AS price
    FROM orders o
    JOIN order_items i
      ON o.id = i.order_id
    JOIN sushi s
      ON i.sushi_id = s.id
    GROUP BY o.user_id
    """,
    tables=tables
)
```

```python
user_id price
      1   4.0
      2   3.0
```

参照: [Python SQL エンジンをゼロから作成する](https://github.com/tobymao/sqlglot/blob/main/posts/python_sql_engine.md)。

## Used By

* [SQLMesh](https://github.com/TobikoData/sqlmesh)
* [Apache Superset](https://github.com/apache/superset)
* [Dagster](https://github.com/dagster-io/dagster)
* [Fugue](https://github.com/fugue-project/fugue)
* [Ibis](https://github.com/ibis-project/ibis)
* [dlt](https://github.com/dlt-hub/dlt)
* [mysql-mimic](https://github.com/kelsin/mysql-mimic)
* [Querybook](https://github.com/pinterest/querybook)
* [Quokka](https://github.com/marsupialtail/quokka)
* [Splink](https://github.com/moj-analytical-services/splink)
* [SQLFrame](https://github.com/eakmanrq/sqlframe)

## Documentation

SQLGlot は API ドキュメントの提供に [pdoc](https://pdoc.dev/) を使用しています。

ホスト版は [SQLGlot ウェブサイト](https://sqlglot.com/) に掲載されています。また、以下のコマンドでローカルにビルドすることもできます。

```
make docs-serve
```

## Run Tests and Lint

```
make style  # Only linter checks
make unit   # Only unit tests (or unit-rs, to use the Rust tokenizer)
make test   # Unit and integration tests (or test-rs, to use the Rust tokenizer)
make check  # Full test suite & linter checks
```

## Benchmarks

[ベンチマーク](https://github.com/tobymao/sqlglot/blob/main/benchmarks/bench.py​​) は Python 3.10.12 で数秒で実行されます。

|           Query |         sqlglot |       sqlglotrs |        sqlfluff |         sqltree |        sqlparse |  moz_sql_parser |        sqloxide |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
|            tpch |   0.00944 (1.0) | 0.00590 (0.625) | 0.32116 (33.98) | 0.00693 (0.734) | 0.02858 (3.025) | 0.03337 (3.532) | 0.00073 (0.077) |
|           short |   0.00065 (1.0) | 0.00044 (0.687) | 0.03511 (53.82) | 0.00049 (0.759) | 0.00163 (2.506) | 0.00234 (3.601) | 0.00005 (0.073) |
|            long |   0.00889 (1.0) | 0.00572 (0.643) | 0.36982 (41.56) | 0.00614 (0.690) | 0.02530 (2.844) | 0.02931 (3.294) | 0.00059 (0.066) |
|           crazy |   0.02918 (1.0) | 0.01991 (0.682) | 1.88695 (64.66) | 0.02003 (0.686) | 7.46894 (255.9) | 0.64994 (22.27) | 0.00327 (0.112) |

```
make bench            # Run parsing benchmark
make bench-optimize   # Run optimization benchmark
```

## Optional Dependencies

SQLGlotは[dateutil](https://github.com/dateutil/dateutil)を使用して、リテラルなtimedelta式を簡略化します。モジュールが見つからない場合、オプティマイザーは以下のような式を簡略化しません。

```sql
x + interval '1' month
```

## Supported Dialects

| Dialect | Support Level |
|---------|---------------|
| Athena | Official |
| BigQuery | Official |
| ClickHouse | Official |
| Databricks | Official |
| Doris | Community |
| Dremio | Community |
| Drill | Community |
| Druid | Community |
| DuckDB | Official |
| Exasol | Community |
| Fabric | Community |
| Hive | Official |
| Materialize | Community |
| MySQL | Official |
| Oracle | Official |
| Postgres | Official |
| Presto | Official |
| PRQL | Community |
| Redshift | Official |
| RisingWave | Community |
| SingleStore | Community |
| Snowflake | Official |
| Spark | Official |
| SQLite | Official |
| StarRocks | Official |
| Tableau | Official |
| Teradata | Community |
| Trino | Official |
| TSQL | Official |

**公式方言** は、SQLGlot コアチームによってメンテナンスされており、バグ修正や機能追加は高い優先度で行われています。

**コミュニティ方言** は、主にコミュニティの貢献によって開発・メンテナンスされています。コミュニティ方言は完全に機能しますが、公式にサポートされている方言に比べて問題解決の優先度が低くなる可能性があります。コミュニティの方言の改善に向けた貢献を歓迎し、奨励しています。



# MEMO
## ドキュメント

```bash
git clone https://github.com/take0a/sqlglot
cd sqlglot
uv sync
uv add pdoc
uv run pdoc/cli.py
```

