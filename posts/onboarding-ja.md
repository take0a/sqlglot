# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Onboarding](#onboarding)
  - [Tokenizer](#tokenizer)
  - [Parser](#parser)
    - [Token Index](#token-index)
    - [Matching text](#matching-text)
    - [Utility methods](#utility-methods)
    - [Command fallback](#command-fallback)
    - [Expression parsing](#expression-parsing)
  - [Generator](#generator)
    - [Generating queries](#generating-queries)
    - [Utility methods](#utility-methods-1)
    - [Pretty print](#pretty-print)
  - [Schema](#schema)
  - [Optimizer](#optimizer)
    - [Optimization rules](#optimization-rules)
      - [Qualify](#qualify)
        - [Identifier normalization](#identifier-normalization)
        - [Qualifying tables \& columns](#qualifying-tables--columns)
      - [Type annotation](#type-annotation)
  - [Dialects](#dialects)
    - [Implementing a custom dialect](#implementing-a-custom-dialect)
  - [Column Level Lineage](#column-level-lineage)
    - [Implementation details](#implementation-details)

## オンボーディング
このドキュメントは、読者にSQLGlotのコードベースとアーキテクチャを理解してもらうことを目的としています。

一般的に、プログラミング言語/コンパイラに関するある程度の背景知識が必要です。良い出発点として、Robert Nystrom著の[Crafting Interpreters](https://craftinginterpreters.com/)が挙げられます。これは、SQLGlotが最初に開発された際の基盤となりました。

SQLGlotのトランスパイルは、大まかに3つのモジュールで構成されます。

- [Tokenizer](#tokenizer): 生のコードを、単語/記号ごとに1つずつ、一連​​の「トークン」に変換します。
- [Parser](#parser): トークンのシーケンスを、生のコードの意味を表す抽象構文木に変換します。
- [Generator](#generator): 抽象構文木をSQLコードに変換します。

SQLGlotは、通常は異なるデータベースシステムに関連付けられている異なる _SQL方言_ 間でSQLをトランスパイルできます。

各方言は独自の構文を持ち、3つのモジュールの基本バージョンをオーバーライドすることで実装されます。方言定義は、3つの基本モジュールのいずれか、またはすべてをオーバーライドできます。

モジュールの基本バージョンは、`sqlglot` 方言（「基本方言」と呼ばれることもあります）を実装しています。この方言は、サポートされている他の方言間で可能な限り多くの共通構文要素に対応するように設計されており、コードの重複を防いでいます。

SQLGlotには、[Optimizer](#optimizer)やExecutorなど、基本的なトランスパイルには必要のないモジュールも含まれています。Optimizerは、抽象構文木を他の用途（列レベルの系統の推論など）のために修正します。ExecutorはSQLコードをPythonで実行しますが、そのエンジンはまだ実験段階であるため、一部の機能が欠けている可能性があります。

このドキュメントの残りの部分では、3つの基本モジュール、方言固有のオーバーライド、およびOptimizerモジュールについて説明します。

## トークナイザー
トークナイザーモジュール（`tokens.py`）は、SQLコードを意味のある情報の最小単位であるトークン（キーワード、識別子など）に分解する役割を担います。このプロセスは、字句解析（lexical analysis）とも呼ばれます。

Pythonは処理速度やパフォーマンスを最大限に高めるように設計されていないため、SQLGlotはパフォーマンス向上のため、同等の[Rust版トークナイザー](https://tobikodata.com/sqlglot-jumps-on-the-rust-bandwagon.html)を`sqlglotrs/tokenizer.rs`で提供しています。

> [!IMPORTANT]
> トークナイザーロジックの変更は、PythonとRustの両方のトークナイザーに反映させる必要があります。2つの実装を類似させ、コード移植時の作業負担を軽減することが目標です。

この例は、トークナイザーを使用してSQLクエリ`SELECT b FROM table WHERE c = 1`をトークナイズした結果を示しています。

```python
from sqlglot import tokenize

tokens = tokenize("SELECT b FROM table WHERE c = 1")
for token in tokens:
    print(token)

# <Token token_type: <TokenType.SELECT: 'SELECT'>, text: 'SELECT', line: 1, col: 6, start: 0, end: 5, comments: []>
# <Token token_type: <TokenType.VAR: 'VAR'>, text: 'b', line: 1, col: 8, start: 7, end: 7, comments: []>
# <Token token_type: <TokenType.FROM: 'FROM'>, text: 'FROM', line: 1, col: 13, start: 9, end: 12, comments: []>
# <Token token_type: <TokenType.TABLE: 'TABLE'>, text: 'table', line: 1, col: 19, start: 14, end: 18, comments: []>
# <Token token_type: <TokenType.WHERE: 'WHERE'>, text: 'WHERE', line: 1, col: 25, start: 20, end: 24, comments: []>
# <Token token_type: <TokenType.VAR: 'VAR'>, text: 'c', line: 1, col: 27, start: 26, end: 26, comments: []>
# <Token token_type: <TokenType.EQ: 'EQ'>, text: '=', line: 1, col: 29, start: 28, end: 28, comments: []>
# <Token token_type: <TokenType.NUMBER: 'NUMBER'>, text: '1', line: 1, col: 31, start: 30, end: 30, comments: []>
```

トークナイザーはクエリをスキャンし、単語（例：`b`）や演算子（例：`=`）などの記号のグループ（または語彙素）をトークンに変換します。例えば、`VAR` トークンは識別子 `b` を表すために使用され、`EQ` トークンは「equals」演算子を表すために使用されます。

各トークンには、その型（`token_type`）、それがカプセル化する語彙素（`text`）、そして語彙素の行（`line`）や列（`col`）などのメタデータが含まれており、これらはエラーを正確に報告するために使用されます。

SQLGlot の `TokenType` 列挙型は、語彙素とその型の間に間接的なレイヤーを提供します。例えば、`!=` と `<>` は「等しくない」を表すためにしばしば互換的に使用されるため、SQLGlot はこれらを `TokenType.NEQ` にマッピングすることでグループ化します。

`Tokenizer.KEYWORDS` と `Tokenizer.SINGLE_TOKENS` は、語彙素を対応する `TokenType` 列挙値にマッピングする 2 つの重要な辞書です。

## パーサー
パーサーモジュールは、トークナイザーによって生成されたトークンのリストを受け取り、それらから[抽象構文木 (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) を構築します。

一般的に、AST は SQL 文の _意味のある_ 構成要素を格納します。例えば、`SELECT` クエリを表現するには、その射影、フィルター、集計式などのみを格納できますが、`SELECT`、`WHERE`、`GROUP BY` キーワードは暗黙的に指定されるため、保持する必要はありません。

SQLGlot の場合、AST は通常、式の _セマンティクス_、つまり意味を捉えた情報を格納します。これにより、異なる方言間で SQL コードをトランスパイルすることが可能になります。この考え方を高レベルで説明したブログ記事は、[こちら](https://tobikodata.com/transpiling_sql1.html) でご覧いただけます。

パーサーは、SQL 文に対応するトークンを順番に処理し、それを表す AST を生成します。例えば、文の最初のトークンが `CREATE` の場合、パーサーは次のトークン（`SCHEMA`、`TABLE`、`VIEW` など）を調べることで、何が作成されるのかを判断します。

各セマンティック概念は、AST 内の _式_ で表現されます。トークンを式に変換することがパーサーの主なタスクです。

AST は SQLGlot の中核概念であるため、その表現、走査、変換に関する詳細なチュートリアルは [こちら](https://github.com/tobymao/sqlglot/blob/main/posts/ast_primer.md) でご覧いただけます。

次のセクションでは、パーサーが SQL 文のトークンリストをどのように処理するかを説明し、その後、方言固有の解析の仕組みについて説明します。

### トークンインデックス
パーサーはトークンリストを順番に処理し、次に消費するトークンを指すインデックス/カーソルを保持します。

現在のトークンの処理が完了すると、パーサーは `_advance()` を呼び出してインデックスをインクリメントし、シーケンス内の次のトークンに移動します。

トークンを消費したが解析できなかった場合、パーサーは `_retreat()` を呼び出してインデックスをデクリメントすることで、前のトークンに戻ることができます。

場合によっては、パーサーの動作は現在のトークンと周囲のトークンの両方に依存します。パーサーは、それぞれ `_prev` プロパティと `_next` プロパティにアクセスすることで、前のトークンと次のトークンを消費せずに「覗き見る」ことができます。

### トークンとテキストの一致
SQL文を解析する際、パーサーはASTを正しく構築するために特定のキーワードセットを識別する必要があります。これは、トークンまたはキーワードのセットを「一致」させることで文の構造を推測することによって行われます。

例えば、ウィンドウ仕様 `SUM(x) OVER (PARTITION BY y)` を解析すると、次のような一致が発生します。
1. `SUM` トークンと `(` トークンが一致し、`x` が識別子として解析されます。
2. `)` トークンが一致します。
2. `OVER` トークンと `(` トークンが一致します。
2. `PARTITION` トークンと `BY` トークンが一致します。
3. パーティション句 `y` が解析されます。
4. `)` トークンが一致します。

`_match` メソッド群は、それぞれ異なる種類の一致処理を実行します。一致した場合は `True` を返し、インデックスを進めます。一致しない場合は `None` を返します。

最もよく使われる `_match` メソッドは次のとおりです。

Method                | Use
-------------------------|-----
**`_match(type)`** | 特定の `TokenType` を持つ単一のトークンの一致を試みます
**`_match_set(types)`** | `TokenType` のセット内の単一のトークンの一致を試みます
**`_match_text_seq(texts)`** | 連続する文字列/テキストのシーケンスの一致を試みます
**`_match_texts(texts)`** | 文字列/テキストのセット内のキーワードの一致を試みます

### `_parse` メソッド
SQLGlot のパーサーは「再帰下降」アプローチを採用しています。つまり、SQL 構文と、組み合わせた SQL 式間の様々な優先順位を理解するために、相互再帰メソッドに依存しています。

例えば、SQL 文 `CREATE TABLE a (col1 INT)` は、作成されるオブジェクトの種類（テーブル）とそのスキーマ（テーブル名、列名、型）を含む `exp.Create` 式に解析されます。スキーマは、各列に対して 1 つの列定義 `exp.ColumnDef` ノードを含む `exp.Schema` ノードに解析されます。

SQL 構造間の関係は、`_parse_create()` のように、名前が「_parse_」で始まるメソッドにエンコードされています。これらを「解析メソッド」と呼びます。

例えば、上記の文を解析するには、エントリポイントメソッド `_parse_create()` が呼び出されます。その後、以下の処理が実行されます。
- `_parse_create()` で、パーサーは `TABLE` が作成されていると判断します。
- テーブルが作成されているため、次にテーブル名が期待されるため、`_parse_table_parts()` が呼び出されます。これは、テーブル名が `catalog.schema.table` のように複数の部分で構成される可能性があるためです。
- テーブル名の処理後、パーサーは `_parse_schema()` を使用してスキーマ定義を解析します。

SQLGlot パーサーの重要な特徴は、解析メソッドが組み合わせ可能であり、引数として渡すことができることです。その仕組みについては次のセクションで説明します。

### ユーティリティメソッド
SQLGlot は、一般的な解析タスクのためのヘルパーメソッドを提供します。コードの重複やトークン/テキストの手動マッチングを削減するために、可能な限りこれらのメソッドを使用する必要があります。

多くのヘルパーメソッドは解析メソッド引数を受け取り、この引数は、解析対象となる句の一部を解析するために呼び出されます。

例えば、これらのヘルパーメソッドは、それぞれ `col1, col2`、`(PARTITION BY y)`、`(colA TEXT, colB INT)` のような SQL オブジェクトのコレクションを解析します。

Method                | Use
-------------------------|-----
**`_parse_csv(parse_method)`** | 適切な `parse_method` 呼び出し可能オブジェクトを使用して、カンマ区切りの値を解析します
**`_parse_wrapped(parse_method)`** | （オプションで）括弧で囲まれた特定の構造を解析します
**`_parse_wrapped_csv(parse_method)`** | （オプションで）括弧で囲まれたカンマ区切りの値を解析します


### コマンドフォールバック
SQL言語仕様と方言のバリエーションは膨大であり、SQLGlotはあらゆる文を解析することはできません。

SQLGlotは、解析できない文に対してエラーを返す代わりに、解析できないコードを`exp.Command`式に格納してフォールバックします。

これにより、SQLGlotは解析できないコードであっても、変更せずにそのまま返します。コードが変更されていないため、方言固有のコンポーネントはトランスパイルされません。

### 方言固有の解析
ベースパーサーの目標は、異なるSQL方言に共通する構造を可能な限り多く表現することです。これにより、パーサーはより柔軟になり、繰り返しが少なく簡潔になります。

方言固有のパーサーの動作は、機能フラグとパーサーオーバーライドの2つの方法で実装されます。

方言間で2つの異なる解析動作が共通している場合、ベースパーサーは両方を実装し、機能フラグを使用して特定の方言でどちらを使用するかを決定します。一方、パーサーオーバーライドは、特定のベースパーサーメソッドを直接置き換えます。

したがって、各方言のパーサークラスでは、以下を指定できます。

- ベースパーサーメソッドの動作を制御するために使用される、`SUPPORTS_IMPLICIT_UNNEST` や `SUPPORTS_PARTITION_SELECTION` などの機能フラグ。

- オブジェクト名として使用できない、`RESERVED_TOKENS` などの、同様の役割を果たすトークンのセット。

- Python ラムダ関数で実装された `token -> Callable` マッピングのセット
- パーサーが左側のトークン（文字列または `TokenType`）に遭遇すると、右側のマッピング関数を呼び出して対応する Expression / AST ノードを作成します。
- ラムダ関数は、式を直接返すか、返すべき式を決定する `_parse_ メソッド` を返します。
- マッピングは、一般的なセマンティックタイプ（例：関数、`ALTER` ステートメントのパーサーなど）ごとにグループ化されており、各タイプには独自の辞書があります。

`token -> Callable` マッピングの例として、文字列キーに基づいて適切な `exp.Func` ノードを構築する `FUNCTIONS` 辞書があります。

```Python3
 FUNCTIONS: t.Dict[str, t.Callable] = {
    "LOG2": lambda args: exp.Log(this=exp.Literal.number(2), expression=seq_get(args, 0)),
    "LOG10": lambda args: exp.Log(this=exp.Literal.number(10), expression=seq_get(args, 0)),
    "MOD": build_mod,
     …,
 }
```

この方言では、`LOG2()` 関数は底 2 の対数を計算します。この対数は SQLGlot では一般的な `exp.Log` 式で表現されます。

対数の底と `log` にユーザーが指定した値は、ノードの引数に格納されます。前者は `this` 引数に整数リテラルとして格納され、後者は `expression` 引数に格納されます。

## ジェネレータ
パーサーが AST を作成した後、ジェネレータモジュールはそれを SQL コードに変換する役割を担います。

各ダイアレクトのジェネレータクラスは、以下を指定できます。

- `ALTER_TABLE_INCLUDE_COLUMN_KEYWORD` などのフラグセット。パーサーと同様に、これらのフラグはジェネレータの基本メソッドの動作を制御します。

- `Expression -> str` マッピングセット
- ジェネレータは AST を再帰的に走査し、各式に対応する文字列を生成します。これは、式 / AST ノードを文字列に変換するパーサーの逆操作と考えることができます。
- マッピングは、一般的なセマンティックタイプ（例：データ型式）ごとにグループ化され、各タイプには独自の辞書があります。

`Expression -> str` マッピングは、以下のいずれかで実装できます。
- `TRANSFORMS` 辞書のエントリ。これは、単一行の生成に最適です。
- `<expr_name>_sql(...)` という形式の関数名。例えば、`exp.SessionParameter` ノード用のSQLを生成するには、方言のジェネレータクラスでメソッド `def sessionparameter_sql(...)` を定義することで、基本ジェネレータメソッドをオーバーライドできます。

### クエリの生成
Generator モジュールのキーメソッドは `sql()` 関数で、各式の文字列表現を生成します。

まず、`TRANSFORMS` 辞書のキーを調べ、`Generator` のメソッド内で `<exprname>_sql()` メソッドを検索することで、式と文字列またはジェネレーターの `Callable` のマッピングを見つけます。

次に、適切な呼び出し可能オブジェクトを呼び出して、対応する SQL コードを生成します。

> [!IMPORTANT]
> 特定の式に `TRANSFORM` エントリと `<expr_name>_sql(...)` メソッドの両方がある場合、ジェネレーターは前者を使用して式を文字列に変換します。

### ユーティリティメソッド
ジェネレーターは、`sql()` メソッドの利便性を向上させる抽象化を含むヘルパーメソッドを定義します。以下に例を示します。

Method                | Use
-------------------------|-----
**`expressions()`** | 一連の引数を使用して、`Expression` のリストの文字列表現を生成します。書式設定に役立ちます。
**`func()`, `rename_func()`** | 関数名と `Expression` を指定すると、式の引数を func の引数として生成し、関数呼び出しの文字列表現を返します。

コードの重複を減らすために、可能な限りこれらの方法を使用する必要があります。

### プリティプリント
プリティプリントとは、フォーマットされ、一貫性のあるスタイルのSQLを生成するプロセスを指します。これにより、可読性、デバッグ、コードレビューが向上します。

SQLGlotは、デフォルトではASTのSQLコードを1行で生成しますが、ユーザーは「プリティ」形式で生成し、改行やインデントを含めるように指定できます。

SQLに空白や改行が正しく埋め込まれていることを確認するのは開発者の責任です。このプロセスを支援するために、Generatorクラスはヘルパーメソッド「sep()」と「seg()」を提供しています。

## 方言
SQLGlot の基本モジュールである Tokenizer、Parser、Generator について説明した際に、方言固有の動作について簡単に触れました。このセクションでは、新しい SQL 方言を指定する方法について詳しく説明します。

[上記で説明したように](#dialect-specific-parsing)、SQLGlot は方言固有のバリエーションを「スーパーセット」方言で橋渡ししようとします。これを _sqlglot_ 方言と呼びます。

その他のすべての SQL 方言には、独自の Dialect サブクラスがあり、その Tokenizer、Parser、Generator コンポーネントは必要に応じて基本モジュールを拡張またはオーバーライドします。

基本方言コンポーネントは、`tokens.py`、`parser.py`、`generator.py` で定義されています。方言の定義は `dialects/<dialect>.py` にあります。

複数の方言で使用される機能を追加する場合、共通部分または重複部分をベースの _sqlglot_ 方言に配置するのが最適です。これにより、他の方言間でのコードの重複を回避できます。方言固有の機能で、繰り返し使用できない（または繰り返し使用すべきでない）機能は、方言のサブクラスで指定できます。

Dialect クラスには、Tokenizer、Parser、および Generator の各コンポーネントから参照可能なフラグが含まれています。コードの重複を避けるため、フラグは少なくとも2つのコンポーネントから参照する必要がある場合にのみ、Dialect に追加されます。

### カスタムダイアレクトの実装
新しいSQLダイアレクトの作成は一見複雑に思えるかもしれませんが、SQLGlotでは実際には非常に簡単です。

この例では、ベースダイアレクトモジュールのコンポーネントを拡張またはオーバーライドする「Custom」という名前の新しいダイアレクトを定義する方法を示します。

```Python
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect
from sqlglot.generator import Generator
from sqlglot.tokens import Tokenizer, TokenType


class Custom(Dialect):
    class Tokenizer(Tokenizer):
        QUOTES = ["'", '"']  # Strings can be delimited by either single or double quotes
        IDENTIFIERS = ["`"]  # Identifiers can be delimited by backticks

        # Associates certain meaningful words with tokens that capture their intent
        KEYWORDS = {
            **Tokenizer.KEYWORDS,
            "INT64": TokenType.BIGINT,
            "FLOAT64": TokenType.DOUBLE,
        }

      class Parser(Parser):
           # Specifies how certain tokens are mapped to function AST nodes
           FUNCTIONS = {
             **parser.Parser.FUNCTIONS,
             "APPROX_PERCENTILE": exp.ApproxQuantile.from_arg_list,
           }

          # Specifies how a specific construct e.g. CONSTRAINT is parsed
          def _parse_constraint(self) -> t.Optional[exp.Expression]:
            return super()._parse_constraint() or self._parse_projection_def()

    class Generator(Generator):
        # Specifies how AST nodes, i.e. subclasses of exp.Expression, should be converted into SQL
        TRANSFORMS = {
            exp.Array: lambda self, e: f"[{self.expressions(e)}]",
        }

        # Specifies how AST nodes representing data types should be converted into SQL
        TYPE_MAPPING = {
            exp.DataType.Type.TINYINT: "INT64",
            exp.DataType.Type.SMALLINT: "INT64",
            ...
        }
```


この例はかなり現実的な出発点ですが、既存の方言の実装を調べて、さまざまなコンポーネントをどのように変更できるかを理解することを強くお勧めします。

## スキーマ
前のセクションでは、トランスパイルに使用する SQLGlot コンポーネントについて説明しました。このセクションと次のセクションでは、列レベルの系統など、その他の SQLGlot 機能を実装するために必要なコンポーネントについて説明します。

スキーマは、データベース スキーマの構造を表します。これには、スキーマに含まれるテーブル/ビュー、それらの列名とデータ型が含まれます。

ファイル `schema.py` は、抽象クラス `Schema` と `AbstractMappingSchema` を定義しており、実装クラスは `MappingSchema` です。

`MappingSchema` オブジェクトは、次の例のように、ネストされた辞書で定義されます。

```Python
schema = MappingSchema({"t": {"x": "int", "y": "datetime"}})
```

最上位の辞書のキーはテーブル名 (`t`)、`t` の辞書のキーは列名 (`x` と `y`)、値は列のデータ型 (`int` と `datetime`) です。

スキーマは、他の SQLGlot モジュール (最も重要なのは、オプティマイザーと列レベルの系統) に必要な情報を提供します。

スキーマ情報は、アスタリスク (`SELECT *`) を上流のテーブルから選択された列名に置き換えるなどのアクションによって AST を強化するために使用されます。

## オプティマイザー
SQLGlot のオプティマイザーモジュール (`optimizer.py`) は、一連の最適化ルールを適用することで、標準化された効率的な SQL クエリを生成します。

最適化には、式の簡素化、冗長な演算の削除、パフォーマンス向上のためのクエリの書き換えなどが含まれます。

オプティマイザーは、パーサーから返された抽象構文木 (AST) を操作し、元のクエリのセマンティクスを維持しながら、よりコンパクトな形式に変換します。

> [!NOTE]
> オプティマイザーは論理的な最適化のみを実行します。基盤となるエンジンは、ほとんどの場合、パフォーマンスの最適化に優れています。

### 最適化ルール
オプティマイザーは基本的に[ルールのリスト](https://sqlglot.com/sqlglot/optimizer.html)を順番に適用します。各ルールはASTを受け取り、正規化および最適化されたバージョンを生成します。

> [!WARNING]
> ルールの順序は重要です。一部のルールは他のルールに依存して適切に動作するため、ルールの順序は重要です。個々のルールを手動で適用することは推奨しません。誤った動作や予測できない動作につながる可能性があります。

最初の基本的な最適化タスクは、他の最適化手順で必要となる AST を標準化することです。これらのルールは正規化を実装します。

#### Qualify
オプティマイザーで最も重要なルールは `qualify` です。これは、すべてのテーブルと列が _normalized正規化_ され _qualified修飾_ されるように AST を書き換える役割を果たします。

##### 識別子の正規化
正規化ステップ（`normalize_identifiers.py`）では、一部の識別子を小文字または大文字に変換し、それぞれのケースで意味が保持されるようにします（例：大文字と小文字の区別を尊重する）。

この変換は、各SQL方言に対応するエンジンが識別子をどのように解決するかを反映しており、ASTの標準化において非常に重要な役割を果たします。

> [!NOTE]
> 一部の方言（例：DuckDB）では、識別子が引用符で囲まれていても大文字と小文字を区別しないため、すべての識別子が正規化されます。

方言によって正規化の方法が異なる場合があります。例えば、Snowflakeでは引用符で囲まれていない識別子は大文字に正規化されます。

次の例は、識別子「Foo」が、デフォルトのsqlglot方言では小文字の「foo」に、Snowflake方言では大文字の「FOO」に正規化される様子を示しています。

```Python
import sqlglot
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

normalize_identifiers("Foo").sql()
# 'foo'

normalize_identifiers("Foo", dialect="snowflake").sql("snowflake")
# 'FOO'
```

##### テーブルと列の修飾
識別子を正規化した後、すべてのテーブル名と列名が修飾されるため、クエリで参照されるデータソースについて曖昧さがなくなります。

つまり、すべてのテーブル参照にエイリアスが割り当てられ、すべての列名にはソーステーブルの名前がプレフィックスとして付加されます。

この例は、クエリ `SELECT col FROM tbl` を修飾する方法を示しています。`tbl` には単一の列 `col` が含まれています。テーブルのスキーマが `qualify()` の引数として渡されることに注意してください。

```Python
import sqlglot
from sqlglot.optimizer.qualify import qualify

schema = {"tbl": {"col": "INT"}}
expression = sqlglot.parse_one("SELECT col FROM tbl")
qualify(expression, schema=schema).sql()

# 'SELECT "tbl"."col" AS "col" FROM "tbl" AS "tbl"'
```

`qualify` ルールには、クエリをさらに正規化するサブルールのセットも用意されています。

例えば、`qualify()` はスターを展開し、各 `SELECT *` を選択された列の射影リストに置き換えることができます。

この例では、前の例のクエリを `SELECT *` を使用するように変更しています。`qualify()` はスターを展開し、前と同じ出力を返します。

```Python
import sqlglot
from sqlglot.optimizer.qualify import qualify


schema = {"tbl": {"col": "INT"}}
expression = sqlglot.parse_one("SELECT * FROM tbl")
qualify(expression, schema=schema).sql()


# 'SELECT "tbl"."col" AS "col" FROM "tbl" AS "tbl"'
```

#### 型推論
一部のSQL演算子の動作は、入力のデータ型によって異なります。

例えば、一部のSQL方言では、`+`は数値の加算または文字列の連結に使用できます。データベースは、列のデータ型に関する知識に基づいて、特定の状況でどの演算を実行するかを決定します。

データベースと同様に、SQLGlotも最適化を実行するために列のデータ型を認識する必要があります。多くの場合、トランスパイルと最適化の両方が適切に機能するには型情報が必要です。

型アノテーションは、少なくとも1つのテーブルの列のデータ型に関するユーザー提供の情報から始まります。次に、SQLGlotはASTを走査し、その型情報を伝播して、各AST式によって返される型を推論します。

AST内のすべての式にアノテーションを正しく付けるには、複数のテーブルの列の型情報を提供する必要がある場合があります。

## 列レベルの系統
列レベルの系統 (CLL) は、SQL コードベースにおいてテーブルが互いに列を選択および変更する際に、列から列へとデータの流れをトレースします。

CLL は、データパイプラインまたはデータベースシステムのさまざまな部分でデータがどのように取得、変換、使用されるかを理解するのに役立つ、データ系統に関する重要な情報を提供します。

SQLGlot の CLL 実装は `lineage.py` で定義されています。これは、互いに `SELECT` を実行するクエリの集合に対して動作します。クエリのグラフ/ネットワークは、2 つのテーブルが互いに `SELECT` を実行しない有向非巡回グラフ (DAG) を形成することを前提としています。

ユーザーは以下を提供する必要があります。
上流の列系統をトレースする「ターゲット」クエリ
ターゲットクエリの上流にあるすべてのテーブルをリンクするクエリ
DAG 内のルートテーブルのスキーマ (列名と型)

この例では、列 `traced_col` の系統をトレースします。

```Python
from sqlglot.lineage import lineage

target_query = “””
WITH cte AS (
  SELECT
    traced_col
  FROM
    intermediate_table
)
SELECT
  traced_col
FROM
  cte
“””

node = lineage(
    column="traced_col",
    sql=target_query,
    sources={"intermediate_table": "SELECT * FROM root_table"},
    schema={"root_table": {"traced_col": "int"}},
)
```

`lineage()` 関数の `sql` 引数はターゲットクエリを受け取り、`sources` 引数はソーステーブル名の辞書と各テーブルを生成するクエリを受け取り、`schema` 引数はグラフのルートテーブルの列名/タイプを受け取ります。

### 実装の詳細
このセクションでは、SQLGlot が前の例の系統をトレースする方法について説明します。

系統をトレースする前に、以下の準備手順を実行する必要があります。

1. `sql` クエリのテーブル参照を `sources` クエリに置き換え、単一のスタンドアロンクエリにします。

この例では、前の例の CTE 内の `intermediate_table` への参照を、それを生成したクエリ `SELECT * FROM root_table` に置き換え、エイリアス `intermediate_table` を付与する方法を示します。

```SQL
WITH cte AS (
  SELECT
    traced_col
  FROM (
    SELECT
      *
    FROM
      root_table
  ) AS intermediate_table
)
SELECT
  traced_col
FROM
  cte
```

2. 前の手順で生成されたクエリを修飾し、`*` を展開して、すべての列参照を対応するソース テーブルで修飾します。

```SQL
WITH cte AS (
  SELECT
    intermediate_table.traced_col
  FROM (
    SELECT
      root_table.traced_col
    FROM
      root_table AS root_table
  ) AS intermediate_table
)
SELECT
  cte.traced_col
FROM
  cte AS cte
```

これらの操作の後、クエリは正規化され、系統をトレースできるようになります。

トレースはトップダウンのプロセスです。つまり、検索は `cte.traced_col`（外側のスコープ）から開始され、内側に向かって走査して、その起点が `intermediate_table.traced_col` であることを見つけます。`intermediate_table.traced_col` は `root_table.traced_col` に基づいています。

検索は最も内側のスコープまで続けられます。その時点ですべての `sources` が尽きたため、`schema` が検索され、ルートテーブル（この例では `root_table`）が存在し、ターゲット列 `traced_col` がそのテーブルに定義されていることが検証されます。

各ステップで、対象の列（つまり `cte.traced_col`、`intermediate_table.traced_col` など）は、上流のノードにリンクされた系統オブジェクト `Node` にラップされます。

このアプローチでは、対象列の系統をトレースする `Nodes` のリンクリストを段階的に構築します: `root_table.traced_col -> intermediate_table.traced_col -> cte.traced_col`。

`lineage()` の出力は、`node.to_html()` 関数を使用して視覚化できます。この関数は、`vis.js` を使用して HTML 系統グラフを生成します。

![Lineage](onboarding_images/lineage_img.png)
