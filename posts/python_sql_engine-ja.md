# Python SQLエンジンをゼロから作成する
[Toby Mao](https://www.linkedin.com/in/toby-mao/)

## はじめに

2021年初頭にSQLGlotの開発を始めた当初は、SparkSQLからPrestoへのSQLクエリの変換、そしてその逆の変換のみを目標としていました。しかし、この1年半で、本格的なSQLエンジンを完成させました。SQLGlotは現在、[18種類のSQL方言](https://github.com/tobymao/sqlglot/blob/main/sqlglot/dialects/__init__.py)間のパースとトランスパイルが可能で、24種類の[TPC-H](https://www.tpc.org/tpch/) SQLクエリをすべて実行できます。パーサーとエンジンはすべてPythonを使ってゼロから開発しました。

この記事では、[なぜ](#why) Python SQLエンジンを作成したのか、そして[どのように](#どのように) 単純なクエリが文字列から実際にデータ変換されるのかを説明します。手順は以下のとおりです。

* [Tokenizing](#tokenizing)
* [Parsing](#parsing)
* [Optimizing](#optimizing)
* [Planning](#planning)
* [Executing](#executing)

## Why?
私は Netflix で [実験およびメトリクス プラットフォーム](https://netflixtechblog.com/reimagining-experimentation-analysis-at-netflix-71356393af21) に携わっていたことがきっかけで、SQLGlot の開発に取り組み始めました。そこでは、データ サイエンティストが SQL ベースのメトリクスを定義および計算できるツールを開発しました。Netflix はデータのクエリに複数のエンジン (Spark、Presto、Druid) に依存していたため、私のチームは Python SQL クエリ ビルダーである [PyPika](https://github.com/kayak/pypika) を中心にメトリクス プラットフォームを構築しました。これにより、定義を複数のエンジン間で再利用できるようになりました。しかし、データ サイエンティスト、特に学術的なバックグラウンドを持つデータ サイエンティストにとっては、プログラムで SQL を生成する Python コードを書くのは困難であることがすぐに明らかになりました。なぜなら、彼らは主に R と SQL に慣れていたからです。当時、Python の SQL パーサーは [sqlparse]([https://github.com/andialbrecht/sqlparse]) しかありませんでした。これは実際にはパーサーではなくトークナイザーなので、ユーザーに生の SQL をプラットフォームに書き込ませることは現実的ではありませんでした。しばらくして、偶然 [Crafting Interpreters](https://craftinginterpreters.com/) を見つけ、これをガイドとして独自の SQL パーサー/トランスパイラーを作成できることに気付きました。

なぜこれをやったかというと、Python SQL エンジンは極端に遅くなるのではないかと思ったからです。

SQLエンジンを構築することになった主な理由は…**単なる**娯楽**でした。SQLクエリを実際に実行するために必要なことをすべて学ぶのは楽しく、実際に動作するのを見るのは非常にやりがいがあります。SQLGlotを使う前は、字句解析器、パーサー、コンパイラの経験は全くありませんでした。

実用的なユースケースとしては、Python SQLエンジンをSQLパイプラインのユニットテストに使用することを計画していました。ビッグデータパイプラインは、多くのエンジンがオープンソースではなく、ローカルで実行できないため、テストが困難です。SQLGlotを使えば、[Snowflake](https://www.snowflake.com/en/)などのウェアハウスをターゲットとしたSQLクエリを、CIでモックPythonデータに対してシームレスに実行できます。すべてがPythonで記述されているため、データのモック作成や任意の[UDF](https://en.wikipedia.org/wiki/User-defined_function)の作成も簡単です。実装は遅く、大量のデータ (100 万行以上) には適していませんが、オーバーヘッドや起動の遅延はほとんどなく、数ミリ秒でテスト データに対してクエリを実行できます。

最後に、実行をサポートするために構築されたコンポーネントは、より高速なエンジンの**基盤**として使用できます。私は、[Apache Calcite](https://github.com/apache/calcite)がJVMの世界にもたらした成果に刺激を受けています。Pythonはデータ処理によく使用されますが、Python用のCalciteはこれまで存在しませんでした。つまり、SQLGlotはそのようなフレームワークを目指していると言えるでしょう。例えば、Python実行エンジンをnumpy/pandas/arrowに置き換えれば、それなりに高性能なクエリエンジンになるのにそれほど手間はかかりません。実装ではパーサー、オプティマイザー、論理プランナーを活用でき、物理的な実行を実装するだけで済みます。Pythonエコシステムでは、高性能なベクトル化計算に関する多くの研究が行われており、純粋なPythonベースの[AST](https://en.wikipedia.org/wiki/Abstract_syntax_tree)/[plan](https://en.wikipedia.org/wiki/Query_plan)から恩恵を受けることができると考えています。クエリ実行のボトルネックがテラバイト規模のデータ処理である場合、解析と計画の高速化は不要です。そのため、ベアメタルパフォーマンスは得られないにもかかわらず、Pythonでの開発の容易さを考えると、SQLを中心としたPythonベースのエコシステムを持つことは有益です。

SQLGlot のツールキットの一部は、現在、以下の用途で使用されています。

* [Ibis](https://github.com/ibis-project/ibis): データラングリングのための軽量で汎用的なインターフェースを提供する Python ライブラリです。
  - Python SQL 式ビルダーを使用し、オプティマイザー/プランナーを活用して SQL をデータフレーム操作に変換します。
* [mysql-mimic](https://github.com/kelsin/mysql-mimic): MySQL サーバーワイヤプロトコルの Pure Python 実装です。
  - SQL を解析/変換し、INFORMATION_SCHEMA クエリを実行します。
* [Quokka](https://github.com/marsupialtail/quokka): プッシュベースのベクトル化クエリエンジンです。
  - SQL を解析および最適化します。
* [Splink](https://github.com/moj-analytical-services/splink): 任意の SQL バックエンドを使用して、高速、正確、かつスケーラブルな確率的データリンケージを実現します。
  - クエリをトランスパイルします。

## How?

次のような単純なクエリを実際に実行するには、多くの手順が必要です。

```sql
SELECT
  bar.a,
  b + 1 AS b
FROM bar
JOIN baz
  ON bar.a = baz.a
WHERE bar.a > 1
```

この記事では、SQLGlot が Python オブジェクトに対してこのクエリを実行するために実行するすべての手順について説明します。

## Tokenizing

最初のステップは、SQL文字列をトークンのリストに変換することです。SQLGlotのトークナイザーは非常にシンプルで、[こちら](https://github.com/tobymao/sqlglot/blob/main/sqlglot/tokens.py)から入手できます。whileループ内で各文字をチェックし、現在のトークンに文字を追加するか、新しいトークンを作成します。

SQLGlotトークナイザーを実行すると、出力が表示されます。

![Tokenizer Output](python_sql_engine_images/tokenizer.png)

各キーワードはSQLGlotトークンオブジェクトに変換されています。各トークンには、エラーメッセージの行/列情報など、いくつかのメタデータが関連付けられています。コメントもトークンの一部であるため、保持されます。

## Parsing

SQL文がトークン化されると、空白文字やその他の書式設定を気にする必要がなくなり、作業がしやすくなります。これで、トークンのリストをASTに変換できるようになりました。SQLGlot [パーサー](https://github.com/tobymao/sqlglot/blob/main/sqlglot/parser.py) は、手書きの [再帰降下](https://en.wikipedia.org/wiki/Recursive_descent_parser) パーサーです。

トークナイザーと同様に、トークンを順番に処理しますが、再帰アルゴリズムを使用します。トークンは、SQLクエリを表す単一のASTノードに変換されます。SQLGlot パーサーはさまざまな方言をサポートするように設計されているため、解析機能をオーバーライドするための多くのオプションが含まれています。

![Parser Output](python_sql_engine_images/parser.png)

ASTは、与えられたSQLクエリの汎用的な表現です。各方言は、独自のジェネレーターをオーバーライドまたは実装することができ、ASTオブジェクトを構文的に正しいSQLに変換できます。

## Optimizing

AST ができたら、それを同等のクエリに変換し、より効率的に同じ結果を得ることができます。クエリを最適化する際、ほとんどのエンジンはまず AST を論理プランに変換し、その後そのプランを最適化します。しかし、私は以下の理由から **AST を直接最適化** することにしました。

1. 入力と出力がどちらも SQL の場合、最適化のデバッグと [検証](https://github.com/tobymao/sqlglot/blob/main/tests/fixtures/optimizer) が容易になります。

2. ルールを個別に適用して、SQL をより望ましい形式に変換できます。

3. 「標準的な SQL」を生成する方法が欲しかったのです。SQL の標準的な表現があれば、2 つのクエリが意味的に同等かどうか（例: `SELECT 1 + 1` と `SELECT 2`）を理解するのに役立ちます。

このアプローチを採用している他のエンジンはまだ見つかっていませんが、この決定には非常に満足しています。オプティマイザは現在、結合順序の変更などの「物理的な最適化」は行いません。追加の統計情報や情報が重要になる可能性があるため、これらの最適化は実行層に委ねられています。

![Optimizer Output](python_sql_engine_images/optimizer.png)

オプティマイザには現在[17個のルール](https://github.com/tobymao/sqlglot/tree/main/sqlglot/optimizer)があります。これらのルールをそれぞれ適用することで、ASTをその場で変換します。これらのルールを組み合わせることで、「標準的な」SQLが作成され、論理プランへの変換と実行が容易になります。

ルールの例をいくつか挙げます。

### qualify\_tables and qualify_columns
- すべての db/catalog 修飾子をテーブルに追加し、別名を強制します。
- 各列が明確であることを確認し、スターを展開します。

```sql
SELECT * FROM x;

SELECT "db"."x" AS "x";
```

### simplify
ブール値と算術式の簡素化。すべての[テストケース](https://github.com/tobymao/sqlglot/blob/main/tests/fixtures/optimizer/simplify.sql)を確認してください。

```sql
((NOT FALSE) AND (x = x)) AND (TRUE OR 1 <> 3);
x = x;

1 + 1;
2;
```

### normalize
すべての述語を [接続正規形](https://en.wikipedia.org/wiki/Conjunctive_normal_form) に変換しようとします。

```sql
-- DNF
(A AND B) OR (B AND C AND D);

-- CNF
(A OR C) AND (A OR D) AND B;
```

### unnest\_subqueries
述語内のサブクエリを結合に変換します。

```sql
-- The subquery can be converted into a left join
SELECT *
FROM x AS x
WHERE (
  SELECT y.a AS a
  FROM y AS y
  WHERE x.a = y.a
) = 1;

SELECT *
FROM x AS x
LEFT JOIN (
  SELECT y.a AS a
  FROM y AS y
  WHERE TRUE
  GROUP BY y.a
) AS "_u_0"
  ON x.a = "_u_0".a
WHERE ("_u_0".a = 1 AND NOT "_u_0".a IS NULL)
```

### pushdown_predicates
フィルターを最も内側のクエリにプッシュダウンします。
```sql
SELECT *
FROM (
  SELECT *
  FROM x AS x
) AS y
WHERE y.a = 1;

SELECT *
FROM (
  SELECT *
  FROM x AS x
  WHERE y.a = 1
) AS y WHERE TRUE
```

### annotate_types
スキーマ情報と関数型定義が与えられた場合、AST 全体のすべての型を推論します。

## Planning
SQL ASTが「最適化」されると、[論理プランへの変換](https://github.com/tobymao/sqlglot/blob/main/sqlglot/planner.py)がはるかに簡単になります。ASTは走査され、5つのステップのいずれかで構成される[DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)に変換されます。各ステップは以下のとおりです。

### Scan
テーブルから列を選択し、投影を適用し、最後にテーブルをフィルターします。

### Sort
式に基づいてテーブルを並べ替えます。

### Set
演算子 union/union all/except/intersect を適用します。

### Aggregate
集計/グループ化を適用します。

### Join
複数のテーブルを結合します。

![Planner Output](python_sql_engine_images/planner.png)

論理計画は非常に単純で、それを物理計画 (実行) に変換するために必要な情報が含まれています。

## Executing
ついに、SQLクエリを実際に実行できるようになりました。[Pythonエンジン](https://github.com/tobymao/sqlglot/blob/main/sqlglot/executor/python.py)は高速ではありませんが、非常に軽量です（約400行）。キューを使ってDAGを反復処理し、各ステップを実行して、中間テーブルを次のステップに渡します。

シンプルさを保つため、式は `eval` で評価されます。SQLGlot は主にトランスパイラとして構築されているため、「Python SQL」方言を簡単に作成できます。つまり、SQL 式 `x + 1` は `scope['x'] + 1` に変換できます。

![Executor Output](python_sql_engine_images/executor.png)

## What's next
SQLGlot の主な焦点は常にパース/トランスパイルにありますが、実行エンジンの開発は継続していく予定です。[TPC-DS](https://www.tpc.org/tpcds/) に合格したいと思っています。もし誰かに先を越されなければ、Pandas/Arrow 実行エンジンの開発にも挑戦するかもしれません。

SQLGlot が、Java における Calcite のように、Python SQL エコシステムの活性化に繋がることを期待しています。

## Special thanks
SQLGlotは、コアコントリビューターの存在なしには今の姿にはなれませんでした。特に、実行エンジンは[Barak Alon](https://github.com/barakalon)と[George Sittas](https://github.com/GeorgeSittas)なしには存在しなかったでしょう。

## Get in touch
SQLGlot についてさらに詳しくお話ししたい場合は、私の [Slack チャンネル](https://tobikodata.com/slack) にご参加ください。