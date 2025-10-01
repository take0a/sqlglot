# SQLGlot の抽象構文木入門

SQLGlot は SQL を解析・変換するための強力なツールですが、習得に時間がかかる場合があります。

この記事は、SQLGlot の抽象構文木の概要、その走査方法、そして変更方法について初心者向けに解説することを目的としています。

## ツリー

SQLGlot は SQL を抽象構文木 (AST) に解析します。

```python
from sqlglot import parse_one

ast = parse_one("SELECT a FROM (SELECT a FROM x) AS x")
```

ASTはSQL文を表すデータ構造です。特定のASTの構造を把握する最良の方法は、Pythonの組み込み関数「repr」を使うことです。

```python
repr(ast)

# Select(
#   expressions=[
#     Column(
#       this=Identifier(this=a, quoted=False))],
#   from=From(
#     this=Subquery(
#       this=Select(
#         expressions=[
#           Column(
#             this=Identifier(this=a, quoted=False))],
#         from=From(
#           this=Table(
#             this=Identifier(this=x, quoted=False)))),
#       alias=TableAlias(
#         this=Identifier(this=x, quoted=False)))))
```

これは内部データ構造のテキスト表現です。その構成要素の一部を以下に説明します。

```
`Select`は式の種類です
  |
Select(
  expressions=[  ------------------------------- `expressions` は `Select` の子キーです
    Column(  ----------------------------------- `Column`は子の式タイプです
      this=Identifier(this=a, quoted=False))],
  from=From(  ---------------------------------- `from` は `Select` の別の子キーです
  ...
```

## ツリーのノード

このツリー内のノードは `sqlglot.Expression` のインスタンスです。ノードは `args` で子ノードを参照し、 `parent` で親ノードを参照します。

```python
ast.args
# {
#    "expressions": [Column(this=...)],
#    "from": From(this=...),
#    ...
# }

ast.args["expressions"][0]
# Column(this=...)

ast.args["expressions"][0].args["this"]
# Identifier(this=...)

ast.args["from"]
# From(this=...)

assert ast.args["expressions"][0].args["this"].parent.parent is ast
```

子要素は次のいずれかになります。
1. Expression インスタンス
2. Expression インスタンスのリスト
3. str や bool などの別の Python オブジェクト。これは常にツリーのリーフノードになります。

このツリーをナビゲートするには、様々なExpression型を理解している必要があります。Expression型を参照する最良の方法は、[expressions.py](../sqlglot/expressions.py)のコードを直接参照することです。あるExpression型の簡略版を見てみましょう。

```python
class Column(Expression):
    arg_types = {
      "this": True,
      "table": False,
      ...
    }
```

`Column` は `Expression` のサブクラスです。

`arg_types` は、可能な子要素を指定するクラス属性です。Expression インスタンスの `args` キーは、そのクラスの `arg_types` キーに対応します。キーが必須の場合、`arg_types` 辞書の値は `True` になります。

一般的な `arg_types` キーには次のようなものがあります。
- "this": これは通常、主子要素に使用されます。`Column` では、"this" は列名の識別子です。
- "expression": これは通常、セカンダリ子要素に使用されます。
- "expressions": これは通常、プライマリ子要素のリストに使用されます。

これらのキーの使用法については厳密なルールはありませんが、すべての式型で利用可能な便利なメソッドのいくつかで役立ちます。
- `Expression.this`: `self.args.get("this")` の省略形
- `Expression.expression`: 同様に、式引数 arg の省略形
- `Expression.expressions`: 同様に、式リスト引数 arg の省略形
- `Expression.name`: `this` のテキスト名

`arg_types` は、子要素の可能な式型を指定しません。これは、特定の AST を走査するコードを記述していて、どのような結果になるかがわからない場合に問題となる可能性があります。よくある対策としては、サンプルクエリを解析して `repr` を出力することが挙げられます。

引数だけを使用して AST をトラバースすることもできますが、プログラムによるトラバース用の高階関数もいくつかあります。

> [!NOTE]
> SQLGlotは、様々な方言のSQLを解析・生成できます。しかし、すべての方言に対応する式型は1セットしかありません。ASTはすべての方言のスーパーセットを表現できると言えるでしょう。
>
> 場合によっては、SQLGlot は SQL を方言から予期しない Expression タイプに解析することがあります。
>
> ```python
> ast = parse_one("SELECT NOW()", dialect="postgres")
>
> repr(ast)
> # Select(
> #   expressions=[
> #     CurrentTimestamp()])
> ```
>
> これは、SQLGlot が方言を標準 AST に収束させようとするためです。つまり、多くの場合、複数の方言を扱うコードを 1 つ記述できます。

## AST の走査

SQL 文を分析するには、このデータ構造を走査する必要があります。これにはいくつかの方法があります。

### 引数

AST の構造を理解している場合は、上記と同様に `Expression.args` を使用できます。ただし、任意の SQL を扱う場合、この方法は制限される可能性があります。

### Walk メソッド

`Expression` の Walk メソッド (`find`、`find_all`、`walk`) は、AST を解析する最もシンプルな方法です。

`find` と `find_all` は、AST 内で特定の Expression 型を検索します。

```python
from sqlglot import exp

ast.find(exp.Select)
# Select(
#   expressions=[
#     Column(
#       this=Identifier(this=a, quoted=False))],
# ...

list(ast.find_all(exp.Select))
# [Select(
#   expressions=[
#     Column(
#       this=Identifier(this=a, quoted=False))],
# ...
```

`find` と `find_all` はどちらも `walk` に基づいて構築されており、よりきめ細かい制御が可能です。

```python
for node in ast.walk():
    ...
```

> [!WARNING]
> ウォーク方式のよくある落とし穴は次のとおりです。
> ```python
> ast.find_all(exp.Table)
> ```
> 一見すると、これはクエリ内のすべてのテーブルを検索する優れた方法のように思えます。しかし、`Table` インスタンスは必ずしもデータベース内のテーブルとは限りません。以下は、この方法が失敗する例です。
> ```python
> ast = parse_one("""
> WITH x AS (
>   SELECT a FROM y
> )
> SELECT a FROM x
> """)
>
> # This is NOT a good way to find all tables in the query!
> for table in ast.find_all(exp.Table):
>     print(table)
>
> # x  -- this is a common table expression, NOT an actual table
> # y
> ```
>
> クエリのより深い意味的理解を必要とする AST のプログラムによるトラバーサルには、「スコープ」が必要です。

### スコープ

スコープは、SQLクエリのよりセマンティックなコンテキストを処理するトラバーサルモジュールです。`walk`メソッドよりも使いにくいですが、より強力です。

```python
from sqlglot.optimizer.scope import build_scope

ast = parse_one("""
WITH x AS (
  SELECT a FROM y
)
SELECT a FROM x
""")

root = build_scope(ast)
for scope in root.traverse():
    print(scope)

# Scope<SELECT a FROM y>
# Scope<WITH x AS (SELECT a FROM y) SELECT a FROM x>
```

クエリ内のすべてのテーブルをより効率的に見つけるためにこれを使用しましょう。

```python
tables = [
    source

    # Traverse the Scope tree, not the AST
    for scope in root.traverse()

    # `selected_sources` contains sources that have been selected in this scope, e.g. in a FROM or JOIN clause.
    # `alias` is the name of this source in this particular scope.
    # `node` is the AST node instance
    # if the selected source is a subquery (including common table expressions),
    #     then `source` will be the Scope instance for that subquery.
    # if the selected source is a table,
    #     then `source` will be a Table instance.
    for alias, (node, source) in scope.selected_sources.items()
    if isinstance(source, exp.Table)
]

for table in tables:
    print(table)

# y  -- Success!
```

`build_scope` は `Scope` クラスのインスタンスを返します。`Scope` には、クエリを検査するための多数のメソッドがあります。これらのメソッドを参照する最良の方法は、[scope.py](../sqlglot/optimizer/scope.py) のコード内を直接参照することです。また、SQLGlot の [optimizer](../sqlglot/optimizer) モジュール全体で Scope の使用例を探すこともできます。

Scope の多くのメソッドは、完全修飾 SQL 式に依存します。例えば、次のクエリの列の系統をトレースするとします。

```python
ast = parse_one("""
SELECT
  a,
  c
FROM (
  SELECT
    a,
    b
  FROM x
) AS x
JOIN (
  SELECT
    b,
    c
  FROM y
) AS y
  ON x.b = y.b
""")
```

外側のクエリだけを見ても、サブクエリの列を見なければ、列 `a` がテーブル `x` から来ていることは明らかではありません。

[qualify](../sqlglot/optimizer/qualify.py) 関数を使用すると、次のように AST 内のすべての列にテーブル名のプレフィックスを付けることができます。

```python
from sqlglot.optimizer.qualify import qualify

qualify(ast)
# SELECT
#   x.a AS a,
#   y.c AS c
# FROM (
#   SELECT
#     x.a AS a,
#     x.b AS b
#   FROM x AS x
# ) AS x
# JOIN (
#   SELECT
#     y.b AS b,
#     y.c AS c
#   FROM y AS y
# ) AS y
#   ON x.b = y.b
```

これで、列のソースをトレースできるようになりました。修飾されたAST内のすべての列に対応するテーブルまたはサブクエリを見つけるには、次のようにします。

```python
from sqlglot.optimizer.scope import find_all_in_scope

root = build_scope(ast)

# `find_all_in_scope` is similar to `Expression.find_all`, except it doesn't traverse into subqueries
for column in find_all_in_scope(root.expression, exp.Column):
    print(f"{column} => {root.sources[column.table]}")

# x.a => Scope<SELECT x.a AS a, x.b AS b FROM x AS x>
# y.c => Scope<SELECT y.b AS b, y.c AS c FROM y AS y>
# x.b => Scope<SELECT x.a AS a, x.b AS b FROM x AS x>
# y.b => Scope<SELECT y.b AS b, y.c AS c FROM y AS y>
```

列系統のトレースの完全な例については、[lineage](../sqlglot/lineage.py) モジュールを参照してください。

> [!NOTE]
> 一部のクエリでは、曖昧さを解消するためにデータベーススキーマが必要です。例:
>
> ```sql
> SELECT a FROM x CROSS JOIN y
> ```
>
> 列 `a` はテーブル `x` または `y` から取得される可能性があります。これらの場合、`schema` を `qualify` に渡す必要があります。

## ツリーの変更

AST を変更したり、ゼロから構築したりすることも可能です。これにはいくつかの方法があります。

### 高水準ビルダーメソッド

SQLGlot には、ORM と同様にプログラムで式を構築するためのメソッドが用意されています。

```python
ast = (
    exp
    .select("a", "b")
    .from_("x")
    .where("b < 4")
    .limit(10)
)
```

> [!WARNING]
> 高水準ビルダーメソッドは、文字列引数を式に変換しようとします。これは非常に便利ですが、文字列の方言を念頭に置いてください。特定の方言で記述されている場合は、`dialect` 引数を設定する必要があります。
>
> 式を引数として渡すことで解析を回避できます。例: `.where("b < 4")` ではなく `.where(exp.column("b") < 4)`

これらのメソッドは、解析したものも含め、任意の AST で使用できます。

```python
ast = parse_one("""
SELECT * FROM (SELECT a, b FROM x)
""")

# To modify the AST in-place, set `copy=False`
ast.args["from"].this.this.select("c", copy=False)

print(ast)
# SELECT * FROM (SELECT a, b, c FROM x)
```

利用可能なすべての高レベルビルダー メソッドとそのパラメーターを参照するのに最適な場所は、いつものように、[expressions.py](../sqlglot/expressions.py) のコード内です。

### 低レベルビルダーメソッド

高レベルビルダーメソッドは、構築したい可能性のあるすべての式に対応しているわけではありません。特定の高レベルメソッドが不足している場合は、低レベルメソッドを使用してください。以下に例を示します。

```python
node = ast.args["from"].this.this

# These all do the same thing:

# high-level
node.select("c", copy=False)
# low-level
node.set("expressions", node.expressions + [exp.column("c")])
node.append("expressions", exp.column("c"))
node.replace(node.copy().select("c"))
```
> [!NOTE]
> 一般的には、`Expression.args` を直接変更するのではなく、`Expression.set` と `Expression.append` を使用する必要があります。`set` と `append` は、`parent` などのノード参照の更新を適切に行います。

AST ノードを直接インスタンス化することもできます。

```python
col = exp.Column(
    this=exp.to_identifier("c")
)
node.append("expressions", col)
```

> [!WARNING]
> SQLGlotは引数の型を検証しないため、SQLを正しく生成できない無効なASTノードをインスタンス化してしまう可能性があります。上記の方法を用いて、ノードの想定される型を注意深く検査してください。

### 変換

`Expression.transform` メソッドは、AST 内のすべてのノードに深さ優先、事前順序で関数を適用します。

```python
def transformer(node):
    if isinstance(node, exp.Column) and node.name == "a":
        # Return a new node to replace `node`
        return exp.func("FUN", node)
    # Or return `node` to do nothing and continue traversing the tree
    return node

print(parse_one("SELECT a, b FROM x").transform(transformer))
# SELECT FUN(a), b FROM x
```

> [!WARNING]
> walk メソッドと同様に、`transform` はスコープを管理しません。複雑な式で列やテーブルを安全に変換するには、おそらく Scope を使用するのがよいでしょう。

## まとめ

SQLGlot は SQL 文を抽象構文木 (AST) に変換します。AST のノードは `sqlglot.Expression` のインスタンスです。

AST を走査する方法は 3 つあります。
1. **args** - 処理する AST の正確な構造がわかっている場合に使用します。
2. **walk メソッド** - 最も簡単な方法です。単純なケースに使用します。
3. **scope** - 最も難しい方法です。クエリのセマンティックコンテキストを処理する必要がある複雑なケースに使用します。

AST を変更する方法は 3 つあります。
1. **高レベルビルダーメソッド** - 処理する AST の正確な構造がわかっている場合に使用します。
2. **低レベルビルダーメソッド** - 構築しようとしている AST に高レベルビルダーメソッドが存在しない場合にのみ使用します。
3. **transform** - 任意の文に対する単純な変換に使用します。

もちろん、これらのメカニズムは組み合わせて使用​​できます。例えば、任意の AST を走査するためにスコープを使用し、それをインプレースで変更するために高水準ビルダーメソッドを使用する必要があるかもしれません。

まだサポートが必要ですか？[お問い合わせください！](../README.md#get-in-touch)
