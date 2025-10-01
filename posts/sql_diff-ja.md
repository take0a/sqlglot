# SQL のセマンティック Diff
*by [Iaroslav Zeigerman](https://github.com/izeigerman)*

## 動機

ソフトウェアは常に変化し進化しており、変更点を特定し、それらの変更点をレビューすることは開発プロセスの不可欠な部分です。SQLコードも例外ではありません。

`git diff` などのテキストベースの diff ツールをコードベースに適用する場合、いくつかの制限があります。まず、挿入と削除しか検出できず、個々のコード部分の移動や更新は検出できません。次に、このようなツールはテキスト行間の変更しか検出できないため、ソースコードのように粒度が細かく詳細なものには粗すぎます。さらに、このような diff の結果は、基盤となるコードのフォーマットに依存し、フォーマットが変更されると結果も異なります。

Git によって生成された次の diff を考えてみましょう。

![Git diff output](sql_diff_images/git_diff_output.png)

クエリの意味は変更されていません。2つの引数 `b` と `c` が入れ替わった（移動された）だけで、クエリの出力には影響がありません。しかし、Git は影響を受けた式全体を、無関係な要素の塊と一緒に置き換えてしまいました。

テキストベースの比較の代わりに、抽象構文木（AST）を比較する方法があります。AST の主な利点は、コード解析によって直接得られるものであり、基盤となるコード構造を任意の粒度で表現できることです。AST を比較することで、非常に正確な比較結果が得られるだけでなく、コードの移動や更新などの変更も検出できます。さらに重要なのは、このアプローチによって、2つのバージョンのソースコードを並べて目視確認する以上のユースケースが実現できることです。

セマンティック比較の旅に出ようと決めたとき、SQL のユースケースとして私が考えていたのは以下の点です。

* **クエリ類似度スコア。** 2 つのクエリの共通部分を特定し、統合や中間テーブル/ステージングテーブルの作成など、どのような機能的な変更を行うかを自動的に提案します。
* **外観/構造上の変更と機能上の変更を区別します。** 例えば、ネストされたクエリを共通テーブル式 (CTE) にリファクタリングする場合、この種の変更はクエリ自体にも結果にも機能的な影響を与えません。
* **データの遡及的なバックフィルの必要性に関する自動提案。** これは、非常に大きなテーブルにデータを入力するパイプラインで特に重要であり、再ステートメント処理は実行時に大量の処理を必要とします。単純なコード移動と実際の変更を区別できれば、変更の影響を評価し、それに応じた提案を行うことができます。

この記事で説明した実装は、[SQLGlot](https://github.com/tobymao/sqlglot/) ライブラリの一部となっています。完全なソースコードは[diff.py](https://github.com/tobymao/sqlglot/blob/main/sqlglot/diff.py)モジュールにあります。SQLglotを選んだのは、シンプルながらも強力なAPI、外部依存性の少なさ、そして何よりもサポートされるSQL方言の豊富さから当然のことでした。

## 解決策の探求

セマンティック比較ツールに限らず、あらゆる比較ツールにおいて、主な課題は比較対象となるエンティティの要素を可能な限り多く一致させることです。このような一致要素の集合が利用可能になれば、変更のシーケンスを導き出すのは容易になります。

要素に一意の識別子（例えば、DOM内の要素ID）が関連付けられている場合、一致の問題は単純です。しかし、比較対象となるSQL構文木には、一致のために使用できる一意のキーもオブジェクト識別子もありません。では、関連するノードのペアをどのように見つければよいのでしょうか？

この問題をより分かりやすく説明するために、次のSQL式を比較してみましょう。`SELECT a + b + c, d, e` と `SELECT a - b + c, e, f` です。それぞれの構文木から個々のノードを一致させる様子は、次のように視覚化できます。

![Figure 1: Example of node matching for two SQL expression trees](sql_diff_images/figure_1.png)
*図 1: 2 つの SQL 式ツリーのノード マッチングの例。*

上記の2つのSQL式ツリーのノードマッチングの図を見ると、以下の変更が私たちのソリューションで捕捉できることがわかります。

* 挿入されたノード: `Sub` と `f`。これらは、ソースASTに一致するノードがない、ターゲットASTのノードです。
* 削除されたノード: `Add` と `d`。これらは、ソースASTに一致するノードがない、ターゲットASTのノードです。
* 残りのノードは変更なしとして識別する必要があります。

ここまでで、ソースツリーのノードとターゲットツリーの対応するノードをマッチングできれば、差分の計算は簡単になることが明らかでしょう。

### 単純な総当たり法

単純な解決策としては、ノードペアの組み合わせをあらゆる順列で試し、何らかのヒューリスティックに基づいてどのペアの組み合わせが最もパフォーマンスが良いかを調べることが挙げられます。このような解決策の実行コストはすぐに脱出速度に達します。両方のツリーにそれぞれ10個のノードしかない場合、そのようなペアの数はおよそ10! ^ 2 = 3.6M ^ 2 ~= 13 * 10^12となります。これは階乗計算量の非常に悪い例です（正確には、実際にはもっと悪い、O(n! ^ 2)ですが、適切な名前が思いつきませんでした）。そのため、このアプローチをこれ以上検討する必要はほとんどありません。

### マイヤーズアルゴリズム

ナイーブなアプローチが実行不可能であることが証明された後、次に私が自問したのは「git diffはどのように動作するのか？」でした。この疑問から、マイヤーズdiffアルゴリズム[1]を発見しました。このアルゴリズムは、文字列のシーケンスを比較するために設計されています。その本質は、最初のシーケンスを2番目のシーケンスに変換する編集の可能なグラフ上で最短経路を探し、変更されていない要素の部分シーケンスが最も長くなる経路に大きな報酬を与えることです。このアルゴリズムをより詳細に説明した資料は数多く存在します。中でも、James Coglanによる一連の[ブログ投稿](https://blog.jcoglan.com/2017/02/12/the-myers-diff-algorithm-part-1/)が最も包括的だと感じました。

そのため、私はツリーを位相順に走査してシーケンスに変換し、結果のシーケンスに Myers アルゴリズムを適用しながら、2 つのノードが等しいかどうかをチェックするカスタム ヒューリスティックを使用するという「素晴らしい」(実際はそうでもない) アイデアを思いつきました。当然のことながら、文字列のシーケンスの比較は階層的なツリー構造の比較とはまったく異なり、ツリーをシーケンスに平坦化することで、多くの関連するコンテキストが失われます。その結果、このアルゴリズムは AST でひどいパフォーマンスになりました。2 つのツリーがほぼ同じであっても、完全に無関係なノードと一致することが多く、全体的に変更のリストが非常に不正確でした。少しいじって等価性ヒューリスティックを微調整して精度を向上させた後、最終的に実装全体を破棄して、最初からやり直しました。

## Change Distiller

最終的に私が採用したアルゴリズムは、Fluriら [2] が作成したChange Distillerです。これは、Chawatheら [3] が説明したコアアイデアを改良したものです。

このアルゴリズムは、2つの高レベルなステップで構成されています。

1. **比較対象となるASTに含まれるノードペア間の適切なマッチングを見つける。** 「適切な」マッチングとは何かを特定することも、このステップの一部です。
2. **最初のステップで構築されたマッチングセットから、いわゆる「編集スクリプト」を生成する。** 編集スクリプトとは、個々のツリーノードに対する編集操作（挿入、削除、更新など）のシーケンスであり、ソースASTに変換として適用すると、最終的にターゲットASTになります。一般的に、シーケンスは短いほど良いとされています。編集スクリプトの長さは、異なるアルゴリズムのパフォーマンスを比較するために使用できますが、これが唯一の重要な指標ではありません。

このセクションの残りの部分では、SQLGlot ライブラリによって提供される AST 実装を使用して、上記の手順を Python で実装する方法について説明します。

### マッチングセットの構築
#### リーフノードのマッチング

マッチングセットの作成は、まずリーフノードのマッチングから始めます。リーフノードとは、子ノード（リテラル、識別子など）を持たないノードです。これらのマッチングを行うために、ソースツリーからすべてのリーフノードを集め、ターゲットツリーのすべてのリーフノードとの直積を生成し、このようにして作成されたペアを比較して類似度スコアを割り当てます。この段階では、基本的なマッチング基準を満たさないペアも除外します。そして、各ノードが複数回マッチングされないことを確認しながら、最も高いスコアを獲得したペアを選択します。

本記事の冒頭で示した例を用いて、マッチング候補の初期セットを構築するプロセスを図2に示します。

![Figure 2: Building a set of candidate matchings between leaf nodes. The third item in each triplet represents a similarity score between two nodes.](sql_diff_images/figure_2.gif)
*図2: リーフノード間の候補マッチングセットの構築。各トリプレットの3番目の項目は、2つのノード間の類似度スコアを表します。*

まず、類似度スコアを分析しましょう。次に、マッチング基準について説明します。

Fluriら[2]が提案した類似度スコアは、各ノード値の[バイグラム](https://en.wikipedia.org/wiki/Bigram)に[ダイス係数](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)を適用したものです。バイグラムとは、文字列からスライディングウィンドウ方式で計算された2つの隣接する要素のシーケンスです。

```python
def bigram(string):
    count = max(0, len(string) - 1)
    return [string[i : i + 2] for i in range(count)]
```

理由はすぐに明らかになりますが、実際には単なるシーケンスではなく、バイグラム ヒストグラムを計算する必要があります。

```python
from collections import defaultdict

def bigram_histo(string):
    count = max(0, len(string) - 1)
    bigram_histo = defaultdict(int)
    for i in range(count):
        bigram_histo[string[i : i + 2]] += 1
    return bigram_histo
```

サイコロ係数の式は次のようになります。

![Dice Coefficient](sql_diff_images/dice_coef.png)

ここで、Xはソースノードのバイグラム、Yは2番目のノードのバイグラムです。これは基本的に、2つのノードに共通するバイグラム要素の数を数え、それを2倍にして、両方のバイグラムの要素の総数で割ります。ここでバイグラムヒストグラムが役立ちます。

```python
def dice_coefficient(source, target):
    source_histo = bigram_histo(source.sql())
    target_histo = bigram_histo(target.sql())

    total_grams = (
        sum(source_histo.values()) + sum(target_histo.values())
    )
    if not total_grams:
        return 1.0 if source == target else 0.0

    overlap_len = 0
    overlapping_grams = set(source_histo) & set(target_histo)
    for g in overlapping_grams:
        overlap_len += min(source_histo[g], target_histo[g])

    return 2 * overlap_len / total_grams
```

ツリーノードからバイグラムを計算するには、まずノードを標準的なSQL表現に変換します。つまり、`Literal(123)`ノードは単に“123”に、`Identifier(“a”)`ノードは単に“a”になります。また、文字列が短すぎてバイグラムを導出できない場合にも対応します。この場合、2つのノードの等価性をチェックするフォールバックを行います。

類似度スコアの計算方法がわかったので、リーフノードのマッチング基準を考慮できます。元の論文[2]では、マッチング基準は次のように定式化されています。

![Matching criteria for leaf nodes](sql_diff_images/matching_criteria_1.png)

2つのノードは、以下の2つの条件が満たされた場合にマッチングされます。

1. ノードラベルが一致する（このケースでは、ラベルはノードタイプのみです）。
2. ノード値の類似度スコアが、ある閾値「f」以上である。論文の著者は、「f」の値を0.6に設定することを推奨しています。

構成要素が揃ったので、リーフノードのマッチングセットを構築できます。まず、マッチング候補のリストを生成します。

```python
from heapq import heappush, heappop

candidate_matchings = []
source_leaves = _get_leaves(self._source)
target_leaves = _get_leaves(self._target)
for source_leaf in source_leaves:
    for target_leaf in target_leaves:
        if _is_same_type(source_leaf, target_leaf):
            similarity_score = dice_coefficient(
                source_leaf, target_leaf
            )
            if similarity_score >= 0.6:
                heappush(
                    candidate_matchings,
                    (
                        -similarity_score,
                        len(candidate_matchings),
                        source_leaf,
                        target_leaf,
                    ),
                )
```

上記の実装では、割り当てられた類似度スコアに基づいて正しい順序を自動的に維持するために、各マッチングペアをヒープにプッシュします。

最後に、スコアが最も高いリーフペアを選択して、初期マッチングセットを構築します。

```python
matching_set = set()
while candidate_matchings:
    _, _, source_leaf, target_leaf = heappop(candidate_matchings)
    if (
        source_leaf in unmatched_source_nodes
        and target_leaf in unmatched_target_nodes
    ):
        matching_set.add((source_leaf, target_leaf))
        unmatched_source_nodes.remove(source_leaf)
        unmatched_target_nodes.remove(target_leaf)
```

一致するセットを確定するには、内部ノードの一致に進む必要があります。

#### 内部ノードのマッチング

内部ノードのマッチングは、リーフノードのマッチングと非常に似ていますが、以下の2つの違いがあります。

* 候補となるノード群をランク付けするのではなく、マッチング基準を満たす最初のノードペアを選択します。
* マッチング基準自体は、内部ノードペアに共通するリーフノードの数を考慮するように拡張されています。

![図3: 内部ノードの型と、過去にマッチングされたリーフノードの数に基づいて内部ノードをマッチングします。](sql_diff_images/figure_3.gif)
*図3: 内部ノードの型と、過去にマッチングされたリーフノードの数に基づいて内部ノードをマッチングします。*

まずはマッチング基準から見ていきましょう。この基準は以下のように定式化されます。

![Matching criteria for inner nodes](sql_diff_images/matching_criteria_2.png)

既におなじみの類似度スコアとノードタイプの基準に加え、中間に新たな基準が加わりました。2つのノードに共通するリーフノードの比率が、ある閾値「t」を超える必要があります。「t」の推奨値も0.6です。リーフノードのマッチングセットは既に揃っているので、共通リーフノードの数を数えるのは非常に簡単です。比較対象となる2つの内部ノードからリーフノードがいくつのマッチングペアを形成するかを数えるだけです。

このマッチング基準には、さらに2つのヒューリスティックが関連付けられています。

* 内部ノードの類似度の重み付け：ノード値間の類似度スコアが閾値「f」を超えないが、共通リーフノードの比率（「t」）が0.8以上の場合、マッチングは成功とみなされます。
* 小さなサブツリーにおける偽陰性率を低減するため、リーフノード数が4以下の内部ノードでは閾値「t」が0.4に引き下げられます。

残りの一致しないノードを反復処理し、概説した基準に基づいて一致するペアを形成するだけです。

```python
leaves_matching_set = matching_set.copy()

for source_node in unmatched_source_nodes.copy():
    for target_node in unmatched_target_nodes:
        if _is_same_type(source_node, target_node):
            source_leaves = set(_get_leaves(source_node))
            target_leaves = set(_get_leaves(target_node))

            max_leaves_num = max(len(source_leaves), len(target_leaves))
            if max_leaves_num:
                common_leaves_num = sum(
                    1 if s in source_leaves and t in target_leaves else 0
                    for s, t in leaves_matching_set
                )
                leaf_similarity_score = common_leaves_num / max_leaves_num
            else:
                leaf_similarity_score = 0.0

            adjusted_t = (
                0.6
                if min(len(source_leaves), len(target_leaves)) > 4
                else 0.4
            )

            if leaf_similarity_score >= 0.8 or (
                leaf_similarity_score >= adjusted_t
                and dice_coefficient(source_node, target_node) >= 0.6
            ):
                matching_set.add((source_node, target_node))
                unmatched_source_nodes.remove(source_node)
                unmatched_target_nodes.remove(target_node)
                break
```

マッチング セットが形成されたら、アルゴリズムの出力となる編集スクリプトの生成に進むことができます。

### 編集スクリプトの生成

この時点で、以下の3つのセットが利用可能になっているはずです。

* 一致したノードペアのセット。
* ソースツリーに残っている一致しないノードのセット。
* ターゲットツリーに残っている一致しないノードのセット。

一致セットからは、3種類の編集を導き出すことができます。ノードの値が更新された場合（**Update**）、ノードがツリー内の別の位置に移動された場合（**Move**）、ノードが変更されなかった場合（**Keep**）です。**Move** の場合と他の2つの場合が排他的ではないことに注意してください。ノードは更新された場合も、親ノード内での位置が変更された場合でも、同じままであった場合も考えられます。ソースツリーの一致しないノードはすべて削除されたノード（**Remove**）であり、ターゲットツリーの一致しないノードは挿入されたノード（**Insert**）です。

最後の2つのケースは実装が非常に簡単です。

```python
edit_script = []

for removed_node in unmatched_source_nodes:
    edit_script.append(Remove(removed_node))
for inserted_node in unmatched_target_nodes:
    edit_script.append(Insert(inserted_node))
```

一致するセットを走査するには、もう少し考慮する必要があります。

```python
for source_node, target_node in matching_set:
    if (
        not isinstance(source_node, LEAF_EXPRESSION_TYPES)
        or source_node == target_node
    ):
        move_edits = generate_move_edits(
            source_node, target_node, matching_set
        )
        edit_script.extend(move_edits)
        edit_script.append(Keep(source_node, target_node))
    else:
        edit_script.append(Update(source_node, target_node))
```

一致するペアがリーフノードのペアを表す場合、更新が行われたかどうかを判断するために、それらが同じであるかどうかを確認します。内部ノードのペアについては、ノードの移動を検出するために、それぞれの子ノードの位置も比較する必要があります。Chawatheら[3]は、[最長共通部分列](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem)(LCS)アルゴリズムを適用することを提案しています。これは当然のことながら、Myers自身[1]によって説明されました。ただし、小さな問題があります。2つの子ノードの等価性を確認する代わりに、2つのノードが一致するセットの一部であるペアを形成するかどうかを確認する必要があります。

この知識があれば、実装は簡単になります。

```python
def generate_move_edits(source, target, matching_set):
    source_children = _get_child_nodes(source)
    target_children = _get_child_nodes(target)

    lcs = set(
        _longest_common_subsequence(
            source_children,
            target_children,
            lambda l, r: (l, r) in matching_set
        )
    )

    move_edits = []
    for node in source_children:
        if node not in lcs and node not in unmatched_source_nodes:
            move_edits.append(Move(node))

    return move_edits
```

ここでは LCS アルゴリズム自体の実装については省略しましたが、簡単に調べることができる実装の選択肢は数多くあります。

### 出力

実装されたアルゴリズムは、次のような出力を生成します。

```python
>>> from sqlglot import parse_one, diff
>>> diff(parse_one("SELECT a + b + c, d, e"), parse_one("SELECT a - b + c, e, f"))

Remove(Add)
Remove(Column(d))
Remove(Identifier(d))
Insert(Sub)
Insert(Column(f))
Insert(Identifier(f))
Keep(Select, Select)
Keep(Add, Add)
Keep(Column(a), Column(a))
Keep(Identifier(a), Identifier(a))
Keep(Column(b), Column(b))
Keep(Identifier(b), Identifier(b))
Keep(Column(c), Column(c))
Keep(Identifier(c), Identifier(c))
Keep(Column(e), Column(e))
Keep(Identifier(e), Identifier(e))
```
上記の出力は省略されていることに注意してください。実際のASTノードの文字列表現は、はるかに冗長です。

この実装は、比較対象のクエリの正規表現を生成するSQLGlotのクエリオプティマイザと組み合わせると特に効果的です。

```python
>>> schema={"t": {"a": "INT", "b": "INT", "c": "INT", "d": "INT"}}
>>> source = """
... SELECT 1 + 1 + a
... FROM t
... WHERE b = 1 OR (c = 2 AND d = 3)
... """
>>> target = """
... SELECT 2 + a
... FROM t
... WHERE (b = 1 OR c = 2) AND (b = 1 OR d = 3)
... """
>>> optimized_source = optimize(parse_one(source), schema=schema)
>>> optimized_target = optimize(parse_one(target), schema=schema)
>>> edit_script = diff(optimized_source, optimized_target)
>>> sum(0 if isinstance(e, Keep) else 1 for e in edit_script)
0
```

### 最適化

このアルゴリズムの最悪ケースの実行時計算量は、O(n^2 * log n^2) と、決して驚くほどではありません。これは、比較対象となるツリーのすべてのリーフノード間の直積を順位付けするリーフマッチング処理が原因です。当然のことながら、大規模なクエリの場合、このアルゴリズムの完了にはかなりの時間がかかります。

実装において、パフォーマンス向上のために実行できる基本的な事項がいくつかあります。

* 個々のノードオブジェクトを参照する際に、セット内の直接参照ではなく、識別子（Python の [id()](https://docs.python.org/3/library/functions.html#id)）を使用します。これにより、コストのかかる再帰ハッシュ計算や等価性チェックを回避できます。
* バイグラムヒストグラムをキャッシュすることで、同じノードに対して複数回計算することを防ぎます。
* 各ツリーの標準的な SQL 文字列表現を 1 回計算する一方で、すべての内部ノードの文字列表現をキャッシュします。これにより、バイグラム計算時の冗長なツリー探索を回避できます。

執筆時点では最初の 2 つの最適化のみが実装されているため、興味のある人なら誰でも貢献する機会があります。

## 代替ソリューション

このセクションでは、私が調査したものの、まだ試していないソリューションについて解説します。

まず、このセクションはTristan Hume氏の[ブログ投稿](https://thume.ca/2017/06/17/tree-diffing/)なしでは完結しません。Tristan氏のソリューションはMyersアルゴリズムと多くの共通点があり、さらに私が考案したものよりもはるかに巧妙なヒューリスティックを採用しています。実装では、[動的計画法](https://en.wikipedia.org/wiki/Dynamic_programming)と[A*探索アルゴリズム](https://en.wikipedia.org/wiki/A*_search_algorithm)を組み合わせて、可能なマッチング空間を探索し、最適なものを選択します。Tristan氏の特定のユースケースではうまく機能しているように見えましたが、Myersアルゴリズムで失敗した経験があったため、別の方法を試してみることにしました。

もう一つの注目すべきアプローチは、FalleriらによるGumtreeアルゴリズム[4]です。この論文を発見したのは、この記事の主題であるアルゴリズムを実装した後のことでした。論文のセクション5.2と5.3では、著者らは2つのアルゴリズムを並べて比較し、12,792組のJavaソースファイルで評価した結果、Gumtreeが実行時パフォーマンスと精度の両方で大幅に優れていると主張しています。このアルゴリズムはサブツリーの高さを考慮しているため、これは驚くべきことではありません。私のテストでは、このコンテキストが役立つシナリオが確かに見られました。さらに、著者らは最悪の場合でも実行時複雑度がO(n^2)になると約束しており、Change DistillerのO(n^2 * log n^2)を考えると、これは非常に魅力的です。いつかこのアルゴリズムを試してみたいと思っています。今後の記事でこのアルゴリズムについて書く機会があるかもしれません。

## 結論

Change Distillerアルゴリズムは、私のテストのほとんどで非常に満足のいく結果をもたらしました。期待に応えられなかったシナリオは、主にASTの異なる部分にある同一（または非常に類似）のサブツリーに関するものでした。これらのケースではノードの不一致が頻繁に発生し、結果として編集スクリプトがやや最適化されていませんでした。

さらに、このアルゴリズムの実行時パフォーマンスには改善の余地がかなりあります。1,000個のリーフノードを持つツリーでは、このアルゴリズムの実行に2秒弱かかります。私の実装にはまだ改善の余地がありますが、期待される成果の大まかな概要はつかめるでしょう。Gumtreeアルゴリズム[4]は、これらの両方の問題に対処するのに役立つようです。近いうちに作業時間を見つけて、2つのアルゴリズムを比較し、SQLで特にどちらのパフォーマンスが優れているかを調べたいと思っています。その間、Change Distillerは確かに仕事をこなしており、この記事の冒頭で述べたいくつかのユースケースに適用することができます。

また、業界の他の方々が同様の問題に直面したことがあるか、そしてどのように対処したかを知りたいです。もし同じような経験をされた方がいらっしゃいましたら、ぜひご経験談をお聞かせください。

## References

[1] Eugene W. Myers. [An O(ND) Difference Algorithm and Its Variations](http://www.xmailserver.org/diff2.pdf). Algorithmica 1(2): 251-266 (1986)

[2] B. Fluri, M. Wursch, M. Pinzger, and H. Gall. [Change Distilling: Tree differencing for fine-grained source code change extraction](https://www.researchgate.net/publication/3189787_Change_DistillingTree_Differencing_for_Fine-Grained_Source_Code_Change_Extraction). IEEE Trans. Software Eng., 33(11):725–743, 2007.

[3] S.S. Chawathe, A. Rajaraman, H. Garcia-Molina, and J. Widom. [Change Detection in Hierarchically Structured Information](http://ilpubs.stanford.edu:8090/115/1/1995-46.pdf). Proc. ACM Sigmod Int’l Conf. Management of Data, pp. 493-504, June 1996

[4] Jean-Rémy Falleri, Floréal Morandat, Xavier Blanc, Matias Martinez, Martin Monperrus. [Fine-grained and Accurate Source Code Differencing](https://hal.archives-ouvertes.fr/hal-01054552/document). Proceedings of the International Conference on Automated Software Engineering, 2014, Västeras, Sweden. pp.313-324, 10.1145/2642937.2642982. hal-01054552
