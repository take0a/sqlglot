from __future__ import annotations

import datetime
import inspect
import logging
import re
import sys
import typing as t
from collections.abc import Collection, Set
from copy import copy
from difflib import get_close_matches
from enum import Enum
from itertools import count

if t.TYPE_CHECKING:
    from sqlglot import exp
    from sqlglot._typing import A, E, T
    from sqlglot.dialects.dialect import DialectType
    from sqlglot.expressions import Expression


CAMEL_CASE_PATTERN = re.compile("(?<!^)(?=[A-Z])")
PYTHON_VERSION = sys.version_info[:2]
logger = logging.getLogger("sqlglot")


class AutoName(Enum):
    """
    This is used for creating Enum classes where `auto()` is the string form
    of the corresponding enum's identifier (e.g. FOO.value results in "FOO").
    これは、`auto()` が対応する列挙型の識別子の文字列形式である Enum クラスを
    作成するために使用されます (例: FOO.value の結果は "FOO" になります)。

    Reference: https://docs.python.org/3/howto/enum.html#using-automatic-values
    """

    def _generate_next_value_(name, _start, _count, _last_values):
        return name


class classproperty(property):
    """
    Similar to a normal property but works for class methods
    通常のプロパティに似ていますが、クラスメソッドで機能します
    """

    def __get__(self, obj: t.Any, owner: t.Any = None) -> t.Any:
        return classmethod(self.fget).__get__(None, owner)()  # type: ignore


def suggest_closest_match_and_fail(
    kind: str,
    word: str,
    possibilities: t.Iterable[str],
) -> None:
    close_matches = get_close_matches(word, possibilities, n=1)

    similar = seq_get(close_matches, 0) or ""
    if similar:
        similar = f" Did you mean {similar}?"

    raise ValueError(f"Unknown {kind} '{word}'.{similar}")


def seq_get(seq: t.Sequence[T], index: int) -> t.Optional[T]:
    """Returns the value in `seq` at position `index`, or `None` if `index` is out of bounds.
    位置 `index` の `seq` の値を返します。`index` が範囲外の場合は `None` を返します。"""
    try:
        return seq[index]
    except IndexError:
        return None


@t.overload
def ensure_list(value: t.Collection[T]) -> t.List[T]: ...


@t.overload
def ensure_list(value: None) -> t.List: ...


@t.overload
def ensure_list(value: T) -> t.List[T]: ...


def ensure_list(value):
    """
    Ensures that a value is a list, otherwise casts or wraps it into one.
    値がリストであることを確認します。そうでない場合は、値をリストにキャストまたはラップします。

    Args:
        value: The value of interest.
            関心のある値

    Returns:
        The value cast as a list if it's a list or a tuple, or else the value wrapped in a list.
        リストまたはタプルの場合はリストとしてキャストされた値、それ以外の場合はリストにラップされた値。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


@t.overload
def ensure_collection(value: t.Collection[T]) -> t.Collection[T]: ...


@t.overload
def ensure_collection(value: T) -> t.Collection[T]: ...


def ensure_collection(value):
    """
    Ensures that a value is a collection (excluding `str` and `bytes`), otherwise wraps it into a list.
    値がコレクションであることを確認します (`str` と `bytes` を除く)。それ以外の場合は、値をリストにラップします。

    Args:
        value: The value of interest.
            関心のある値

    Returns:
        The value if it's a collection, or else the value wrapped in a list.
        コレクションの場合は値、そうでない場合はリストにラップされた値。
    """
    if value is None:
        return []
    return (
        value if isinstance(value, Collection) and not isinstance(value, (str, bytes)) else [value]
    )


def csv(*args: str, sep: str = ", ") -> str:
    """
    Formats any number of string arguments as CSV.
    任意の数の文字列引数を CSV としてフォーマットします。

    Args:
        args: The string arguments to format.
            フォーマットする文字列引数。
        sep: The argument separator.
            引数の区切り文字。

    Returns:
        The arguments formatted as a CSV string.
        CSV 文字列としてフォーマットされた引数。
    """
    return sep.join(arg for arg in args if arg)


def subclasses(
    module_name: str,
    classes: t.Type | t.Tuple[t.Type, ...],
    exclude: t.Set[t.Type] = set(),
) -> t.List[t.Type]:
    """
    Returns all subclasses for a collection of classes, possibly excluding some of them.
    クラスのコレクションのすべてのサブクラスを返します (一部が除外される場合もあります)。

    Args:
        module_name: The name of the module to search for subclasses in.
            サブクラスを検索するモジュールの名前。
        classes: Class(es) we want to find the subclasses of.
            サブクラスを検索するクラス。
        exclude: Classes we want to exclude from the returned list.
            返されるリストから除外するクラス。

    Returns:
        The target subclasses.
        ターゲットのサブクラス。
    """
    return [
        obj
        for _, obj in inspect.getmembers(
            sys.modules[module_name],
            lambda obj: inspect.isclass(obj) and issubclass(obj, classes) and obj not in exclude,
        )
    ]


def apply_index_offset(
    this: exp.Expression,
    expressions: t.List[E],
    offset: int,
    dialect: DialectType = None,
) -> t.List[E]:
    """
    Applies an offset to a given integer literal expression.
    指定された整数リテラル式にオフセットを適用します。

    Args:
        this: The target of the index.
            インデックスのターゲット。
        expressions: The expression the offset will be applied to, wrapped in a list.
            オフセットが適用される式。リストにラップされます。
        offset: The offset that will be applied.
            適用されるオフセット。
        dialect: the dialect of interest.
            対象の方言。

    Returns:
        The original expression with the offset applied to it, wrapped in a list. If the provided
        `expressions` argument contains more than one expression, it's returned unaffected.
        オフセットが適用された元の式がリストにラップされます。
        指定された `expressions` 引数に複数の式が含まれている場合、そのまま返されます。
    """
    if not offset or len(expressions) != 1:
        return expressions

    expression = expressions[0]

    from sqlglot import exp
    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.optimizer.simplify import simplify

    if not this.type:
        annotate_types(this, dialect=dialect)

    if t.cast(exp.DataType, this.type).this not in (
        exp.DataType.Type.UNKNOWN,
        exp.DataType.Type.ARRAY,
    ):
        return expressions

    if not expression.type:
        annotate_types(expression, dialect=dialect)

    if t.cast(exp.DataType, expression.type).this in exp.DataType.INTEGER_TYPES:
        logger.info("Applying array index offset (%s)", offset)
        expression = simplify(expression + offset)
        return [expression]

    return expressions


def camel_to_snake_case(name: str) -> str:
    """Converts `name` from camelCase to snake_case and returns the result.
    `name` を camelCase から snake_case に変換し、結果を返します。"""
    return CAMEL_CASE_PATTERN.sub("_", name).upper()


def while_changing(expression: Expression, func: t.Callable[[Expression], E]) -> E:
    """
    Applies a transformation to a given expression until a fix point is reached.
    固定点に達するまで、指定された式に変換を適用します。

    Args:
        expression: The expression to be transformed.
            変換する式。
        func: The transformation to be applied.
            適用する変換。

    Returns:
        The transformed expression.
        変形された式。
    """

    while True:
        start_hash = hash(expression)
        expression = func(expression)
        end_hash = hash(expression)

        if start_hash == end_hash:
            break

    return expression


def tsort(dag: t.Dict[T, t.Set[T]]) -> t.List[T]:
    """
    Sorts a given directed acyclic graph in topological order.
    指定された有向非巡回グラフを位相順序でソートします。

    Args:
        dag: The graph to be sorted.
            ソートするグラフ。

    Returns:
        A list that contains all of the graph's nodes in topological order.
        グラフのすべてのノードをトポロジカルな順序で含むリスト。
    """
    result = []

    for node, deps in tuple(dag.items()):
        for dep in deps:
            if dep not in dag:
                dag[dep] = set()

    while dag:
        current = {node for node, deps in dag.items() if not deps}

        if not current:
            raise ValueError("Cycle error")

        for node in current:
            dag.pop(node)

        for deps in dag.values():
            deps -= current

        result.extend(sorted(current))  # type: ignore

    return result


def find_new_name(taken: t.Collection[str], base: str) -> str:
    """
    Searches for a new name.
    新しい名前を検索します。

    Args:
        taken: A collection of taken names.
            使用されている名前のコレクション。
        base: Base name to alter.
            変更するベース名。

    Returns:
        The new, available name.
        新しい、使用可能な名前。
    """
    if base not in taken:
        return base

    i = 2
    new = f"{base}_{i}"
    while new in taken:
        i += 1
        new = f"{base}_{i}"

    return new


def is_int(text: str) -> bool:
    return is_type(text, int)


def is_float(text: str) -> bool:
    return is_type(text, float)


def is_type(text: str, target_type: t.Type) -> bool:
    try:
        target_type(text)
        return True
    except ValueError:
        return False


def name_sequence(prefix: str) -> t.Callable[[], str]:
    """Returns a name generator given a prefix (e.g. a0, a1, a2, ... if the prefix is "a").
    プレフィックスが指定された名前ジェネレーターを返します (例: プレフィックスが "a" の場合は a0、a1、a2、...)。"""
    sequence = count()
    return lambda: f"{prefix}{next(sequence)}"


def object_to_dict(obj: t.Any, **kwargs) -> t.Dict:
    """Returns a dictionary created from an object's attributes.
    オブジェクトの属性から作成された辞書を返します。"""
    return {
        **{k: v.copy() if hasattr(v, "copy") else copy(v) for k, v in vars(obj).items()},
        **kwargs,
    }


def split_num_words(
    value: str, sep: str, min_num_words: int, fill_from_start: bool = True
) -> t.List[t.Optional[str]]:
    """
    Perform a split on a value and return N words as a result with `None` used for words that don't exist.
    値を分割し、結果として N 個の単語を返します。存在しない単語には `None` が使用されます。

    Args:
        value: The value to be split.
            分割する値。
        sep: The value to use to split on.
            分割に使用する値。
        min_num_words: The minimum number of words that are going to be in the result.
            結果に含まれる単語の最小数。
        fill_from_start: Indicates that if `None` values should be inserted at the start or end of the list.
            `None` の場合、リストの先頭または末尾に値を挿入する必要があることを示します。

    Examples:
        >>> split_num_words("db.table", ".", 3)
        [None, 'db', 'table']
        >>> split_num_words("db.table", ".", 3, fill_from_start=False)
        ['db', 'table', None]
        >>> split_num_words("db.table", ".", 1)
        ['db', 'table']

    Returns:
        The list of words returned by `split`, possibly augmented by a number of `None` values.
        `split` によって返される単語のリスト。`None` 値がいくつか追加される可能性があります。
    """
    words = value.split(sep)
    if fill_from_start:
        return [None] * (min_num_words - len(words)) + words
    return words + [None] * (min_num_words - len(words))


def is_iterable(value: t.Any) -> bool:
    """
    Checks if the value is an iterable, excluding the types `str` and `bytes`.
    `str` および `bytes` 型を除いて、値が反復可能かどうかを確認します。

    Examples:
        >>> is_iterable([1,2])
        True
        >>> is_iterable("test")
        False

    Args:
        value: The value to check if it is an iterable.
            反復可能かどうかを確認する値。

    Returns:
        A `bool` value indicating if it is an iterable.
        反復可能かどうかを示す `bool` 値。
    """
    from sqlglot import Expression

    return hasattr(value, "__iter__") and not isinstance(value, (str, bytes, Expression))


def flatten(values: t.Iterable[t.Iterable[t.Any] | t.Any]) -> t.Iterator[t.Any]:
    """
    Flattens an iterable that can contain both iterable and non-iterable elements. Objects of
    type `str` and `bytes` are not regarded as iterables.
    反復可能な要素と反復不可能な要素の両方を含む反復可能オブジェクトをフラット化します。
    `str` 型および `bytes` 型のオブジェクトは反復可能オブジェクトとはみなされません。

    Examples:
        >>> list(flatten([[1, 2], 3, {4}, (5, "bla")]))
        [1, 2, 3, 4, 5, 'bla']
        >>> list(flatten([1, 2, 3]))
        [1, 2, 3]

    Args:
        values: The value to be flattened.
            平坦化される値。

    Yields:
        Non-iterable elements in `values`.
        `values` 内の反復不可能な要素。
    """
    for value in values:
        if is_iterable(value):
            yield from flatten(value)
        else:
            yield value


def dict_depth(d: t.Dict) -> int:
    """
    Get the nesting depth of a dictionary.
    辞書のネストの深さを取得します。

    Example:
        >>> dict_depth(None)
        0
        >>> dict_depth({})
        1
        >>> dict_depth({"a": "b"})
        1
        >>> dict_depth({"a": {}})
        2
        >>> dict_depth({"a": {"b": {}}})
        3
    """
    try:
        return 1 + dict_depth(next(iter(d.values())))
    except AttributeError:
        # d doesn't have attribute "values"
        return 0
    except StopIteration:
        # d.values() returns an empty sequence
        return 1


def first(it: t.Iterable[T]) -> T:
    """Returns the first element from an iterable (useful for sets).
    反復可能オブジェクトから最初の要素を返します (セットに便利です)。"""
    return next(i for i in it)


def to_bool(value: t.Optional[str | bool]) -> t.Optional[str | bool]:
    if isinstance(value, bool) or value is None:
        return value

    # Coerce the value to boolean if it matches to the truthy/falsy values below
    value_lower = value.lower()
    if value_lower in ("true", "1"):
        return True
    if value_lower in ("false", "0"):
        return False

    return value


def merge_ranges(ranges: t.List[t.Tuple[A, A]]) -> t.List[t.Tuple[A, A]]:
    """
    Merges a sequence of ranges, represented as tuples (low, high) whose values
    belong to some totally-ordered set.
    完全に順序付けられたセットに属する値を持つタプル (low、high) として表される
    範囲のシーケンスを結合します。

    Example:
        >>> merge_ranges([(1, 3), (2, 6)])
        [(1, 6)]
    """
    if not ranges:
        return []

    ranges = sorted(ranges)

    merged = [ranges[0]]

    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def is_iso_date(text: str) -> bool:
    try:
        datetime.date.fromisoformat(text)
        return True
    except ValueError:
        return False


def is_iso_datetime(text: str) -> bool:
    try:
        datetime.datetime.fromisoformat(text)
        return True
    except ValueError:
        return False


# Interval units that operate on date components
DATE_UNITS = {"day", "week", "month", "quarter", "year", "year_month"}


def is_date_unit(expression: t.Optional[exp.Expression]) -> bool:
    return expression is not None and expression.name.lower() in DATE_UNITS


K = t.TypeVar("K")
V = t.TypeVar("V")


class SingleValuedMapping(t.Mapping[K, V]):
    """
    Mapping where all keys return the same value.
    すべてのキーが同じ値を返すマッピング。

    This rigamarole is meant to avoid copying keys, which was originally intended
    as an optimization while qualifying columns for tables with lots of columns.
    この面倒な処理は、キーのコピーを回避するためのもので、元々は多数の列を持つテーブルで
    列を修飾する際の最適化を目的としていました。
    """

    def __init__(self, keys: t.Collection[K], value: V):
        self._keys = keys if isinstance(keys, Set) else set(keys)
        self._value = value

    def __getitem__(self, key: K) -> V:
        if key in self._keys:
            return self._value
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> t.Iterator[K]:
        return iter(self._keys)
