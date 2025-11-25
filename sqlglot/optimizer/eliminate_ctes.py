from sqlglot.optimizer.scope import Scope, build_scope


def eliminate_ctes(expression):
    """
    Remove unused CTEs from an expression.
    式から未使用の CTE を削除します。

    Example:
        >>> import sqlglot
        >>> sql = "WITH y AS (SELECT a FROM x) SELECT a FROM z"
        >>> expression = sqlglot.parse_one(sql)
        >>> eliminate_ctes(expression).sql()
        'SELECT a FROM z'

    Args:
        expression (sqlglot.Expression): expression to optimize
            最適化する式
    Returns:
        sqlglot.Expression: optimized expression
        sqlglot.Expression: 最適化された式
    """
    root = build_scope(expression)

    if root:
        ref_count = root.ref_count()

        # Traverse the scope tree in reverse so we can remove chains of unused CTEs
        # スコープツリーを逆順に走査して、未使用のCTEのチェーンを削除します。
        for scope in reversed(list(root.traverse())):
            if scope.is_cte:
                count = ref_count[id(scope)]
                if count <= 0:
                    cte_node = scope.expression.parent
                    with_node = cte_node.parent
                    cte_node.pop()

                    # Pop the entire WITH clause if this is the last CTE
                    # これが最後のCTEである場合はWITH句全体をポップします
                    if with_node and len(with_node.expressions) <= 0:
                        with_node.pop()

                    # Decrement the ref count for all sources this CTE selects from
                    # このCTEが選択するすべてのソースの参照カウントをデクリメントします
                    for _, source in scope.selected_sources.values():
                        if isinstance(source, Scope):
                            ref_count[id(source)] -= 1

    return expression
