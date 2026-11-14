def backup_mcts_path(path: list, leaf_value: float, leaf_player: int) -> list:
    """
    backprop in MCTS
    Returns updated copies of the path records in their original order.
    """
    # path => list of dicts[id, action, player, N, W, Q, P] , leaf val, leaf player 
    ans = [node.copy() for node in path]
    
    for node in ans:
        node["N"] += 1
        if leaf_player == node["player"]:
            v = leaf_value
        else:
            v = -leaf_value
        node["W"] += v
        node["Q"] = node["W"]/node["N"]

    return ans