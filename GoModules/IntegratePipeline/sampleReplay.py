import copy

def sample_recent_replay_batch(
    games: list | tuple,
    window_size: int,
    position_indices: list | tuple,
) -> tuple[list, int]:
    """
    Returns independent sampled records and the eligible position count.
    """
    recent = games[-window_size:]
    pool = []
    
    for game in recent:
        for record in game:
            pool.append(record)
            
    ans = []
    for idx in position_indices:
        ans.append(copy.deepcopy(pool[idx]))

    return ans, int(len(pool))