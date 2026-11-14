import numpy as np
from collections import deque

def get_liberties_groups(board : list, x, y) -> tuple:
    r, c = len(board), len(board[0])
    directions =  [(0, 1), (1, 0), (0, -1), (-1, 0)]
    liberties, groups = [], []
    color = board[x][y]

    if color == 0:
        return [], []

    visited = [[0]*c for _ in range(r)]

    q = deque()
    visited[x][y] = 1
    q.append([x, y])
    groups.append([x, y])
    
    
    while q:
        x, y = q.popleft()

        for dx, dy in directions:
            xx, yy = x + dx, y + dy
            
            if not (0 <= xx < r and 0 <= yy < c) or visited[xx][yy]:
                continue

            if board[xx][yy] == 0:
                liberties.append([xx, yy])
                visited[xx][yy] = 1

            if board[xx][yy] == color:
                q.append([xx, yy])
                visited[xx][yy] = 1
                groups.append([xx, yy])
        
    return liberties, groups



# [Claude] added: board fingerprint for the ko check
def position_key(board) -> bytes:
    return np.asarray(board, dtype=np.int8).tobytes()


def legal_go_action_mask(board: list, player: int, seen_positions: list, consecutive_passes: int, move_count: int) -> np.ndarray:
    """
    Returns: a Boolean NumPy array of legality flags, with pass last.
    """
    r, c = len(board), len(board[0])
    mask = np.zeros(r*c+1, dtype=bool)
    seen = seen_positions if isinstance(seen_positions, (set, frozenset)) else {position_key(x) for x in seen_positions}  # [Claude]

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    #terminated
    if consecutive_passes >= 2 or move_count >= 2*r*c:
        return mask

    b = np.array(board)
    
    #non terminated
    for i in range(r):
        for j in range(c):
            
            if b[i][j] != 0:
                continue

            #move
            candidate = b.copy()
            candidate[i][j] = player

            
            #capture
            checked = set()
            
            for dx, dy in directions:
                xx, yy = i + dx, j + dy

                if not(0 <= xx < r and 0 <= yy < c):
                    continue

                if candidate[xx][yy] != -player:
                    continue

                if (xx, yy) in checked:
                    continue
                
                liberties, groups = get_liberties_groups(candidate, xx, yy)
                checked.update(tuple(o) for o in groups)
                
                if len(liberties) == 0:
                    for gx, gy in groups:
                        candidate[gx][gy] = 0
            

            #suicide
            liberties, groups = get_liberties_groups(candidate, i, j)
            if len(liberties) == 0:
                continue

            #ko memory
            if position_key(candidate) in seen:  # [Claude] replaces the array_equal scan ; makes it faster
                continue

            mask[i*r + j] = True

    
    mask[-1] = True
    return mask