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


def resolve_go_action(board: list, player: int, action: int, consecutive_passes: int, move_count: int) -> tuple:
    """
    Returns: a tuple: next board, player, captures, consecutive passes, move count, terminal.
    """
    r, c = len(board), len(board[0])
    move_count += 1
    terminated = False
    board = np.array(board)
    
    if action == r*c:
        consecutive_passes += 1
        if move_count >= 2*r*c or consecutive_passes >= 2:
            terminated = True
        return (board, -player, 0, consecutive_passes, move_count, terminated)

    directions =  [(0, 1), (1, 0), (0, -1), (-1, 0)]

    x, y = action//r, action%r
    board[x][y] = player
    count = 0
    
    for dx, dy in directions:
        xx, yy = x + dx, y + dy    

        if not(0<=xx<r and 0<=yy<c):
            continue

        # [Claude] fix: only enemy groups can be captured
        if board[xx][yy] != -player:
            continue
        
        liberties, groups = get_liberties_groups(board, xx, yy)
        if len(liberties) == 0:
            for gx, gy in groups:
                board[gx][gy] = 0
                count += 1

    
    if move_count >= 2*r*c or consecutive_passes >= 2:
        terminated = True
    
    return (board, -player, count, 0, move_count, terminated)
        
