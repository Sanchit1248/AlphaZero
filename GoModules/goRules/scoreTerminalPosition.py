import numpy as np
from collections import deque

def tromp_taylor_result(board: list, komi: float, consecutive_passes: int, move_count: int) -> tuple:
    """
    tuple: terminal, Black score, White score, winner; unfinished scores and winner are None.
    """
    blackscore, whitescore = 0, 0
    r, c = len(board), len(board[0])
    
    if consecutive_passes < 2 and move_count < 2*r*c:
        return (False, None, None, None)

    
    for i in range(r):
        for j in range(c):
            if board[i][j] == 1:
                blackscore += 1

            elif board[i][j] == -1:
                whitescore += 1

    whitescore += komi

    q = deque()
    directions = [(0,1), (1,0), (0,-1), (-1,0)]
    visited = [[0]*c for _ in range(r)]
    
    for i in range(r):
        for j in range(c):
            
            if board[i][j] == 0 and visited[i][j] == 0:
                
                q.append([i, j])
                visited[i][j] = 1
                boundary = set()
                regionsize = 1
                
                while q:
                    x, y = q.popleft()
                    
                    for dx, dy in directions:
                        xx, yy = x + dx, y + dy

                        if 0 <= xx < r and 0 <= yy < c and visited[xx][yy] == 0:

                            if board[xx][yy] == 1:
                                boundary.add(1)
                            elif board[xx][yy] == -1:
                                boundary.add(-1)
                            else:
                                regionsize += 1
                                visited[xx][yy] = 1
                                q.append([xx, yy])

                if boundary == {1}:
                    blackscore += regionsize
                elif boundary == {-1}:
                    whitescore += regionsize



    if blackscore > whitescore:
        winner = 1
    elif blackscore < whitescore:
        winner = -1
    else:
        winner = 0

    return (True, float(blackscore), float(whitescore), winner)

                    


    