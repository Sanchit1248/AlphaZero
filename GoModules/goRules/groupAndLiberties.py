import numpy as np
from collections import deque

def go_group_liberties(board: list, row: int, col: int) -> tuple:
    """
    group and liberties.
    """
    r = len(board)
    c = len(board[0])
    
    if board[row][col] == 0:
        return ([], [])


    group = []
    liberties = []
    
    movement = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    
    color = board[row][col]
    
    visited = [[0]*c for _ in range(r)]

    q = deque()
    q.append([row, col])
    visited[row][col] = 1
    group.append([row, col])
    
    while q:
        crow, ccol = q.popleft()
        
        for dx, dy in movement:
            xx = crow + dx
            yy = ccol + dy
            
            if 0 <= xx < r and 0 <= yy < c and visited[xx][yy] == 0:
                neigh = board[xx][yy]
    
                if neigh == 0:
                    liberties.append([xx, yy])
                    visited[xx][yy] = 1
                    
                elif neigh == color:
                    group.append([xx, yy])
                    q.append([xx, yy])
                    visited[xx][yy] = 1
 

    group = sorted(group)
    liberties = sorted(liberties)
    return (group, liberties)