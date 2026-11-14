
import torch

def encode_go_history(history: list[torch.Tensor], to_play: int) -> torch.Tensor:
    """
    Returns a float32 tensor of shape    (17, board_size, board_size) on the history device.
    """

    planes = []

    for board in history:
        planes.append((board == to_play).float())
        planes.append((board == -to_play).float())


    for _ in range(8 - len(history)):
        planes.append(torch.zeros_like(history[0], dtype=torch.float32))
        planes.append(torch.zeros_like(history[0], dtype=torch.float32))

    
    if to_play == 1:
        planes.append(torch.ones_like(history[0], dtype=torch.float32))
    else:
        planes.append(torch.zeros_like(history[0], dtype=torch.float32))

    return torch.stack(planes)
