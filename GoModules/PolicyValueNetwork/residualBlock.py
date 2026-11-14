import torch
import torch.nn.functional as F

def alpha_go_residual_block(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    conv2_weight: torch.Tensor,
    scale2: torch.Tensor,
    shift2: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Returns: the float32 residual output, with the same shape and device as x.
    """
    x = x.to(dtype=torch.float32)
    conv1_weight = conv1_weight.to(dtype=torch.float32)
    conv2_weight = conv2_weight.to(dtype=torch.float32)

    def batch_norm(z, scale, shift):
        mean = z.mean(dim=(0, 2, 3), keepdim=True)
        var = z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        scale = scale.to(dtype=torch.float32).view(1, -1, 1, 1)
        shift = shift.to(dtype=torch.float32).view(1, -1, 1, 1)

        return scale * (z - mean) / torch.sqrt(var + epsilon) + shift


    h = F.conv2d(x, conv1_weight, stride=1, padding=1)
    h = batch_norm(h, scale1, shift1)
    h = torch.relu(h)

    y = F.conv2d(h, conv2_weight, stride=1, padding=1)
    y = batch_norm(y, scale2, shift2)
    y = torch.relu(x + y)

    return y
