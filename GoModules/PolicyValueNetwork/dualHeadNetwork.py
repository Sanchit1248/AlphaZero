
import torch
import torch.nn.functional as F

def compact_policy_value_network(
    x: torch.Tensor,
    trunk_weight: torch.Tensor,
    trunk_scale: torch.Tensor,
    trunk_shift: torch.Tensor,
    residual_parameters: list[dict],
    head_parameters: dict,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    device = x.device

    def f32(t):
        return t.to(device=device, dtype=torch.float32)

    def batch_norm(z, scale, shift):
        mean = z.mean(dim=(0, 2, 3), keepdim=True)
        var = z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        scale = f32(scale).view(1, -1, 1, 1)
        shift = f32(shift).view(1, -1, 1, 1)

        return scale * (z - mean) / torch.sqrt(var + epsilon) + shift

    x = f32(x)


    # 1. TRUNK

    h = F.conv2d(x, f32(trunk_weight), stride=1, padding=1)
    h = batch_norm(h, trunk_scale, trunk_shift)
    h = F.relu(h)


    # 2. RESIDUAL TOWER

    for k in residual_parameters:

        identity = h

        residual = F.conv2d(h, f32(k["conv1_weight"]), stride=1, padding=1)
        residual = batch_norm(residual, k["scale1"], k["shift1"])
        residual = F.relu(residual)

        residual = F.conv2d(residual, f32(k["conv2_weight"]), stride=1, padding=1)
        residual = batch_norm(residual, k["scale2"], k["shift2"])

        h = F.relu(identity + residual)


    
    # 3. POLICY 

    p = F.conv2d(h, f32(head_parameters["policy_conv_weight"]), stride=1, padding=0)
    p = batch_norm(p, head_parameters["policy_scale"], head_parameters["policy_shift"])
    p = F.relu(p)
    p = p.flatten(start_dim=1)

    policy_logits = F.linear(p,f32(head_parameters["policy_linear_weight"]), f32(head_parameters["policy_linear_bias"]))


    # 4. VALUE 

    v = F.conv2d(h, f32(head_parameters["value_conv_weight"]), stride=1, padding=0)
    v = batch_norm(v, head_parameters["value_scale"], head_parameters["value_shift"])
    
    v = F.relu(v)
    v = v.flatten(start_dim=1)
    v = F.linear(v, f32(head_parameters["value_hidden_weight"]), f32(head_parameters["value_hidden_bias"]))
    
    v = F.relu(v)    
    v = F.linear(v, f32(head_parameters["value_output_weight"]), f32(head_parameters["value_output_bias"]))

    values = torch.tanh(v).squeeze(-1)
    return policy_logits, values
