
import torch
import torch.nn.functional as F

def policy_value_heads(
    features: torch.Tensor,
    policy_conv_weight: torch.Tensor,
    policy_scale: torch.Tensor,
    policy_shift: torch.Tensor,
    policy_linear_weight: torch.Tensor,
    policy_linear_bias: torch.Tensor,
    value_conv_weight: torch.Tensor,
    value_scale: torch.Tensor,
    value_shift: torch.Tensor,
    value_hidden_weight: torch.Tensor,
    value_hidden_bias: torch.Tensor,
    value_output_weight: torch.Tensor,
    value_output_bias: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    device = features.device
    features = features.to(torch.float32)

    def f32(t):
        return t.to(device=device, dtype=torch.float32)

    def batch_norm(z, scale, shift):
        mean = z.mean(dim=(0, 2, 3), keepdim=True)
        var = z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

        scale = f32(scale).view(1, -1, 1, 1)
        shift = f32(shift).view(1, -1, 1, 1)

        return scale * (z - mean) / torch.sqrt(var + epsilon) + shift

    
    # POLICY (1x1 conv so output size preserved) B C H W
    p = F.conv2d(features, f32(policy_conv_weight)) #B 2 H W (output channels = 2 ALPHAGO choice)
    p = batch_norm(p, policy_scale, policy_shift) #B 2 H W 
    p = F.relu(p) #B 2 H W
    p = p.flatten(start_dim=1) # B 2*H*W
    policy_logits = F.linear(p, f32(policy_linear_weight), f32(policy_linear_bias)) # B 2*H*W => B H*W + 1

    # VALUE (1x1 conv so output size presreved) B C H W
    v = F.conv2d(features, f32(value_conv_weight)) # B 1 H W (output =1)
    v = batch_norm(v, value_scale, value_shift) # B 1 H W
    v = F.relu(v)
    v = v.flatten(start_dim=1) # B H*W
    v = F.linear(v, f32(value_hidden_weight), f32(value_hidden_bias)) # B H*W => B 256
    v = F.relu(v)
    v = F.linear(v, f32(value_output_weight), f32(value_output_bias)) # B 256 => B 1
    
    values = torch.tanh(v).squeeze(-1) # B
    return policy_logits, values
