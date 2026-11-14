
import torch
import torch.nn.functional as F

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    B = features.shape[0]

    # Create independent parameter copies for autograd
    pw = policy_weight.detach().clone().requires_grad_(True)
    pb = policy_bias.detach().clone().requires_grad_(True)
    vw = value_weight.detach().clone().requires_grad_(True)
    vb = value_bias.detach().clone().requires_grad_(True)

    # 1. POLICY HEAD
    log_probs = F.log_softmax(features @ pw.T + pb, dim=-1)

    # 2. VALUE HEAD
    predicted_values = torch.tanh(features @ vw + vb)

    # 3. Loss

    lp = -(target_policies * log_probs).sum() / B

    lv = ((predicted_values - target_values) ** 2).sum() / B
    
    l2 = l2_coefficient * (pw.square().sum() + pb.square().sum() + vw.square().sum() + vb.square().sum())

    loss = lp + lv + l2


    # 5. CALCULATE GRADIENTS
    params = (pw, pb, vw, vb)
    gradients = torch.autograd.grad(loss, params)

    # UPDATE
    
    updated = [
        (p - learning_rate * g).detach()
        for p, g in zip(params, gradients)
    ]

    return (loss.detach(), *updated)
