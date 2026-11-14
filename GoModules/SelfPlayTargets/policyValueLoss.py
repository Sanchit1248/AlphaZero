
import torch
import torch.nn.functional as F

def alpha_go_policy_value_loss(
    policy_logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    parameters: list,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    B = policy_logits.shape[0]

    #lp
    log_probs = F.log_softmax(policy_logits, dim=-1)
    lp = -((target_policies * log_probs).sum(dim=-1).sum()) / B
                                        #across A, across B
    
    #lv
    lv = ((predicted_values - target_values) ** 2).sum() / B

    #l2
    if len(parameters) > 0:
        lr = l2_coefficient * sum((p ** 2).sum() for p in parameters)
    else:
        lr = policy_logits.new_zeros(())

    #l
    total = lp + lv + lr

    return total, lp, lv, lr
