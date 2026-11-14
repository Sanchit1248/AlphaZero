
import numpy as np

def visit_count_policy(
    visits: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float,
    uniform_draw: float | None,
) -> tuple[np.ndarray, int]:

    ans = np.zeros(visits.size, dtype=np.float64)
    legal_mask = np.asarray(legal_mask, dtype=bool)

    legal_visits = visits[legal_mask]

    # Temperature = 0: deterministic selection
    if temperature == 0:
        maxi = np.max(legal_visits)

        for i, x in enumerate(visits):
            if x == maxi and legal_mask[i]:
                ans[i] = 1.0
                return ans, int(i)

    # All legal visits are zero: uniform policy
    if np.all(legal_visits == 0):
        term = np.ones(len(legal_visits), dtype=np.float64)
    
    else:
        # Numerically stable temperature scaling
        positive = legal_visits > 0

        log_weights = np.log(legal_visits[positive]) / temperature
        log_weights -= log_weights.max()

        term = np.zeros(len(legal_visits), dtype=np.float64)
        term[positive] = np.exp(log_weights)

    # Normalize
    term = term / term.sum()

    ans[legal_mask] = term

    sumi = 0.0
    for i, x in enumerate(ans):
        sumi += x

        if sumi > uniform_draw:
            return ans, int(i)

    return ans, -1
