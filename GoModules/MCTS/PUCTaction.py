import numpy as np

def select_puct_action(visits: np.ndarray, total_values: np.ndarray, priors: np.ndarray, legal_mask: np.ndarray, cpuct: float) -> tuple[int, np.ndarray]:
    """
    Returns: a tuple of the Python action index and the float64 score array.
    """
    # N visits, W total val,P priors,M legalmask, cpuct
    q = np.zeros(visits.size, dtype = np.float64)
    select = np.zeros(visits.size, dtype = np.float64)

    for i, x in enumerate(visits):
        if x == 0:
            continue
        else:
            q[i] = total_values[i]/visits[i]


    t = visits.sum()

    for i, x in enumerate(visits):
        if legal_mask[i] == False:
            select[i] = -np.inf
        else:
            select[i] = q[i] + cpuct*priors[i]*np.sqrt(t)/(1+visits[i])


    val = int(np.argmax(select))
    return (val, select)
