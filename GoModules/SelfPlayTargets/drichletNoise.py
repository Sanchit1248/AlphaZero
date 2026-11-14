import numpy as np

def mix_root_dirichlet_noise(
    priors: np.ndarray,
    noise: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Returns the mixed float64 prior array in the original action order.
    """
    ans = np.zeros(priors.size, dtype = np.float64)
    legal_mask = np.asarray(legal_mask, dtype = bool)
    
    legal_priors = priors[legal_mask]
    legal_priors = legal_priors/legal_priors.sum()
    
    legal_noise = noise[legal_mask]
    legal_noise = legal_noise/legal_noise.sum()
    
    new_prior = (1-epsilon)*legal_priors + epsilon*legal_noise
    ans[legal_mask] = new_prior
    return ans
