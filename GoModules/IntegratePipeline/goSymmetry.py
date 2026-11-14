
import numpy as np

def apply_go_symmetry(
    state_planes: np.ndarray,
    policy: np.ndarray,
    value: float,
    transform_id: int,
) -> tuple[np.ndarray, np.ndarray, float]:

    n = state_planes.shape[-1]

    k = transform_id % 4
    reflect = transform_id >= 4

    state = state_planes.copy()
    policy_board = policy[:-1].reshape(n, n).copy()

    # Left-right reflection
    if reflect:
        state = np.flip(state, axis=-1)
        policy_board = np.flip(policy_board, axis=-1)

    # Counterclockwise rotation
    state = np.rot90(state, k=k, axes=(-2, -1))
    policy_board = np.rot90(policy_board, k=k)

    # Flatten policy and append unchanged pass probability
    policy_ans = np.concatenate([
        policy_board.flatten(),
        policy[-1:]
    ])

    return state.copy(), policy_ans.copy(), value
