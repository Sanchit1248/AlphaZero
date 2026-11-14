
import numpy as np

def neural_guided_mcts(
    root_state: str,
    transitions: dict,
    legal_actions: dict,
    players: dict,
    policy_logits: dict,
    value_predictions: dict,
    terminal_values: dict,
    simulations: int,
    cpuct: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    nodes = {}

    # --------------------------------
    # 1. EXPANSION
    # --------------------------------

    def expand_state(state):

        actions = legal_actions[state]

        logits = np.asarray(
            policy_logits[state], dtype=np.float64
        ).copy()

        # Stable softmax
        logits -= logits.max()

        priors = np.exp(logits)
        priors /= priors.sum()

        nodes[state] = {
            "N": np.zeros(len(actions), dtype=np.int64),
            "W": np.zeros(len(actions), dtype=np.float64),
            "P": priors
        }

    # --------------------------------
    # 2. SELECTION
    # --------------------------------

    def select_edge(state):

        N = nodes[state]["N"]
        W = nodes[state]["W"]
        P = nodes[state]["P"]

        Q = np.zeros(len(N), dtype=np.float64)

        np.divide(W, N, out=Q, where=N > 0)

        total_visits = N.sum()

        scores = (
            Q + cpuct * P * np.sqrt(total_visits) / (1 + N)
        )

        return int(np.argmax(scores))

    # --------------------------------
    # 3. GET CHILD
    # --------------------------------

    def get_child(state, action):

        edges = transitions[state]

        if action in edges:
            return edges[action]

        return edges[str(action)]

    # --------------------------------
    # 4. INITIALIZE ROOT
    # --------------------------------

    expand_state(root_state)

    # --------------------------------
    # 5. MCTS SIMULATIONS
    # --------------------------------

    for _ in range(simulations):

        state = root_state
        path = []

        # DESCENT
        while state in nodes and state not in terminal_values:

            idx = select_edge(state)

            action = legal_actions[state][idx]

            path.append((state, idx))

            state = get_child(state, action)

        # EVALUATE LEAF
        leaf_player = players[state]

        if state in terminal_values:

            leaf_value = terminal_values[state]

        else:

            expand_state(state)

            leaf_value = value_predictions[state]

        # BACKUP
        for parent_state, action_idx in path:

            if players[parent_state] == leaf_player:
                v = leaf_value
            else:
                v = -leaf_value

            nodes[parent_state]["N"][action_idx] += 1
            nodes[parent_state]["W"][action_idx] += v

    # --------------------------------
    # 6. FINAL ROOT STATISTICS
    # --------------------------------

    root = nodes[root_state]

    visits = root["N"].copy()
    total_values = root["W"]

    Q = np.zeros(len(visits), dtype=np.float64)

    np.divide(
        total_values,
        visits,
        out=Q,
        where=visits > 0
    )

    # --------------------------------
    # 7. FINAL POLICY
    # --------------------------------

    policy = np.zeros(len(visits), dtype=np.float64)

    if temperature == 0:

        idx = int(np.argmax(visits))
        policy[idx] = 1.0

    else:

        positive = visits > 0

        log_visits = np.log(
            visits[positive].astype(np.float64)
        )

        weights = np.exp(
            (log_visits - log_visits.max()) / temperature
        )

        policy[positive] = weights / weights.sum()

    return visits, Q, policy
