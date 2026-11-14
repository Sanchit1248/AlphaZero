# AlphaZero from Scratch: Tic-Tac-Toe → Go

A from-scratch implementation of the AlphaZero algorithm in Python and PyTorch. It is built up in stages: first on Tic-Tac-Toe, where every part is easy to inspect, then on Go.

AlphaZero learns a game with no human data. It plays against itself, using Monte Carlo Tree Search (MCTS) guided by a neural network, then trains the network on the results of those games.

For diagrams of how the pieces connect, see **[FLOWCHART.md](FLOWCHART.md)**.

---

## Project status

| Part | Where | Status |
|------|-------|--------|
| Tic-Tac-Toe game rules | `tictactoe.ipynb` | ✅ done |
| MCTS with random rollouts | `tictactoe.ipynb` (`MCTSrandom`) | ✅ done, plays well |
| Policy/value ResNet | `tictactoe.ipynb`, `model.ipynb` | ✅ done |
| AlphaZero MCTS (PUCT + network) | `tictactoe.ipynb` (`MCTS`) | ✅ done |
| Self-play + training loop | `tictactoe.ipynb` (`AlphaZero`) | 🚧 in progress |
| Go rules, encoding, MCTS and loss building blocks | `Modules/` | 🚧 in progress |
| Go game state and network wrappers (`state.py`, `net.py`) | — | ⏳ not written yet (needed by `tests/`) |

---

## How it works

```
         ┌──────────────── selfPlay() ────────────────┐
model ──►│ MCTS search guided by the network           │──► (board, MCTS probabilities, result)
         └────────────────────────────────────────────┘                  │
             ▲                                                            ▼
             └──────────── train(): fit the network to those targets ◄────┘
```

1. **Self-play.** For each move, MCTS runs many searches. At each leaf, the network provides
   * a **policy**: prior probabilities over moves, used to decide what to explore, and
   * a **value**: an estimate of who wins, used instead of a random playout.

   The visit counts of the root's children become the move probabilities. A move is sampled from them, and the position is saved.
2. **Labelling.** When the game ends, every saved position gets the result from the point of view of the player who was to move there: +1, 0 or −1.
3. **Training.** The network learns to
   * match the MCTS move probabilities (**cross-entropy** policy loss), and
   * predict the game result (**MSE** value loss).
4. **Repeat.** A better network guides MCTS better, which produces better games, which give better training data.

The trained model is only the network. When playing, it can be used on its own (fast) or inside MCTS (stronger; this is how AlphaZero plays).

### Key conventions

* **The player to move is always `+1`.** Before searching, the board is multiplied by the current player (`change_perspective`), so one network can play both sides.
* **Values are from the point of view of the player to move at that node.** That is why values are negated at every level in `backprop`, and why the parent uses `1 − q` in the selection score.
* **Board encoding (Tic-Tac-Toe):** 3 planes, one each for opponent stones, empty squares and own stones. The input shape is `(3, 3, 3)`.

---

## Repository layout

```
GO/
├── tictactoe.ipynb        # main notebook: game → MCTS → ResNet → AlphaZero MCTS → training
├── mcts.ipynb             # scratch notebook for MCTS
├── model.ipynb            # scratch notebook for the ResNet
├── FLOWCHART.md           # pipeline diagrams
├── Modules/               # Go building blocks (pure functions)
│   ├── goRules/
│   │   ├── goAction.py              # resolve_go_action: play a move, captures, passes, termination
│   │   ├── groupAndLiberties.py     # go_group_liberties: flood-fill a group and its liberties
│   │   ├── legalActionMask.py       # legal_go_action_mask: legal moves incl. suicide and ko
│   │   └── scoreTerminalPosition.py # tromp_taylor_result: area scoring with komi
│   ├── PolicyValueNetwork/
│   │   ├── encodeBoard.py           # encode_go_history: 17-plane AlphaGo Zero input
│   │   ├── residualBlock.py         # alpha_go_residual_block
│   │   ├── policyValueHead.py       # policy_value_heads
│   │   └── dualHeadNetwork.py       # compact_policy_value_network: trunk + tower + heads
│   ├── MCTS/
│   │   ├── initEdge.py              # initialize_mcts_edges: N, W, Q, P per edge
│   │   ├── PUCTaction.py            # select_puct_action
│   │   └── backupMCTS.py            # backup_mcts_path
│   ├── SelfPlayTargets/
│   │   ├── drichletNoise.py         # mix_root_dirichlet_noise
│   │   ├── searchPolicy.py          # visit_count_policy with temperature
│   │   └── policyValueLoss.py       # alpha_go_policy_value_loss (policy + value + L2)
│   └── IntegratePipeline/
│       ├── deterministicMCTS.py     # neural_guided_mcts over a given transition table
│       ├── goSymmetry.py            # apply_go_symmetry: 8 board symmetries for augmentation
│       ├── sampleReplay.py          # sample_recent_replay_batch
│       └── policyTrainStep.py       # policy_value_training_step
└── tests/
    ├── test_rules.py      # captures, ko, random-game rule checks
    ├── test_state.py      # GoState (needs state.py)
    └── test_net.py        # PolicyValueNet vs Modules reference (needs net.py)
```

---

## Getting started

### Requirements

* Python 3.10+
* `numpy`, `torch`, `tqdm`, `jupyter`
* `pytest` (for the tests)

```bash
pip install numpy torch tqdm jupyter pytest
```

### Tic-Tac-Toe notebook

Open `tictactoe.ipynb` and run the cells **from top to bottom**. Several cells redefine the same names as the project progresses (the random-rollout MCTS, then the network-guided MCTS), so running cells out of order can leave an older version of a class defined.

Train:

```python
game = TicTacToe()
model = ResNet(game, 4, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

args = {
    'C': 2,
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
}

AlphaZero(model, optimizer, game, args).learn()   # saves model_<i>.pt each iteration
```

Play against the trained model:

```python
model = ResNet(game, 4, 64)
model.load_state_dict(torch.load("model_2.pt"))
model.eval()
mcts = MCTS(game, {'C': 2, 'num_searches': 100}, model)
# then run the play cell: the AI moves with np.argmax(mcts.search(neutral_state))
```

### Tests

```bash
python -m pytest tests/test_rules.py
```

`test_state.py` and `test_net.py` import `state.GoState` and `net.PolicyValueNet`. Those files don't exist yet, so these two tests will fail until they are added.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Node** | One board position in the search tree |
| **Expand** | Create child nodes for a node's legal moves |
| **Prior (P)** | The network's probability for a move, used to guide exploration |
| **Q** | The average value of a move so far |
| **N** | Visit count |
| **PUCT** | Selection rule: `Q + C · P · √N_parent / (1 + N_child)` |
| **Rollout** | A random playout to the end of the game (random-rollout MCTS only) |
| **Policy target** | MCTS visit distribution, which the network learns to imitate |
| **Value target** | Final game result from the point of view of the player to move |
| **Dirichlet noise** | Random noise mixed into the root priors during self-play, for more exploration |
| **Temperature** | Controls how greedy move selection is from visit counts |
| **Komi** | Points given to White in Go to make up for Black moving first |

---

## References

* Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero), Nature 2017.
* Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero), Science 2018.
