# AlphaZero from Scratch

An AlphaZero implementation in Python and PyTorch, built step by step: first on **Tic-Tac-Toe** (complete and trained), then on **Go** (in progress).

AlphaZero learns a game with no human data. It plays against itself using Monte Carlo Tree Search (MCTS) guided by a neural network, then trains the network on those games. See **[FLOWCHART.md](FLOWCHART.md)** for diagrams of the pipeline.

## Results (Tic-Tac-Toe)

Trained for 3 iterations × 200 self-play games, then evaluated over 100 games against a random player, alternating who starts:

| Agent | Wins | Draws | Losses |
|---|---|---|---|
| Network + MCTS (100 searches) | 92 | 8 | 0 |
| Network only (no search) | 83 | 11 | 6 |

In games played against it by hand, it has always won or drawn.

## Repository

| Path | Contents |
|---|---|
| `TicTacToe.ipynb` | The main notebook: game rules → random-rollout MCTS → ResNet → AlphaZero MCTS → self-play training → loss curve → play / evaluation |
| `model_*_TicTacToe.pt` | Trained checkpoints (one per iteration), plus optimizer states and `loss_history_TicTacToe.npy` |
| `MCTS.ipynb`, `Model.ipynb` | Scratch notebooks |
| `GoModules/` | Go building blocks: rules (captures, ko, Tromp-Taylor scoring), 17-plane board encoding, network layers, MCTS and self-play helpers |
| `tests/` | Rule tests. `test_state.py` and `test_net.py` need `state.py` / `net.py`, which aren't written yet |

## How it works

1. **Self-play.** For each move, MCTS runs many searches. At each leaf the network gives a **policy** (which moves to explore) and a **value** (who is winning). Root visit counts become move probabilities. During self-play, Dirichlet noise and a temperature add exploration.
2. **Labels.** When the game ends, each position gets the result from the view of the player to move: +1, 0 or −1.
3. **Training.** The network learns to match the MCTS probabilities (cross-entropy) and the results (MSE), with L2 regularisation through Adam's `weight_decay`.
4. **Repeat.** A better network gives a better search, which gives better training data.

Key convention: **the player to move is always +1**. For Tic-Tac-Toe, the board is multiplied by the player before searching. For Go, the state keeps the real colour, because komi makes the two sides asymmetric.

## Running it

```bash
pip install numpy torch tqdm matplotlib jupyter
```

Open `TicTacToe.ipynb` and run the cells top to bottom. Skip the interactive play cells: they wait for keyboard input.

- **Train:** the training cell (~20 min per iteration at 200 games). Uses the GPU if available.
- **Loss curve:** the cell after training. It also works later, from the saved `.npy` file.
- **Play:** the last play cell loads `model_2_TicTacToe.pt`. Set `human = -1` to let the bot move first.
- **Evaluate:** the Kaggle-style agent cells play the bot against a random player.

## Go: status

The Go game class (built on `GoModules/`) works with the same MCTS and training code, unchanged. Self-play is currently slow (~10 s per 5×5 game, ~40 s per 9×9 game) because the rules are pure Python and the network sees one board at a time. Next steps: batched parallel self-play, faster rules, then 9×9 training evaluated against GNU Go.

