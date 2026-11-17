# AlphaZero from Scratch

An AlphaZero implementation in Python and PyTorch, built step by step: first on **Tic-Tac-Toe** (complete and trained), then on **9×9 Go** (trained and evaluated against GNU Go, see [Go: findings](#go-findings)).

AlphaZero learns a game with no human data. It plays against itself using Monte Carlo Tree Search (MCTS) guided by a neural network, then trains the network on those games. See **[FLOWCHART.md](FLOWCHART.md)** for diagrams of the pipeline.

## Results (Tic-Tac-Toe)

Trained for 3 iterations × 200 self-play games, then evaluated over 100 games against a random player, alternating who starts:

| Agent | Wins | Draws | Losses |
|---|---|---|---|
| Network + MCTS (100 searches) | 92 | 8 | 0 |
| Network only (no search) | 83 | 11 | 6 |

In games played against it by hand, it has always won or drawn.

## Go: findings

**Setup:** 9×9 board, komi 7.5, Tromp-Taylor scoring. A 6-block × 96-channel ResNet trained for 34 iterations (about 13,000 self-play games) with batched self-play, a replay buffer and 8-fold symmetry augmentation.

**Evaluation:** 100 games against GNU Go 3.8 (levels 1–10, 10 games each, alternating colours), scored two ways:

| Scoring | Bot wins |
|---|---|
| Our strict Tromp-Taylor rules | 32 / 100 |
| GNU Go's own scoring (dead stones removed) | **0 / 100** |

**What happened: reward hacking.** Strict Tromp-Taylor counts every stone left on the board as alive. Ending a game early with dead stones still on the board keeps the opponent's area from counting as territory, and komi then often decides the game for White. In self-play, both sides were equally weak, so **passing early as White** kept being rewarded, and the bot learned that instead of territory and life and death. Nearly every "win" under our scoring was a short White game (29–53 moves), and GNU Go scored all of them as losses.

**Lessons:**
- A self-play agent optimises the rules exactly as written, including their loopholes.
- Falling loss curves didn't mean a stronger bot. Only games against an independent opponent revealed the problem.
- Evaluation must use scoring that real players accept (GNU Go's `final_score` here).
- Fixes for a future run: only allow passing when no sensible move is left, forbid filling your own eyes, and play games out until dead stones are captured.

**Engineering along the way:**
- Batched self-play: 4–9× faster on 5×5, ~1.5× on 9×9.
- Lazy node expansion: verified to produce identical moves.
- Profiling showed the Python rule checks and the PUCT loop take ~80% of the time on 9×9; the network takes ~5%.
- Replay buffer + symmetries cut the value head's overfitting gap (epoch-0 vs last-epoch loss) from 0.36 to 0.04.

## Repository

| Path | Contents |
|---|---|
| `TicTacToe.ipynb` | Game rules → random-rollout MCTS → ResNet → AlphaZero MCTS → self-play training → loss curve → play / evaluation |
| `TicTacToeModels/` | Trained Tic-Tac-Toe checkpoints, optimizer states and loss history |
| `MCTS.ipynb`, `Model.ipynb` | Scratch notebooks |
| `GoModules/` | Go building blocks: rules (captures, superko, Tromp-Taylor scoring), 17-plane board encoding, network layers, MCTS and self-play helpers, board symmetries |
| `tests/` | Rule tests. `test_state.py` and `test_net.py` need `state.py` / `net.py`, which aren't written yet |

## How it works

1. **Self-play.** For each move, MCTS runs many searches. At each leaf the network gives a **policy** (which moves to explore) and a **value** (who is winning). Root visit counts become move probabilities. Dirichlet noise and a temperature add exploration.
2. **Labels.** When the game ends, each position gets the result from the view of the player to move: +1, 0 or −1.
3. **Training.** The network learns to match the MCTS probabilities (cross-entropy) and the results (MSE), with L2 regularisation through Adam's `weight_decay`.
4. **Repeat.** A better network gives a better search, which gives better training data.

Key convention: **the player to move is always +1**. For Tic-Tac-Toe, the board is multiplied by the player before searching. For Go, the state keeps the real colour, because komi makes the two sides asymmetric.

## Running it

```bash
pip install numpy torch tqdm matplotlib jupyter
```

Open `TicTacToe.ipynb` and run the cells top to bottom. Skip the interactive play cells: they wait for keyboard input.

The notebook's load paths still point to the main folder; point them at `TicTacToeModels/` to load the trained checkpoints.
