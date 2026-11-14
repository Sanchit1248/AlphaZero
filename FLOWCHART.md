# AlphaZero Pipeline: Flowcharts

These diagrams show how the pieces in `tictactoe.ipynb` fit together. The Go version in `Modules/` follows the same structure on a larger board.

The diagrams use [Mermaid](https://mermaid.js.org/). They render on GitHub and in VS Code's Markdown preview (with a Mermaid extension).

---

## 1. The whole training loop

`learn()` alternates between two phases: self-play, which generates data, and training, which updates the network. MCTS runs only during self-play.

```mermaid
flowchart TD
    Start(["learn()"]) --> Iter{"next iteration?<br/>num_iterations"}
    Iter -- yes --> Eval["model.eval()"]
    Eval --> SP["selfPlay() × num_selfPlay_iterations<br/>(uses MCTS + model)"]
    SP --> Mem[("memory<br/>(board, MCTS probs, result)")]
    Mem --> TrainMode["model.train()"]
    TrainMode --> Train["train(memory) × num_epochs<br/>(network only, no MCTS)"]
    Train --> Save["save model_i.pt<br/>and optimizer_i.pt"]
    Save --> Iter
    Iter -- no --> Done(["trained model"])
```

The cycle that makes it improve:

```mermaid
flowchart LR
    Net["network"] -- "priors + values" --> MCTS["MCTS search"]
    MCTS -- "stronger moves" --> Games["self-play games"]
    Games -- "training targets" --> Train["train()"]
    Train -- "updated weights" --> Net
```

---

## 2. Self-play: one game

The AI plays both sides. Each position is saved, and once the game ends, every saved position is labelled with the result from the point of view of the player who was to move there.

```mermaid
flowchart TD
    A(["selfPlay()"]) --> B["state = empty board<br/>player = 1"]
    B --> C["neutral_state = change_perspective(state, player)<br/>player to move is always 1"]
    C --> D["action_probs = mcts.search(neutral_state)"]
    D --> E["memory.append(neutral_state, action_probs, player)"]
    E --> F["sample action ~ action_probs"]
    F --> G["state = get_next_state(state, action, player)"]
    G --> H{"get_val_and_terminated<br/>game over?"}
    H -- no --> I["player = opponent"] --> C
    H -- yes --> J["for each saved position:<br/>result = value if same player as last mover<br/>else −value"]
    J --> K(["return (encoded board, action_probs, result)"])
```

Example of how the results are assigned (X wins on the 5th move):

| Move | To play | Result stored |
|------|---------|---------------|
| 1    | X       | +1            |
| 2    | O       | −1            |
| 3    | X       | +1            |
| 4    | O       | −1            |
| 5    | X       | +1            |

---

## 3. MCTS: one search (the AlphaZero version)

`search()` repeats this `num_searches` times, then returns the visit counts of the root's children as probabilities.

```mermaid
flowchart TD
    S(["one search"]) --> R["node = root"]
    R --> FE{"node has children?<br/>is_fully_expanded()"}
    FE -- yes --> Sel["node = select()<br/>child with highest PUCT"] --> FE
    FE -- no --> T{"game over at node?<br/>get_val_and_terminated"}
    T -- yes --> TV["value = −result<br/>(real game result)"]
    T -- no --> NN["policy, value = model(encoded node.state)"]
    NN --> Mask["policy = softmax → mask illegal moves → renormalise"]
    Mask --> Exp["expand(policy)<br/>create ALL children, each with its prior"]
    TV --> BP
    Exp --> BP["backprop(value)<br/>add to value_sum, visit_count += 1<br/>flip sign at each level"]
    BP --> End(["next search"])
```

Selection score for each child (PUCT):

```
q     = 1 − ((child.value_sum / child.visit_count) + 1) / 2      # 0 if unvisited
score = q + C · child.prior · √(parent.visit_count) / (1 + child.visit_count)
```

* `q` is how good the move has been so far, converted to the parent's point of view and scaled to [0, 1].
* The second term is an exploration bonus. It is largest for moves the network likes (high prior) that haven't been visited much.

### How it differs from the random-rollout MCTS

| Step       | Random-rollout MCTS (`MCTSrandom`) | AlphaZero MCTS (`MCTS`)                  |
|------------|------------------------------------|------------------------------------------|
| Expand     | one random untried move per search | all legal moves at once, with priors     |
| Evaluate   | random playout to the end          | network value head                       |
| Select     | UCB: `q + C·√(ln N / n)`           | PUCT: `q + C·P·√N / (1 + n)`             |

---

## 4. Training: where the losses come from

MCTS does **not** appear here. By the time `train()` runs, MCTS has finished, and its output is stored data. Gradients flow only through the network.

```mermaid
flowchart LR
    subgraph Memory["memory (from self-play)"]
        Board["encoded board"]
        PT["MCTS probabilities<br/>(policy target)"]
        VT["game result<br/>(value target)"]
    end

    Board --> Model["model (ResNet)"]
    Model --> OP["out_policy<br/>(logits)"]
    Model --> OV["out_value<br/>(tanh, −1…1)"]

    OP --> CE["cross_entropy"]
    PT --> CE
    OV --> MSE["mse_loss"]
    VT --> MSE

    CE --> L["loss = policy_loss + value_loss"]
    MSE --> L
    L --> BW["loss.backward()"]
    BW --> Step["optimizer.step()"]
    Step -. "updates weights" .-> Model
```

* **Policy loss**: learn to suggest the moves that the search ended up preferring.
* **Value loss**: learn to predict who wins from this position.

---

## 5. Playing against the trained model

The trained model is **just the network**. You can use it on its own, or wrap it in MCTS (which is how AlphaZero plays).

```mermaid
flowchart TD
    P(["AI's turn"]) --> N["neutral_state = change_perspective(state, player)"]
    N --> Choice{"how to pick the move?"}
    Choice -- "network only<br/>(fast, weaker)" --> A["policy = model(neutral_state)<br/>mask illegal moves → argmax"]
    Choice -- "MCTS + network<br/>(slower, stronger)" --> B["probs = MCTS(game, args, model).search(neutral_state)<br/>argmax"]
    A --> Play["get_next_state(state, action, player)"]
    B --> Play
```

---

## 6. The board is always seen from "my" point of view

Every node and every network input treats the player to move as `+1` and the opponent as `−1`. This lets one network play both sides.

```mermaid
flowchart LR
    Real["real board<br/>X = +1, O = −1<br/>O to move"] -- "× −1<br/>change_perspective" --> Neutral["neutral board<br/>O = +1, X = −1<br/>'I' am +1"]
    Neutral --> Search["MCTS / network"]
    Search --> Move["move index"]
    Move -- "played with the real player value" --> Real
```

Because each tree level is the opposite player, values are flipped in two places:

* in **`backprop`**: `−value` is passed up to the parent;
* in **`get_ucb`**: `1 − …` converts the child's score into the parent's point of view.
