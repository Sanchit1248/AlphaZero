# AlphaZero Pipeline

Diagrams of the pipeline in `TicTacToe.ipynb`. The Go version uses the same structure. The diagrams use [Mermaid](https://mermaid.js.org/), which GitHub renders automatically.

## 1. Training loop

MCTS runs only during self-play. Training uses only the network and the stored data.

```mermaid
flowchart TD
    Start(["learn()"]) --> Iter{"next iteration?"}
    Iter -- yes --> SP["selfPlay() × num_selfPlay_iterations<br/>model.eval(), MCTS + network"]
    SP --> Mem[("memory<br/>(board, MCTS probs, result)")]
    Mem --> Train["train(memory) × num_epochs<br/>model.train(), network only"]
    Train --> Save["save model, optimizer, loss history"]
    Save --> Iter
    Iter -- no --> Done(["trained model"])
```

## 2. Self-play: one game

```mermaid
flowchart TD
    A(["selfPlay()"]) --> B["board from the mover's view<br/>change_perspective"]
    B --> C["probs = mcts.search(board)<br/>(Dirichlet noise at the root)"]
    C --> D["save (board, probs, player)"]
    D --> E["sample move from probs^(1/temperature)"]
    E --> F{"game over?"}
    F -- no --> B
    F -- yes --> G["label each saved position:<br/>+1 if its player won, −1 if lost, 0 draw"]
    G --> H(["return training examples"])
```

## 3. MCTS: one search

`search()` repeats this `num_searches` times, then returns the root's visit counts as probabilities.

```mermaid
flowchart TD
    S(["node = root"]) --> FE{"node has children?"}
    FE -- yes --> Sel["node = child with highest PUCT"] --> FE
    FE -- no --> T{"game over here?"}
    T -- yes --> TV["value = real result"]
    T -- no --> NN["policy, value = network(node)"]
    NN --> Exp["mask illegal moves,<br/>create all children with their priors"]
    TV --> BP["backprop: update visits and values,<br/>flipping the sign at each level"]
    Exp --> BP
```

```
PUCT = q + C · prior · √(parent visits) / (1 + child visits)
q    = 1 − (child's average value + 1) / 2      (the child's value, from the parent's view)
```

## 4. Where the losses come from

```mermaid
flowchart LR
    Board["board"] --> Model["network"]
    Model --> OP["policy logits"]
    Model --> OV["value"]
    PT["MCTS probs (target)"] --> CE["cross-entropy"]
    OP --> CE
    VT["game result (target)"] --> MSE["MSE"]
    OV --> MSE
    CE --> L["loss"]
    MSE --> L
    L --> Opt["backward + Adam step<br/>(weight_decay = L2)"]
```

## 5. Playing and evaluation

The trained model is just the network. It can play on its own, or guide MCTS (stronger).

```mermaid
flowchart LR
    Obs["board + whose turn"] --> Persp["view from the mover"]
    Persp --> Q{"search?"}
    Q -- no --> P["network policy"]
    Q -- yes --> M["MCTS(network) visit counts"]
    P --> Mask["mask illegal moves → argmax"]
    M --> Mask
    Mask --> Move(["move"])
```
