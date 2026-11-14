import os, sys, random
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state import GoState
from Modules.goRules.scoreTerminalPosition import tromp_taylor_result


def test_initial():
    s = GoState.initial(5, 0.5)
    m = s.legal_mask()
    assert m.shape == (26,) and m.all()
    x = s.encode()
    assert x.shape == (17, 5, 5) and x.dtype == torch.float32
    assert x[16].all() and not x[:16].any()
    print("ok  initial state")


def test_play_is_immutable():
    s0 = GoState.initial(5, 0.5)
    s1 = s0.play(12)
    assert s0.board.sum() == 0 and s0.player == 1 and s0.move_count == 0
    assert s1.board[2, 2] == 1 and s1.player == -1 and s1.move_count == 1
    assert len(s0.seen) == 1 and len(s1.seen) == 2
    try:
        s1.play(12)
        raise SystemExit("occupied point accepted")
    except ValueError:
        pass
    print("ok  play returns a new state, illegal moves raise")


def test_encode_layout():
    s = GoState.initial(5, 0.5).play(0)
    x = s.encode()
    assert x[0].sum() == 0 and x[1, 0, 0] == 1
    assert x[2:4].sum() == 0
    assert not x[16].any()
    s = s.play(1)
    x = s.encode()
    assert x[0, 0, 0] == 1 and x[1, 0, 1] == 1
    assert x[2, 0, 0] == 1 and x[3].sum() == 0
    assert x[16].all()
    print("ok  encode plane layout (own, opponent, newest first, colour plane)")


def test_history_capped():
    s = GoState.initial(5, 0.5)
    for a in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        s = s.play(a)
    assert len(s.history) == 8
    assert s.encode().shape == (17, 5, 5)
    print("ok  history capped at 8 boards")


def test_random_games(n=5, games=200, seed=0, komi=0.5):
    rng = random.Random(seed)
    for _ in range(games):
        s = GoState.initial(n, komi)
        while not s.terminal:
            m = s.legal_mask()
            legal = [a for a in np.flatnonzero(m) if a != n * n]
            a = n * n if (not legal or rng.random() < 0.03) else rng.choice(legal)
            s = s.play(int(a))
        assert s.legal_mask().sum() == 0
        _, b, w, winner = tromp_taylor_result(s.board, komi, s.passes, s.move_count)
        v = s.terminal_value()
        assert v == (0.0 if winner == 0 else (1.0 if winner == s.player else -1.0))
        assert len(s.seen) <= s.move_count + 1
    print(f"ok  {games} random {n}x{n} games through GoState")


def test_terminal_value_sign():
    s = GoState.initial(5, 0.5)
    for a in (12, 25, 25):
        s = s.play(a)
    assert s.terminal and s.player == -1
    assert s.terminal_value() == -1.0
    print("ok  terminal value is from the side to move (Black's lone stone beats White's komi)")


if __name__ == "__main__":
    test_initial()
    test_play_is_immutable()
    test_encode_layout()
    test_history_capped()
    test_terminal_value_sign()
    test_random_games()
    print("ALL PASSED")
