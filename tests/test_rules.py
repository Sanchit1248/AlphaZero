import os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.goRules.goAction import resolve_go_action, get_liberties_groups
from Modules.goRules.legalActionMask import legal_go_action_mask, position_key
from Modules.goRules.scoreTerminalPosition import tromp_taylor_result


def test_capture_regression():
    b = [[0, 1, -1, 0, 0],
         [-1, 1, -1, 0, 0],
         [1, -1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    assert legal_go_action_mask(b, 1, [np.array(b)], 0, 10)[0]
    nb, _, cap, *_ = resolve_go_action(b, 1, 0, 0, 10)
    assert cap == 1 and nb[1][0] == 0
    assert nb[0][0] == nb[0][1] == nb[1][1] == 1
    print("ok  capture regression")


def test_ko():
    b = [[0, 1, -1, 0, 0],
         [1, -1, 0, -1, 0],
         [0, 1, -1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
    seen = [np.array(b)]
    nb, _, cap, *_ = resolve_go_action(b, 1, 1 * 5 + 2, 0, 5)
    assert cap == 1 and nb[1][1] == 0
    seen.append(nb.copy())
    m = legal_go_action_mask(nb, -1, seen, 0, 6)
    assert not m[1 * 5 + 1] and m[-1]
    print("ok  ko retake blocked")


def play_random_game(n, rng, komi, stats):
    board = np.zeros((n, n), dtype=int)
    player, passes, mc = 1, 0, 0
    seen = [board.copy()]
    keys = {position_key(board)}
    while True:
        mask = legal_go_action_mask(board, player, keys, passes, mc)
        assert np.array_equal(mask, legal_go_action_mask(board, player, seen, passes, mc))
        legal = [a for a in np.flatnonzero(mask) if a != n * n]
        action = n * n if (not legal or rng.random() < 0.03) else rng.choice(legal)

        nb, nxt, cap, passes2, mc2, term = resolve_go_action(board, player, int(action), passes, mc)

        if action != n * n:
            assert np.sum(nb == player) == np.sum(board == player) + 1
            assert np.sum(nb == -player) == np.sum(board == -player) - cap
            assert not any(np.array_equal(nb, s) for s in seen)
            _, libs = get_liberties_groups(nb, *divmod(int(action), n))
            assert libs
            stats["captures"] += cap > 0

        board, player, passes, mc = nb, nxt, passes2, mc2
        seen.append(board.copy())
        keys.add(position_key(board))
        if term:
            break

    assert legal_go_action_mask(board, player, keys, passes, mc).sum() == 0
    res = tromp_taylor_result(board, komi, passes, mc)
    assert res[0] and res[3] in (1, -1, 0)
    stats["games"] += 1
    stats["moves"] += mc


def run(n, games, seed, komi):
    rng = random.Random(seed)
    stats = dict(games=0, moves=0, captures=0)
    t = time.time()
    for _ in range(games):
        play_random_game(n, rng, komi, stats)
    print(f"ok  {games} random {n}x{n} games: {stats['moves']} moves, "
          f"{stats['captures']} capturing moves, {time.time() - t:.1f}s")


if __name__ == "__main__":
    test_capture_regression()
    test_ko()
    run(5, 300, 1, 0.5)
    run(9, 10, 2, 7.5)
    print("ALL PASSED")
