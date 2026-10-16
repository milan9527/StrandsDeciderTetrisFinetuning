from dataclasses import replace

import pytest

from backend.engine import Game, absolute_cells, features, fits, lock_piece


@pytest.mark.parametrize(
    "piece,count", [("I", 17), ("O", 9), ("T", 34), ("S", 17), ("Z", 17), ("J", 34), ("L", 34)]
)
def test_empty_board_reachable_landings(piece, count):
    game = Game(42)
    game.active = piece
    candidates = game.candidates()
    assert len(candidates) == count
    assert len({tuple(map(tuple, p.public()["cells"])) for p in candidates}) == count
    for candidate in candidates:
        assert candidate.path[0] == dict(
            zip(("rotation", "x", "y"), game.spawn(piece), strict=True)
        )
        previous = None
        for step in candidate.path:
            assert fits(game.board, piece, step["rotation"], step["x"], step["y"])
            if previous:
                changes = [k for k in step if step[k] != previous[k]]
                assert len(changes) == 1
                if changes[0] == "y":
                    assert step["y"] == previous["y"] + 1
                if changes[0] == "x":
                    assert abs(step["x"] - previous["x"]) == 1
            previous = step
        assert not fits(game.board, piece, candidate.rotation, candidate.x, candidate.y + 1)


def test_seeded_seven_bag_has_each_piece_and_replays():
    a, b = Game(2026), Game(2026)
    assert a.public() == b.public()
    a._fill_queue()
    sequence = [a.active] + a.queue
    for i in (0, 7):
        assert sorted(sequence[i : i + 7]) == sorted("IOTSZJL")
    assert sequence != [Game(99).active] + Game(99).queue


def test_tetris_clear_scores_and_keeps_board_size():
    game = Game(42)
    game.active = "I"
    for y in range(16, 20):
        game.board[y] = [2] * 10
        game.board[y][4] = 0
    candidate = next(p for p in game.candidates() if p.x == 4 and p.rotation == 1)
    assert candidate.metrics["lines_cleared"] == 4
    move = game.apply(candidate)
    assert move["cleared_rows"] == [16, 17, 18, 19]
    assert game.lines == 4 and game.score == 800 and game.pieces == 1
    assert game.board == [[0] * 10 for _ in range(20)]


def test_single_clear_moves_rows_down_and_uses_level_before_clear():
    game = Game(42)
    game.active = "I"
    game.lines = 9
    game.board[18][8] = 6
    game.board[19] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    candidate = next(p for p in game.candidates() if p.x == 0 and p.rotation == 0)
    game.apply(candidate)
    assert game.board[19][8] == 6
    assert game.score == 100
    assert game.public()["level"] == 2


def test_geometrically_fitting_but_unreachable_cavity_is_not_candidate():
    game = Game(1)
    game.active = "O"
    game.board[17][:6] = [1] * 6
    game.board[18][6] = game.board[19][6] = 1
    assert fits(game.board, "O", 0, 0, 18)
    assert not any(p.x == 0 and p.y == 18 for p in game.candidates())


def test_holes_are_empty_cells_under_blocks_not_open_columns():
    board = [[0] * 10 for _ in range(20)]
    board[17][2] = board[19][2] = 1
    result = features(board)
    assert result["holes"] == 1
    assert result["max_height"] == 3
    assert result["aggregate_height"] == 3
    assert result["bumpiness"] == 6


def test_invalid_placement_never_mutates_board():
    game = Game(5)
    candidate = game.candidates()[0]
    before = game.public()
    with pytest.raises(ValueError):
        game.apply(replace(candidate, x=99))
    assert game.public() == before
    with pytest.raises(ValueError):
        lock_piece(game.board, game.active, *game.spawn(game.active))


def test_spawn_blocked_and_piece_limit_stop():
    game = Game(5)
    for x, y in absolute_cells(game.active, *game.spawn(game.active)):
        game.board[y][x] = 1
    assert game.candidates() == []
    bounded = Game(5, max_pieces=1)
    bounded.apply(bounded.candidates()[0])
    assert bounded.over and bounded.finish_reason == "piece_limit"
    assert bounded.candidates() == []


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_cell_conservation_across_complete_games(seed):
    game = Game(seed, max_pieces=60)
    while not game.over:
        candidates = game.candidates()
        if not candidates:
            break
        # Deliberately mechanical test selection, not a production policy.
        game.apply(candidates[(game.pieces * 3) % len(candidates)])
        assert len(game.board) == 20
        assert all(len(row) == 10 for row in game.board)
        assert sum(bool(v) for row in game.board for v in row) == 4 * game.pieces - 10 * game.lines
