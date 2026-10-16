"""Deterministic game rules. The engine never ranks or selects a placement."""

import random
from collections import deque
from dataclasses import dataclass

WIDTH, HEIGHT = 10, 20
PIECES = ("I", "O", "T", "S", "Z", "J", "L")
COLORS = {piece: i + 1 for i, piece in enumerate(PIECES)}
BASES = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((1, 0), (0, 1), (1, 1), (2, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}


def rotations(cells):
    result = []
    for _ in range(4):
        min_x, min_y = min(x for x, _ in cells), min(y for _, y in cells)
        cells = tuple(sorted((x - min_x, y - min_y) for x, y in cells))
        if cells not in result:
            result.append(cells)
        cells = tuple((-y, x) for x, y in cells)
    return tuple(result)


SHAPES = {piece: rotations(cells) for piece, cells in BASES.items()}


def absolute_cells(piece, rotation, x, y):
    return [[x + dx, y + dy] for dx, dy in SHAPES[piece][rotation]]


def fits(board, piece, rotation, x, y):
    return all(
        0 <= cx < WIDTH and 0 <= cy < HEIGHT and not board[cy][cx]
        for cx, cy in absolute_cells(piece, rotation, x, y)
    )


def features(board):
    heights, holes = [], 0
    for x in range(WIDTH):
        occupied = [y for y in range(HEIGHT) if board[y][x]]
        top = occupied[0] if occupied else HEIGHT
        heights.append(HEIGHT - top)
        holes += sum(not board[y][x] for y in range(top, HEIGHT))
    return {
        "holes": holes,
        "max_height": max(heights),
        "aggregate_height": sum(heights),
        "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:], strict=False)),
        "column_heights": heights,
    }


def lock_piece(board, piece, rotation, x, y):
    if not fits(board, piece, rotation, x, y):
        raise ValueError("Placement collides with the board")
    if fits(board, piece, rotation, x, y + 1):
        raise ValueError("Placement has not landed")
    landed = [row[:] for row in board]
    for cx, cy in absolute_cells(piece, rotation, x, y):
        landed[cy][cx] = COLORS[piece]
    cleared = [y for y, row in enumerate(landed) if all(row)]
    remaining = [row for row in landed if not all(row)]
    result = [[0] * WIDTH for _ in cleared] + remaining
    return result, cleared, landed


@dataclass(frozen=True)
class Placement:
    id: str
    piece: str
    rotation: int
    x: int
    y: int
    path: tuple
    metrics: dict

    def public(self):
        return {
            "id": self.id,
            "piece": self.piece,
            "rotation": self.rotation,
            "x": self.x,
            "y": self.y,
            "cells": absolute_cells(self.piece, self.rotation, self.x, self.y),
            "metrics": self.metrics,
        }

    def description(self):
        m = self.metrics
        return (
            f"x={self.x}, rotation={self.rotation}, landing_y={self.y}; "
            f"clears={m['lines_cleared']}, holes_after={m['holes']}, "
            f"holes_change={m['holes_delta']}, max_height={m['max_height']}, "
            f"sum_heights={m['aggregate_height']}, bumpiness={m['bumpiness']}, "
            f"next_piece_can_spawn={m['next_spawn_ok']}"
        )


class Game:
    def __init__(self, seed=42, max_pieces=200):
        self.seed = seed
        self.max_pieces = max_pieces
        self.rng = random.Random(seed)
        self.board = [[0] * WIDTH for _ in range(HEIGHT)]
        self.queue = []
        self._fill_queue()
        self.active = self.queue.pop(0)
        self.score = 0
        self.lines = 0
        self.pieces = 0
        self.over = False
        self.finish_reason = None

    def _fill_queue(self):
        while len(self.queue) < 7:
            bag = list(PIECES)
            self.rng.shuffle(bag)
            self.queue.extend(bag)

    @staticmethod
    def spawn(piece):
        width = max(x for x, _ in SHAPES[piece][0]) + 1
        return 0, (WIDTH - width) // 2, 0

    def candidates(self):
        """BFS from spawn; each candidate is reachable without passing through blocks."""
        if self.over:
            return []
        start = self.spawn(self.active)
        if not fits(self.board, self.active, *start):
            return []
        parents = {start: None}
        pending = deque([start])
        landings = []
        while pending:
            rotation, x, y = state = pending.popleft()
            if not fits(self.board, self.active, rotation, x, y + 1):
                landings.append(state)
            neighbors = (
                (rotation, x - 1, y),
                (rotation, x + 1, y),
                (rotation, x, y + 1),
                ((rotation + 1) % len(SHAPES[self.active]), x, y),
            )
            for neighbor in neighbors:
                if neighbor not in parents and fits(self.board, self.active, *neighbor):
                    parents[neighbor] = state
                    pending.append(neighbor)
        holes_before = features(self.board)["holes"]
        candidates = []
        for index, (rotation, x, y) in enumerate(sorted(landings)):
            after, cleared, _ = lock_piece(self.board, self.active, rotation, x, y)
            metrics = features(after)
            metrics.update(
                lines_cleared=len(cleared),
                holes_delta=metrics["holes"] - holes_before,
                next_spawn_ok=fits(after, self.queue[0], *self.spawn(self.queue[0])),
            )
            path, state = [], (rotation, x, y)
            while state is not None:
                r, px, py = state
                path.append({"rotation": r, "x": px, "y": py})
                state = parents[state]
            candidates.append(
                Placement(
                    f"p{index:03}", self.active, rotation, x, y, tuple(reversed(path)), metrics
                )
            )
        return candidates

    def apply(self, candidate):
        # Revalidate against the current board, including reachability, before changing state.
        actual = next((p for p in self.candidates() if p == candidate), None)
        if actual is None:
            raise ValueError("Stale or unreachable placement")
        before = [row[:] for row in self.board]
        self.board, cleared, landed = lock_piece(
            self.board, candidate.piece, candidate.rotation, candidate.x, candidate.y
        )
        points = (0, 100, 300, 500, 800)[len(cleared)] * (1 + self.lines // 10)
        self.score += points
        self.lines += len(cleared)
        self.pieces += 1
        self.active = self.queue.pop(0)
        self._fill_queue()
        if not fits(self.board, self.active, *self.spawn(self.active)):
            self.over, self.finish_reason = True, "top_out"
        elif self.pieces >= self.max_pieces:
            self.over, self.finish_reason = True, "piece_limit"
        return {
            **candidate.public(),
            "path": list(candidate.path),
            "board_before": before,
            "board_landed": landed,
            "cleared_rows": cleared,
            "points": points,
            "piece_number": self.pieces,
        }

    def public(self):
        return {
            "seed": self.seed,
            "board": [row[:] for row in self.board],
            "width": WIDTH,
            "height": HEIGHT,
            "active": self.active,
            "active_cells": absolute_cells(self.active, *self.spawn(self.active)),
            "next": self.queue[:5],
            "shapes": {p: [list(cell) for cell in SHAPES[p][0]] for p in PIECES},
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces,
            "level": 1 + self.lines // 10,
            "max_pieces": self.max_pieces,
            "over": self.over,
            "finish_reason": self.finish_reason,
            "features": features(self.board),
        }
