"""Build and independently audit engine-labelled choice data without model calls."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from backend.decider import question, state_text
from backend.engine import Game
from backend.policy import dominates, quality_vector

SPLITS = {"train": (10000, 96), "validation": (20000, 16),
          "calibration": (30000, 16), "test": (40000, 16)}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def state_key(state):
    """Ignore piece colours, score and seed: identical observable boards stay together."""
    data = json.loads(state)
    return digest(json.dumps({k: data[k] for k in (
        "board_rows_top_to_bottom", "current_piece", "next_piece"
    )}, sort_keys=True))


def labelled_groups(candidates, rng):
    """Only label strict metric dominance; no heuristic 'best move' labels."""
    champions = [(c, [p for p in candidates if dominates(c, p)]) for c in candidates]
    champions = [(c, worse) for c, worse in champions if worse]
    if not champions:
        return []
    winner, worse = rng.choice(champions)
    pair = [winner, rng.choice(worse)]
    # A different, larger group teaches the deployed 2..12-option comparison format.
    wide = [(c, worse) for c, worse in champions if len(worse) >= 2]
    if not wide:
        return [pair]
    winner, worse = rng.choice(wide)
    n = rng.randint(2, min(11, len(worse)))
    return [pair, [winner, *rng.sample(worse, n)]]


def examples_for_game(game, candidates, rng, split):
    state = state_text(game)
    rows = []
    for group in labelled_groups(candidates, rng):
        gold = group[0].id
        for policy in ("original", "model"):
            ordered = rng.sample(group, len(group))
            q = question(ordered, policy)
            # Names have no stable relationship to rank, placement ID, or ordering.
            names = rng.sample(range(1000), len(ordered))
            ids = [f"p{n:03}" for n in names]
            rows.append({
                "kind": "choice", "state": state, "instructions": q["instructions"],
                "options": [[name, q["criteria"][p.id]] for name, p in zip(ids, ordered, strict=True)],
                "label": next(i for i, p in enumerate(ordered) if p.id == gold),
                "task": f"tetris_dominance_{policy}", "weight": 1.0,
                "_meta": {
                    "split": split, "seed": game.seed, "turn": game.pieces + 1,
                    "state_key": state_key(state), "policy": policy,
                    "candidate_ids": [p.id for p in ordered],
                    "label_source": "engine_strict_metric_dominance",
                    "label_scope": "one_step_metrics_not_long_term_optimality",
                },
            })
    return rows


def trajectory_choice(candidates, rng, mode):
    """Diversify visited boards. These sampling policies NEVER supply labels."""
    if mode == 0:
        return rng.choice(candidates)
    if mode == 1:
        frontier = [c for c in candidates if not any(dominates(p, c) for p in candidates)]
        return rng.choice(frontier)
    if rng.random() < 0.1:
        return rng.choice(candidates)
    return min(candidates, key=quality_vector)


def build(output, counts=None, pieces=40):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be absent or empty; datasets are immutable.")
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {"schema_version": 1, "teacher": None, "pieces_per_game": pieces,
                "label_source": "engine_strict_metric_dominance", "splits": {}}
    for split, (start, default) in SPLITS.items():
        count = (counts or {}).get(split, default)
        seeds = list(range(start, start + count))
        generated = visited = duplicate = 0
        path = output / f"{split}.jsonl"
        with path.open("w") as stream:
            for seed in seeds:
                game, rng = Game(seed, pieces), random.Random(seed)
                while not game.over:
                    candidates = game.candidates()
                    if not candidates:
                        break
                    key = state_key(state_text(game))
                    visited += 1
                    if key not in seen:
                        seen.add(key)
                        for row in examples_for_game(game, candidates, rng, split):
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                            generated += 1
                    else:
                        duplicate += 1
                    game.apply(trajectory_choice(candidates, rng, seed % 3))
        manifest["splits"][split] = {
            "seeds": seeds, "rows": generated, "visited_states": visited,
            "duplicate_states_skipped": duplicate,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(json.dumps({"split": split, **manifest["splits"][split]}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = audit(output)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return manifest


def restore(state):
    data = json.loads(state)
    board = data["board_rows_top_to_bottom"]
    if len(board) != 20 or any(len(row) != 10 or set(row) - {".", "#"} for row in board):
        raise ValueError("Invalid board")
    game = Game()
    game.board = [[int(cell == "#") for cell in row] for row in board]
    game.active = data["current_piece"]
    game.queue[0] = data["next_piece"]
    return game


def audit(folder):
    """Recompute reachability, descriptions and dominance, and detect split leakage."""
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    owners, seed_owners, report = {}, {}, {}
    for split in SPLITS:
        path = folder / f"{split}.jsonl"
        info = manifest["splits"][split]
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        for seed in info["seeds"]:
            if seed in seed_owners:
                raise ValueError("Seed leakage")
            seed_owners[seed] = split
        histogram, tasks = Counter(), Counter()
        cache, count = {}, 0
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                meta = row["_meta"]
                key = state_key(row["state"])
                if key in owners and owners[key] != split:
                    raise ValueError("Board leakage")
                owners[key] = split
                if (meta["state_key"] != key or meta["split"] != split
                        or meta["seed"] not in info["seeds"]):
                    raise ValueError("Invalid provenance")
                if key not in cache:
                    cache[key] = {p.id: p for p in restore(row["state"]).candidates()}
                chosen = [cache[key][pid] for pid in meta["candidate_ids"]]
                opts, label = row["options"], row["label"]
                if (row["kind"] != "choice" or not 2 <= len(opts) <= 12
                        or len(chosen) != len(opts) or type(label) is not int
                        or not 0 <= label < len(opts)
                        or len({p.id for p in chosen}) != len(chosen)
                        or len({p[0] for p in opts}) != len(opts)):
                    raise ValueError("Invalid options/label")
                q = question(chosen, meta["policy"])
                if row["instructions"] != q["instructions"] or [
                    text for _, text in opts
                ] != list(q["criteria"].values()):
                    raise ValueError("Candidate facts do not match the engine")
                if not all(i == label or dominates(chosen[label], p)
                           for i, p in enumerate(chosen)):
                    raise ValueError("Label is not supported by strict dominance")
                histogram[len(opts)] += 1
                tasks[row["task"]] += 1
                count += 1
        if count != info["rows"] or count == 0:
            raise ValueError(f"{split}: row count mismatch or empty dataset")
        report[split] = {"rows": count, "unique_states": len(cache),
                         "option_counts": dict(histogram), "tasks": dict(tasks)}
    return {"all_labels_recomputed": True, "no_cross_split_board_or_seed_leakage": True,
            "splits": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/sft/data"))
    parser.add_argument("--pieces", type=int, default=40)
    parser.add_argument("--train-seeds", type=int, default=96)
    parser.add_argument("--heldout-seeds", type=int, default=16)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.output), indent=2))
    else:
        if min(args.pieces, args.train_seeds, args.heldout_seeds) < 1:
            parser.error("Counts must be positive")
        counts = {s: args.heldout_seeds for s in SPLITS}
        counts["train"] = args.train_seeds
        build(args.output, counts, args.pieces)


if __name__ == "__main__":
    main()
