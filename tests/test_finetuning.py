import hashlib
import json

import pytest

from backend.engine import Game
from backend.policy import quality_vector
from finetuning.data import SPLITS, audit, build, examples_for_game, state_key
from finetuning.runtime import UPSTREAM_COMMIT, permute_row, use_upstream


@pytest.fixture
def dataset(tmp_path):
    build(tmp_path / "data", {split: 2 for split in SPLITS}, pieces=5)
    return tmp_path / "data"


def rewrite(folder, split, rows):
    path = folder / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["splits"][split]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["splits"][split]["rows"] = len(rows)
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_generated_labels_replay_and_split_boards_do_not_overlap(dataset):
    result = audit(dataset)
    assert result["all_labels_recomputed"]
    assert result["no_cross_split_board_or_seed_leakage"]
    assert all(info["rows"] for info in result["splits"].values())


def test_incorrect_label_rejected_even_with_valid_file_checksum(dataset):
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    rows[0]["label"] = (rows[0]["label"] + 1) % len(rows[0]["options"])
    rewrite(dataset, "train", rows)
    with pytest.raises(ValueError, match="strict dominance"):
        audit(dataset)


def test_cross_split_board_leak_rejected_even_with_new_seed(dataset):
    leaked = json.loads((dataset / "train.jsonl").read_text().splitlines()[0])
    leaked["_meta"].update(split="test", seed=40000)
    rewrite(dataset, "test", [leaked])
    with pytest.raises(ValueError, match="Board leakage"):
        audit(dataset)


def test_corrupted_candidate_description_rejected(dataset):
    rows = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
    rows[0]["options"][0][1] = "Clear 99 rows"
    rewrite(dataset, "train", rows)
    with pytest.raises(ValueError, match="facts"):
        audit(dataset)


def test_permutation_remaps_label_and_real_candidate_identity(dataset):
    row = json.loads((dataset / "train.jsonl").read_text().splitlines()[0])
    changed = permute_row(row, list(reversed(range(len(row["options"])))), rename=True)
    assert changed["_meta"]["candidate_ids"][changed["label"]] == row["_meta"]["candidate_ids"][row["label"]]
    assert changed["options"][changed["label"]][1] == row["options"][row["label"]][1]
    assert changed["options"][changed["label"]][0] != row["options"][row["label"]][0]


def test_state_key_ignores_score_but_distinguishes_next_piece():
    from backend.decider import state_text

    game = Game(1)
    key = state_key(state_text(game))
    game.lines = 100
    assert state_key(state_text(game)) == key
    game.queue[0] = "T" if game.queue[0] != "T" else "O"
    assert state_key(state_text(game)) != key


def test_equal_metric_options_do_not_receive_arbitrary_gold_label():
    import random

    game = Game(1)
    game.active = "O"
    candidates = game.candidates()
    equal = [p for p in candidates if p.x in (0, 8)]
    assert len(equal) == 2
    assert quality_vector(equal[0]) == quality_vector(equal[1])
    assert examples_for_game(game, equal, random.Random(1), "train") == []


def test_data_generation_is_reproducible_and_wont_overwrite(dataset, tmp_path):
    other = tmp_path / "copy"
    build(other, {split: 2 for split in SPLITS}, pieces=5)
    for split in SPLITS:
        assert (dataset / f"{split}.jsonl").read_bytes() == (other / f"{split}.jsonl").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        build(dataset)


def test_exported_upstream_rejects_changed_or_extra_python_files(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    source = tmp_path / "source.py"
    source.write_text("# pinned source\n")
    manifest = {"commit": UPSTREAM_COMMIT, "files": {
        "source.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }}
    (tmp_path / ".sft-source.json").write_text(json.dumps(manifest))
    assert use_upstream(tmp_path) == UPSTREAM_COMMIT
    source.write_text("# changed source\n")
    with pytest.raises(ValueError, match="file changed"):
        use_upstream(tmp_path)
    source.write_text("# pinned source\n")
    (tmp_path / "extra.py").write_text("# unreviewed\n")
    with pytest.raises(ValueError, match="file set changed"):
        use_upstream(tmp_path)
