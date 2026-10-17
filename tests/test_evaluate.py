"""Saved evaluation choices must map shuffled options back to the test data."""

import gzip
import json

import pytest

from finetuning import evaluate


def test_saved_predictions_use_original_option_indices(tmp_path, monkeypatch):
    rows = [
        {"task": "example", "state": "a", "options": [["a", "A"], ["b", "B"], ["c", "C"]],
         "label": 0},
        {"task": "example", "state": "b", "options": [["a", "A"], ["b", "B"]],
         "label": 1},
    ]
    results = iter([
        {"predictions": [0, 0], "accuracy": 0.5},
        {"predictions": [2, 0], "accuracy": 1.0},
    ])
    monkeypatch.setattr(evaluate, "predict", lambda *args: next(results))
    path = tmp_path / "predictions.jsonl.gz"
    report = evaluate.choice_report(None, rows, "cpu", 2, path)
    with gzip.open(path, "rt") as stream:
        saved = [json.loads(line) for line in stream]
    assert [r["reversed_predicted_original_index"] for r in saved] == [0, 1]
    assert [r["correct"] for r in saved] == [True, False]
    assert report["same_actual_choice_fraction"] == 0.5
    assert report["by_task"] == {"example": {"rows": 2, "correct": 1}}
    assert "predictions" not in report["canonical"]


def test_incomplete_predictions_cannot_be_saved_as_complete(tmp_path, monkeypatch):
    rows = [{"task": "example", "state": "a", "options": [["a", "A"], ["b", "B"]],
             "label": 0}]
    results = iter([{"predictions": [0]}, {"predictions": []}])
    monkeypatch.setattr(evaluate, "predict", lambda *args: next(results))
    path = tmp_path / "predictions.jsonl.gz"
    with pytest.raises(ValueError, match="shorter"):
        evaluate.choice_report(None, rows, "cpu", 2, path)
    assert not path.exists()
