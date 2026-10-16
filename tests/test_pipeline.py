"""Dataset integrity, failure recovery, and honest release gates (no mock model accuracy claims)."""

import copy
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from finetuning import datasets, pipeline


def example(index=0, **changes):
    return {"state": {"case": index}, "instructions": "Choose the lower cost.",
            "options": [["a", "Cost 10"], ["b", "Cost 20"]],
            "label": 0, **changes}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_jsonl_csv_and_gzip_share_the_same_labels(tmp_path):
    rows = [example(1), example(2)]
    rows[1].pop("label")
    rows[1]["answer"] = "a"
    rows[1]["options"] = {"a": "Cost 10", "b": "Cost 20"}
    jsonl = write_rows(tmp_path / "rows.jsonl", rows)
    gz = tmp_path / "rows.jsonl.gz"
    with gzip.open(gz, "wt") as stream:
        stream.write(jsonl.read_text())
    csv_path = tmp_path / "rows.csv"
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*rows)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, int)) else v
                             for k, v in row.items()})
    loaded = list(datasets.load(jsonl))
    assert loaded == list(datasets.load(gz))
    assert [r["label"] for r in datasets.load(csv_path)] == [r["label"] for r in loaded]
    assert [datasets.state_id(r) for r in datasets.load(csv_path)] == [
        datasets.state_id(r) for r in loaded
    ]


@pytest.mark.parametrize("changes", [
    {"label": True}, {"label": "0"}, {"label": -1}, {"label": 2}, {"answer": "missing"},
    {"answer": "b"}, {"options": [["a", "one"], ["a", "two"]]},
    {"options": [["a", "same"], ["b", "same"]]}, {"options": [["a", "only"]]},
    {"options": [["a", "same words"], ["b", "same  words"]]},
    {"options": [["a\nfake", "one"], ["b", "two"]]},
    {"state": "  "}, {"state": {"value": float("nan")}}, {"instructions": ""},
    {"kind": "score"}, {"weight": True}, {"weight": 2}, {"group_id": ""},
])
def test_invalid_data_is_rejected_before_training(changes):
    with pytest.raises(ValueError):
        datasets.normalize(example(**changes))


def test_parse_error_includes_file_and_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(example()) + "\nnot json\n")
    with pytest.raises(ValueError, match="broken.jsonl:2"):
        list(datasets.load(path))


def test_duplicate_and_conflicting_labels_ignore_option_names_and_order():
    first = example()
    second = example(options=[["renamed_b", "Cost 20"], ["renamed_a", "Cost 10"]], label=1)
    with pytest.raises(ValueError, match="Duplicate question"):
        datasets.validate_rows({"source": [first, second]})
    second["label"] = 0
    with pytest.raises(ValueError, match="Conflicting labels"):
        datasets.validate_rows({"source": [first, second]})


def test_split_keeps_transitively_linked_groups_together():
    rows = [example(1, group_id="A"), example(2, group_id="A"),
            example(2, group_id="B", instructions="Pick the cheaper one."),
            example(3, group_id="B")]
    rows += [example(i) for i in range(4, 30)]
    splits = datasets.split_rows(rows, [0.7, 0.1, 0.1, 0.1], 17)
    assert splits == datasets.split_rows(rows, [0.7, 0.1, 0.1, 0.1], 17)
    related = [split for split, items in splits.items() if any(r.get("group_id") for r in items)]
    assert len(related) == 1
    assert sum(len(items) for items in splits.values()) == len(rows)
    assert datasets.validate_rows(splits)["no_cross_split_state_or_group_leakage"]


@pytest.mark.parametrize("same_state", [True, False])
def test_presplit_data_rejects_leakage(same_state):
    rows = {split: [example(i, group_id=str(i))] for i, split in enumerate(datasets.SPLITS)}
    if same_state:
        rows["test"][0]["state"] = rows["train"][0]["state"]
    else:
        rows["test"][0]["group_id"] = rows["train"][0]["group_id"]
    with pytest.raises(ValueError, match="Cross-split"):
        datasets.validate_rows(rows)


def test_independent_states_are_required_for_four_splits():
    rows = [example(i, group_id="one-session") for i in range(20)]
    with pytest.raises(ValueError, match="four independent"):
        datasets.split_rows(rows, [0.7, 0.1, 0.1, 0.1], 17)


def test_snapshot_is_immutable_and_audit_detects_tampering(tmp_path):
    original = write_rows(tmp_path / "source.jsonl", [example(i) for i in range(20)])
    copied = datasets.snapshot(str(original), tmp_path / "input")
    datasets.prepare(copied, tmp_path / "data")
    original.write_text("changed outside the run")
    assert len(list(datasets.load(copied))) == 20
    assert datasets.audit(tmp_path / "data")["labels_semantically_verified"] is False
    (tmp_path / "data/test.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.audit(tmp_path / "data")


def test_engine_manifest_preserves_strict_label_audit(tmp_path):
    from finetuning.data import build
    from tests.test_finetuning import rewrite

    source, output = tmp_path / "engine", tmp_path / "prepared"
    build(source, {split: 2 for split in datasets.SPLITS}, pieces=5)
    assert datasets.prepare(source, output)["all_labels_recomputed"]
    assert (source / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    rows[0]["label"] = (rows[0]["label"] + 1) % len(rows[0]["options"])
    rewrite(output, "train", rows)
    with pytest.raises(ValueError, match="strict dominance"):
        datasets.audit(output)


def test_s3_prefix_downloads_only_split_files(tmp_path, monkeypatch):
    import boto3

    calls = []

    class S3:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, **kwargs):
            assert kwargs == {"Bucket": "datasets", "Prefix": "my-data/"}
            return [{"Contents": [{"Key": "my-data/" + name} for name in
                                  ("train.jsonl", "test.csv.gz", "manifest.json",
                                   "credentials.txt", "nested/train.jsonl", "../escape.jsonl")]}]

        def download_file(self, bucket, key, destination):
            calls.append((bucket, key))
            Path(destination).write_text("sample")

    monkeypatch.setattr(boto3, "client", lambda service: S3())
    datasets.snapshot("s3://datasets/my-data/", tmp_path / "snapshot")
    assert {k for _, k in calls} == {
        "my-data/train.jsonl", "my-data/test.csv.gz", "my-data/manifest.json"
    }
    calls.clear()
    file = datasets.snapshot("s3://datasets/private/train.jsonl", tmp_path / "single")
    assert file.name == "train.jsonl"
    assert calls == [("datasets", "private/train.jsonl")]


def test_config_relative_paths_preserve_virtual_environment_symlink(tmp_path):
    env = tmp_path / "venv/bin"
    env.mkdir(parents=True)
    (env / "python").symlink_to(sys.executable)
    path = tmp_path / "config.json"
    pipeline.write(path, {"runtime": {"python": "./venv/bin/python"},
                          "model": {"checkpoint": "./checkpoint"}})
    config = pipeline.configure(path)
    assert config["runtime"]["python"] == str(env / "python")
    assert config["model"]["checkpoint"] == str(tmp_path / "checkpoint")
    pipeline.write(path, {"training": {"learning_rate": 0.1}})
    with pytest.raises(ValueError, match="Unknown settings"):
        pipeline.configure(path)


def test_invalid_training_settings_fail_early():
    config = pipeline.configure()
    config["data"]["location"] = "/tmp/data.jsonl"
    config["model"]["checkpoint"] = "/tmp/model"
    for section, key, value in [
        ("training", "epochs", 0), ("training", "lr", float("nan")),
        ("quality", "max_accuracy_drop", -0.1), ("runtime", "device", "cpu"),
        ("data", "fractions", [0.7, 0.2, 0.2, 0.1]),
    ]:
        bad = copy.deepcopy(config)
        bad[section][key] = value
        with pytest.raises(ValueError):
            pipeline.validate_config(bad)


def test_resume_verifies_completed_stages_and_archives_failed_attempts(tmp_path):
    config = pipeline.configure()
    runner = pipeline.Runner(tmp_path, config, False, False)

    def failing():
        (tmp_path / "output").mkdir()
        (tmp_path / "output/partial").write_text("failed attempt")
        raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        runner.stage("train", ["output"], failing)
    assert json.loads((tmp_path / "pipeline.json").read_text())["status"] == "failed"
    resumed = pipeline.Runner(tmp_path, config, False, True)

    def succeed():
        assert not (tmp_path / "output").exists()
        (tmp_path / "output").mkdir()
        (tmp_path / "output/complete").write_text("final checkpoint")

    resumed.stage("train", ["output"], succeed)
    assert next((tmp_path / "failed").glob("train-*/output/partial")).read_text() == "failed attempt"
    resumed.stage("train", ["output"], lambda: pytest.fail("Completed stage must not rerun"))
    (tmp_path / "output/complete").write_text("corrupt")
    with pytest.raises(ValueError, match="Artifact changed"):
        resumed.stage("train", ["output"], lambda: pytest.fail("Corrupt output must not rerun silently"))


def test_resume_rejects_changed_configuration_or_code(tmp_path, monkeypatch):
    config = pipeline.configure()
    pipeline.Runner(tmp_path, config, False, False)
    changed = copy.deepcopy(config)
    changed["training"]["epochs"] = 2
    with pytest.raises(ValueError, match="unchanged"):
        pipeline.Runner(tmp_path, changed, False, True)
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {"changed.py": "new"})
    with pytest.raises(ValueError, match="unchanged"):
        pipeline.Runner(tmp_path, config, False, True)


def test_command_failure_stops_pipeline_before_training(tmp_path, monkeypatch):
    config = pipeline.configure()
    config["data"]["location"] = str(write_rows(tmp_path / "rows.jsonl", [example(i) for i in range(20)]))
    config["model"]["checkpoint"] = "/unused"
    output = tmp_path / "run"
    output.mkdir()
    runner = pipeline.Runner(output, config, False, False)
    monkeypatch.setattr(pipeline, "setup", lambda _: (sys.executable, tmp_path))
    stages = []

    def fail(name, command):
        stages.append(name)
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(runner, "command", fail)
    with pytest.raises(RuntimeError, match="preflight failed"):
        pipeline.execute(runner, False)
    assert stages == ["preflight"]
    assert not (output / "run-v1").exists()


def test_subprocess_exit_is_reported_with_log(tmp_path):
    runner = pipeline.Runner(tmp_path, pipeline.configure(), False, False)
    with pytest.raises(RuntimeError, match="exited 7; see"):
        runner.command("intentional-failure", [sys.executable, "-c", "print('diagnostic'); exit(7)"])
    assert "diagnostic" in (tmp_path / "logs/intentional-failure.log").read_text()


def reports(folder, before=0.8, after=0.7):
    def save(path, value):
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        pipeline.write(target, value)

    save("run-v1/run.json", {"run_kind": "sft", "completed_steps": 3, "train_rows": 20})
    save("verification/report.json", {"checkpoint_reloaded": True, "predictions_match": True})
    for name, accuracy in (("baseline", before), ("adapted", after)):
        save(f"eval-v1/{name}/report.json", {
            "data_manifest": {"label_source": "user_supplied"},
            "evaluation_kind": "full_test",
            "heldout_choices": {
                "canonical": {"accuracy": accuracy, "rows": 10},
                "reversed_and_renamed": {"accuracy": accuracy, "rows": 10},
                "same_actual_choice_fraction": 0.9,
            }, "calibration": {"split": "calibration"},
        })


def test_accuracy_regression_fails_quality_but_smoke_cannot_claim_quality(tmp_path):
    reports(tmp_path)
    config = pipeline.configure()
    failed = pipeline.quality_report(tmp_path, config, False)
    assert failed["pipeline_verified"] is True
    assert failed["quality_passed"] is False
    assert failed["quality_gates"]["no_excess_accuracy_regression"] is False
    assert pipeline.quality_report(tmp_path, config, True)["quality_passed"] is None
    config["quality"]["max_accuracy_drop"] = 0.1
    assert pipeline.quality_report(tmp_path, config, False)["quality_passed"] is True
    config["quality"]["min_accuracy"] = 0.95
    assert pipeline.quality_report(tmp_path, config, False)["quality_passed"] is False


def test_full_evaluation_rejects_smoke_or_failed_reload(tmp_path):
    reports(tmp_path)
    path = tmp_path / "eval-v1/adapted/report.json"
    report = json.loads(path.read_text())
    report["evaluation_kind"] = "smoke_subset"
    pipeline.write(path, report)
    with pytest.raises(ValueError, match="smoke subset"):
        pipeline.quality_report(tmp_path, pipeline.configure(), False)
    report["evaluation_kind"] = "full_test"
    pipeline.write(path, report)
    pipeline.write(tmp_path / "verification/report.json",
                   {"checkpoint_reloaded": False, "predictions_match": False})
    with pytest.raises(ValueError, match="reload verification"):
        pipeline.quality_report(tmp_path, pipeline.configure(), False)


def test_game_gate_requires_both_policies_matching_seeds_limits_and_replay(tmp_path):
    reports(tmp_path)
    config = pipeline.configure()
    config["evaluation"]["games"] = True
    with pytest.raises(ValueError, match="configured game policies/seeds"):
        pipeline.quality_report(tmp_path, config, False)
    records = [{
        "policy": policy, "seed": seed, "pieces": 50, "max_pieces": 50, "lines": 10,
        "board_replay_matches": True, "raw_choice_matches_applied_placement": True,
    } for policy in ("original", "model") for seed in config["evaluation"]["seeds"]]
    for name in ("baseline", "adapted"):
        path = tmp_path / f"eval-v1/{name}/report.json"
        report = json.loads(path.read_text())
        report["games"] = records
        pipeline.write(path, report)
    assert pipeline.quality_report(tmp_path, config, False)["games"]["adapted"]["mean_lines"] == 10
    path = tmp_path / "eval-v1/adapted/report.json"
    records[0]["max_pieces"] = 100
    pipeline.write(path, {**report, "games": records})
    with pytest.raises(ValueError, match="piece limit"):
        pipeline.quality_report(tmp_path, config, False)
    records[0]["board_replay_matches"] = False
    pipeline.write(path, {**report, "games": records})
    with pytest.raises(ValueError, match="replay failed"):
        pipeline.quality_report(tmp_path, config, False)


def test_changed_external_base_is_rejected(tmp_path):
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"original")
    (tmp_path / "preflight").mkdir()
    pipeline.write(tmp_path / "preflight/report.json",
                   {"external_model_sha256": {str(base): datasets.digest(base)}})
    pipeline.verify_external_inputs(tmp_path)
    base.write_bytes(b"changed")
    with pytest.raises(ValueError, match="model changed"):
        pipeline.verify_external_inputs(tmp_path)


def test_artifact_verification_rejects_path_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("data")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="outside"):
        pipeline.verify_hashes(run, {"../outside": datasets.digest(outside)})


def test_validate_only_cli_needs_no_model_and_preserves_snapshot_on_resume(tmp_path):
    source = write_rows(tmp_path / "rows.jsonl", [example(i) for i in range(20)])
    output = tmp_path / "run"
    command = [sys.executable, str(pipeline.ROOT / "finetune.py"),
               "--data", str(source), "--output", str(output), "--validate-only"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    source.write_text("source changed after snapshot")
    result = subprocess.run([*command, "--resume"], check=True, capture_output=True, text=True)
    assert "verified_existing" in result.stdout
    assert json.loads((output / "pipeline.json").read_text())["status"] == "data_validated"
    assert datasets.audit(output / "data")["splits"]["train"]["rows"] == 14
