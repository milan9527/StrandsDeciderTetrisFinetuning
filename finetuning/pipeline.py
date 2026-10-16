"""One-command continued Decider choice SFT, with immutable inputs and verification."""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from finetuning.datasets import audit, digest, prepare, snapshot
from finetuning.runtime import UPSTREAM_COMMIT, use_upstream

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "model": {"checkpoint": None, "base_model": None},
    "data": {"location": None, "mode": "auto", "fractions": [0.7, 0.1, 0.1, 0.1], "seed": 17},
    "runtime": {"python": None, "upstream": str(ROOT / "artifacts/sft/upstream"),
                "device": "cuda", "auto_setup": True, "timeout_minutes": 180},
    "training": {"epochs": 1, "batch_size": 2, "accumulation": 8, "lr": 2e-5,
                 "head_lr": 1e-4, "eval_every": 200, "seed": 17, "cpu_threads": 4},
    "evaluation": {"batch_size": 2, "games": "auto",
                   "seeds": [50000, 50001, 50002, 50003, 50004], "pieces": 50},
    "quality": {"min_accuracy": 0.0, "max_accuracy_drop": 0.0, "min_order_consistency": 0.0},
}


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def configure(path=None):
    config = copy.deepcopy(DEFAULTS)
    supplied = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown configuration sections")
    base = Path(path).resolve().parent if path else Path.cwd()
    for section, values in supplied.items():
        if not isinstance(values, dict) or set(values) - set(config[section]):
            raise ValueError(f"Unknown settings in {section}")
        config[section].update(values)
    for section, key in (("model", "checkpoint"), ("model", "base_model"), ("runtime", "python"),
                         ("runtime", "upstream"), ("data", "location")):
        value = config[section][key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{section}.{key} must be a nonempty string or null")
    for section, key in (("model", "base_model"), ("runtime", "python"), ("runtime", "upstream")):
        if config[section][key]:
            # Resolving a venv's python symlink discards the virtual environment.
            config[section][key] = os.path.abspath(base / Path(config[section][key]).expanduser())
    checkpoint = config["model"]["checkpoint"]
    if checkpoint and (checkpoint.startswith(("/", ".", "~")) or (base / checkpoint).exists()):
        config["model"]["checkpoint"] = str((base / Path(checkpoint).expanduser()).resolve())
    location = config["data"]["location"]
    if location and not location.startswith("s3://"):
        config["data"]["location"] = str((base / Path(location).expanduser()).resolve())
    return config


def validate_config(config, smoke=False, validate_only=False):
    if not config["runtime"]["upstream"]:
        raise ValueError("runtime.upstream is required")
    if config["runtime"]["device"] not in ("cuda", "cpu"):
        raise ValueError("runtime.device must be cuda or cpu")
    if config["runtime"]["device"] == "cpu" and not (smoke or validate_only):
        raise ValueError("CPU is supported for --smoke and --validate-only; full training requires CUDA")
    if type(config["runtime"]["auto_setup"]) is not bool:
        raise ValueError("runtime.auto_setup must be boolean")
    if config["data"]["mode"] not in ("auto", "choice", "tetris"):
        raise ValueError("data.mode must be auto, choice, or tetris")
    for section, keys in (
        ("runtime", ("timeout_minutes",)),
        ("training", ("epochs", "batch_size", "accumulation", "eval_every", "cpu_threads")),
        ("evaluation", ("batch_size", "pieces")),
    ):
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section in ("data", "training"):
        if type(config[section]["seed"]) is not int or not 0 <= config[section]["seed"] < 2**32:
            raise ValueError(f"{section}.seed must be an integer in [0, 2**32)")
    for key in ("lr", "head_lr"):
        value = config["training"][key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key, value in config["quality"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"quality.{key} must be a number in [0, 1]")
    fractions = config["data"]["fractions"]
    if not isinstance(fractions, list) or len(fractions) != 4 or any(
        type(f) not in (int, float) or not math.isfinite(f) or f <= 0 for f in fractions
    ) or not math.isclose(sum(fractions), 1):
        raise ValueError("data.fractions must be four positive numbers summing to 1")
    evaluation = config["evaluation"]
    if evaluation["games"] != "auto" and type(evaluation["games"]) is not bool:
        raise ValueError("evaluation.games must be auto, true, or false")
    if not isinstance(evaluation["seeds"], list) or not evaluation["seeds"] or any(
        type(s) is not int or not 0 <= s < 2**31 for s in evaluation["seeds"]
    ) or len(set(evaluation["seeds"])) != len(evaluation["seeds"]):
        raise ValueError("evaluation.seeds must be distinct, nonnegative integers")
    if not config["data"]["location"]:
        raise ValueError("Supply --data or data.location")
    if not validate_only and not config["model"]["checkpoint"]:
        raise ValueError("Set model.checkpoint to the original Decider LoRA/pointer checkpoint")


def source_hashes():
    paths = [p for folder in ("finetuning", "backend", "scripts")
             for p in (ROOT / folder).rglob("*.py")]
    paths += [ROOT / "finetune.py", ROOT / "finetuning/requirements.txt"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def hashes(folder, paths):
    return {str(p.relative_to(folder)): digest(p)
            for directory in paths for p in sorted((folder / directory).rglob("*")) if p.is_file()}


def verify_hashes(folder, manifest):
    if not manifest:
        raise ValueError("Missing artifact checksum manifest")
    for name, expected in manifest.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError(f"Artifact changed or is outside the run: {name}")


def verify_external_inputs(folder):
    """Adapters are copied; a local frozen base remains external and must not drift."""
    report = json.loads((folder / "preflight/report.json").read_text())
    for name, expected in report["external_model_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Source/base model changed since preflight: {name}")


class Runner:
    def __init__(self, folder, config, smoke, resume):
        self.folder, self.config, self.smoke = folder, config, smoke
        fingerprint = hashlib.sha256(json.dumps({"config": config, "smoke": smoke}, sort_keys=True).encode()).hexdigest()
        self.path = folder / "pipeline.json"
        if resume:
            self.state = json.loads(self.path.read_text())
            if self.state["fingerprint"] != fingerprint or self.state["code_sha256"] != source_hashes():
                raise ValueError("Resume requires unchanged configuration and pipeline source code")
        else:
            self.state = {"schema": 1, "fingerprint": fingerprint, "code_sha256": source_hashes(),
                          "started_at": datetime.now(UTC).isoformat(),
                          "run_kind": "smoke_only" if smoke else "sft", "stages": {}}
            write(folder / "config.resolved.json", config)
        (folder / "logs").mkdir(exist_ok=True)
        self.state["status"] = "running"
        write(self.path, self.state)

    def stage(self, name, paths, function):
        prior = self.state["stages"].get(name, {})
        if prior.get("status") == "complete":
            verify_hashes(self.folder, prior["sha256"])
            print(json.dumps({"stage": name, "status": "verified_existing"}), flush=True)
            return
        existing = [self.folder / p for p in paths if (self.folder / p).exists()]
        if existing:
            archive = self.folder / "failed" / f"{name}-{time.time_ns()}"
            archive.mkdir(parents=True)
            for path in existing:
                path.rename(archive / path.name)
        self.state.update(status="running", current_stage=name)
        self.state["stages"][name] = {"status": "running", "started_at": time.time()}
        write(self.path, self.state)
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        try:
            function()
            manifest = hashes(self.folder, paths)
            if not manifest:
                raise ValueError(f"{name} produced no artifacts")
            self.state["stages"][name].update(status="complete", sha256=manifest)
            write(self.path, self.state)
        except BaseException as exc:
            self.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.state["stages"][name]["status"] = "failed"
            write(self.path, self.state)
            raise

    def command(self, name, command):
        log = self.folder / "logs" / f"{name}.log"
        started = time.monotonic()
        timeout = self.config["runtime"]["timeout_minutes"] * 60
        with log.open("a") as stream:
            stream.write(json.dumps({"command": list(map(str, command))}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                list(map(str, command)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1",
                                             "TOKENIZERS_PARALLELISM": "false"},
            )
            try:
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError(f"{name} exceeded {timeout} seconds")
                    try:
                        code = process.wait(timeout=min(30, remaining))
                        if code:
                            raise RuntimeError(f"{name} exited {code}; see {log}")
                        break
                    except subprocess.TimeoutExpired:
                        print(json.dumps({"stage": name, "elapsed_seconds": round(time.monotonic() - started),
                                          "log": str(log)}), flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


def setup(runner):
    settings = runner.config["runtime"]
    upstream = Path(settings["upstream"])
    if not upstream.exists():
        if not settings["auto_setup"]:
            raise ValueError("Pinned upstream checkout is missing; enable runtime.auto_setup")
        upstream.parent.mkdir(parents=True, exist_ok=True)
        runner.command("setup-upstream", ["git", "clone", "https://github.com/strands-labs/strands-decider.git", upstream])
        runner.command("setup-upstream", ["git", "-C", upstream, "checkout", UPSTREAM_COMMIT])
    use_upstream(upstream)
    supplied = settings["python"]
    python = Path(supplied) if supplied else ROOT / f"artifacts/training-env-{settings['device']}/bin/python"
    if supplied:
        if not python.is_file():
            raise ValueError(f"Training interpreter does not exist: {python}")
    else:
        if not settings["auto_setup"]:
            raise ValueError("Supply runtime.python or enable runtime.auto_setup")
        if not shutil.which("uv"):
            raise ValueError("Install uv or provide runtime.python with training dependencies")
        if settings["device"] == "cuda" and (platform.system() != "Linux" or platform.machine() != "x86_64"):
            raise ValueError("Automatic CUDA setup targets Linux x86_64; supply a prepared runtime.python")
        env_dir = python.parent.parent
        if not python.exists():
            runner.command("setup-environment", ["uv", "venv", env_dir, "--python", "3.11"])
        index = "cu126" if settings["device"] == "cuda" else "cpu"
        runner.command("setup-environment", ["uv", "pip", "install", "--python", python,
                                             "torch==2.7.1", "--index-url",
                                             f"https://download.pytorch.org/whl/{index}"])
        runner.command("setup-environment", ["uv", "pip", "install", "--python", python,
                                             "-r", ROOT / "finetuning/requirements.txt"])
        if settings["device"] == "cuda":
            runner.command("setup-environment", ["uv", "pip", "install", "--python", python,
                                                 "flash-linear-attention==0.5.2"])
    return str(python), upstream


def quality_report(folder, config, smoke):
    training = json.loads((folder / "run-v1/run.json").read_text())
    reports = {name: json.loads((folder / f"eval-v1/{name}/report.json").read_text())
               for name in ("baseline", "adapted")}
    a, b = reports.values()
    if a["data_manifest"] != b["data_manifest"]:
        raise ValueError("Baseline and adapted datasets differ")
    if not smoke and any(r["evaluation_kind"] != "full_test" for r in reports.values()):
        raise ValueError("A smoke subset cannot pass full evaluation gates")
    before, after = a["heldout_choices"], b["heldout_choices"]
    improvement = after["canonical"]["accuracy"] - before["canonical"]["accuracy"]
    gates = {
        "minimum_accuracy": after["canonical"]["accuracy"] >= config["quality"]["min_accuracy"],
        "no_excess_accuracy_regression": improvement + 1e-12 >= -config["quality"]["max_accuracy_drop"],
        "minimum_order_consistency": after["same_actual_choice_fraction"] >= config["quality"]["min_order_consistency"],
    }
    reload = json.loads((folder / "verification/report.json").read_text())
    if not reload["checkpoint_reloaded"] or not reload["predictions_match"]:
        raise ValueError("Saved checkpoint did not pass reload verification")
    games = {}
    for name, report in reports.items():
        records = report.get("games", [])
        if not all(g["board_replay_matches"] and g["raw_choice_matches_applied_placement"] for g in records):
            raise ValueError("A full-game replay failed")
        if records:
            games[name] = {
                "games": len(records), "mean_pieces": sum(g["pieces"] for g in records) / len(records),
                "mean_lines": sum(g["lines"] for g in records) / len(records),
            }
    game_setting = config["evaluation"]["games"]
    expected_games = (a["data_manifest"]["label_source"] == "engine_strict_metric_dominance"
                      if game_setting == "auto" else game_setting) and not smoke
    if expected_games or any(r.get("games") for r in reports.values()):
        expected = {(policy, seed) for policy in ("original", "model")
                    for seed in config["evaluation"]["seeds"]}
        for report in reports.values():
            records = report.get("games", [])
            actual = {(g["policy"], g["seed"]) for g in records}
            if actual != expected or len(records) != len(expected):
                raise ValueError("Baseline and adapted must cover the configured game policies/seeds")
            if any(g["max_pieces"] != config["evaluation"]["pieces"] for g in records):
                raise ValueError("Game evaluations must use the configured piece limit")
    return {"run_kind": training["run_kind"], "pipeline_verified": True,
            "quality_passed": None if smoke else all(gates.values()),
            "quality_gates": gates, "thresholds": config["quality"],
            "accuracy_delta": improvement, "baseline": before, "adapted": after,
            "training_steps": training["completed_steps"], "training_rows": training["train_rows"],
            "calibration": b["calibration"], "checkpoint_reloaded": True,
            "checkpoint": "eval-v1/adapted/calibrated",
            "label_source": a["data_manifest"]["label_source"], "games": games,
            "scope": "plumbing_only_not_efficacy" if smoke else "held_out_choice_data_and_optional_tetris_games"}


def markdown(report):
    status = "SMOKE ONLY：流程检查，不证明微调效果" if report["run_kind"] == "smoke_only" else (
        "达到配置的质量门槛" if report["quality_passed"] else "训练完成，但未达到配置的质量门槛"
    )
    lines = ["# Decider SFT 验证报告", "", status, "",
             f"训练样本：{report['training_rows']}；更新步数：{report['training_steps']}。",
             f"标签来源：`{report['label_source']}`。通用数据的标签语义正确性需要数据提供者负责。",
             "", "| 指标 | 原版 | 微调后 |", "| --- | ---: | ---: |"]
    for title, key in (("测试准确率", "canonical"), ("反序并改名后的准确率", "reversed_and_renamed")):
        lines.append(f"| {title} | {report['baseline'][key]['accuracy']:.2%} | {report['adapted'][key]['accuracy']:.2%} |")
    lines.append(f"| 实际选项一致率 | {report['baseline']['same_actual_choice_fraction']:.2%} | {report['adapted']['same_actual_choice_fraction']:.2%} |")
    lines += ["", f"测试行数：{report['adapted']['canonical']['rows']}。"
              "这些指标针对本次数据分布，不代表未知任务或长期策略表现。",
              "", "保存的检查点已由另一个进程重新加载，并核对推理结果。",
              f"检查点：`{report['checkpoint']}`。加载时需要同一基座及固定版本的上游代码。",
              "本流程不发布线上模型，也不启动或停止 EC2。", ""]
    return "\n".join(lines)


def execute(runner, validate_only):
    folder, config = runner.folder, runner.config

    def data():
        raw = snapshot(config["data"]["location"], folder / "input")
        prepare(raw, folder / "data", config["data"]["fractions"],
                config["data"]["seed"], config["data"]["mode"])

    runner.stage("data", ["input", "data"], data)
    audit(folder / "data")
    if validate_only:
        runner.state["status"] = "data_validated"
        write(runner.path, runner.state)
        return 0
    manifest = json.loads((folder / "data/manifest.json").read_text())
    tetris = manifest["label_source"] == "engine_strict_metric_dominance"
    game_setting = config["evaluation"]["games"]
    games = (tetris if game_setting == "auto" else game_setting) and not runner.smoke
    if games and not tetris:
        raise ValueError("Tetris game evaluation requires engine-audited data")
    if games:
        used = {s for split in manifest["splits"].values() for s in split["seeds"]}
        if used.intersection(config["evaluation"]["seeds"]):
            raise ValueError("Game evaluation seeds overlap a dataset split")
    python, upstream = setup(runner)
    common = ["--upstream", upstream, "--data", folder / "data", "--device", config["runtime"]["device"]]
    source = ["--checkpoint", config["model"]["checkpoint"]]
    if config["model"]["base_model"]:
        source += ["--base-model", config["model"]["base_model"]]
    runner.stage("preflight", ["preflight"], lambda: runner.command(
        "preflight", [python, "-m", "finetuning.checks", "preflight", *common, *source,
                      "--output", folder / "preflight"],
    ))
    verify_external_inputs(folder)
    training = dict(config["training"])
    if runner.smoke:
        training.update(epochs=1, batch_size=1, accumulation=1)
    train_args = [item for k, v in training.items() for item in ("--" + k.replace("_", "-"), str(v))]
    runner.stage("train", ["run-v1"], lambda: runner.command(
        "train", [python, "-m", "finetuning.train", *common, *source, *train_args,
                  "--output", folder / "run-v1", *(["--smoke"] if runner.smoke else [])],
    ))
    eval_args = ["--batch-size", str(config["evaluation"]["batch_size"])]
    if runner.smoke:
        eval_args += ["--limit", "2"]
    if games:
        eval_args += ["--pieces", str(config["evaluation"]["pieces"]),
                      "--seeds", *map(str, config["evaluation"]["seeds"])]
    else:
        eval_args += ["--skip-games"]
    for name, checkpoint in (("baseline", "source"), ("adapted", "best")):
        runner.stage(name, [f"eval-v1/{name}"], lambda name=name, checkpoint=checkpoint: runner.command(
            name, [python, "-m", "finetuning.evaluate", *common, *eval_args,
                   "--checkpoint", folder / "run-v1" / checkpoint,
                   "--output", folder / "eval-v1" / name,
                   "--model-name", "strands-decider-" + ("baseline" if name == "baseline" else "sft"),
                   *(["--calibrate"] if name == "adapted" else [])],
        ))
    runner.stage("reload", ["verification"], lambda: runner.command(
        "reload", [python, "-m", "finetuning.checks", "reload", *common,
                   "--checkpoint", folder / "eval-v1/adapted/calibrated",
                   "--reference", folder / "eval-v1/adapted/report.json",
                   "--batch-size", str(config["evaluation"]["batch_size"]),
                   "--output", folder / "verification"],
    ))
    report = quality_report(folder, config, runner.smoke)
    write(folder / "report.json", report)
    (folder / "REPORT.md").write_text(markdown(report))
    final_hashes = hashes(folder, ["input", "data", "preflight", "run-v1", "eval-v1", "verification"])
    final_hashes.update({name: digest(folder / name) for name in ("config.resolved.json", "report.json", "REPORT.md")})
    write(folder / "SHA256.json", final_hashes)
    verify_hashes(folder, final_hashes)
    status = "smoke_verified" if runner.smoke else "complete" if report["quality_passed"] else "quality_failed"
    runner.state.update(status=status, finished_at=datetime.now(UTC).isoformat())
    write(runner.path, runner.state)
    print(json.dumps({"status": status, "report": str(folder / "REPORT.md"),
                      "checkpoint": str(folder / report["checkpoint"])}), flush=True)
    return 2 if status == "quality_failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", help="Local JSONL/CSV, split directory, or S3 URI")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", help="Existing training interpreter; never modified")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse verified stages; restart an interrupted stage")
    parser.add_argument("--verify-artifacts", type=Path, help="Check saved file hashes without loading a model")
    args = parser.parse_args()
    if args.verify_artifacts:
        folder = args.verify_artifacts.resolve()
        verify_hashes(folder, json.loads((folder / "SHA256.json").read_text()))
        print(json.dumps({"artifact_checksums_verified": True, "model_inference_rerun": False}))
        return 0
    config = configure(args.config)
    if args.data:
        config["data"]["location"] = args.data if args.data.startswith("s3://") else str(Path(args.data).expanduser().resolve())
    if args.python:
        config["runtime"]["python"] = os.path.abspath(Path(args.python).expanduser())
    if args.device:
        config["runtime"]["device"] = args.device
    validate_config(config, args.smoke, args.validate_only)
    if args.resume and not args.output:
        parser.error("--resume requires the existing --output directory")
    folder = (args.output or ROOT / "artifacts/runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")).resolve()
    for protected in (config["model"]["checkpoint"], config["model"]["base_model"]):
        if protected and Path(protected).is_dir() and folder.is_relative_to(Path(protected).resolve()):
            raise ValueError("Output cannot be inside original model directories")
    if not args.resume and folder.exists() and any(folder.iterdir()):
        raise ValueError("Output already contains files; choose a new directory or --resume")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is already using this output directory") from exc
        runner = Runner(folder, config, args.smoke, args.resume)
        try:
            return execute(runner, args.validate_only)
        except BaseException as exc:
            runner.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write(runner.path, runner.state)
            raise


if __name__ == "__main__":
    sys.exit(main())
