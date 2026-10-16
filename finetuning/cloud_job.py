"""ECS job runner: bounded subprocesses and periodic private S3 checkpoint backups."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import boto3


def main():
    root = Path.cwd()
    bucket, prefix = os.environ["SFT_BUCKET"], os.environ["SFT_PREFIX"]
    s3 = boto3.client("s3", region_name="us-east-1")
    done = threading.Event()
    deadline = time.monotonic() + int(os.getenv("SFT_MAX_SECONDS", "6600"))
    output = root / "artifacts/sft"
    (output / "logs").mkdir(parents=True, exist_ok=True)
    progress = output / "job-status.json"
    state = {"status": "starting", "started_at": time.time(), "stages": []}
    uploaded = {}
    upload_lock = threading.Lock()

    def sync():
        with upload_lock:
            for folder in ("run-v1", "eval-v1", "gpu-smoke", "logs"):
                for path in (output / folder).rglob("*"):
                    if not path.is_file():
                        continue
                    # Avoid partial checkpoint snapshots while save_pretrained writes.
                    before = path.stat()
                    if time.time() - before.st_mtime < 5:
                        continue
                    relative = str(path.relative_to(output))
                    key = (before.st_size, before.st_mtime_ns)
                    if uploaded.get(relative) != key:
                        s3.upload_file(str(path), bucket, f"{prefix}/results/{relative}")
                        if path.stat().st_mtime_ns == before.st_mtime_ns:
                            uploaded[relative] = key
            progress.write_text(json.dumps(state, indent=2) + "\n")
            s3.upload_file(str(progress), bucket, f"{prefix}/results/job-status.json")

    def backup():
        while not done.wait(45):
            try:
                sync()
            except Exception as exc:
                print(json.dumps({"backup_error": type(exc).__name__}), flush=True)

    thread = threading.Thread(target=backup, daemon=True)
    thread.start()
    upstream = root / "upstream"
    shared = ["--upstream", str(upstream), "--data", str(output / "data-v1")]
    training = [sys.executable, "-m", "finetuning.train", *shared,
                "--checkpoint", "/opt/models/decider", "--base-model", "/opt/models/base"]
    stages = [
        ("gpu-smoke", [*training, "--output", str(output / "gpu-smoke"),
                       "--smoke", "--batch-size", "1", "--accumulation", "1"]),
        ("train", [*training, "--output", str(output / "run-v1")]),
        ("baseline", [sys.executable, "-m", "finetuning.evaluate", *shared,
                      "--checkpoint", str(output / "run-v1/source"),
                      "--output", str(output / "eval-v1/baseline"),
                      "--model-name", "strands-decider-v19-baseline"]),
        ("adapted", [sys.executable, "-m", "finetuning.evaluate", *shared,
                     "--checkpoint", str(output / "run-v1/best"),
                     "--output", str(output / "eval-v1/adapted"), "--calibrate"]),
    ]
    try:
        for stage, command in stages:
            state.update(status="running", stage=stage)
            print(json.dumps({"stage": stage, "event": "start"}), flush=True)
            started = time.monotonic()
            with (output / "logs" / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise TimeoutError("SFT job exceeded its bounded runtime") from None
            state["stages"].append({"stage": stage, "exit_code": code,
                                    "seconds": round(time.monotonic() - started, 2)})
            print(json.dumps(state["stages"][-1]), flush=True)
            sync()
            if code:
                raise RuntimeError(f"{stage} exited with {code}; inspect saved logs")
        state["status"] = "complete"
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        state["finished_at"] = time.time()
        done.set()
        thread.join(timeout=60)
        # All subprocesses are now stopped. Flush even the most recently written files.
        time.sleep(6)
        sync()
        hashes = {}
        for folder in ("run-v1", "eval-v1", "gpu-smoke"):
            for path in (output / folder).rglob("*"):
                if path.is_file():
                    hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
        s3.put_object(Bucket=bucket, Key=f"{prefix}/results/SHA256.json",
                      Body=json.dumps(hashes, indent=2).encode())
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
