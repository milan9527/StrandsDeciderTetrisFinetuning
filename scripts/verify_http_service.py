"""Run an actual HTTP -> queue -> worker -> CLI audit round trip without GPU/model weights."""

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait_until(function, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = function()
            if result:
                return result
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    raise RuntimeError("HTTP verification timed out; inspect the API and worker logs")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def verify(folder):
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    data = folder / "data"
    data.mkdir()
    (data / "choices.jsonl").write_bytes((ROOT / "examples/choices.jsonl").read_bytes())
    (folder / "pipeline.json").write_text(json.dumps({
        "runtime": {"device": "cpu", "auto_setup": False},
        "evaluation": {"games": False},
    }))
    config = folder / "service.json"
    config.write_text(json.dumps({
        "storage_dir": str(folder / "storage"), "pipeline_config": "pipeline.json",
        "local_data_roots": [str(data)], "s3_data_prefixes": [],
    }))
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]
    token = secrets.token_urlsafe(32)
    base_url = f"http://127.0.0.1:{port}"
    environment = {**os.environ, "DECIDER_API_TOKEN": token}
    processes = []
    with ExitStack() as stack:
        api_log = stack.enter_context((folder / "api.log").open("ab"))
        worker_log = stack.enter_context((folder / "worker.log").open("ab"))

        def api():
            process = subprocess.Popen(
                [sys.executable, "-m", "finetuning.http_api", "--config", str(config),
                 "--port", str(port)], cwd=ROOT, env=environment,
                stdout=api_log, stderr=subprocess.STDOUT,
            )
            processes.append(process)
            return process

        client = stack.enter_context(httpx.Client(
            base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=5,
        ))
        try:
            frontend = api()
            wait_until(lambda: client.get("/healthz").status_code == 200)
            unauthorized = httpx.get(base_url + "/v1/jobs", timeout=5).status_code
            assert unauthorized == 401
            request = {"data": str(data / "choices.jsonl"), "mode": "validate"}
            first = client.post("/v1/jobs", json=request, headers={"Idempotency-Key": "verify-1"})
            first.raise_for_status()
            job_id = first.json()["id"]
            repeated = client.post("/v1/jobs", json=request, headers={"Idempotency-Key": "verify-1"})
            assert repeated.json()["id"] == job_id

            # Verify a queued request survives an actual API process restart.
            stop(frontend)
            api()
            wait_until(lambda: client.get("/healthz").status_code == 200)
            url = f"/v1/jobs/{job_id}"
            assert client.get(url).json()["status"] == "queued"
            worker = subprocess.Popen(
                [sys.executable, "-m", "finetuning.worker", "--config", str(config)],
                cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT,
                env={k: v for k, v in os.environ.items() if k != "DECIDER_API_TOKEN"},
            )
            processes.append(worker)
            wait_until(lambda: client.get("/readyz").status_code == 200)
            job = wait_until(lambda: (
                value if (value := client.get(url).json())["status"] not in ("queued", "running")
                else None
            ))
            assert job["status"] == "succeeded" and job["pipeline_status"] == "data_validated"
            assert job["report"] is None
            manifest = client.get(url + "/artifacts/output/data/manifest.json")
            manifest.raise_for_status()
            local = folder / "storage/jobs" / job_id / "output/data/manifest.json"
            assert manifest.content == local.read_bytes()
            log = client.get(url + "/logs").json()
            assert '"stage": "data"' in log["text"]
            result = {
                "verified_at": datetime.now(UTC).isoformat(), "job_id": job_id,
                "transport": "real_loopback_http", "authorization_required": unauthorized == 401,
                "idempotent_submission": True, "api_restart_preserved_queue": True,
                "worker_ready": True, "status": job["status"],
                "pipeline_status": job["pipeline_status"], "mode": "validate",
                "artifact_download_sha256": hashlib.sha256(manifest.content).hexdigest(),
                "logs_downloaded": True, "gpu_training_executed": False,
            }
            (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            for process in reversed(processes):
                stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / (
        "http-verification-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    ))
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
