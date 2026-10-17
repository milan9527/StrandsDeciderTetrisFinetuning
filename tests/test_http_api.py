"""Exercise the public API and real CLI data validation, plus worker lifecycle failures."""

import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psutil")

from fastapi.testclient import TestClient  # noqa: E402

from finetuning import worker as worker_module  # noqa: E402
from finetuning.http_api import create_app  # noqa: E402
from finetuning.jobs import Settings, Store  # noqa: E402
from finetuning.pipeline import DEFAULTS, ROOT  # noqa: E402
from finetuning.worker import Worker  # noqa: E402

TOKEN = "test-secret-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def service(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    source = data / "choices.jsonl"
    source.write_bytes((ROOT / "examples/choices.jsonl").read_bytes())
    config = copy.deepcopy(DEFAULTS)
    config["runtime"]["device"] = "cpu"
    settings = Settings(tmp_path / "storage", config, (data,), ("s3://allowed/datasets/",))
    with TestClient(create_app(settings, TOKEN)) as client:
        yield settings, client, source


def submit(client, source, **kwargs):
    return client.post("/v1/jobs", headers=AUTH,
                       json={"data": str(source), "mode": "validate", **kwargs})


def test_authentication_and_private_schema(service):
    _, client, source = service
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/jobs").status_code == 401
    assert client.get("/openapi.json").status_code == 401
    assert client.post("/v1/jobs", json={"data": str(source)}).status_code == 401
    assert client.get("/v1/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/openapi.json", headers=AUTH).json()["info"]["version"] == "1.0.0"
    assert client.get("/readyz", headers=AUTH).status_code == 503


def test_actual_cli_validation_and_download(service):
    settings, client, source = service
    response = submit(client, source)
    assert response.status_code == 202
    job_id = response.json()["id"]
    url = f"/v1/jobs/{job_id}"
    assert client.get(url + "/logs", headers=AUTH).json()["next_offset"] == 0
    Worker(settings).run(once=True)
    job = client.get(url, headers=AUTH).json()
    assert (job["status"], job["pipeline_status"], job["mode"], job["exit_code"]) == (
        "succeeded", "data_validated", "validate", 0
    )
    assert job["report"] is None  # Data validation is not model training or efficacy.
    files = client.get(url + "/artifacts", headers=AUTH).json()["artifacts"]
    assert "output/data/manifest.json" in {f["path"] for f in files}
    assert not any("/.lock" in f["path"] for f in files)
    manifest = client.get(url + "/artifacts/output/data/manifest.json", headers=AUTH)
    assert manifest.status_code == 200
    assert manifest.json()["schema"] == "decider_choice_v1"
    log = client.get(url + "/logs?limit=10", headers=AUTH).json()
    assert log["next_offset"] == 10
    rest = client.get(url + "/logs?offset=10", headers=AUTH).json()
    assert '"stage": "data"' in log["text"] + rest["text"]
    # A new frontend sees the same durable job records and artifacts.
    with TestClient(create_app(settings, TOKEN)) as restarted:
        assert restarted.get(url, headers=AUTH).json()["status"] == "succeeded"
    assert client.get(url + "/artifacts", headers=AUTH).status_code == 200


def test_invalid_data_fails_without_claiming_training_success(service):
    settings, client, source = service
    source.write_text('{"invalid": true}\n')
    job_id = submit(client, source).json()["id"]
    Worker(settings).run(once=True)
    job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
    assert job["status"] == "failed"
    assert job["exit_code"] != 0
    assert job["pipeline_status"] == "failed"
    assert job["report"] is None


@pytest.mark.parametrize("overrides", [
    {"runtime": {"python": "/tmp/execute-arbitrary-code"}},
    {"model": {"checkpoint": "/private/model"}},
    {"data": {"location": "/etc/passwd"}},
    {"training": {"extra_argument": "--unsafe"}},
    {"training": {"epochs": True}},
    {"quality": {"min_accuracy": 2}},
    {"evaluation": {"games": "anything"}},
])
def test_rejects_execution_overrides_and_invalid_config(service, overrides):
    _, client, source = service
    response = submit(client, source, overrides=overrides)
    assert response.status_code == 422
    assert client.get("/v1/jobs", headers=AUTH).json()["jobs"] == []


def test_data_allowlist_and_symlink_escape(service, tmp_path):
    _, client, source = service
    outside = tmp_path / "secret.jsonl"
    outside.write_text("private")
    link = source.parent / "linked.jsonl"
    link.symlink_to(outside)
    for data in (outside, link, "/etc/passwd", "s3://allowed/datasets-other/rows.jsonl",
                 "s3://another/datasets/rows.jsonl", "https://example.com/data.jsonl"):
        assert submit(client, data).status_code == 422
    directory = source.parent / "split"
    directory.mkdir()
    (directory / "train.jsonl").symlink_to(outside)
    assert submit(client, directory).status_code == 422
    assert submit(client, "s3://allowed/datasets/browser/").status_code == 202


def test_idempotency_queue_limit_and_cancel(service):
    settings, client, source = service
    settings.max_pending = 1
    body = {"data": str(source), "mode": "validate"}
    headers = {**AUTH, "Idempotency-Key": "client-request-1"}
    first = client.post("/v1/jobs", json=body, headers=headers)
    again = client.post("/v1/jobs", json=body, headers=headers)
    assert first.status_code == again.status_code == 202
    assert first.json()["id"] == again.json()["id"]
    changed = {**body, "overrides": {"training": {"epochs": 2}}}
    assert client.post("/v1/jobs", json=changed, headers=headers).status_code == 409
    assert submit(client, source).status_code == 429
    job_id = first.json()["id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH).json()["status"] == "cancelled"
    Worker(settings).run(once=True)
    assert not (Store(settings).folder(job_id) / "worker.log").exists()
    assert submit(client, source).status_code == 202


def test_concurrent_idempotent_submission_is_atomic(service):
    settings, _, source = service
    store = Store(settings)
    request = settings.resolve_request(str(source), "validate", {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: store.submit(request, "same-request"), range(12)))
    assert len({j["id"] for j in jobs}) == 1
    assert len(store.list()) == 1


def test_artifacts_and_logs_cannot_escape_job(service, tmp_path):
    settings, client, source = service
    job_id = submit(client, source).json()["id"]
    folder = Store(settings).folder(job_id)
    secret = tmp_path / "secret"
    secret.write_text("do not return")
    (folder / "leak.txt").symlink_to(secret)
    (folder / ".private").write_text("hidden")
    url = f"/v1/jobs/{job_id}"
    for name in ("leak.txt", ".private", "%2e%2e%2fsecret", "/etc/passwd"):
        assert client.get(url + "/artifacts/" + name, headers=AUTH).status_code == 404
    assert client.get(url + "/logs?name=../secret", headers=AUTH).status_code == 422
    assert client.get("/v1/jobs/not-an-id", headers=AUTH).status_code == 404
    assert client.get(url + "/artifacts", headers=AUTH).json()["artifacts"] == []


def replace_pipeline(monkeypatch, script):
    """Inject a child executable for lifecycle tests, not model accuracy."""
    original = subprocess.Popen

    def launch(command, **kwargs):
        if "--execute" in command:
            folder = command[command.index("--execute") + 1]
            return original([sys.executable, "-c", script, folder], **kwargs)
        return original(command, **kwargs)

    monkeypatch.setattr(worker_module.subprocess, "Popen", launch)


@pytest.mark.parametrize(("code", "pipeline_status", "expected"), [
    (2, "quality_failed", "quality_failed"),
    (2, "failed", "failed"),
    (0, "running", "failed"),
])
def test_worker_distinguishes_quality_failure_and_bad_exit(service, monkeypatch,
                                                         code, pipeline_status, expected):
    settings, client, source = service
    script = (
        "import sys,json,pathlib; sys.stdin.buffer.read(1); "
        "p=pathlib.Path(sys.argv[1])/'output'; p.mkdir(); "
        f"(p/'pipeline.json').write_text(json.dumps({{'status': {pipeline_status!r}}})); "
        f"sys.exit({code})"
    )
    replace_pipeline(monkeypatch, script)
    job_id = submit(client, source).json()["id"]
    Worker(settings).run(once=True)
    job = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
    assert (job["status"], job["exit_code"]) == (expected, code)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail("Timed out waiting for process state")


def test_cancel_running_job_kills_descendants(service, monkeypatch):
    settings, client, source = service
    script = """
import sys,subprocess,pathlib,time
sys.stdin.buffer.read(1)
folder = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c',
    "import pathlib,sys,time; time.sleep(1.5); pathlib.Path(sys.argv[1]).touch()",
    str(folder/'should-not-exist')])
(folder/'child-started').touch()
time.sleep(60)
"""
    replace_pipeline(monkeypatch, script)
    job_id = submit(client, source).json()["id"]
    worker = Worker(settings)
    thread = threading.Thread(target=worker.run, kwargs={"once": True})
    thread.start()
    folder = Store(settings).folder(job_id)
    try:
        wait_for(lambda: (folder / "child-started").exists())
        assert client.get("/readyz", headers=AUTH).status_code == 200
        response = client.post(f"/v1/jobs/{job_id}/cancel", headers=AUTH)
        assert response.json()["status"] == "cancelling"
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert Store(settings).get(job_id)["status"] == "cancelled"
        time.sleep(1.6)
        assert not (folder / "should-not-exist").exists()
    finally:
        worker.stop.set()
        thread.join(timeout=30)


def test_single_worker_lock_and_recovery(service):
    settings, client, source = service
    job_id = submit(client, source).json()["id"]
    store = Store(settings)
    store.claim()  # Simulate a crash before the CLI was launched.
    with (settings.storage / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another worker"):
            Worker(settings).run(once=True)
    Worker(settings).run(once=True)
    assert store.get(job_id)["status"] == "interrupted"
    assert not (store.folder(job_id) / "output").exists()


def test_recovery_terminates_recorded_orphan(service):
    import psutil

    settings, client, source = service
    job_id = submit(client, source).json()["id"]
    store = Store(settings)
    store.claim()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             start_new_session=True)
    store.process(job_id, child.pid, psutil.Process(child.pid).create_time())
    # Reap while recovery waits for the process group to disappear.
    reaper = threading.Thread(target=child.wait)
    reaper.start()
    try:
        Worker(settings).run(once=True)
        reaper.join(timeout=5)
        assert child.returncode == -signal.SIGTERM
        assert store.get(job_id)["status"] == "interrupted"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        reaper.join(timeout=5)


def test_settings_paths_resolve_relative_to_service_config(tmp_path):
    (tmp_path / "pipeline.json").write_text(json.dumps({
        "model": {"checkpoint": "./models/decider", "base_model": "./models/base"},
    }))
    path = tmp_path / "service.json"
    path.write_text(json.dumps({
        "pipeline_config": "pipeline.json", "storage_dir": "storage",
        "local_data_roots": ["data"], "s3_data_prefixes": ["s3://bucket/data/"],
    }))
    settings = Settings.load(path)
    assert settings.storage == tmp_path / "storage"
    assert settings.pipeline["model"]["base_model"] == str(tmp_path / "models/base")
    assert settings.local_roots == (tmp_path / "data",)
