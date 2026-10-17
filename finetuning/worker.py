"""One GPU worker per storage directory, with durable queue and process-group cancellation."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from finetuning.jobs import Settings, Store, read_json
from finetuning.pipeline import ROOT, write


def signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def terminate_group(process):
    signal_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        signal_group(process.pid, signal.SIGKILL)
        process.wait()
    # Reap any descendants still holding the GPU or log files.
    signal_group(process.pid, signal.SIGKILL)


class Worker:
    def __init__(self, settings):
        self.settings = settings
        self.store = Store(settings)
        self.stop = threading.Event()

    def heartbeat(self, job_id=None):
        write(self.settings.storage / "worker.json",
              {"pid": os.getpid(), "updated_at": time.time(), "job_id": job_id})

    def recover(self):
        # A previous worker may have died while its separate CLI process kept running.
        for job in self.store.active():
            pid = job["pid"]
            if pid:
                alive = False
                try:
                    process = psutil.Process(pid)
                    alive = process.create_time() == job["process_created"]
                except psutil.NoSuchProcess:
                    pass
                if not alive:
                    # The CLI may have died before its subprocesses. Verify the job marker
                    # before signalling an otherwise potentially reused process group.
                    for process in psutil.process_iter(["pid", "uids"]):
                        try:
                            if process.uids().real == os.getuid() and os.getpgid(process.pid) == pid \
                                    and process.environ().get("DECIDER_HTTP_JOB_ID") == job["id"]:
                                alive = True
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                            continue
                if alive:
                    signal_group(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 25
                    while time.monotonic() < deadline:
                        self.heartbeat(job["id"])
                        try:
                            os.killpg(pid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(0.2)
                    signal_group(pid, signal.SIGKILL)
            self.store.finish(job["id"], "interrupted", error="Worker stopped before recording a result")

    def run_job(self, job):
        job_id = job["id"]
        folder = self.store.folder(job_id)
        process = None
        try:
            # Recheck allowlists/symlinks in case local files changed while queued.
            self.settings.data_location(job["request"]["config"]["data"]["location"])
            write(folder / "config.json", job["request"]["config"])
            command = [
                sys.executable, "-m", "finetuning.worker", "--execute", str(folder),
                "--mode", job["request"]["mode"],
            ]
            env = {k: v for k, v in os.environ.items() if k != "DECIDER_API_TOKEN"}
            env.update(PYTHONUNBUFFERED="1", DECIDER_MANAGED_PROCESS_GROUP="1",
                       DECIDER_HTTP_JOB_ID=job_id)
            with (folder / "worker.log").open("ab") as stream:
                process = subprocess.Popen(
                    command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE, start_new_session=True, env=env,
                )
                # The child waits for this gate: a crash before persisting its PID cannot
                # leave an unrecorded training process running.
                self.store.process(job_id, process.pid, psutil.Process(process.pid).create_time())
                process.stdin.write(b"1")
                process.stdin.close()
                interrupted = False
                while process.poll() is None:
                    self.heartbeat(job_id)
                    current = self.store.get(job_id)
                    if self.stop.is_set() or current["status"] == "cancelling":
                        interrupted = self.stop.is_set()
                        terminate_group(process)
                        break
                    self.stop.wait(0.25)
                code = process.wait()
                signal_group(process.pid, signal.SIGKILL)
            state = read_json(folder / "output/pipeline.json") or {}
            expected = {"train": "complete", "smoke": "smoke_verified", "validate": "data_validated"}
            if interrupted:
                status, error = "interrupted", "Worker shutdown interrupted this job"
            elif code == 0 and state.get("status") == expected[job["request"]["mode"]]:
                status, error = "succeeded", None
            elif code == 2 and state.get("status") == "quality_failed":
                status, error = "quality_failed", None
            else:
                status = "failed"
                error = state.get("error") or f"Pipeline exited {code}; inspect worker.log"
            self.store.finish(job_id, status, code, error)
        except Exception as exc:
            if process is not None:
                terminate_group(process)
            self.store.finish(job_id, "failed", error=f"{type(exc).__name__}: {exc}")

    def run(self, once=False):
        with (self.settings.storage / ".worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another worker already owns this storage/GPU queue") from exc
            try:
                self.recover()
                while not self.stop.is_set():
                    self.heartbeat()
                    job = self.store.claim()
                    if job:
                        self.run_job(job)
                    if once:
                        break
                    if not job:
                        self.stop.wait(0.5)
            finally:
                (self.settings.storage / "worker.json").unlink(missing_ok=True)


def execute(folder, mode):
    if sys.stdin.buffer.read(1) != b"1":
        return 1

    def interrupt(signum, frame):
        raise KeyboardInterrupt("HTTP job terminated")

    signal.signal(signal.SIGTERM, interrupt)
    # Invoke the actual public CLI, keeping all existing pipeline validation/gates.
    sys.argv = [str(ROOT / "finetune.py"), "--config", str(folder / "config.json"),
                "--output", str(folder / "output")]
    if mode != "train":
        sys.argv.append("--validate-only" if mode == "validate" else "--smoke")
    from finetuning.pipeline import main as pipeline_main

    return pipeline_main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["train", "validate", "smoke"], default="train",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        return execute(args.execute, args.mode)
    if not args.config:
        parser.error("--config is required")
    worker = Worker(Settings.load(args.config))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: worker.stop.set())
    worker.run(args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
