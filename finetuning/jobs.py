"""Persistent, single-host HTTP jobs; all training still uses finetune.py."""

import copy
import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from finetuning.datasets import EXTENSIONS, SPLITS
from finetuning.pipeline import configure, validate_config

TERMINAL = {"succeeded", "quality_failed", "failed", "interrupted", "cancelled"}
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
OVERRIDES = {
    "data": {"mode", "fractions", "seed"},
    "training": {"epochs", "batch_size", "accumulation", "lr", "head_lr",
                 "eval_every", "seed", "cpu_threads"},
    "evaluation": {"batch_size", "games", "seeds", "pieces"},
    "quality": {"min_accuracy", "max_accuracy_drop", "min_order_consistency"},
}


class Conflict(ValueError):
    pass


class QueueFull(ValueError):
    pass


@dataclass
class Settings:
    storage: Path
    pipeline: dict
    local_roots: tuple[Path, ...] = ()
    s3_prefixes: tuple[str, ...] = ()
    max_pending: int = 100

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = json.loads(path.read_text())
        allowed = {"storage_dir", "pipeline_config", "local_data_roots",
                   "s3_data_prefixes", "max_pending_jobs"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("Unknown HTTP service configuration")

        def local(value):
            return (path.parent / Path(value).expanduser()).resolve()

        roots = values.get("local_data_roots", [])
        prefixes = values.get("s3_data_prefixes", [])
        if not isinstance(roots, list) or not isinstance(prefixes, list):
            raise ValueError("Data allowlists must be lists")
        for prefix in prefixes:
            parsed = urlparse(prefix)
            if parsed.scheme != "s3" or not parsed.netloc or not prefix.endswith("/") \
                    or parsed.query or parsed.fragment:
                raise ValueError("S3 allowlist entries must be s3://bucket/prefix/")
        maximum = values.get("max_pending_jobs", 100)
        if type(maximum) is not int or not 1 <= maximum <= 10000:
            raise ValueError("max_pending_jobs must be in [1, 10000]")
        return cls(
            local(values["storage_dir"]), configure(local(values["pipeline_config"])),
            tuple(local(p) for p in roots), tuple(prefixes), maximum,
        )

    def data_location(self, location):
        if not isinstance(location, str) or not location.strip():
            raise ValueError("data must be a nonempty location")
        parsed = urlparse(location)
        if parsed.scheme == "s3":
            if not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
                raise ValueError("Use an S3 object or dataset prefix")
            if not any(location.startswith(prefix) for prefix in self.s3_prefixes):
                raise ValueError("S3 location is outside the configured data prefixes")
            return location
        if parsed.scheme:
            raise ValueError("Only local paths and s3:// data locations are supported")
        path = Path(location)
        if not path.is_absolute():
            raise ValueError("Local data paths must be absolute")
        path = path.resolve()

        def check(candidate):
            if not any(candidate.resolve().is_relative_to(root) for root in self.local_roots):
                raise ValueError("Local data is outside the configured data roots")

        check(path)
        if path.is_file():
            if not path.name.endswith(EXTENSIONS):
                raise ValueError("Data files must be JSONL, CSV, or gzip")
        elif path.is_dir():
            allowed = {"manifest.json"} | {s + e for s in SPLITS for e in EXTENSIONS}
            for child in path.iterdir():
                if child.name in allowed:
                    check(child)
        else:
            raise ValueError("Local data does not exist")
        return str(path)

    def resolve_request(self, data, mode, overrides):
        if mode not in ("train", "validate", "smoke"):
            raise ValueError("mode must be train, validate, or smoke")
        config = copy.deepcopy(self.pipeline)
        if not isinstance(overrides, dict) or set(overrides) - set(OVERRIDES):
            raise ValueError("Only data, training, evaluation, and quality may be overridden")
        for section, values in overrides.items():
            if not isinstance(values, dict) or set(values) - OVERRIDES[section]:
                raise ValueError(f"Unsupported overrides in {section}")
            config[section].update(values)
        config["data"]["location"] = self.data_location(data)
        validate_config(config, smoke=mode == "smoke", validate_only=mode == "validate")
        return {"mode": mode, "config": config}


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.root = settings.storage
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, request TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, idempotency_key TEXT UNIQUE,
                    created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                    exit_code INTEGER, error TEXT, pid INTEGER, process_created REAL
                )
            """)
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def folder(self, job_id):
        if not JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self.root / "jobs" / job_id

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError("Job not found")
        result = dict(row)
        result["request"] = json.loads(result["request"])
        return result

    def get(self, job_id):
        self.folder(job_id)
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit=50, offset=0):
        with self.connection() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id LIMIT ? OFFSET ?", (limit, offset)
            )]

    def submit(self, request, key=None):
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if key:
                prior = db.execute("SELECT * FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
                if prior:
                    if prior["fingerprint"] != fingerprint:
                        raise Conflict("Idempotency-Key was already used for a different request")
                    return self.decode(prior)
            pending = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running','cancelling')"
            ).fetchone()[0]
            if pending >= self.settings.max_pending:
                raise QueueFull("Job queue is full")
            job_id = uuid.uuid4().hex
            folder = self.folder(job_id)
            folder.mkdir()
            try:
                db.execute(
                    "INSERT INTO jobs(id,status,request,fingerprint,idempotency_key,created_at) "
                    "VALUES (?, 'queued', ?, ?, ?, ?)",
                    (job_id, payload, fingerprint, key, time.time()),
                )
            except BaseException:
                shutil.rmtree(folder)
                raise
        return self.get(job_id)

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?",
                       (time.time(), row["id"]))
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def active(self):
        with self.connection() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM jobs WHERE status IN ('running','cancelling')"
            )]

    def process(self, job_id, pid, created):
        with self.connection() as db:
            db.execute("UPDATE jobs SET pid=?, process_created=? WHERE id=?", (pid, created, job_id))

    def finish(self, job_id, status, code=None, error=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            # A concurrent accepted cancellation must not turn into success.
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["status"] == "cancelling":
                status, error = "cancelled", None
            db.execute(
                "UPDATE jobs SET status=?, exit_code=?, error=?, finished_at=? WHERE id=?",
                (status, code, error, time.time(), job_id),
            )

    def cancel(self, job_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            if row["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?",
                           (time.time(), job_id))
            elif row["status"] == "running":
                db.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
        return self.get(job_id)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
