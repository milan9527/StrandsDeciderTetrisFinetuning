"""Authenticated HTTP frontend. Run the GPU worker separately against the same storage."""

import argparse
import hmac
import os
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from finetuning.jobs import Conflict, QueueFull, Settings, Store, read_json


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    data: str = Field(min_length=1, max_length=4096)
    mode: Literal["train", "validate", "smoke"] = "train"
    overrides: dict = Field(default_factory=dict)


def create_app(settings=None, token=None):
    settings = settings or Settings.load(os.environ["DECIDER_SERVICE_CONFIG"])
    token = token if token is not None else os.environ.get("DECIDER_API_TOKEN", "")
    if len(token) < 32:
        raise ValueError("DECIDER_API_TOKEN must contain at least 32 characters")
    store = Store(settings)
    security = HTTPBearer(auto_error=False)

    def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]):
        if credentials is None or not hmac.compare_digest(
            credentials.credentials.encode(), token.encode()
        ):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Decider Fine-tuning API", version="1.0.0", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.store = store
    authorized = [Depends(authenticate)]

    def get_job(job_id):
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    def view(job):
        folder = store.folder(job["id"])
        pipeline = read_json(folder / "output/pipeline.json")
        report = read_json(folder / "output/report.json")
        return {
            **{k: job[k] for k in ("id", "status", "created_at", "started_at", "finished_at",
                                   "exit_code", "error")},
            "mode": job["request"]["mode"],
            "data": job["request"]["config"]["data"]["location"],
            "pipeline_status": pipeline.get("status") if pipeline else None,
            "current_stage": pipeline.get("current_stage") if pipeline else None,
            "report": report,
            "links": {name: f"/v1/jobs/{job['id']}/{name}" for name in ("logs", "artifacts")},
        }

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/readyz", dependencies=authorized)
    def ready():
        heartbeat = read_json(settings.storage / "worker.json") or {}
        if time.time() - heartbeat.get("updated_at", 0) > 15:
            raise HTTPException(503, "GPU worker is not responding")
        return {"status": "ready", "worker": heartbeat}

    @app.get("/openapi.json", dependencies=authorized, include_in_schema=False)
    def schema():
        return app.openapi()

    @app.post("/v1/jobs", status_code=202, dependencies=authorized)
    def submit(body: Submission,
               idempotency_key: Annotated[str | None, Header(max_length=128)] = None):
        if idempotency_key is not None and (
            not idempotency_key or not idempotency_key.isascii() or not idempotency_key.isprintable()
        ):
            raise HTTPException(422, "Idempotency-Key must be printable ASCII")
        try:
            request = settings.resolve_request(body.data, body.mode, body.overrides)
            return view(store.submit(request, idempotency_key))
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except QueueFull as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "30"}) from exc
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/jobs", dependencies=authorized)
    def list_jobs(limit: Annotated[int, Query(ge=1, le=100)] = 50,
                  offset: Annotated[int, Query(ge=0)] = 0):
        return {"jobs": [view(j) for j in store.list(limit, offset)]}

    @app.get("/v1/jobs/{job_id}", dependencies=authorized)
    def status(job_id: str):
        return view(get_job(job_id))

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202, dependencies=authorized)
    def cancel(job_id: str):
        get_job(job_id)
        return view(store.cancel(job_id))

    def artifact_path(job_id, name):
        get_job(job_id)
        root = store.folder(job_id).resolve()
        relative = Path(name)
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root) \
                or any(part.startswith(".") for part in relative.parts) or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return path

    @app.get("/v1/jobs/{job_id}/logs", dependencies=authorized)
    def logs(job_id: str, name: str = "worker.log",
             offset: Annotated[int, Query(ge=0)] = 0,
             limit: Annotated[int, Query(ge=1, le=1048576)] = 65536):
        get_job(job_id)
        if name != "worker.log" and not (
            name.startswith("output/logs/") and name.endswith(".log")
        ):
            raise HTTPException(422, "Select worker.log or output/logs/<stage>.log")
        # A newly queued job has no log yet; clients can immediately start polling.
        if name == "worker.log" and not (store.folder(job_id) / name).exists():
            return {"text": "", "offset": offset, "next_offset": offset}
        path = artifact_path(job_id, name)
        with path.open("rb") as stream:
            stream.seek(offset)
            content = stream.read(limit)
            next_offset = stream.tell()
        return {"text": content.decode("utf-8", errors="replace"), "offset": offset,
                "next_offset": next_offset}

    @app.get("/v1/jobs/{job_id}/artifacts", dependencies=authorized)
    def artifacts(job_id: str):
        get_job(job_id)
        root = store.folder(job_id).resolve()
        files = []
        for path in sorted(root.rglob("*")):
            name = path.relative_to(root)
            if any(p.startswith(".") for p in name.parts) or path.is_symlink():
                continue
            if path.is_file() and path.resolve().is_relative_to(root):
                files.append({"path": str(name), "bytes": path.stat().st_size})
        return {"artifacts": files}

    @app.get("/v1/jobs/{job_id}/artifacts/{name:path}", dependencies=authorized)
    def download(job_id: str, name: str):
        return FileResponse(artifact_path(job_id, name), filename=Path(name).name,
                            media_type="application/octet-stream")

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(create_app(Settings.load(args.config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
