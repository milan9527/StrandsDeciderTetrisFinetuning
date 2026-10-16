"""Run SFT on the explicitly selected existing GPU, then restore its demo service."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from finetuning.runtime import UPSTREAM_COMMIT

REGION = "us-east-1"
INSTANCE = "i-0daca64f56056aa1c"
CLUSTER = "strands-decider-demo"
SERVICE = "strands-decider-demo-gpu-model-ondemand"
CONTROLLER = "strands-decider-demo-gpu-controller"
RULE = "strands-decider-demo-gpu-Schedule-lm9Gw5Oyy0P0"


def client(service):
    return boto3.client(service, region_name=REGION)


def save(folder, state):
    (folder / "run.json").write_text(json.dumps(state, indent=2) + "\n")


def archive_sources(folder):
    """Commit a separate source snapshot; leave the user's working tree untouched."""
    package = folder / "package"
    package.mkdir()
    for name in ("backend", "finetuning", "tests"):
        shutil.copytree(name, package / name, ignore=shutil.ignore_patterns("__pycache__"))
    (package / "scripts").mkdir()
    for name in ("__init__.py", "run_live.py"):
        shutil.copy2(Path("scripts") / name, package / "scripts" / name)
    shutil.copy2("pyproject.toml", package)
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.name=Tetris SFT", "-c", "user.email=sft@localhost",
         "commit", "-qm", "Snapshot reviewed Tetris SFT sources"],
    ):
        subprocess.run(command, cwd=package, check=True)
    project_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=package, text=True
    ).strip()
    upstream = package / "upstream"
    upstream.mkdir()
    upstream_tar = folder / "upstream.tar"
    with upstream_tar.open("wb") as stream:
        subprocess.run(["git", "-C", "artifacts/sft/upstream", "archive", UPSTREAM_COMMIT],
                       stdout=stream, check=True)
    with tarfile.open(upstream_tar) as archive:
        archive.extractall(upstream, filter="data")
    manifest = {"commit": UPSTREAM_COMMIT, "files": {
        str(p.relative_to(upstream)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in upstream.rglob("*") if p.is_file()
    }}
    (upstream / ".sft-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copytree("artifacts/sft/data-v1", package / "artifacts/sft/data-v1")
    bundle = folder / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted(package.iterdir()):
            if path.name != ".git":
                archive.add(path, arcname=path.name)
    return bundle, project_commit


def prepare(folder):
    if (folder / "run.json").exists():
        raise ValueError("A prepared run already exists; use a new state directory")
    folder.mkdir(parents=True, exist_ok=True)
    ecs = client("ecs")
    service = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"][0]
    source = ecs.describe_task_definition(taskDefinition=service["taskDefinition"])["taskDefinition"]
    account = client("sts").get_caller_identity()["Account"]
    job = "tetris-sft-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    bucket = f"strands-decider-tetris-training-{account}"
    state = {
        "job": job, "instance": INSTANCE, "cluster": CLUSTER, "bucket": bucket,
        "prefix": job, "region": REGION, "prepared_at": time.time(),
        "original": {
            "desired_count": service["desiredCount"], "task_definition": service["taskDefinition"],
            "controller_concurrency": client("lambda").get_function_concurrency(
                FunctionName=CONTROLLER
            ).get("ReservedConcurrentExecutions"),
            "schedule_state": client("events").describe_rule(Name=RULE)["State"],
        },
        "status": "preparing",
    }
    save(folder, state)
    bundle, commit = archive_sources(folder)
    state.update(project_commit=commit, upstream_commit=UPSTREAM_COMMIT,
                 bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
    s3 = client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        s3.create_bucket(Bucket=bucket)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    })
    s3.put_bucket_encryption(Bucket=bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}],
    })
    s3.upload_file(str(bundle), bucket, f"{job}/bundle.tar.gz")
    role = f"{job}-task"
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": account}},
    }]}
    arn = client("iam").create_role(RoleName=role, AssumeRolePolicyDocument=json.dumps(trust))["Role"]["Arn"]
    policy = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
        "Resource": f"arn:aws:s3:::{bucket}/{job}/*",
    }]}
    client("iam").put_role_policy(RoleName=role, PolicyName="RunArtifacts", PolicyDocument=json.dumps(policy))
    state["task_role"] = role
    save(folder, state)
    # Reuse the cached CUDA image, weights and cache volume. No SSH, host agent or new EC2.
    bootstrap = (
        "import boto3,os,pathlib,tarfile,hashlib\n"
        "root=pathlib.Path('/var/cache/decider/tetris-sft')/os.environ['SFT_PREFIX']\n"
        "root.mkdir(parents=True,exist_ok=False)\n"
        "archive=root/'bundle.tar.gz'\n"
        "boto3.client('s3',region_name='us-east-1').download_file("
        "os.environ['SFT_BUCKET'],os.environ['SFT_PREFIX']+'/bundle.tar.gz',str(archive))\n"
        "assert hashlib.sha256(archive.read_bytes()).hexdigest()==os.environ['SFT_BUNDLE_SHA256']\n"
        "with tarfile.open(archive) as t:t.extractall(root,filter='data')\n"
        "os.chdir(root)\n"
        "os.execv('/usr/local/bin/python',['python','-m','finetuning.cloud_job'])\n"
    )
    original_container = source["containerDefinitions"][0]
    container = {
        "name": "train", "image": original_container["image"], "cpu": 4096, "memory": 14000,
        "essential": True, "user": "10001", "entryPoint": ["python", "-c"],
        "command": [bootstrap], "stopTimeout": 120,
        "resourceRequirements": [{"type": "GPU", "value": "1"}],
        "mountPoints": original_container["mountPoints"],
        "environment": [
            {"name": key, "value": value} for key, value in {
                "SFT_BUCKET": bucket, "SFT_PREFIX": job, "AWS_DEFAULT_REGION": REGION,
                "SFT_BUNDLE_SHA256": state["bundle_sha256"], "SFT_MAX_SECONDS": "6600",
                "TRITON_CACHE_DIR": f"/var/cache/decider/tetris-sft/{job}-triton",
                "CUDA_CACHE_PATH": f"/var/cache/decider/tetris-sft/{job}-cuda",
                "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
            }.items()
        ],
        "logConfiguration": {
            **original_container["logConfiguration"],
            "options": {**original_container["logConfiguration"]["options"],
                        "awslogs-stream-prefix": "tetris-sft"},
        },
    }
    definition = ecs.register_task_definition(
        family="strands-decider-tetris-sft", networkMode="host", requiresCompatibilities=["EC2"],
        taskRoleArn=arn, executionRoleArn=source["executionRoleArn"],
        volumes=source["volumes"], containerDefinitions=[container],
    )["taskDefinition"]["taskDefinitionArn"]
    state.update(task_definition=definition, status="prepared")
    save(folder, state)
    print(json.dumps({"job": job, "instance": INSTANCE, "status": state["status"]}), flush=True)


def restore(folder, state):
    ecs = client("ecs")
    if state.get("task"):
        task = ecs.describe_tasks(cluster=CLUSTER, tasks=[state["task"]])["tasks"][0]
        if task["lastStatus"] != "STOPPED":
            ecs.stop_task(cluster=CLUSTER, task=state["task"], reason="Tetris SFT bounded run cleanup")
    original = state["original"]
    ecs.update_service(cluster=CLUSTER, service=SERVICE, desiredCount=original["desired_count"])
    if original["controller_concurrency"] is None:
        client("lambda").delete_function_concurrency(FunctionName=CONTROLLER)
    else:
        client("lambda").put_function_concurrency(
            FunctionName=CONTROLLER, ReservedConcurrentExecutions=original["controller_concurrency"]
        )
    if original["schedule_state"] == "ENABLED":
        client("events").enable_rule(Name=RULE)
    state["restoration_requested_at"] = time.time()
    save(folder, state)
    print(json.dumps({"event": "original_service_restoration_requested"}), flush=True)


def run(folder):
    state = json.loads((folder / "run.json").read_text())
    if state["status"] != "prepared":
        raise ValueError("Run must be prepared and not previously started")
    ecs = client("ecs")
    current = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"][0]
    if (current["desiredCount"] != state["original"]["desired_count"]
            or current["taskDefinition"] != state["original"]["task_definition"]):
        raise RuntimeError("Original service changed since preparation")
    state.update(status="starting", started_at=time.time())
    save(folder, state)
    try:
        client("events").disable_rule(Name=RULE)
        client("lambda").put_function_concurrency(
            FunctionName=CONTROLLER, ReservedConcurrentExecutions=0
        )
        # Allow a controller invocation that was already running to finish (timeout 45s).
        time.sleep(46)
        ecs.update_service(cluster=CLUSTER, service=SERVICE, desiredCount=0)
        draining = time.monotonic() + 240
        while True:
            service = ecs.describe_services(cluster=CLUSTER, services=[SERVICE])["services"][0]
            if not service["runningCount"] and not service["pendingCount"]:
                break
            if time.monotonic() > draining:
                raise TimeoutError("Original model service did not drain")
            time.sleep(10)
        response = ecs.run_task(
            cluster=CLUSTER, taskDefinition=state["task_definition"], launchType="EC2", count=1,
            placementConstraints=[{"type": "memberOf", "expression": f"ec2InstanceId == {INSTANCE}"}],
            startedBy=state["job"],
        )
        if response.get("failures"):
            raise RuntimeError(response["failures"])
        state.update(task=response["tasks"][0]["taskArn"], status="running")
        save(folder, state)
        print(json.dumps({"event": "task_started", "task": state["task"]}), flush=True)
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            task = ecs.describe_tasks(cluster=CLUSTER, tasks=[state["task"]])["tasks"][0]
            if task["lastStatus"] == "STOPPED":
                state.update(status="stopped", stopped_reason=task.get("stoppedReason"),
                             containers=[{"name": c["name"], "exit_code": c.get("exitCode"),
                                          "reason": c.get("reason")} for c in task["containers"]])
                save(folder, state)
                print(json.dumps(state["containers"]), flush=True)
                return
            time.sleep(30)
        raise TimeoutError("Training task exceeded 120 minutes")
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        save(folder, state)
        raise
    finally:
        restore(folder, state)


def download(folder):
    state = json.loads((folder / "run.json").read_text())
    s3 = client("s3")
    prefix = state["prefix"] + "/results/"
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=state["bucket"], Prefix=prefix):
        for obj in page.get("Contents", []):
            destination = folder / "results" / obj["Key"][len(prefix):]
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(state["bucket"], obj["Key"], str(destination))
            count += 1
    checksum = folder / "results/SHA256.json"
    if checksum.exists():
        for name, expected in json.loads(checksum.read_text()).items():
            if hashlib.sha256((folder / "results" / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
    print(json.dumps({"downloaded_files": count, "final_manifest_checked": checksum.exists()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "restore", "download"])
    parser.add_argument("--state-dir", type=Path, default=Path("artifacts/sft/aws-existing/job-v1"))
    args = parser.parse_args()
    if args.action == "restore":
        restore(args.state_dir, json.loads((args.state_dir / "run.json").read_text()))
    else:
        {"prepare": prepare, "run": run, "download": download}[args.action](args.state_dir)


if __name__ == "__main__":
    main()
