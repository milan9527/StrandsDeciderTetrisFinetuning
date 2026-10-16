"""Import user-labelled choice data, split related states together, and audit snapshots."""

import csv
import gzip
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

SPLITS = ("train", "validation", "calibration", "test")
EXTENSIONS = (".jsonl", ".jsonl.gz", ".csv", ".csv.gz")
SCHEMA = "decider_choice_v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def state_id(row):
    return hashlib.sha256(canonical(row["state"]).encode()).hexdigest()


def normalize(row):
    if not isinstance(row, dict):
        raise ValueError("Each example must be a JSON object")
    if row.get("kind", "choice") != "choice":
        raise ValueError("This pipeline supports kind=choice only, not chat messages, noul or score")
    state = row.get("state")
    if not isinstance(state, (str, dict, list)) or not state or (isinstance(state, str) and not state.strip()):
        raise ValueError("state must be nonempty text, a JSON object, or a JSON array")
    canonical(state)  # Reject non-finite JSON values as well.
    instructions = row.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be nonempty text")
    options = row.get("options")
    if isinstance(options, dict):
        options = list(map(list, options.items()))
    if not isinstance(options, list) or not 2 <= len(options) <= 24:
        raise ValueError("options must contain 2..24 [id, description] pairs")
    if any(not isinstance(o, (list, tuple)) or len(o) != 2
           or any(not isinstance(s, str) or not s.strip() for s in o) for o in options):
        raise ValueError("Each option needs a nonempty string ID and description")
    ids, descriptions = [o[0] for o in options], [" ".join(o[1].split()) for o in options]
    if any(i != i.strip() or "\n" in i or "\r" in i or "\t" in i for i in ids):
        raise ValueError("Option IDs must be single-line text without surrounding whitespace")
    if len(set(ids)) != len(ids) or len(set(descriptions)) != len(descriptions):
        raise ValueError("Option IDs and descriptions must each be unique")
    label = row.get("label")
    if "answer" in row:
        if row["answer"] not in ids:
            raise ValueError("answer must name an existing option ID")
        index = ids.index(row["answer"])
        if label is not None and (type(label) is not int or label != index):
            raise ValueError("label and answer disagree")
        label = index
    if type(label) is not int or not 0 <= label < len(options):
        raise ValueError("label must be a zero-based integer index into options")
    task = row.get("task", "custom")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be nonempty text")
    if type(row.get("weight", 1)) not in (int, float) or row.get("weight", 1) != 1:
        raise ValueError("This pipeline requires weight=1 for comparable evaluation metrics")
    variants = row.get("instruction_variants", [])
    if not isinstance(variants, list) or any(not isinstance(v, str) or not v.strip() for v in variants):
        raise ValueError("instruction_variants must be a list of nonempty strings")
    group = row.get("group_id")
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise ValueError("group_id must be nonempty text when supplied")
    result = {"kind": "choice", "state": state, "instructions": instructions,
              "options": [list(o) for o in options], "label": label, "task": task,
              "weight": 1.0, "instruction_variants": variants}
    if group is not None:
        result["group_id"] = group
    return result


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        if ".csv" in Path(path).suffixes:
            for number, raw in enumerate(csv.DictReader(stream), 2):
                row = {k: v for k, v in raw.items() if v not in ("", None)}
                try:
                    for key in ("options", "label", "weight", "instruction_variants"):
                        if key in row:
                            row[key] = json.loads(row[key])
                    yield normalize(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{Path(path).name}:{number}: {exc}") from exc
        else:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield normalize(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{Path(path).name}:{number}: {exc}") from exc


def snapshot(location, destination):
    """Download/copy only the requested dataset, never arbitrary sibling S3 objects."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if location.startswith("s3://"):
        import boto3

        uri = urlparse(location)
        if not uri.netloc or not uri.path.strip("/") or uri.query or uri.fragment:
            raise ValueError("Use s3://bucket/key.jsonl or s3://bucket/dataset-prefix/")
        s3 = boto3.client("s3")
        key = uri.path.lstrip("/")
        if any(key.endswith(ext) for ext in EXTENSIONS):
            path = destination / Path(key).name
            s3.download_file(uri.netloc, key, str(path))
            return path
        prefix = key.rstrip("/") + "/"
        allowed = {"manifest.json"} | {s + e for s in SPLITS for e in EXTENSIONS}
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=uri.netloc, Prefix=prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"][len(prefix):]
                if name in allowed:
                    s3.download_file(uri.netloc, obj["Key"], str(destination / name))
        return destination
    source = Path(location).expanduser().resolve()
    if source.is_file():
        if not any(source.name.endswith(ext) for ext in EXTENSIONS):
            raise ValueError("Data files must be JSONL, CSV, or their .gz variants")
        shutil.copy2(source, destination / source.name)
        return destination / source.name
    if source.is_dir():
        allowed = {"manifest.json"} | {s + e for s in SPLITS for e in EXTENSIONS}
        for path in source.iterdir():
            if path.name in allowed and path.is_file():
                shutil.copy2(path, destination / path.name)
        return destination
    raise ValueError(f"Data location does not exist: {source}")


def identifiers(row):
    keys = ["state:" + state_id(row)]
    if row.get("group_id"):
        keys.append("group:" + row["group_id"])
    return keys


def validate_rows(splits):
    owners, questions, report = {}, {}, {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"{split}: empty split")
        tasks, counts = Counter(), Counter()
        for raw in rows:
            row = normalize(raw)
            for key in identifiers(row):
                if key in owners and owners[key] != split:
                    raise ValueError(f"Cross-split state/group leakage involving {split}")
                owners[key] = split
            # Ignore option IDs/order and task names when finding duplicate questions.
            key = canonical([state_id(row), row["instructions"].strip(),
                             sorted(" ".join(o[1].split()) for o in row["options"])])
            gold = " ".join(row["options"][row["label"]][1].split())
            if key in questions:
                reason = "Conflicting labels" if questions[key] != gold else "Duplicate question"
                raise ValueError(f"{split}: {reason}; deduplicate the source data")
            questions[key] = gold
            tasks[row["task"]] += 1
            counts[len(row["options"])] += 1
        report[split] = {"rows": len(rows), "tasks": dict(tasks), "option_counts": dict(counts)}
    return {"schema": SCHEMA, "labels_semantically_verified": False,
            "label_source": "user_supplied", "no_cross_split_state_or_group_leakage": True,
            "splits": report}


def split_rows(rows, fractions, seed):
    if len(fractions) != 4 or any(not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Four split fractions must be positive and sum to 1")
    validate_rows({"source": rows})
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in identifiers(row):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row)
    if len(groups) < 4:
        raise ValueError("At least four independent state/group components are required")
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    result = {s: [] for s in SPLITS}
    # Give each split a whole component first; allocate the rest by remaining target size.
    for i, group in enumerate(ordered):
        chosen = i if i < 4 else max(range(4), key=lambda j: fractions[j] * len(rows) - len(result[SPLITS[j]]))
        result[SPLITS[chosen]].extend(group)
    return result


def prepare(source, output, fractions=(0.7, 0.1, 0.1, 0.1), seed=17, mode="auto"):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Prepared dataset directory already exists")
    original = json.loads((source / "manifest.json").read_text()) if source.is_dir() and (source / "manifest.json").exists() else {}
    is_tetris = original.get("label_source") == "engine_strict_metric_dominance"
    if mode == "tetris" and not is_tetris:
        raise ValueError("Tetris mode requires the original engine dataset and its manifest")
    if is_tetris and mode != "choice":
        from finetuning.data import audit as tetris_audit

        tetris_audit(source)
        output.mkdir(parents=True)
        for name in ("manifest.json", *(s + ".jsonl" for s in SPLITS)):
            shutil.copy2(source / name, output / name)
        report = tetris_audit(output)
    else:
        if source.is_file():
            splits = split_rows(list(load(source)), fractions, seed)
        else:
            splits = {}
            for split in SPLITS:
                matches = [source / (split + e) for e in EXTENSIONS if (source / (split + e)).is_file()]
                if len(matches) != 1:
                    raise ValueError(f"Directory needs exactly one {split}.jsonl/.csv file (optionally gzip)")
                splits[split] = list(load(matches[0]))
        report = validate_rows(splits)
        output.mkdir(parents=True)
        manifest = {"schema": SCHEMA, "label_source": "user_supplied",
                    "split_seed": seed, "requested_fractions": list(fractions),
                    "split_strategy": ("connected_components_of_state_and_optional_group_id"
                                       if source.is_file() else "provided_splits_checked_for_leakage"),
                    "splits": {}}
        for split, rows in splits.items():
            path = output / (split + ".jsonl")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            manifest["splits"][split] = {"rows": len(rows), "sha256": digest(path)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        if manifest.get("label_source") != "engine_strict_metric_dominance":
            raise ValueError("Unknown dataset manifest schema")
        from finetuning.data import audit as tetris_audit

        return tetris_audit(folder)
    splits = {}
    for split in SPLITS:
        path = folder / (split + ".jsonl")
        if digest(path) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split}: checksum mismatch")
        splits[split] = list(load(path))
        if len(splits[split]) != manifest["splits"][split]["rows"]:
            raise ValueError(f"{split}: row count mismatch")
    return validate_rows(splits)
