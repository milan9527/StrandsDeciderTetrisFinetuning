"""Utilities shared by training and evaluation; torch stays outside app dependencies."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM_COMMIT = "6039cf8b64c6074e75804000c07ac3d4cce795f7"


def use_upstream(path):
    path = Path(path).resolve()
    if (path / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True):
            raise ValueError("Upstream checkout must be clean")
    else:
        # GPU service images intentionally omit git. A git-archive export carries
        # its pinned commit and every source-file hash, checked before importing.
        manifest = json.loads((path / ".sft-source.json").read_text())
        revision = manifest["commit"]
        for name, expected in manifest["files"].items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Upstream archive file changed: {name}")
        actual = {str(p.relative_to(path)) for p in path.rglob("*.py")}
        expected = {name for name in manifest["files"] if name.endswith(".py")}
        if actual != expected:
            raise ValueError("Upstream archive Python file set changed")
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {revision}")
    sys.path[:0] = [str(path / "src"), str(path)]
    return revision


def read_rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def permute_row(row, order, rename=False):
    if sorted(order) != list(range(len(row["options"]))):
        raise ValueError("Not a permutation")
    result = copy.deepcopy(row)
    result["options"] = [result["options"][i] for i in order]
    result["label"] = order.index(row["label"])
    if "_meta" in result:
        result["_meta"]["candidate_ids"] = [row["_meta"]["candidate_ids"][i] for i in order]
    if rename:
        for i, option in enumerate(result["options"]):
            option[0] = f"renamed_{i}"
    return result


def enable_adaptation(model):
    """Keep loaded weights; enable only existing LoRA and the original pointer head."""
    if model.config.head_type != "pointer" or not model.config.use_lora:
        raise ValueError("This adaptation requires the v19 pointer head and existing LoRA")
    model.requires_grad_(False)
    adapter = []
    for name, parameter in model.torso.named_parameters():
        if "lora_A." in name or "lora_B." in name:
            parameter.requires_grad_(True)
            adapter.append(parameter)
    head = list(model.head.parameters())
    for parameter in head:
        parameter.requires_grad_(True)
    if not adapter or not head:
        raise ValueError("Missing trainable LoRA or head parameters")
    model.config.temperature = 1.0
    model.config.temperature_by_kind = {}
    return adapter, head


def stage_source(source, destination, base_model=None):
    """Copy the adapter/head; only rewrite the copy's base path for offline loading."""
    from strands_decider.modeling import checkpoint_dir, config_path

    source, destination = Path(checkpoint_dir(str(source))), Path(destination)
    if destination.exists():
        raise ValueError("Staged source already exists")
    destination.mkdir(parents=True)
    for name in ("head.safetensors", "slot_head.pt", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt", "provenance.json"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    shutil.copytree(source / "lora", destination / "lora")
    config = json.loads(Path(config_path(str(source))).read_text())
    if base_model:
        config["base_model"] = str(Path(base_model).resolve())
        config["base_revision"] = None
    (destination / "strands_decider_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        "source": str(source.resolve()),
        "copied_weights": {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in destination.rglob("*") if p.suffix in (".safetensors", ".pt")
        },
    }


def weight_digest(parameters):
    import torch

    result = hashlib.sha256()
    for parameter in parameters:
        result.update(parameter.detach().cpu().contiguous().view(-1).view(
            torch.uint8
        ).numpy().tobytes())
    return result.hexdigest()


def collator(model, train=False, seed=0):
    from training.data.collate import CollatorConfig, SystemOneCollator

    return SystemOneCollator(model.tokenizer, CollatorConfig(
        max_length=model.config.max_length, num_slots=model.config.num_slots,
        head_type="pointer", shuffle_options=True, seed=seed,
    ), train=train)


def check_lengths(model, rows):
    """Reject overlength rows instead of training on silently truncated game facts."""
    from strands_decider.prompting import build_prompt
    from training.data.format import Example

    maximum = 0
    for row in rows:
        example = Example.from_dict(row)
        text, _ = build_prompt(example.state, example.to_question())
        size = len(model.tokenizer.encode(text))
        maximum = max(maximum, size)
        if size > model.config.max_length:
            raise ValueError(f"Overlength training/eval row: {size} tokens")
    return maximum


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def predict(model, rows, device, batch_size=4):
    import torch
    from training.data.format import Example

    render = collator(model)
    predictions, losses = [], []
    model.eval()
    temperature = model.config.temperature_by_kind.get("choice", model.config.temperature)
    with torch.inference_mode():
        for group in batches(rows, batch_size):
            data = {k: v.to(device) for k, v in render(
                [Example.from_dict(row) for row in group]
            ).items()}
            result = model(**data, temperature=temperature)
            predictions.extend(result["log_probs"].argmax(-1).cpu().tolist())
            losses.append((float(result["loss"]), len(group)))
    return {
        "accuracy": sum(p == row["label"] for p, row in zip(predictions, rows, strict=True)) / len(rows),
        "mean_nll": sum(loss * n for loss, n in losses) / len(rows),
        "predictions": predictions, "rows": len(rows),
    }
