"""Tokenizer/hardware preflight and real reload verification for pipeline checkpoints."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

from finetuning.datasets import audit, digest
from finetuning.runtime import check_lengths, predict, read_rows, use_upstream, weight_digest


def environment(device):
    import torch

    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "tokenizers")}
    for name, expected in (("torch", "2.7.1"), ("transformers", "5.17.0"), ("peft", "0.21.0")):
        if packages[name].split("+")[0] != expected:
            raise ValueError(f"Training runtime requires {name}=={expected}, got {packages[name]}")
    if packages["tokenizers"] not in ("0.23.2", "0.23.3"):
        raise ValueError("Expected tokenizers 0.23.2 or 0.23.3")
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA with bf16 support is required")
        if importlib.util.find_spec("fla") is None:
            raise ValueError("Install flash-linear-attention in the training environment")
        free, total = torch.cuda.mem_get_info()
        if free < 18 * 1024**3:
            raise ValueError("At least 18 GiB free GPU memory is required; this script never stops services")
        return {"packages": packages, "gpu": torch.cuda.get_device_name(),
                "free_bytes": free, "total_bytes": total}
    return {"packages": packages, "gpu": None}


def preflight(args):
    from transformers import AutoTokenizer

    use_upstream(args.upstream)
    from strands_decider.modeling import StrandsDeciderConfig, checkpoint_dir, config_path

    audit(args.data)
    env = environment(args.device)
    checkpoint = Path(checkpoint_dir(args.checkpoint))
    config = StrandsDeciderConfig.from_json(config_path(str(checkpoint)))
    if not config.use_lora or config.head_type != "pointer":
        raise ValueError("Continued SFT requires an existing LoRA checkpoint with a pointer head")
    if not (checkpoint / "lora/adapter_model.safetensors").is_file():
        raise ValueError("Missing pretrained LoRA weights")
    if not any((checkpoint / p).is_file() for p in ("head.safetensors", "slot_head.pt")):
        raise ValueError("Missing pretrained pointer head")
    if args.base_model and not args.base_model.is_dir():
        raise ValueError("--base-model must point to the local base-model directory")
    if not args.base_model and str(config.base_model).startswith("/") and not Path(config.base_model).is_dir():
        raise ValueError("Checkpoint's base path is unavailable; configure base_model for this host")
    model = SimpleNamespace(config=config, tokenizer=AutoTokenizer.from_pretrained(str(checkpoint)))
    lengths = {}
    for split in ("train", "validation", "calibration", "test"):
        rows = read_rows(args.data / (split + ".jsonl"))
        lengths[split] = {"rows": len(rows), "max_tokens": check_lengths(model, rows)}
    source_hashes = {str(p.relative_to(checkpoint)): digest(p)
                     for p in checkpoint.rglob("*") if p.suffix in (".safetensors", ".pt")}
    base = args.base_model or Path(config.base_model)
    external_files = list(checkpoint.rglob("*"))
    if base.is_dir():
        external_files += list(base.rglob("*"))
    external_hashes = {str(p.absolute()): digest(p) for p in sorted(external_files)
                       if p.is_file() and p.suffix in (".safetensors", ".pt", ".bin", ".json", ".txt")}
    return {"environment": env, "token_lengths": lengths, "max_length": config.max_length,
            "num_slots": config.num_slots, "source_weight_sha256": source_hashes,
            "checkpoint": str(checkpoint), "external_model_sha256": external_hashes}


def reload_check(args):
    import torch

    use_upstream(args.upstream)
    from strands_decider.modeling import StrandsDeciderModel

    report = json.loads(args.reference.read_text())
    count = report["reload_probe"]["rows"]
    rows = read_rows(args.data / "test.jsonl")[:count]
    torch.set_num_threads(4)
    model = StrandsDeciderModel.load(args.checkpoint)
    model.to(args.device)
    if args.device == "cpu":
        model.float()
    check_lengths(model, rows)
    result = predict(model, rows, args.device, args.batch_size)
    if result["predictions"] != report["reload_probe"]["predictions"]:
        raise ValueError("Reloaded checkpoint changed probe predictions")
    if not math.isfinite(result["mean_nll"]) or not math.isclose(
        result["mean_nll"], report["reload_probe"]["mean_nll"], rel_tol=0.01, abs_tol=1e-4,
    ):
        raise ValueError(
            "Reloaded checkpoint changed probe probabilities/loss: "
            f"expected mean_nll={report['reload_probe']['mean_nll']}, "
            f"actual mean_nll={result['mean_nll']} (rel_tol=0.01, abs_tol=1e-4)"
        )
    return {"checkpoint_reloaded": True, "predictions_match": True, "probe": result,
            "head_sha256": weight_digest(model.head.parameters()),
            "lora_sha256": weight_digest(p for n, p in model.torso.named_parameters() if "lora_" in n)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "reload"])
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "reload" and args.reference is None:
        parser.error("reload requires --reference")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"preflight": preflight, "reload": reload_check}[args.action](args)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
