"""Continue v19 LoRA + pointer-head weights with supervised choice loss."""

import argparse
import importlib.metadata
import importlib.util
import json
import math
import random
from pathlib import Path
from time import perf_counter

from finetuning.data import audit
from finetuning.runtime import (
    batches,
    check_lengths,
    collator,
    enable_adaptation,
    predict,
    read_rows,
    stage_source,
    use_upstream,
    weight_digest,
)


def run(args):
    import torch

    revision = use_upstream(args.upstream)
    from strands_decider.modeling import StrandsDeciderModel
    from training.data.format import Example

    audit(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Run directory must be absent or empty")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; --device cpu --smoke is for plumbing checks")
    if args.device == "cuda":
        if not torch.cuda.is_bf16_supported() or importlib.util.find_spec("fla") is None:
            raise RuntimeError("CUDA training requires bf16 support and flash-linear-attention")
        free, _ = torch.cuda.mem_get_info()
        if free < 18 * 1024**3:
            raise RuntimeError("Training requires at least 18 GiB free GPU memory")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("Full training requires CUDA; CPU is restricted to --smoke")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    source = stage_source(args.checkpoint, args.output / "source", args.base_model)
    model = StrandsDeciderModel.load(str(args.output / "source"))
    if args.device == "cpu":
        model.float()
    model.to(args.device)
    initial_head = weight_digest(model.head.parameters())
    initial_lora = weight_digest(p for n, p in model.torso.named_parameters() if "lora_" in n)
    adapter, head = enable_adaptation(model)
    assert initial_head == weight_digest(head), "The pretrained head was changed during setup"
    assert initial_lora == weight_digest(adapter), "The pretrained LoRA was changed during setup"
    inner = model.torso.base_model.model
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.torso.config.use_cache = False
    train_rows = read_rows(args.data / "train.jsonl")
    validation = read_rows(args.data / "validation.jsonl")
    if args.smoke:
        # Real model/engine-labelled rows, but this run is explicitly NOT an efficacy experiment.
        train_rows = [r for r in train_rows if len(r["options"]) == 2][:args.batch_size]
        validation = [r for r in validation if len(r["options"]) == 2][:2]
    max_tokens = check_lengths(model, train_rows + validation)
    baseline = predict(model, validation, args.device, args.batch_size)
    baseline.pop("predictions")
    print(json.dumps({"event": "baseline_validation", **baseline}), flush=True)
    optim = torch.optim.AdamW([
        {"params": adapter, "lr": args.lr}, {"params": head, "lr": args.head_lr},
    ], weight_decay=0.01)
    render = collator(model, train=True, seed=args.seed)
    steps_per_epoch = math.ceil(len(train_rows) / (args.batch_size * args.accumulation))
    total_steps = 1 if args.smoke else steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: min((step + 1) / warmup, max(
            0.0, (total_steps - step) / max(1, total_steps - warmup)
        ))
    )
    report = {
        "run_kind": "smoke_only" if args.smoke else "sft",
        "upstream_commit": revision, "source": source,
        "data_manifest": json.loads((args.data / "manifest.json").read_text()),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_rows": len(train_rows), "validation_rows": len(validation),
        "max_tokens": max_tokens, "initial_head_sha256": initial_head,
        "initial_lora_sha256": initial_lora,
        "trainable_lora_parameters": sum(p.numel() for p in adapter),
        "trainable_head_parameters": sum(p.numel() for p in head),
        "baseline_validation": baseline, "history": [], "best_checkpoint": None,
        "calibration": "not_yet_fitted",
        "environment": {
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "peft", "tokenizers", "safetensors"
            )},
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    rng, step, best, started = random.Random(args.seed), 0, -1, perf_counter()
    # Test and calibration files are deliberately never read by the optimizer/checkpoint selector.
    for epoch in range(1 if args.smoke else args.epochs):
        ordered = rng.sample(train_rows, len(train_rows))
        model.train()
        for accumulation in batches(ordered, args.batch_size * args.accumulation):
            optim.zero_grad(set_to_none=True)
            running = 0.0
            for micro in batches(accumulation, args.batch_size):
                batch = {k: v.to(args.device) for k, v in render(
                    [Example.from_dict(row) for row in micro]
                ).items()}
                result = model(**batch)
                fraction = len(micro) / len(accumulation)
                loss = result["loss"] * fraction
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                running += float(loss.detach())
            gradient = {}
            for name, params in (("lora", adapter), ("head", head)):
                norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
                if not norms or not all(torch.isfinite(n) for n in norms):
                    raise RuntimeError(f"Missing/non-finite {name} gradients")
                gradient[name] = float(torch.stack(norms).norm())
                if gradient[name] == 0:
                    raise RuntimeError(f"Zero {name} gradient")
            torch.nn.utils.clip_grad_norm_(adapter + head, 1.0)
            optim.step()
            scheduler.step()
            step += 1
            event = {"step": step, "epoch": epoch + 1, "loss": running,
                     "gradient_norms": gradient, "wall_seconds": round(perf_counter() - started, 2)}
            with (args.output / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(event) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps(event), flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                result = predict(model, validation, args.device, args.batch_size)
                result.pop("predictions")
                event = {"step": step, "validation": result}
                report["history"].append(event)
                print(json.dumps(event), flush=True)
                if result["accuracy"] > best:
                    best = result["accuracy"]
                    model.save_pretrained(str(args.output / "best"))
                    report["best_checkpoint"] = str(args.output / "best")
                    report["best_step"] = step
                (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
                model.train()
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    report.update(
        completed_steps=step, final_head_sha256=weight_digest(head),
        final_lora_sha256=weight_digest(adapter),
        wall_seconds=round(perf_counter() - started, 2),
        max_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
    )
    if report["final_head_sha256"] == initial_head or report["final_lora_sha256"] == initial_lora:
        raise RuntimeError("Training failed to update both LoRA and head")
    model.save_pretrained(str(args.output / "last"))
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"completed_steps": step, "run_kind": report["run_kind"],
                      "output": str(args.output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.eval_every, args.cpu_threads) < 1:
        parser.error("Counts must be positive")
    if args.lr <= 0 or args.head_lr <= 0:
        parser.error("Learning rates must be positive")
    run(args)


if __name__ == "__main__":
    main()
