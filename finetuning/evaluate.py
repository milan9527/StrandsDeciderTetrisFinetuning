"""Independent held-out choice tests, order probes, calibration, and full-game replay."""

import argparse
import asyncio
import gzip
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from backend.session import Session
from finetuning.datasets import audit
from finetuning.runtime import (
    batches,
    check_lengths,
    collator,
    permute_row,
    predict,
    read_rows,
    use_upstream,
)
from scripts.run_live import verify_export


def choice_report(model, rows, device, batch_size, predictions_path=None):
    result = predict(model, rows, device, batch_size)
    reversed_rows = [permute_row(r, list(reversed(range(len(r["options"])))), rename=True)
                     for r in rows]
    swapped = predict(model, reversed_rows, device, batch_size)
    by_task = defaultdict(lambda: {"correct": 0, "rows": 0})
    same = 0
    records = []
    for number, (row, first, second) in enumerate(zip(
        rows, result["predictions"], swapped["predictions"], strict=True
    )):
        stats = by_task[row["task"]]
        stats["rows"] += 1
        stats["correct"] += first == row["label"]
        same += first == len(row["options"]) - 1 - second
        records.append({
            "row": number, "task": row["task"], "gold_index": row["label"],
            "predicted_index": first,
            "reversed_predicted_original_index": len(row["options"]) - 1 - second,
            "correct": first == row["label"],
        })
    if predictions_path is not None:
        with gzip.open(predictions_path, "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    result.pop("predictions")
    swapped.pop("predictions")
    return {"canonical": result, "reversed_and_renamed": swapped,
            "same_actual_choice_fraction": same / len(rows), "by_task": dict(by_task)}


def calibrate(model, rows, device, batch_size, scope="engine_dominance_questions_only_not_game_win_probability"):
    """Fit choice temperature by NLL on the dedicated calibration split only."""
    import torch
    from training.data.format import Example

    render = collator(model)
    logits, labels = [], []
    model.eval()
    with torch.inference_mode():
        for group in batches(rows, batch_size):
            batch = {k: v.to(device) for k, v in render(
                [Example.from_dict(row) for row in group]
            ).items()}
            out = model(**batch, temperature=1.0)
            for i, row in enumerate(group):
                logits.append(out["logits"][i, :len(row["options"])].cpu().float())
                labels.append(row["label"])
    temperatures = torch.logspace(-1, 1, 81).tolist()
    losses = [
        sum(float(-torch.log_softmax(v / t, 0)[label])
            for v, label in zip(logits, labels, strict=True)) / len(rows)
        for t in temperatures
    ]
    best = min(range(len(losses)), key=losses.__getitem__)
    model.config.temperature_by_kind = {"choice": temperatures[best]}
    model.config.temperature = 1.0
    return {"method": "choice_temperature_grid_minimum_nll",
            "scope": scope,
            "split": "calibration", "rows": len(rows), "temperature": temperatures[best],
            "calibration_nll": losses[best]}


class LocalClient:
    provider = "decider"
    model_id = "strands-decider-tetris-sft"

    def __init__(self, model, device, model_name):
        from strands_decider.infer import EngineConfig, SystemOneEngine

        self.model_id = model_name
        self.engine = SystemOneEngine(model, EngineConfig(
            device=device, use_prefix_cache=False, max_batch=2,
            model_name=self.model_id, strict_window=True, max_question_fraction=1.0,
        ))

    async def evaluate(self, payload):
        from strands_decider.schema import SystemOneRequest

        started = perf_counter()
        response = self.engine.evaluate(SystemOneRequest.model_validate(payload))
        return {"provider": "decider", "payload": payload,
                "response": response.model_dump(mode="json"),
                "elapsed_ms": round((perf_counter() - started) * 1000, 2)}


async def games(model, device, seeds, pieces, folder, model_name="strands-decider-tetris-sft"):
    client = LocalClient(model, device, model_name)
    summaries = []
    for policy in ("original", "model"):
        for seed in seeds:
            session = Session(client, seed, pieces, interval_ms=0, idle_seconds=86400, policy=policy)
            session.start()
            await session.task
            exported = {"snapshot": session.snapshot(), "history": session.history}
            if session.status != "finished":
                raise RuntimeError(f"Game failed: {session.error}")
            summary = verify_export(exported)
            name = f"{policy}-{seed}.json.gz"
            with gzip.open(folder / name, "wt") as stream:
                json.dump(exported, stream)
            summaries.append({**summary, "receipt": name})
            print(json.dumps({"policy": policy, "seed": seed, "pieces": summary["pieces"],
                              "lines": summary["lines"],
                              "dominated": summary["stats"]["dominated_moves"]}), flush=True)
    return summaries


def run(args):
    import torch

    use_upstream(args.upstream)
    from strands_decider.modeling import StrandsDeciderModel

    audit(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output must be absent or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = StrandsDeciderModel.load(str(args.checkpoint))
    model.to(args.device)
    if args.device == "cpu":
        model.float()
    test = read_rows(args.data / "test.jsonl")
    if getattr(args, "limit", 0):
        test = test[:args.limit]
    check_lengths(model, test)
    report = {"checkpoint": str(args.checkpoint),
              "data_manifest": json.loads((args.data / "manifest.json").read_text()),
              "evaluation_kind": "smoke_subset" if getattr(args, "limit", 0) else "full_test"}
    # This evaluates a frozen checkpoint; it does not select one against the test split.
    if args.calibrate:
        calibration_rows = read_rows(args.data / "calibration.jsonl")
        if getattr(args, "limit", 0):
            calibration_rows = calibration_rows[:args.limit]
        check_lengths(model, calibration_rows)
        scope = ("engine_dominance_questions_only_not_game_win_probability"
                 if report["data_manifest"]["label_source"] == "engine_strict_metric_dominance"
                 else "user_labelled_choice_dataset_only")
        report["calibration"] = calibrate(model, calibration_rows, args.device, args.batch_size, scope)
        # Never mutate the trained/baseline checkpoint during evaluation.
        model.save_pretrained(str(args.output / "calibrated"))
    report["heldout_choices"] = choice_report(
        model, test, args.device, args.batch_size, args.output / "predictions.jsonl.gz"
    )
    probe_rows = test[:min(16, len(test))]
    report["reload_probe"] = predict(model, probe_rows, args.device, args.batch_size)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.skip_games:
        manifest = report["data_manifest"]
        if manifest.get("label_source") != "engine_strict_metric_dominance":
            raise ValueError("Full Tetris games require an engine-audited Tetris dataset")
        used = {seed for split in manifest["splits"].values() for seed in split["seeds"]}
        if any(seed in used for seed in args.seeds):
            raise ValueError("Full-game evaluation seeds must be outside all dataset splits")
        report["games"] = asyncio.run(games(
            model, args.device, args.seeds, args.pieces, args.output, args.model_name
        ))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "heldout_choices": report["heldout_choices"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--skip-games", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[50000, 50001, 50002, 50003, 50004])
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--model-name", default="strands-decider-tetris-sft")
    parser.add_argument("--limit", type=int, default=0, help="Smoke evaluation only; never efficacy evidence")
    args = parser.parse_args()
    if min(args.batch_size, args.pieces) < 1 or args.limit < 0:
        parser.error("Counts must be positive")
    run(args)


if __name__ == "__main__":
    main()
