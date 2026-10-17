"""Measure an unchanged Decider on discovery sets; never read final test partitions."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from finetuning.checks import environment
from finetuning.runtime import (
    check_lengths,
    permute_row,
    predict,
    read_rows,
    stage_source,
    use_upstream,
    weight_digest,
)


def label_family(task, label):
    if task == "model_routing":
        return "defer" if label == "defer" else "route_to_model"
    if task == "browser_controls":
        return "click_control" if label.startswith("e") else label
    return label


def interval(correct, count):
    z = 1.96
    fraction = correct / count
    center = (fraction + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / (
        1 + z * z / count
    )
    return [center - radius, center + radius]


def run(args):
    import torch

    from finetuning.datasets import validate_rows

    use_upstream(args.upstream)
    from strands_decider.modeling import StrandsDeciderModel

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Screening output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    env = environment("cuda")
    torch.set_num_threads(4)
    source = stage_source(args.checkpoint, args.output / "source", args.base_model)
    model = StrandsDeciderModel.load(str(args.output / "source"))
    model.to("cuda")
    initial = {"head": weight_digest(model.head.parameters()),
               "lora": weight_digest(p for n, p in model.torso.named_parameters() if "lora_" in n)}
    provenance = json.loads((args.data / "provenance.json").read_text())
    results = {}
    campaign_start = perf_counter()
    for task, info in provenance["tasks"].items():
        path = args.data / task / "discovery.jsonl"
        expected = info["splits"]["discovery"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Discovery input changed: {task}")
        rows = read_rows(path)
        if len(rows) != expected["rows"]:
            raise ValueError("Discovery row count changed")
        validate_rows({"discovery": rows})
        maximum = check_lengths(model, rows)
        print(json.dumps({"task": task, "event": "start", "rows": len(rows)}), flush=True)
        started = perf_counter()
        first = predict(model, rows, "cuda", args.batch_size)
        reversed_rows = [permute_row(row, list(reversed(range(len(row["options"])))), rename=True)
                         for row in rows]
        check_lengths(model, reversed_rows)
        second = predict(model, reversed_rows, "cuda", args.batch_size)
        groups = defaultdict(lambda: {"rows": 0, "correct": 0})
        records, agreement, errors = [], 0, []
        for number, (row, left, right) in enumerate(zip(
            rows, first["predictions"], second["predictions"], strict=True
        )):
            mapped = len(row["options"]) - 1 - right
            family = label_family(task, row["_source"]["gold"])
            correct = left == row["label"]
            groups[family]["rows"] += 1
            groups[family]["correct"] += correct
            agreement += left == mapped
            record = {
                "row": number, "task": task, "state": row["state"], "instructions": row["instructions"],
                "options": row["options"], "gold_index": row["label"], "predicted_index": left,
                "reversed_predicted_original_index": mapped, "correct": correct,
                "gold_description": row["options"][row["label"]][1],
                "predicted_description": row["options"][left][1],
                "source": row["_source"], "group_id": row["group_id"],
            }
            records.append(record)
            if not correct:
                errors.append(record)
        first.pop("predictions")
        second.pop("predictions")
        correct = sum(r["correct"] for r in records)
        result = {
            "partition": "discovery", "rows": len(rows), "max_tokens": maximum,
            "canonical": first, "reversed_and_renamed": second,
            "correct": correct, "accuracy_wilson_95": interval(correct, len(rows)),
            "same_actual_choice_fraction": agreement / len(rows),
            "mean_uniform_random_accuracy": sum(1 / len(row["options"]) for row in rows) / len(rows),
            "macro_accuracy": sum(g["correct"] / g["rows"] for g in groups.values()) / len(groups),
            "by_label_family": dict(groups), "wall_seconds": round(perf_counter() - started, 2),
            "label_source": info["label_source"], "reference_scenarios": info["reference_scenarios"],
            "candidate_for_finetuning": first["accuracy"] < 0.8,
            "order_instability": agreement / len(rows) < 0.9,
        }
        if task in ("claim_support_ocnli", "support_routing_banking", "retrieval_answerability",
                    "completion_receipts"):
            result["majority_label_accuracy"] = max(Counter(
                row["_source"]["gold"] for row in rows
            ).values()) / len(rows)
        folder = args.output / task
        folder.mkdir()
        (folder / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        (folder / "predictions.jsonl").write_text("".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ))
        (folder / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False) + "\n")
        results[task] = result
        print(json.dumps({"task": task, **result}, ensure_ascii=False), flush=True)
    final = {"head": weight_digest(model.head.parameters()),
             "lora": weight_digest(p for n, p in model.torso.named_parameters() if "lora_" in n)}
    if initial != final:
        raise ValueError("Baseline weights changed during screening")
    summary = {
        "run_kind": "baseline_discovery_screening", "environment": env, "source": source,
        "weights_unchanged": True, "weight_sha256": initial, "tasks": results,
        "candidate_threshold": "discovery canonical accuracy < 0.80",
        "selection_uses_final_test": False, "finetuning_performed": False,
        "wall_seconds": round(perf_counter() - campaign_start, 2),
        "reference": provenance["reference"], "limitations": provenance["limitations"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    main()
