"""Prepare six Jev-inspired decision scenarios with a separate discovery partition."""

import argparse
import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from finetuning.datasets import canonical, validate_rows

SEED = 20261017
REFERENCE = "https://github.com/wuyoscar/jev-skill/blob/7696a76bf3cada4e9518db4a7a3b2202fe76e1e5/README.zh.md"
SOURCES = {
    "ocnli-train.jsonl": "https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/b53efdee17257a5c33993cf6fcf8ffff0497ea0e/data/ocnli/train.50k.json",
    "ocnli-dev.jsonl": "https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/b53efdee17257a5c33993cf6fcf8ffff0497ea0e/data/ocnli/dev.json",
    "banking-train.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data/train.csv",
    "banking-test.csv": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data/test.csv",
    "squad-train.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "squad-dev.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}
BANKING = {
    "card_payment_wrong_exchange_rate": "A card purchase used an incorrect currency exchange rate.",
    "extra_charge_on_statement": "An unexpected small extra charge, such as a card authorization or verification charge.",
    "card_payment_fee_charged": "A fee or surcharge was charged for paying a merchant by card.",
    "card_payment_not_recognised": "A card purchase or merchant transaction is not recognized by the customer.",
    "pending_card_payment": "A card purchase remains pending or has not settled.",
    "request_refund": "The customer wants to request a refund for a card purchase.",
    "Refund_not_showing_up": "A refund was already arranged or issued but has not appeared in the account.",
    "declined_card_payment": "A particular card purchase was declined or rejected.",
    "transaction_charged_twice": "The same transaction was charged more than once.",
    "reverted_card_payment?": "A card purchase was reversed or reverted and its money returned.",
    "card_not_working": "The physical card does not work in general, rather than a particular declined purchase.",
    "balance_not_updated_after_bank_transfer": "An incoming bank transfer has not been reflected in the balance.",
}
NLI = {
    "entailment": "支持：仅根据原文，主张必然成立。",
    "contradiction": "矛盾：原文与主张不能同时成立。",
    "neutral": "未知：原文不足以确定主张成立或不成立。",
}
COMPLETION = {
    "verified": "The claim is verified by all required current-version checks and the artifact receipt.",
    "contradicted": "Current-version evidence explicitly establishes a failed check or a missing artifact.",
    "unknown": "Completion is not established: required current-version evidence is missing, pending, or stale.",
}
COMPLETION_INSTRUCTIONS = (
    "Verify the completion claim against the required checks and artifact. For each check, use the "
    "latest receipt for target_revision; ignore receipts for other revisions. Artifact evidence must "
    "also name target_revision. Any current failed check or explicit artifact absence contradicts "
    "completion. Otherwise, missing or pending evidence means unknown. Verify only if every required "
    "check passed and the current artifact exists. Logs and the agent's assertion are not proof."
)
ROUTING_INSTRUCTIONS = (
    "Choose the cheapest feasible model using only these capability cards. It must be available, "
    "support the required modality, fit input_tokens + output_tokens, meet minimum quality, fit "
    "p95_ms + queue_ms within the latency limit, and fit the budget. Total charge = "
    "(input_tokens * input_price + output_tokens * output_price) / 1000 + setup_charge. "
    "Prices are charge units per 1000 tokens. Break charge ties by lower total latency, then model ID. "
    "Choose defer if no model meets every requirement."
)
BROWSER_INSTRUCTIONS = (
    "Choose one next read-only action. First refresh if observed_revision differs from current_revision. "
    "Otherwise wait if loading is true. Otherwise click the unique visible, enabled view control in "
    "active_panel matching the requested order_id, document_type, and language. Ignore hidden, disabled, "
    "other-panel, wrong-document and payment controls. Choose blocked if no matching control exists."
)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_key(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


def choice(task, state, instructions, options, gold, group, rng, source):
    keys = list(options)
    rng.shuffle(keys)
    ids = rng.sample(range(100, 10000), len(keys))
    return {
        "kind": "choice", "task": task, "state": state, "instructions": instructions,
        "options": [[f"c{number}", options[key]] for number, key in zip(ids, keys, strict=True)],
        "label": keys.index(gold), "group_id": f"{task}:{group}",
        "_source": {**source, "gold": gold},
    }


def completion_oracle(state):
    statuses = []
    for check in state["required_checks"]:
        rows = [r for r in state["check_receipts"]
                if r["check"] == check and r["revision"] == state["target_revision"]]
        statuses.append(max(rows, key=lambda r: r["sequence"])["status"] if rows else None)
    artifact = state["artifact_receipt"]
    current_artifact = artifact and artifact["revision"] == state["target_revision"]
    if "failed" in statuses or (current_artifact and not artifact["exists"]):
        return "contradicted"
    if any(status != "passed" for status in statuses) or not current_artifact:
        return "unknown"
    return "verified"


def routing_oracle(state):
    request = state["request"]
    feasible = []
    for model in state["models"]:
        cost_times_1000 = (request["input_tokens"] * model["input_price"]
                          + request["output_tokens"] * model["output_price"]
                          + 1000 * model["setup_charge"])
        latency = model["p95_ms"] + model["queue_ms"]
        if (model["available"] and request["modality"] in model["modalities"]
                and model["max_tokens"] >= request["input_tokens"] + request["output_tokens"]
                and model["quality"] >= request["minimum_quality"]
                and latency <= request["latency_limit_ms"]
                and cost_times_1000 <= request["budget"] * 1000):
            feasible.append((cost_times_1000, latency, model["id"]))
    return min(feasible)[2] if feasible else "defer"


def browser_oracle(state):
    if state["observed_revision"] != state["current_revision"]:
        return "refresh"
    if state["loading"]:
        return "wait"
    matches = [control["id"] for control in state["controls"]
               if control["visible"] and control["enabled"] and control["action"] == "view"
               and control["panel"] == state["active_panel"]
               and all(control[k] == v for k, v in state["request"].items())]
    if len(matches) > 1:
        raise ValueError("Ambiguous browser label")
    return matches[0] if matches else "blocked"


def synthetic(task, rng, number):
    if task == "completion_receipts":
        revision, old_revision = (f"rev-{rng.getrandbits(32):08x}" for _ in range(2))
        checks = rng.sample(["unit", "integration", "lint", "build", "schema", "replay", "smoke"],
                            rng.randint(3, 6))
        receipts = []
        for check in checks:
            receipts.append({"check": check, "revision": old_revision,
                             "sequence": rng.randrange(1, 40), "status": rng.choice(["passed", "failed"])})
            if rng.random() < 0.8:
                receipts.append({"check": check, "revision": revision,
                                 "sequence": rng.randrange(40, 80), "status": "failed"})
                receipts.append({"check": check, "revision": revision,
                                 "sequence": rng.randrange(80, 120), "status": "passed"})
        target = ("verified", "contradicted", "unknown")[number % 3]
        if target == "verified":
            for check in checks:
                receipts.append({"check": check, "revision": revision,
                                 "sequence": rng.randrange(130, 200), "status": "passed"})
        elif target == "contradicted":
            receipts.append({"check": rng.choice(checks), "revision": revision,
                             "sequence": rng.randrange(130, 200), "status": "failed"})
        else:
            missing = rng.choice(checks)
            receipts = [r for r in receipts if not (r["check"] == missing and r["revision"] == revision)]
            if rng.random() < 0.5:
                receipts.append({"check": missing, "revision": revision,
                                 "sequence": rng.randrange(130, 200), "status": "running"})
        artifact = {"revision": revision, "exists": True, "name": "metrics.json"}
        if target != "verified" and rng.random() < 0.5:
            for check in checks:
                receipts.append({"check": check, "revision": revision,
                                 "sequence": rng.randrange(210, 300), "status": "passed"})
            artifact = ({"revision": revision, "exists": False, "name": "metrics.json"}
                        if target == "contradicted" else
                        {"revision": old_revision, "exists": True, "name": "metrics.json"})
        rng.shuffle(receipts)
        state = {
            "goal": "Run all required checks and produce the evaluation artifact for target_revision.",
            "target_revision": revision, "required_checks": checks,
            "agent_claim": "All required checks succeeded; the artifact is ready.",
            "check_receipts": receipts,
            "artifact_receipt": artifact,
        }
        return state, COMPLETION_INSTRUCTIONS, COMPLETION, completion_oracle(state)
    if task == "model_routing":
        while True:
            request = {"modality": rng.choice(["text", "image"]),
                       "input_tokens": rng.choice([2000, 4000, 8000, 12000]),
                       "output_tokens": rng.choice([500, 1000, 2000]),
                       "minimum_quality": rng.choice([70, 80, 90]),
                       "latency_limit_ms": rng.choice([800, 1200, 2000]),
                       "budget": rng.choice([300, 600, 1000])}
            models = [{"id": f"m{code}", "available": rng.random() < 0.85,
                       "modalities": rng.choice([["text"], ["text", "image"]]),
                       "max_tokens": rng.choice([4096, 8192, 16384, 32768]),
                       "quality": rng.randrange(65, 99), "p95_ms": rng.randrange(200, 1400, 50),
                       "queue_ms": rng.randrange(0, 600, 50), "input_price": rng.randrange(5, 80, 5),
                       "output_price": rng.randrange(10, 160, 10), "setup_charge": rng.randrange(0, 200, 25)}
                      for code in rng.sample(range(100, 1000), 6)]
            state = {"request": request, "models": models}
            gold = routing_oracle(state)
            if (gold == "defer") == (number % 5 == 0):
                break
        options = {m["id"]: f"Route to model {m['id']} under its supplied capability and price card."
                   for m in models}
        options["defer"] = "Defer: none of the listed models meets all requirements."
        return state, ROUTING_INSTRUCTIONS, options, gold
    revision = rng.randrange(100, 10000)
    request = {"order_id": f"order-{rng.randrange(1000, 10000)}",
               "document_type": rng.choice(["invoice", "receipt", "shipping_label"]),
               "language": rng.choice(["en", "fr", "de"])}
    panel = rng.choice(["orders", "history", "preview"])
    controls = []
    for code in rng.sample(range(100, 1000), 8):
        controls.append({
            "id": f"e{code}", "order_id": rng.choice([request["order_id"], f"order-{rng.randrange(1000, 10000)}"]),
            "document_type": rng.choice(["invoice", "receipt", "shipping_label"]),
            "language": rng.choice(["en", "fr", "de"]), "visible": rng.random() < 0.8,
            "enabled": rng.random() < 0.8, "panel": rng.choice(["orders", "history", "preview"]),
            "action": rng.choice(["view", "view", "pay"]),
        })
    # Construct at most one matching control; the rest remain realistic distractors.
    for control in controls:
        if all(control[k] == v for k, v in request.items()):
            control["order_id"] += "-other"
    mode = number % 10
    if mode < 6:
        controls[0].update(**request, visible=True, enabled=True, panel=panel, action="view")
    rng.shuffle(controls)
    state = {"request": request, "active_panel": panel, "current_revision": revision,
             "observed_revision": revision - (mode in (6, 7)), "loading": mode == 8,
             "controls": controls}
    options = {c["id"]: f"Click observed control {c['id']}." for c in controls}
    options.update(refresh="Refresh the stale page observation.", wait="Wait for the page to finish loading.",
                   blocked="No allowed matching view control is present.")
    return state, BROWSER_INSTRUCTIONS, options, browser_oracle(state)


def balanced(rows, per_class, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row["_source"]["gold"]].append(row)
    if not groups or min(map(len, groups.values())) < per_class:
        raise ValueError(f"Insufficient class counts: {dict((k, len(v)) for k, v in groups.items())}")
    result = []
    for label in sorted(groups):
        rng.shuffle(groups[label])
        result.extend(groups[label][:per_class])
    rng.shuffle(result)
    return result


def public_partitions(train, test, counts, rng):
    """Assign entire evidence/article groups before balancing each partition."""
    owners = {}
    parts = defaultdict(list)
    test_groups = {row["group_id"] for row in test}
    seen = set()
    candidates = test + [row for row in train if row["group_id"] not in test_groups]
    labels = defaultdict(set)
    for row in candidates:
        question = fingerprint([row["state"], sorted(o[1] for o in row["options"])])
        labels[question].add(row["_source"]["gold"])
    for row in candidates:
        question = fingerprint([row["state"], sorted(o[1] for o in row["options"])])
        if question in seen or len(labels[question]) != 1:
            continue
        seen.add(question)
        group = row["group_id"]
        if group not in owners:
            if group in test_groups:
                owners[group] = "test"
            else:
                bucket = int(fingerprint([SEED, group])[:8], 16) % 100
                owners[group] = ("train" if bucket < 65 else "validation" if bucket < 77
                                 else "calibration" if bucket < 89 else "discovery")
        parts[owners[group]].append(row)
    return {split: balanced(parts[split], count, rng) for split, count in counts.items()}


def public_data(raw, rng):
    nli = {}
    for part in ("train", "dev"):
        rows = []
        for line in (raw / f"ocnli-{part}.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["label"] not in NLI:
                continue
            state = {"原文": item["sentence1"], "主张": item["sentence2"]}
            rows.append(choice("claim_support_ocnli", state,
                               "仅根据给出的原文判断主张是得到支持、与原文矛盾，还是信息不足。不要用常识补充原文没有的信息。",
                               NLI, item["label"], fingerprint(item["sentence1"]), rng,
                               {"dataset": "OCNLI", "official_split": part, "id": item["id"]}))
        nli[part] = rows
    yield "claim_support_ocnli", public_partitions(
        nli["train"], nli["dev"],
        {"train": 500, "validation": 50, "calibration": 50, "discovery": 50, "test": 100}, rng)

    banking = {}
    for part in ("train", "test"):
        rows = []
        for number, item in enumerate(csv.DictReader((raw / f"banking-{part}.csv").open()), 2):
            if item["category"] not in BANKING:
                continue
            rows.append(choice(
                "support_routing_banking", item["text"],
                "Route this customer message to the best matching support queue. Use the distinctions "
                "in the queue descriptions; do not infer unstated transactions.",
                BANKING, item["category"], fingerprint(text_key(item["text"])), rng,
                {"dataset": "BANKING77 (12-queue subset)", "official_split": part, "csv_line": number}))
        banking[part] = rows
    # Text duplicates are removed globally. Grouping by normalized text makes the
    # per-label exact split safe without the variance of tiny hash partitions.
    seen = set()
    bank_parts = {s: [] for s in ("train", "validation", "calibration", "discovery", "test")}
    for part in ("test", "train"):
        by_label = defaultdict(list)
        for row in banking[part]:
            if row["group_id"] not in seen:
                seen.add(row["group_id"])
                by_label[row["_source"]["gold"]].append(row)
        for label in BANKING:
            rows = by_label[label]
            rng.shuffle(rows)
            allocation = [("test", 20)] if part == "test" else [
                ("train", 60), ("validation", 10), ("calibration", 10), ("discovery", 10)]
            if len(rows) < sum(n for _, n in allocation):
                raise ValueError(f"Too few independent banking examples for {label}: {len(rows)}")
            offset = 0
            for split, count in allocation:
                bank_parts[split].extend(rows[offset:offset + count])
                offset += count
    for rows in bank_parts.values():
        rng.shuffle(rows)
    yield "support_routing_banking", bank_parts

    squad = {}
    for part in ("train", "dev"):
        rows = []
        for article in json.loads((raw / f"squad-{part}.json").read_text())["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                if not 100 <= len(context) <= 2200:
                    continue
                for item in paragraph["qas"]:
                    if not item["is_impossible"]:
                        if not item["answers"] or not all(
                            context[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                            for a in item["answers"]
                        ):
                            raise ValueError("SQuAD answer span does not match the source context")
                    gold = "missing" if item["is_impossible"] else "present"
                    rows.append(choice(
                        "retrieval_answerability", {"question": item["question"], "retrieved_passage": context},
                        "Does this passage contain enough explicit information to answer the question? "
                        "A related topic is not sufficient. Do not answer from outside knowledge.",
                        {"present": "The passage contains an answer to the specific question.",
                         "missing": "The passage does not contain the requested answer; retrieve more evidence."},
                        gold, fingerprint(article["title"]), rng,
                        {"dataset": "SQuAD 2.0", "official_split": part, "id": item["id"],
                         "article": article["title"]}))
        squad[part] = rows
    yield "retrieval_answerability", public_partitions(
        squad["train"], squad["dev"],
        {"train": 800, "validation": 100, "calibration": 100, "discovery": 80, "test": 100}, rng)


def build(raw, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    raw.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        sources[name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tasks = {}

    def save(task, splits, kind, scenario):
        audit = validate_rows(splits)
        task_dir = output / task
        task_dir.mkdir()
        files = {}
        for split, rows in splits.items():
            path = task_dir / f"{split}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            files[split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "labels": dict(Counter(row["_source"]["gold"] for row in rows))}
        tasks[task] = {"label_source": kind, "reference_scenarios": scenario, "splits": files}
        (task_dir / "discovery-audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    for task, scenario in (("completion_receipts", [3, 4]), ("model_routing", [19]),
                           ("browser_controls", [27])):
        splits = {}
        for index, (split, count) in enumerate(
            [("train", 1600), ("validation", 200), ("calibration", 200), ("test", 200), ("discovery", 160)]
        ):
            generator = random.Random(SEED + index * 1000 + len(task))
            rows = []
            for number in range(count):
                state, instructions, options, gold = synthetic(task, generator, number)
                rows.append(choice(task, state, instructions, options, gold, fingerprint(state), generator,
                                   {"dataset": "generated_policy_cases", "partition_seed": SEED + index * 1000 + len(task),
                                    "case": number, "oracle": task}))
            splits[split] = rows
        save(task, splits, "deterministic_synthetic_policy", scenario)
    for task, splits in public_data(raw, rng):
        save(task, splits, "original_public_dataset_labels", {
            "claim_support_ocnli": [52], "support_routing_banking": [40],
            "retrieval_answerability": [58],
        }[task])
    provenance = {"seed": SEED, "reference": REFERENCE, "sources": sources, "tasks": tasks,
                  "protocol": "Use discovery only to select tasks. Do not evaluate test during screening.",
                  "limitations": [
                      "Synthetic cases test these explicit policies, not live browser or agent performance.",
                      "BANKING77 is a 12-queue subset, not the full 77-intent benchmark.",
                      "OCNLI final test is sampled from public official dev; no-consensus labels are excluded.",
                      "SQuAD uses 100..2200-character passages and article-level partition isolation.",
                      "Public-data pretraining overlap and near-duplicate semantics cannot be excluded.",
                  ]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({task: {s: info["rows"] for s, info in value["splits"].items()}
                      for task, value in tasks.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("artifacts/scenario-screening/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.raw, args.output)


if __name__ == "__main__":
    main()
