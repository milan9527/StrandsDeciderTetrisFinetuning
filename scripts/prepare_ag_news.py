"""Build a reproducible, balanced AG News choice example from published labels."""

import argparse
import csv
import hashlib
import html
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from finetuning.datasets import validate_rows

REVISION = "03836ce08fce38daac53e9b4255ef16e0b9f6303"
SOURCE = f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{REVISION}/data/ag_news_csv"
CATEGORIES = {
    1: "World: international affairs, politics, diplomacy, and conflicts.",
    2: "Sports: athletes, teams, matches, tournaments, and sporting events.",
    3: "Business: companies, markets, finance, trade, and the economy.",
    4: "Science and Technology: computing, technology, science, and research.",
}
INSTRUCTIONS = "Assign this news article to the single most appropriate news category."


def clean(text):
    return " ".join(html.unescape(text.replace("\\n", " ")).split())


def key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\W+", " ", text).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, seed=20261016):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty or new output directory")
    raw = output / "raw"
    data = output / "choices"
    raw.mkdir(parents=True)
    data.mkdir()
    source_files = {}
    pools = {}
    for split in ("train", "test"):
        path = raw / f"{split}.csv"
        url = f"{SOURCE}/{split}.csv"
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
        rows = []
        with path.open(newline="", encoding="utf-8") as stream:
            for number, (label, title, description) in enumerate(csv.reader(stream), 1):
                title, description = clean(title), clean(description)
                rows.append({
                    "category": int(label), "title": title, "description": description,
                    "source_split": split, "source_row": number,
                })
        pools[split] = rows
        source_files[split] = {"url": url, "sha256": sha(path), "rows": len(rows)}

    # Deduplicate the complete official test pool first, then exclude its titles
    # and article bodies from the training pool, including unselected test rows.
    seen_titles, seen_bodies = set(), set()
    unique, discarded = {}, {}
    for split in ("test", "train"):
        unique[split] = []
        discarded[split] = 0
        for row in pools[split]:
            title_key, body_key = key(row["title"]), key(row["description"])
            if not title_key or not body_key or title_key in seen_titles or body_key in seen_bodies:
                discarded[split] += 1
                continue
            seen_titles.add(title_key)
            seen_bodies.add(body_key)
            unique[split].append(row)

    rng = random.Random(seed)
    selected = {name: [] for name in ("train", "validation", "calibration", "test")}
    for category in CATEGORIES:
        training = [r for r in unique["train"] if r["category"] == category]
        testing = [r for r in unique["test"] if r["category"] == category]
        rng.shuffle(training)
        rng.shuffle(testing)
        if len(training) < 500 or len(testing) < 50:
            raise ValueError("Insufficient independent articles for a balanced split")
        selected["train"].extend(training[:400])
        selected["validation"].extend(training[400:450])
        selected["calibration"].extend(training[450:500])
        selected["test"].extend(testing[:50])

    splits, summary = {}, {}
    for split, articles in selected.items():
        rng.shuffle(articles)
        rows = []
        for article in articles:
            categories = list(CATEGORIES)
            rng.shuffle(categories)
            ids = rng.sample(range(100, 1000), 4)
            title_key = key(article["title"])
            rows.append({
                "kind": "choice",
                "state": {"title": article["title"], "description": article["description"]},
                "instructions": INSTRUCTIONS,
                "options": [[f"c{identifier}", CATEGORIES[category]]
                            for identifier, category in zip(ids, categories, strict=True)],
                "label": categories.index(article["category"]),
                "task": "ag_news",
                "group_id": "ag-news:" + hashlib.sha256(title_key.encode()).hexdigest(),
                "_source": {k: article[k] for k in ("source_split", "source_row", "category")},
            })
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        splits[split] = rows
        summary[split] = {
            "rows": len(rows), "sha256": sha(path),
            "categories": dict(Counter(r["_source"]["category"] for r in rows)),
        }
    audit = validate_rows(splits)
    manifest = {
        "dataset": "AG News", "seed": seed, "source_revision": REVISION,
        "source": source_files, "splits": summary, "discarded_duplicates_or_empty": discarded,
        "labels": "Original published AG News labels; no teacher or generated labels.",
        "selection": "400/50/50 articles per class from official train; 50 per class from official test.",
        "deduplication": "Global normalized-title and normalized-body deduplication, official test first.",
        "limitations": "Small benchmark subset; near-duplicate stories and pretraining overlap are not ruled out.",
        "dataset_page": "http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
        "paper": "https://arxiv.org/abs/1509.01626",
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"output": str(output), "splits": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    build(args.output, args.seed)


if __name__ == "__main__":
    main()
