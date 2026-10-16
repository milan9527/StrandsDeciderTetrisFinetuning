"""Verify the committed game receipts and report file hashes without model inference."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.run_live import verify_export


def verify(folder):
    manifest = json.loads((folder / "FILES.sha256.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Report checksum mismatch: {name}")
    count = moves = 0
    settings = {}
    for model in ("baseline", "adapted"):
        report = json.loads((folder / model / "report.json").read_text())
        settings[model] = [
            (game["seed"], game["policy"], game["max_pieces"]) for game in report["games"]
        ]
        for expected in report["games"]:
            with gzip.open(folder / model / expected["receipt"], "rt") as source:
                actual = verify_export(json.load(source))
            for key in ("seed", "policy", "pieces", "lines", "score", "finish_reason", "stats"):
                if actual[key] != expected[key]:
                    raise ValueError(f"Game report mismatch: {model}/{expected['receipt']}: {key}")
            count += 1
            moves += actual["pieces"]
    if not settings["baseline"] or sorted(settings["baseline"]) != sorted(settings["adapted"]):
        raise ValueError("Model evaluation settings differ or games are missing")
    return {"verified_files": len(manifest), "verified_games": count, "verified_moves": moves}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports", type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/2026-10-16",
    )
    print(json.dumps(verify(parser.parse_args().reports)))


if __name__ == "__main__":
    main()
