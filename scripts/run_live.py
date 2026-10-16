"""Run one bounded real-model game and verify its exported decisions and board replay."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from backend.decider import validate_result
from backend.engine import Game
from backend.providers import gpt_request, normalize_gpt


def verify_export(data):
    snapshot = data["snapshot"]
    assert snapshot["policy"] in {"original", "model"}, "Only the two current demo versions are supported"
    game = Game(snapshot["game"]["seed"], snapshot["game"]["max_pieces"])
    live_moves = forced_moves = policy_moves = requests = 0
    models = set()
    provider = snapshot.get("provider", "decider")
    for turn in data["history"]:
        assert turn["board_before"] == game.board
        requests += len(turn["calls"])
        for receipt in turn["calls"]:
            assert receipt.get("provider", "decider") == provider
            observed = json.loads(receipt["payload"]["state"])
            assert observed["board_rows_top_to_bottom"] == [
                "".join("#" if cell else "." for cell in row) for row in game.board
            ]
            assert observed["current_piece"] == game.active
            assert observed["next_piece"] == game.queue[0]
            validate_result(receipt["response"], receipt["payload"]["questions"], provider=provider)
            if provider == "gpt6":
                assert receipt["provider_request"] == gpt_request(receipt["payload"])
                assert normalize_gpt(
                    receipt["provider_response"], receipt["payload"]["questions"]
                ) == receipt["response"]
        if turn["outcome"] != "applied":
            continue
        decision = turn["decision"]
        candidates = game.candidates()
        assert decision["exclusions"] == []
        assert decision["eligible_count"] == len(candidates)
        if decision["source"] == "live":
            live_moves += 1
            assert turn["calls"]
            last = turn["calls"][-1]["response"]
            assert last["answers"]["landing"]["choice"] == decision["selected"]
            assert last["answers"]["landing"]["probabilities"] == decision["probabilities"]
            if decision["probabilities"] is not None:
                assert set(last["answers"]["landing"]["probabilities"]) == {
                    p["id"] for p in decision["finalists"]
                }
            else:
                assert provider == "gpt6" and decision["confidence"] is None
            models.add(last["model"])
        else:
            assert decision["source"] == "forced"
            assert len(game.candidates()) == 1
            assert not turn["calls"]
            forced_moves += 1
        selected = next(p for p in game.candidates() if p.id == decision["selected"])
        assert selected.public() == decision["selected_placement"]
        move = game.apply(selected)
        assert move["cells"] == turn["move"]["cells"]
        assert game.board == turn["board_after"]
        assert game.lines == turn["lines"] and game.score == turn["score"]
    assert game.public() == snapshot["game"]
    assert requests == snapshot["stats"]["successful_requests"]
    assert live_moves == snapshot["stats"]["live_moves"]
    assert forced_moves == snapshot["stats"]["forced_moves"]
    assert policy_moves == snapshot["stats"].get("policy_moves", 0)
    assert live_moves + forced_moves + policy_moves == game.pieces
    return {
        "verified_at": datetime.now(UTC).isoformat(),
        "seed": game.seed,
        "max_pieces": game.max_pieces,
        "pieces": game.pieces,
        "lines": game.lines,
        "score": game.score,
        "finish_reason": game.finish_reason,
        "status": snapshot["status"],
        "models": sorted(models),
        "verified_live_moves": live_moves,
        "forced_moves": forced_moves,
        "policy_moves": policy_moves,
        "policy": snapshot.get("policy", "legacy_model"),
        "provider": provider,
        "successful_requests": requests,
        "stats": snapshot["stats"],
        "raw_choice_matches_applied_placement": True,
        "board_replay_matches": True,
        "model_observed_current_board_and_next_piece": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8012")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--policy", choices=["original", "model"], default="model")
    parser.add_argument("--provider", choices=["decider", "jev", "gpt6"], default="decider")
    parser.add_argument("--output", type=Path, default=Path("artifacts/live"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=15, trust_env=False) as client:
        health = client.get("/api/health", params={"provider": args.provider})
        health.raise_for_status()
        assert health.json()["connected"], f"{args.provider} is not configured or connected"
        response = client.post(
            "/api/games",
            json={
                "seed": args.seed, "max_pieces": args.pieces,
                "interval_ms": 0, "policy": args.policy, "provider": args.provider,
            },
        )
        response.raise_for_status()
        game_id = response.json()["id"]
        endpoint = f"/api/games/{game_id}"
        started = time.monotonic()
        state = response.json()
        try:
            client.post(endpoint + "/control", json={"action": "run"}).raise_for_status()
            previous = -1
            while time.monotonic() - started < 240:
                response = client.get(endpoint)
                response.raise_for_status()
                state = response.json()
                if state["game"]["pieces"] // 10 != previous:
                    previous = state["game"]["pieces"] // 10
                    print(
                        json.dumps(
                            {
                                "pieces": state["game"]["pieces"],
                                "lines": state["game"]["lines"],
                                "phase": state["phase"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if state["status"] in ("finished", "error"):
                    break
                time.sleep(0.15)
            else:
                client.post(endpoint + "/control", json={"action": "pause"}).raise_for_status()
                raise TimeoutError("Game exceeded the verification time budget")
            exported = client.get(endpoint + "/export")
            exported.raise_for_status()
            data = exported.json()
            raw = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            record_path = args.output / f"seed-{args.seed}.json"
            record_path.write_text(raw)
            summary = verify_export(data)
            summary["wall_seconds"] = round(time.monotonic() - started, 3)
            summary["receipt_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            summary["receipt_file"] = record_path.name
            (args.output / f"seed-{args.seed}-summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            assert state["status"] == "finished", state.get("error")
        finally:
            client.delete(endpoint).raise_for_status()


if __name__ == "__main__":
    main()
