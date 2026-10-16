import asyncio
import copy
import json

import httpx
import pytest

from backend.decider import DeciderClient, DeciderError, Settings, select_placement, validate_result
from backend.engine import Game


def reply(questions):
    return {
        "model": "test-only-strands-decider",
        "source": "live",
        "answers": {
            key: {
                "type": "choice",
                "choice": next(iter(q["criteria"])),
                "confidence": 0.75,
                "probabilities": {option: float(i == 0) for i, option in enumerate(q["criteria"])},
            }
            for key, q in questions.items()
        },
    }


class FakeClient:
    """Explicit offline test double. Never available as a runtime backend."""

    def __init__(self):
        self.calls = []

    async def evaluate(self, payload):
        self.calls.append(copy.deepcopy(payload))
        return {"payload": payload, "response": reply(payload["questions"]), "elapsed_ms": 1}

    async def health(self):
        return {"connected": True, "model": "test-only-strands-decider", "transport": "test"}

    async def close(self):
        pass


def test_tournament_considers_every_candidate_and_preserves_actual_final_choice():
    async def exercise():
        game = Game(42)
        game.active = "T"
        candidates = game.candidates()
        client = FakeClient()
        chosen, decision = await select_placement(game, candidates, client, lambda _: None)
        assert len(client.calls) == 2
        initial = client.calls[0]["questions"]
        assert set().union(*(set(q["criteria"]) for q in initial.values())) == {
            p.id for p in candidates
        }
        assert all(
            2 <= len(q["criteria"]) <= 12 for p in client.calls for q in p["questions"].values()
        )
        assert chosen.id == decision["calls"][-1]["response"]["answers"]["landing"]["choice"]
        assert len(decision["finalists"]) == 3
        assert game.pieces == 0

    asyncio.run(exercise())


def test_single_candidate_is_explicitly_forced_without_a_model_call():
    async def exercise():
        game = Game()
        candidate = game.candidates()[0]
        client = FakeClient()
        chosen, result = await select_placement(game, [candidate], client, lambda _: None)
        assert chosen == candidate and result["source"] == "forced"
        assert result["probabilities"] is None and not client.calls

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "corruption",
    ["missing", "nan", "bool", "foreign", "sum", "wrong_winner", "replay", "not_decider"],
)
def test_invalid_response_is_rejected(corruption):
    questions = {"landing": {"type": "choice", "criteria": {"a": "first", "b": "second"}}}
    result = reply(questions)
    answer = result["answers"]["landing"]
    if corruption == "missing":
        result["answers"] = {}
    elif corruption == "nan":
        answer["probabilities"]["a"] = float("nan")
    elif corruption == "bool":
        answer["confidence"] = True
    elif corruption == "foreign":
        answer["choice"] = "invented"
    elif corruption == "sum":
        answer["probabilities"]["b"] = 0.5
    elif corruption == "wrong_winner":
        answer["choice"] = "b"
    elif corruption == "replay":
        result["source"] = "example"
    else:
        result["model"] = "some-other-model"
    with pytest.raises(DeciderError):
        validate_result(result, questions, gateway=True)


def test_gateway_authentication_refresh_and_real_mode(tmp_path):
    credentials = tmp_path / "login.json"
    credentials.write_text(json.dumps({"username": "test", "password": "never-output-this"}))
    paths, decision_count = [], 0

    def handler(request):
        nonlocal decision_count
        paths.append(request.url.path)
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/api/decide":
            decision_count += 1
            if decision_count == 1:
                return httpx.Response(401)
            body = json.loads(request.content)
            assert body["mode"] == "live"
            assert "password" not in body
            return httpx.Response(200, json=reply(body["questions"]))
        return httpx.Response(200, json={})

    async def exercise():
        http = httpx.AsyncClient(
            base_url="https://test.invalid", transport=httpx.MockTransport(handler)
        )
        client = DeciderClient(Settings("gateway", "https://test.invalid", str(credentials)), http)
        await client.evaluate(
            {
                "state": "test",
                "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            }
        )
        await client.close()

    asyncio.run(exercise())
    assert paths.count("/api/auth/login") == 2
    assert paths.count("/api/decide") == 2
    assert paths[-1] == "/api/auth/logout"
