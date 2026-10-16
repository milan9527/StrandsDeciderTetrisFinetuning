import asyncio
import json
import random
from pathlib import Path

import pytest

from backend.decider import select_placement
from backend.engine import Game
from backend.policy import dominates
from backend.session import Session
from tests.test_decider import FakeClient

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/decision-regression.json").read_text()
)


def failure_game():
    game = Game(FIXTURE["seed"], FIXTURE["max_pieces"])
    for selected in FIXTURE["placements_before_turn_seven"]:
        game.apply(next(p for p in game.candidates() if p.id == selected))
    return game


def test_actual_turn_seven_failure_remains_legal_despite_diagnostic():
    game = failure_game()
    candidates = {p.id: p for p in game.candidates()}
    assert dominates(candidates["p019"], candidates["p034"])
    # Reporting a dominated move must not alter legality or state.
    assert candidates == {p.id: p for p in game.candidates()}


def test_retired_policy_cannot_create_session_or_select_a_move():
    async def exercise():
        game = failure_game()
        client = FakeClient()
        with pytest.raises(ValueError, match="Unknown decision policy"):
            Session(client, 42, 10, policy="guarded")
        with pytest.raises(ValueError, match="Unknown decision policy"):
            await select_placement(
                game, game.candidates(), client, lambda _: None, policy="guarded"
            )
        assert not client.calls

    asyncio.run(exercise())


@pytest.mark.parametrize("policy", ["original", "model"])
def test_every_legal_candidate_reaches_model_without_strategy_filter(policy):
    async def exercise():
        game = failure_game()
        client = FakeClient()
        candidates = game.candidates()
        chosen, decision = await select_placement(
            game, candidates, client, lambda _: None, policy=policy
        )
        presented = set().union(*(
            set(q["criteria"]) for q in client.calls[0]["questions"].values()
        ))
        assert presented == {p.id for p in candidates}
        assert {"p019", "p034"} <= presented
        assert decision["exclusions"] == []
        assert decision["eligible_count"] == decision["candidate_count"] == len(candidates)
        assert decision["source"] == "live"
        assert chosen.id == decision["calls"][-1]["response"]["answers"]["landing"]["choice"]

    asyncio.run(exercise())


@pytest.mark.parametrize("policy", ["original", "model"])
def test_pure_model_failure_is_visible_and_is_not_silently_replaced(policy):
    async def exercise():
        game = failure_game()
        candidates = {p.id: p for p in game.candidates()}
        # Two candidates with a seed ordering that puts the bad move first for this fake client.
        pair = [candidates["p034"], candidates["p019"]]
        ordered = pair[:]
        random.Random(game.seed * 1009 + game.pieces).shuffle(ordered)
        if ordered[0].id != "p034":
            pair.reverse()
        chosen, decision = await select_placement(
            game, pair, FakeClient(), lambda _: None, policy=policy
        )
        assert chosen.id == "p034"
        assert decision["source"] == "live" and decision["dominated_by"] == ["p019"]
        assert decision["exclusions"] == []

    asyncio.run(exercise())


def test_two_concurrent_versions_keep_distinct_inputs_and_original_receipt():
    async def exercise():
        class YieldingClient(FakeClient):
            async def evaluate(self, payload):
                await asyncio.sleep(0)
                return await super().evaluate(payload)

        clients = {policy: YieldingClient() for policy in ("original", "model")}

        async def choose(policy, client):
            game = Game(42)
            return await select_placement(
                game, game.candidates(), client, lambda _: None, policy=policy
            )

        results = await asyncio.gather(*(choose(p, c) for p, c in clients.items()))
        original = clients["original"].calls[0]
        assert original == FIXTURE["original_first_request"]
        plain = clients["model"].calls[0]
        assert original["state"] == plain["state"]
        original_options = original["questions"]["landing"]["criteria"]
        plain_options = plain["questions"]["landing"]["criteria"]
        assert list(original_options) == list(plain_options)
        assert all(text.startswith("x=") for text in original_options.values())
        assert all(text.startswith("Clear ") for text in plain_options.values())
        assert {result[1]["policy"] for result in results} == set(clients)
        assert results[0][1]["exclusions"] == results[1][1]["exclusions"] == []

    asyncio.run(exercise())
