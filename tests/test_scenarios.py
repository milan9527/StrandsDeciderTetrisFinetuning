"""Boundary cases where an incorrect oracle would produce misleading SFT labels."""

from copy import deepcopy

from scripts.prepare_scenarios import browser_oracle, completion_oracle, routing_oracle


def test_completion_uses_current_revision_and_latest_receipt():
    state = {
        "target_revision": "new", "required_checks": ["unit"],
        "check_receipts": [
            {"check": "unit", "revision": "new", "sequence": 2, "status": "passed"},
            {"check": "unit", "revision": "old", "sequence": 99, "status": "failed"},
            {"check": "unit", "revision": "new", "sequence": 1, "status": "failed"},
        ],
        "artifact_receipt": {"revision": "new", "exists": True},
    }
    assert completion_oracle(state) == "verified"
    state["artifact_receipt"]["revision"] = "old"
    assert completion_oracle(state) == "unknown"
    state["artifact_receipt"] = {"revision": "new", "exists": False}
    assert completion_oracle(state) == "contradicted"


def test_completion_failure_is_not_hidden_by_other_missing_evidence():
    state = {
        "target_revision": "new", "required_checks": ["unit", "integration"],
        "check_receipts": [{"check": "unit", "revision": "new", "sequence": 1, "status": "failed"}],
        "artifact_receipt": None,
    }
    assert completion_oracle(state) == "contradicted"
    state["check_receipts"][0]["status"] = "passed"
    assert completion_oracle(state) == "unknown"


def test_routing_checks_total_cost_and_inclusive_limits():
    request = {"modality": "image", "input_tokens": 2000, "output_tokens": 1000,
               "minimum_quality": 80, "latency_limit_ms": 1000, "budget": 100}
    base = {"available": True, "modalities": ["text", "image"], "max_tokens": 3000,
            "quality": 80, "p95_ms": 800, "queue_ms": 200, "setup_charge": 0}
    state = {"request": request, "models": [
        {**base, "id": "cheap_input", "input_price": 1, "output_price": 99},  # cost 101
        {**base, "id": "fits", "input_price": 40, "output_price": 20},  # cost 100
        {**base, "id": "unavailable", "input_price": 1, "output_price": 1, "available": False},
    ]}
    assert routing_oracle(state) == "fits"
    state["models"][1]["queue_ms"] += 1
    assert routing_oracle(state) == "defer"


def test_browser_requires_fresh_visible_matching_control():
    request = {"order_id": "A", "document_type": "invoice", "language": "en"}
    target = {**request, "id": "target", "visible": True, "enabled": True,
              "panel": "modal", "action": "view"}
    state = {"current_revision": 3, "observed_revision": 3, "loading": False,
             "request": request, "active_panel": "modal",
             "controls": [{**target, "id": "behind_modal", "panel": "main"},
                          {**target, "id": "payment", "action": "pay"}, target]}
    assert browser_oracle(state) == "target"
    stale = deepcopy(state)
    stale.update(observed_revision=2, loading=True)
    assert browser_oracle(stale) == "refresh"
    state["loading"] = True
    assert browser_oracle(state) == "wait"
    state["loading"] = False
    target["enabled"] = False
    assert browser_oracle(state) == "blocked"
