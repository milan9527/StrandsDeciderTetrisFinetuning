"""Supported input formats and diagnostics that never filter or replace a model choice."""

POLICIES = ("original", "model")


def quality_vector(candidate):
    m = candidate.metrics
    return (
        not m["next_spawn_ok"],
        m["holes"],
        -m["lines_cleared"],
        m["max_height"],
        m["aggregate_height"],
        m["bumpiness"],
    )


def dominates(left, right):
    """One-step feature dominance; this is not a proof of long-term game optimality."""
    a, b = quality_vector(left), quality_vector(right)
    return a != b and all(x <= y for x, y in zip(a, b, strict=True))
