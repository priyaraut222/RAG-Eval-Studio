"""
Indicative per-token pricing used to estimate evaluation run cost.

These numbers are NOT fetched live — they're a static snapshot for
cost *estimation* only, so a run's "Estimated Cost" card is directional
(useful for comparing configurations against each other) rather than
a billing-accurate figure. Update `PRICING` if you need it to track
current provider list prices more closely.
"""

from __future__ import annotations

# $ per 1,000,000 tokens
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "heuristic-v1": {"input": 0.0, "output": 0.0},
}

_DEFAULT_RATE = {"input": 0.0, "output": 0.0}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return an estimated dollar cost for one call, given token counts."""
    rates = PRICING.get(model, _DEFAULT_RATE)
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
