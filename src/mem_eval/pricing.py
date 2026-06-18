"""Synthetic, deterministic pricing for usage/cost accounting (PRD §7.6).

Prices are fixed constants so cost-per-query is reproducible and comparable
across backends on the same machine. Real backends would record real USD; for
the in-box offline baselines we price token counts deterministically.
"""

from __future__ import annotations

# USD per 1K tokens (illustrative, stable across runs)
PROMPT_PER_1K = 0.0030
COMPLETION_PER_1K = 0.0150
EMBEDDING_PER_1K = 0.0001


def price(prompt_tokens: int = 0, completion_tokens: int = 0, embedding_tokens: int = 0) -> float:
    return (
        prompt_tokens / 1000 * PROMPT_PER_1K
        + completion_tokens / 1000 * COMPLETION_PER_1K
        + embedding_tokens / 1000 * EMBEDDING_PER_1K
    )


__all__ = ["price", "PROMPT_PER_1K", "COMPLETION_PER_1K", "EMBEDDING_PER_1K"]
