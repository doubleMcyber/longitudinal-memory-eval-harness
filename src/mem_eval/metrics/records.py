"""Per-query evaluation record — the input to every metric (PRD §7).

The orchestrator runs a backend over the suite and produces one ``EvalRecord``
per query (resolving returned items to gold fact ids via provenance). Metric
modules are pure functions over lists of these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mem_eval.adapters.base import Usage


@dataclass
class EvalRecord:
    query_id: str
    category: str
    intent: str | None
    gold_support: frozenset[str]
    gold_superseded: frozenset[str]
    retrieved: tuple[str | None, ...]  # resolved fact ids in rank order (None = unmappable)
    retrieved_count: int  # number of items returned (incl. unmappable)
    superseded_returned: int  # returned items that are superseded as-of the query
    answer: str | None
    gold_answer: str
    answer_correct: bool
    latency_ms: float
    usage: Usage = field(default_factory=Usage)

    @property
    def retrieved_facts(self) -> set[str]:
        return {r for r in self.retrieved if r is not None}


def normalized_answer_correct(answer: str | None, gold_answer: str) -> bool:
    """Offline, deterministic stand-in for the LLM answer judge (PRD §7.3, §14):
    the gold value must appear in the synthesized answer (case-insensitive).

    NOTE: this is an AUXILIARY, text-based signal. The four PRD-§7 quality
    metrics (recall/precision/contradiction/staleness) are graded ID-based via
    provenance and are the exact, gameable-proof labels (PRD §8). Because this
    judge is a substring match, a distractor that coincidentally carries the
    gold value string could in principle earn spurious answer credit; in the v1
    baselines the top-ranked item is the gold fact in each category, so this is
    latent, not active. Treat answer_accuracy as indicative, not authoritative.
    """
    if not answer or not gold_answer:
        return False
    return gold_answer.strip().lower() in answer.strip().lower()


__all__ = ["EvalRecord", "normalized_answer_correct"]
