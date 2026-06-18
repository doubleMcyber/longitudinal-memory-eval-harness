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


# Answer grading lives in mem_eval.grading.judge (pluggable AnswerJudge); the
# orchestrator computes answer_correct via the configured judge. The four PRD-§7
# quality metrics remain ID-based (provenance) — the exact, gameable-proof labels.

__all__ = ["EvalRecord"]
