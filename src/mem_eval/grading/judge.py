"""Robust, pluggable answer grading (PRD §7.3, §14).

The v1 harness graded synthesized answers with a bare case-insensitive substring
match. That is brittle: "5 5 5 1 2 3 4", "NYC" vs "New York City", "March 2025"
vs "2025-03", and trailing punctuation all break naive matching. A benchmark
labs trust needs grading that is robust to surface variation while staying
deterministic and offline by default.

This module defines an ``AnswerJudge`` protocol and three implementations:

* ``NormalizingJudge`` (default) — casefold + punctuation/whitespace normalization,
  numeric tolerance, light date canonicalization, alias sets, and WHOLE-token/phrase
  containment (not raw substring, so "42" does not match "4200" nor "class"
  "classify"). Deterministic, offline. Limitation: containment cannot disambiguate
  an answer that mentions several candidate values ("changed from August to July");
  use ``LLMJudge`` with a real model for semantic adjudication of such answers.
* ``ExactJudge`` — strict normalized equality (for ablations / strict suites).
* ``LLMJudge`` — wraps a ``complete_fn(prompt) -> str`` so a real LLM (e.g. Claude)
  can adjudicate semantic equivalence. Its default ``complete_fn`` is a
  deterministic offline stub that delegates to ``NormalizingJudge`` (so the gate
  stays offline); production callers inject a real model. The judge model id is
  recorded in the scorecard env for reproducibility.

The four PRD-§7 quality metrics remain ID-based (provenance), exact and
gameable-proof; answer grading only affects the answer-accuracy signal and the
answer arm of contradiction-resolution.
"""

from __future__ import annotations

import re
from typing import Callable, Protocol, runtime_checkable

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MONTHS = {
    m: f"{i:02d}"
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


def normalize(text: str) -> str:
    """Casefold, replace non-alphanumeric runs with single spaces, strip."""
    return _PUNCT_RE.sub(" ", text.casefold()).strip()


def _as_number(text: str) -> float | None:
    nums = _NUM_RE.findall(text)
    if len(nums) == 1:
        try:
            return float(nums[0])
        except ValueError:
            return None
    return None


def _canon_date(text: str) -> str | None:
    """Canonicalize a few common date forms to a comparable token, e.g.
    'March' -> 'm03', 'March 2025' -> '2025-m03', '2025-03-19' -> '2025-03-19'."""
    n = normalize(text)
    iso = re.search(r"\b(\d{4})[ -](\d{1,2})(?:[ -](\d{1,2}))?\b", n)
    if iso:
        y, mo, d = iso.group(1), int(iso.group(2)), iso.group(3)
        return f"{y}-{mo:02d}" + (f"-{int(d):02d}" if d else "")
    toks = n.split()
    months = [t for t in toks if t in _MONTHS]
    if len(set(months)) > 1:
        return None  # ambiguous: multiple distinct months -> let the LLM judge decide
    if months:
        month = months[0]
        year = next((t for t in toks if re.fullmatch(r"\d{4}", t)), None)
        return (f"{year}-" if year else "") + f"m{_MONTHS[month]}"
    return None


@runtime_checkable
class AnswerJudge(Protocol):
    name: str

    def judge(self, answer: str | None, gold: str, aliases: tuple[str, ...] = ()) -> bool:
        ...


class NormalizingJudge:
    name = "normalizing"

    def __init__(self, numeric_rel_tol: float = 1e-9, numeric_abs_tol: float = 0.0,
                 use_dates: bool = True) -> None:
        self.numeric_rel_tol = numeric_rel_tol
        self.numeric_abs_tol = numeric_abs_tol
        self.use_dates = use_dates

    def _match_one(self, answer_norm: str, answer_raw: str, gold: str) -> bool:
        gold_norm = normalize(gold)
        if not gold_norm:
            return False
        # 1) WHOLE-token/phrase containment (handles "the answer is Berlin.").
        #    Deliberately NOT a raw substring test: a raw substring would mis-grade
        #    "42" vs "4200", "class" vs "classify", "engine" vs "engineer".
        if gold_norm == answer_norm or f" {gold_norm} " in f" {answer_norm} ":
            return True
        # 2) numeric tolerance
        ga, aa = _as_number(gold), _as_number(answer_raw)
        if ga is not None and aa is not None:
            import math

            if math.isclose(ga, aa, rel_tol=self.numeric_rel_tol, abs_tol=self.numeric_abs_tol):
                return True
        # 3) date canonicalization
        if self.use_dates:
            gd, ad = _canon_date(gold), _canon_date(answer_raw)
            if gd is not None and ad is not None and gd == ad:
                return True
        return False

    def judge(self, answer: str | None, gold: str, aliases: tuple[str, ...] = ()) -> bool:
        if not answer or not gold:
            return False
        answer_norm = normalize(answer)
        for candidate in (gold, *aliases):
            if self._match_one(answer_norm, answer, candidate):
                return True
        return False


class ExactJudge:
    name = "exact"

    def judge(self, answer: str | None, gold: str, aliases: tuple[str, ...] = ()) -> bool:
        if not answer or not gold:
            return False
        a = normalize(answer)
        return any(a == normalize(g) for g in (gold, *aliases))


def _offline_llm_stub(prompt: str) -> str:  # pragma: no cover - exercised via LLMJudge
    """Deterministic offline stand-in for a real LLM judge call. Keeps the gate
    offline; production injects a real complete_fn."""
    return "YES" if "EQUIVALENT" in prompt else "NO"


class LLMJudge:
    """Semantic-equivalence judge backed by a pluggable ``complete_fn``.

    Real usage: ``LLMJudge(complete_fn=lambda p: anthropic_complete(p), model="claude-...")``.
    Offline default: delegates equivalence to ``NormalizingJudge`` and renders a
    YES/NO via the stub so the call path is exercised deterministically."""

    name = "llm"

    def __init__(self, complete_fn: Callable[[str], str] | None = None,
                 model: str = "offline-stub") -> None:
        self._offline = complete_fn is None
        self.model = model
        self._fallback = NormalizingJudge()
        self.complete_fn = complete_fn or _offline_llm_stub

    def judge(self, answer: str | None, gold: str, aliases: tuple[str, ...] = ()) -> bool:
        if not answer or not gold:
            return False
        if self._offline:
            # Deterministic offline path: decide via the normalizing fallback, then
            # round-trip through the stub completion so the call path is exercised.
            equivalent = self._fallback.judge(answer, gold, aliases)
            prompt = f"EQUIVALENT gold={gold!r} answer={answer!r}" if equivalent else \
                     f"DIFFERENT gold={gold!r} answer={answer!r}"
            return self.complete_fn(prompt).strip().upper().startswith("YES")
        # real model path
        alias_txt = f" (also accept: {', '.join(aliases)})" if aliases else ""
        prompt = (
            "You are grading a memory benchmark answer. Reply YES if the ANSWER conveys "
            f"the GOLD value, else NO.\nGOLD: {gold}{alias_txt}\nANSWER: {answer}\nReply YES or NO."
        )
        return self.complete_fn(prompt).strip().upper().startswith("YES")


DEFAULT_JUDGE = NormalizingJudge()


__all__ = [
    "AnswerJudge",
    "NormalizingJudge",
    "ExactJudge",
    "LLMJudge",
    "DEFAULT_JUDGE",
    "normalize",
]
