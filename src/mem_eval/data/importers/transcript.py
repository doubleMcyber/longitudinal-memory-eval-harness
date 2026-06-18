"""Real conversation-log importer (PRD §9).

Maps external transcripts into the Session schema and combines them with a
manual annotation sidecar (query -> gold answer + supporting item refs + as_of +
category) to produce a Scenario that runs end-to-end through the *same* metrics.

Annotation format: JSON in v1 (stdlib-only so the gate has no extra deps). If a
``.yaml``/``.yml`` sidecar is given and PyYAML is importable, it is parsed too;
otherwise JSON is assumed. (PRD names a YAML sidecar; the schema is identical.)

The optional LLM-assisted label proposal (PRD §9 tier b) is a documented stub:
it pre-fills a sidecar skeleton explicitly flagged low-confidence for human
review and does not invent gold.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from mem_eval.adapters.base import Session, Turn
from mem_eval.data.schema import Query, Scenario

# The shipped real-log corpus (D3): a meaningfully-sized, hand-authored real-STYLE
# corpus + multi-annotator labels, used by the external-validity study. It is a
# curated stand-in for human-collected production logs (see corpus README); the
# importer/format below is exactly what a genuine labeled corpus would plug into.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_CORPUS_DIR = os.path.join(_REPO_ROOT, "corpora", "real_logs_v1")
DEFAULT_CORPUS = os.path.join(DEFAULT_CORPUS_DIR, "corpus.json")
DEFAULT_ANNOTATORS = os.path.join(DEFAULT_CORPUS_DIR, "annotators.json")


def import_sessions(raw_sessions: list[dict]) -> list[Session]:
    """raw_sessions: [{session_id, timestamp(ISO), turns:[{role,text[,turn_id]}]}]."""
    sessions = []
    for rs in raw_sessions:
        ts = datetime.fromisoformat(rs["timestamp"])
        turns = []
        for i, t in enumerate(rs["turns"]):
            tid = t.get("turn_id") or f"{rs['session_id']}-t{i:03d}"
            turns.append(Turn(turn_id=tid, role=t.get("role", "user"), text=t["text"]))
        sessions.append(Session(session_id=rs["session_id"], timestamp=ts, turns=turns))
    return sessions


def load_annotation(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "YAML sidecar requested but PyYAML is not installed; use a JSON sidecar."
            ) from exc
        with open(path) as fh:
            return yaml.safe_load(fh)
    with open(path) as fh:
        return json.load(fh)


def _ref(r: dict) -> str:
    """A supporting-item reference (session, turn) -> stable id used for grading."""
    return f"{r['session']}:{r['turn']}"


def _query_from_dict(qa: dict) -> Query:
    """Build a Query from an annotation dict. Gold is taken verbatim from the
    structured sidecar (support/superseded refs + answer); it is never inferred
    from the transcript text."""
    return Query(
        query_id=qa["query_id"],
        category=qa["category"],
        prompt=qa["prompt"],
        as_of=datetime.fromisoformat(qa["as_of"]),
        gold_answer=qa["gold_answer"],
        gold_support=tuple(_ref(r) for r in qa.get("support", [])),
        gold_superseded=tuple(_ref(r) for r in qa.get("superseded", [])),
        intent=qa.get("intent"),
        k=qa.get("k"),
        answer_aliases=tuple(qa.get("answer_aliases", [])),
    )


def _turn_to_fact(sessions: list[Session]) -> dict[tuple[str, str], str]:
    """Every turn gets a stable grading id so any returned item resolves to a ref."""
    return {
        (s.session_id, t.turn_id): f"{s.session_id}:{t.turn_id}"
        for s in sessions
        for t in s.turns
    }


def build_scenario_from_logs(
    transcript_path: str,
    sidecar_path: str,
    scenario_id: str = "real-sample",
) -> Scenario:
    with open(transcript_path) as fh:
        transcript = json.load(fh)
    sessions = import_sessions(transcript["sessions"])
    ann = load_annotation(sidecar_path)
    queries = [_query_from_dict(qa) for qa in ann["queries"]]
    category = ann.get("category") or (queries[0].category if queries else "real_logs")
    return Scenario(
        scenario_id=scenario_id,
        category=category,
        sessions=sessions,
        facts=[],  # real logs carry no constructed structured layer
        queries=queries,
        turn_to_fact=_turn_to_fact(sessions),
    )


# --- D3: meaningfully-sized labeled corpus (multiple scenarios in one file) ---


def _scenario_from_record(rec: dict) -> Scenario:
    sessions = import_sessions(rec["sessions"])
    queries = [_query_from_dict(qa) for qa in rec.get("queries", [])]
    return Scenario(
        scenario_id=rec["scenario_id"],
        category=rec.get("category") or (queries[0].category if queries else "real_logs"),
        sessions=sessions,
        facts=[],  # hand-labeled refs only; no reconstructed temporal fact graph
        queries=queries,
        turn_to_fact=_turn_to_fact(sessions),
    )


def load_corpus(corpus_path: str = DEFAULT_CORPUS) -> dict:
    with open(corpus_path) as fh:
        return json.load(fh)


def build_suite_from_corpus(corpus_path: str = DEFAULT_CORPUS) -> "Suite":
    """Materialize the real-log corpus as a Suite that runs through the SAME
    runner/metrics as the synthetic suites (the precondition for rank-correlating
    the two — D3 external validity)."""
    from mem_eval.data.schema import Suite

    corpus = load_corpus(corpus_path)
    scenarios = [_scenario_from_record(rec) for rec in corpus["scenarios"]]
    return Suite(
        suite="real_logs_v1",
        dataset_version=corpus.get("corpus_version", "real-logs-v1"),
        seed=0,
        scenarios=scenarios,
        config_name="real",
    )


def corpus_candidate_refs(corpus_path: str = DEFAULT_CORPUS) -> dict[str, list[str]]:
    """scenario_id -> all (session:turn) reference ids in that scenario. This is
    the per-scenario candidate set over which inter-annotator support agreement is
    measured (PRD §9)."""
    corpus = load_corpus(corpus_path)
    out: dict[str, list[str]] = {}
    for rec in corpus["scenarios"]:
        refs = [f"{s['session_id']}:{t['turn_id']}"
                for s in rec["sessions"] for t in s["turns"]]
        out[rec["scenario_id"]] = refs
    return out


def corpus_iaa(corpus_path: str = DEFAULT_CORPUS,
               annotators_path: str = DEFAULT_ANNOTATORS) -> dict:
    """Inter-annotator agreement over the whole corpus (pooled across scenarios).
    Returns Cohen's kappa on support membership + gold-answer agreement (PRD §9)."""
    from mem_eval.grading.agreement import corpus_agreement

    with open(annotators_path) as fh:
        annotators = json.load(fh)["annotators"]
    return corpus_agreement(corpus_candidate_refs(corpus_path), annotators)


def transcript_candidate_refs(transcript_path: str) -> list[str]:
    """All (session, turn) reference ids in a transcript — the candidate set over
    which inter-annotator support agreement is measured (PRD §9)."""
    sessions = import_sessions(_load_json(transcript_path)["sessions"])
    return [f"{s.session_id}:{t.turn_id}" for s in sessions for t in s.turns]


def _load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def propose_labels(transcript_path: str) -> dict:
    """LLM-assisted label proposal — STUB (PRD §9 tier b).

    Returns a sidecar skeleton flagged low-confidence so a human must review and
    fill the gold. It does NOT fabricate gold answers/support."""
    with open(transcript_path) as fh:
        transcript = json.load(fh)
    return {
        "low_confidence": True,
        "note": "LLM-assisted proposals are a documented stub; fill gold by hand (PRD §9).",
        "queries": [],
        "_sessions_seen": [s["session_id"] for s in transcript["sessions"]],
    }


__all__ = [
    "import_sessions",
    "load_annotation",
    "build_scenario_from_logs",
    "transcript_candidate_refs",
    "propose_labels",
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_CORPUS",
    "DEFAULT_ANNOTATORS",
    "load_corpus",
    "build_suite_from_corpus",
    "corpus_candidate_refs",
    "corpus_iaa",
]
