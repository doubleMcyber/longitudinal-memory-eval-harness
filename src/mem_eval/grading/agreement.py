"""Inter-annotator agreement for real-log labeling (PRD §9).

Real-log gold is only as trustworthy as its labeling. Before a hand-annotated
suite is admitted, the harness measures how much independent annotators agree —
Cohen's kappa on the binary "is this turn a supporting item?" decision (chance-
corrected), plus exact-match agreement on the gold answer string. Low agreement
flags an ambiguous query or guideline, exactly as a serious benchmark requires.
"""

from __future__ import annotations

from itertools import combinations

from mem_eval.grading.judge import normalize


def cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa between two equal-length sequences of categorical labels.
    1.0 = perfect, 0.0 = chance, <0 = worse than chance."""
    if len(a) != len(b):
        raise ValueError("label sequences must be equal length")
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0  # only one category present and they agree
    return (po - pe) / (1 - pe)


def _support_vectors(annotators: list[dict], candidate_refs: list[str]) -> dict[str, list[int]]:
    """For each annotator, a binary vector over (query, candidate_turn): 1 if the
    annotator marked that turn as supporting the query. Pooled across shared queries."""
    # queries present in every annotator (so vectors align)
    query_sets = [
        {q["query_id"] for q in ann["queries"]} for ann in annotators
    ]
    shared = sorted(set.intersection(*query_sets)) if query_sets else []
    vectors: dict[str, list[int]] = {}
    for ann in annotators:
        by_q = {q["query_id"]: q for q in ann["queries"]}
        vec: list[int] = []
        for qid in shared:
            support = {f"{r['session']}:{r['turn']}" for r in by_q[qid].get("support", [])}
            vec.extend(1 if ref in support else 0 for ref in candidate_refs)
        vectors[ann["annotator_id"]] = vec
    return vectors


def _answer_agreement(annotators: list[dict]) -> float:
    query_sets = [{q["query_id"] for q in ann["queries"]} for ann in annotators]
    shared = sorted(set.intersection(*query_sets)) if query_sets else []
    if not shared:
        return 0.0
    answers = [{q["query_id"]: q.get("gold_answer", "") for q in ann["queries"]} for ann in annotators]
    agree_pairs = total_pairs = 0
    for qid in shared:
        for i, j in combinations(range(len(annotators)), 2):
            total_pairs += 1
            if normalize(answers[i][qid]) == normalize(answers[j][qid]):
                agree_pairs += 1
    return agree_pairs / total_pairs if total_pairs else 0.0


def inter_annotator_agreement(multi: dict, candidate_refs: list[str]) -> dict:
    """multi: {"annotators": [{"annotator_id", "queries":[{query_id, gold_answer, support}]}]}.
    Returns average pairwise Cohen's kappa on support membership + gold-answer
    exact-match agreement."""
    annotators = multi["annotators"]
    if len(annotators) < 2:
        raise ValueError("inter-annotator agreement needs >= 2 annotators")
    vectors = _support_vectors(annotators, candidate_refs)
    ids = [a["annotator_id"] for a in annotators]
    kappas = [cohen_kappa(vectors[i], vectors[j]) for i, j in combinations(ids, 2)]
    return {
        "support_kappa": sum(kappas) / len(kappas) if kappas else 1.0,
        "pairwise_kappa": {f"{i}|{j}": cohen_kappa(vectors[i], vectors[j])
                            for i, j in combinations(ids, 2)},
        "answer_agreement": _answer_agreement(annotators),
        "n_annotators": len(annotators),
    }


def corpus_agreement(scenario_refs: dict[str, list[str]], annotators: list[dict]) -> dict:
    """Inter-annotator agreement over a multi-scenario corpus (D3).

    ``scenario_refs``: scenario_id -> candidate turn refs (all turns in scenario).
    ``annotators``: ``[{"annotator_id", "labels": [{scenario_id, query_id,
    gold_answer, support: [{session, turn}]}]}]``.

    Each annotator becomes one long binary vector over every
    (scenario, query, candidate_turn) cell shared by all annotators: 1 iff that
    annotator marked the turn as supporting that query. Cohen's kappa is averaged
    over annotator pairs; gold-answer agreement is exact-match (normalized)."""
    if len(annotators) < 2:
        raise ValueError("inter-annotator agreement needs >= 2 annotators")

    # the (scenario_id, query_id) cells present for EVERY annotator (so vectors align)
    label_keys = [
        {(lab["scenario_id"], lab["query_id"]) for lab in ann["labels"]} for ann in annotators
    ]
    shared = sorted(set.intersection(*label_keys))

    vectors: dict[str, list[int]] = {}
    for ann in annotators:
        by_key = {(lab["scenario_id"], lab["query_id"]): lab for lab in ann["labels"]}
        vec: list[int] = []
        for sid, qid in shared:
            support = {f"{r['session']}:{r['turn']}" for r in by_key[(sid, qid)].get("support", [])}
            vec.extend(1 if ref in support else 0 for ref in scenario_refs[sid])
        vectors[ann["annotator_id"]] = vec

    ids = [a["annotator_id"] for a in annotators]
    kappas = {f"{i}|{j}": cohen_kappa(vectors[i], vectors[j]) for i, j in combinations(ids, 2)}

    # gold-answer exact-match agreement, averaged over shared cells x annotator pairs
    ans = {a["annotator_id"]: {(lab["scenario_id"], lab["query_id"]): lab.get("gold_answer", "")
                               for lab in a["labels"]} for a in annotators}
    agree = total = 0
    for key in shared:
        for i, j in combinations(ids, 2):
            total += 1
            if normalize(ans[i][key]) == normalize(ans[j][key]):
                agree += 1

    return {
        "support_kappa": sum(kappas.values()) / len(kappas) if kappas else 1.0,
        "pairwise_kappa": kappas,
        "answer_agreement": agree / total if total else 0.0,
        "n_annotators": len(annotators),
        "n_shared_queries": len(shared),
    }


__all__ = ["cohen_kappa", "inter_annotator_agreement", "corpus_agreement"]
