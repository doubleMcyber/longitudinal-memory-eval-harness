"""Inter-annotator agreement tests (PRD §9) — Wave 4."""

from __future__ import annotations

import json
import os

from mem_eval.data.importers.transcript import transcript_candidate_refs
from mem_eval.grading.agreement import cohen_kappa, inter_annotator_agreement

_HERE = os.path.dirname(__file__)
_TRANSCRIPT = os.path.join(_HERE, "fixtures", "real_logs", "transcript.json")
_MULTI = os.path.join(_HERE, "fixtures", "real_logs", "multi_annotator.json")


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # all-same category and agree -> defined as 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    # systematic disagreement -> negative
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0.0


def test_cohen_kappa_known_value():
    # 2x2: a=[1,1,0,0,1], b=[1,0,0,0,1]; po=4/5=0.8
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 0, 0, 1]
    k = cohen_kappa(a, b)
    assert 0.0 < k < 1.0


def test_inter_annotator_agreement_on_sample():
    with open(_MULTI) as fh:
        multi = json.load(fh)
    refs = transcript_candidate_refs(_TRANSCRIPT)
    result = inter_annotator_agreement(multi, refs)
    # annotators agree on answers but disagree on one support turn -> kappa in (0,1)
    assert result["answer_agreement"] == 1.0
    assert 0.0 < result["support_kappa"] < 1.0
    assert result["n_annotators"] == 2
    assert "ann1|ann2" in result["pairwise_kappa"]


def test_candidate_refs_cover_all_turns():
    refs = transcript_candidate_refs(_TRANSCRIPT)
    assert "r-s0:r-s0-t000" in refs
    assert len(refs) == 6  # 3 sessions x 2 turns
