"""Depth-scaling analysis tests (PRD §6.5, §7.7) — Wave 3."""

from __future__ import annotations

from mem_eval.runner.analyze import depth_curve


def test_depth_curve_shape_and_determinism():
    depths = (8, 16, 32)
    c1 = depth_curve("naive_rag", depths=depths, k=10)
    c2 = depth_curve("naive_rag", depths=depths, k=10)
    assert [r["depth"] for r in c1] == list(depths)
    assert all(r["sessions"] == r["depth"] for r in c1)
    assert c1 == c2  # deterministic


def test_naive_finds_needle_at_depth_floor_does_not():
    depths = (8, 16, 32, 64)
    naive = depth_curve("naive_rag", depths=depths, k=10)
    floor = depth_curve("no_memory", depths=depths, k=10)
    assert all(r["recall_at_k"] > 0.0 for r in naive)   # needle stays retrievable at depth
    assert all(r["recall_at_k"] == 0.0 for r in floor)  # floor never finds it


def test_storage_grows_with_depth():
    curve = depth_curve("naive_rag", depths=(8, 64), k=10)
    assert curve[-1]["bytes"] > curve[0]["bytes"]
