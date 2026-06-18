"""External-validity study (D3): does the synthetic ranking predict the real one?"""

from mem_eval.validity.study import (
    COMPOSITE_AXES,
    DEFAULT_PANEL,
    composite_score,
    external_validity,
    render_validity_report,
    run_default_study,
    score_panel,
)

__all__ = [
    "COMPOSITE_AXES",
    "DEFAULT_PANEL",
    "composite_score",
    "score_panel",
    "external_validity",
    "render_validity_report",
    "run_default_study",
]
