"""
Intraday feedback generation.

The daily feedback module compares Qlib metrics such as 1day excess return.
This module instead summarizes intraday metrics emitted by the custom runner.
"""

from __future__ import annotations

from quantaalpha.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from quantaalpha.core.experiment import Experiment


class IntradayHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Minimal intraday feedback adapter for the first integration stage."""

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        result = getattr(exp, "result", {}) or {}
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

        if metrics:
            metric_lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
            observations = "Intraday runner returned the following metrics:\n" + "\n".join(metric_lines[:20])
            reason = "Stub intraday evaluation completed; metrics were recorded successfully."
            decision = True
        else:
            observations = "Intraday runner completed without producing metrics."
            reason = "No usable metrics were returned by the intraday runner."
            decision = False

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=f"Target hypothesis: {hypothesis.hypothesis}",
            new_hypothesis="",
            reason=reason,
            decision=decision,
        )
