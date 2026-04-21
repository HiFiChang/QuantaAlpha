"""
Intraday feedback generation.

This module mirrors the daily feedback flow more closely than the original
stub: it structures intraday evaluation outputs, compares the current round
against prior best results, and asks the LLM to produce actionable next-step
feedback for continued factor mining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined

from quantaalpha.core.experiment import Experiment
from quantaalpha.core.prompts import Prompts
from quantaalpha.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from quantaalpha.log import logger
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.utils import convert2bool

MAX_JSON_PARSE_RETRIES = 3
INTRADAY_FEEDBACK_PROMPTS = Prompts(file_path=Path(__file__).parent / "prompts" / "prompts.yaml")

PRIMARY_METRICS = [
    "annual_long_short_ret_sz",
    "sharp_long_short_sz",
    "ic_wcor_sz",
    "ir_wcor_sz",
    "annual_long_ret_sz",
    "sharp_long_sz",
    "coverage",
    "stock_mean_stock_num_sz",
    "stock_mean_flu_sz",
]
STRUCTURE_METRICS = [
    "group_monotonicity",
    "group_tail_spread",
    "ls_nav_return",
    "ls_positive_ratio",
    "time_ic_positive_ratio",
    "time_ir_positive_ratio",
]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _format_metric_value(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.6f}"


def _load_result_frame(summary: dict[str, Any], key: str) -> pd.DataFrame | None:
    path = ((summary.get("result_paths") or {}).get(key)) if isinstance(summary, dict) else None
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read intraday result frame {csv_path}: {exc}")
        return None


def _compute_structure_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    group_nav = _load_result_frame(summary, "df_intraday_group_sz")
    if group_nav is not None and not group_nav.empty:
        numeric_cols = [c for c in group_nav.columns if str(c) != "date"]
        if numeric_cols:
            final_row = pd.to_numeric(group_nav[numeric_cols].iloc[-1], errors="coerce")
            mean_row = pd.to_numeric(group_nav[numeric_cols].mean(axis=0), errors="coerce")
            x = pd.Series(range(len(numeric_cols)), dtype=float)
            metrics["group_monotonicity"] = float(mean_row.corr(x, method="spearman")) if mean_row.notna().sum() > 1 else 0.0
            metrics["group_tail_spread"] = float(final_row.iloc[-1] - final_row.iloc[0]) if len(final_row) >= 2 else 0.0

    ls_nav = _load_result_frame(summary, "df_intraday_long_short_log_sz")
    if ls_nav is not None and not ls_nav.empty and "log_ls_return" in ls_nav.columns:
        series = pd.to_numeric(ls_nav["log_ls_return"], errors="coerce").dropna()
        if not series.empty:
            metrics["ls_nav_return"] = float(series.iloc[-1] / series.iloc[0] - 1.0) if series.iloc[0] != 0 else 0.0
            metrics["ls_positive_ratio"] = float((series.diff().dropna() > 0).mean()) if len(series) > 1 else 0.0

    time_ic = _load_result_frame(summary, "df_time_ic_sz")
    if time_ic is not None and not time_ic.empty and "rank_ic" in time_ic.columns:
        series = pd.to_numeric(time_ic["rank_ic"], errors="coerce").dropna()
        if not series.empty:
            metrics["time_ic_positive_ratio"] = float((series > 0).mean())

    time_ir = _load_result_frame(summary, "df_time_ir_sz")
    if time_ir is not None and not time_ir.empty and "rank_ir" in time_ir.columns:
        series = pd.to_numeric(time_ir["rank_ir"], errors="coerce").dropna()
        if not series.empty:
            metrics["time_ir_positive_ratio"] = float((series > 0).mean())

    return metrics


def _summary_structure_table(summary: dict[str, Any]) -> str:
    structure = summary.get("structure_metrics", {}) if isinstance(summary, dict) else {}
    rows = []
    for metric_name in STRUCTURE_METRICS:
        rows.append((metric_name, _format_metric_value(structure.get(metric_name))))
    table = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
    return table.to_string()


def _summary_rank_key(summary: dict[str, Any]) -> tuple[float, ...]:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    structure = summary.get("structure_metrics", {}) if isinstance(summary, dict) else {}
    annual_ls = _safe_float(metrics.get("annual_long_short_ret_sz")) or float("-inf")
    sharpe_ls = _safe_float(metrics.get("sharp_long_short_sz")) or float("-inf")
    ic = _safe_float(metrics.get("ic_wcor_sz")) or float("-inf")
    ir = _safe_float(metrics.get("ir_wcor_sz")) or float("-inf")
    group_mono = _safe_float(structure.get("group_monotonicity")) or float("-inf")
    ls_nav_return = _safe_float(structure.get("ls_nav_return")) or float("-inf")
    ls_positive_ratio = _safe_float(structure.get("ls_positive_ratio")) or float("-inf")
    coverage = _safe_float(metrics.get("coverage")) or float("-inf")
    liquidity = _safe_float(metrics.get("stock_mean_flu_sz")) or float("-inf")
    stock_num = _safe_float(metrics.get("stock_mean_stock_num_sz")) or float("-inf")
    return (
        annual_ls,
        sharpe_ls,
        ic,
        ir,
        group_mono,
        ls_nav_return,
        ls_positive_ratio,
        coverage,
        liquidity,
        stock_num,
    )


def _summary_metrics_table(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    rows = []
    for metric_name in PRIMARY_METRICS:
        rows.append((metric_name, _format_metric_value(metrics.get(metric_name))))
    table = pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
    return table.to_string()


def _build_comparison_table(
    current_summaries: list[dict[str, Any]],
    sota_summary: dict[str, Any] | None,
) -> str:
    rows: list[dict[str, Any]] = []
    for idx, summary in enumerate(current_summaries, start=1):
        metrics = summary.get("metrics", {})
        rows.append(
            {
                "rank": idx,
                "factor": summary.get("source_factor_name") or summary.get("factor_name", f"factor_{idx}"),
                "runtime_factor": summary.get("factor_name", ""),
                "annual_ls": _safe_float(metrics.get("annual_long_short_ret_sz")),
                "sharpe_ls": _safe_float(metrics.get("sharp_long_short_sz")),
                "ic": _safe_float(metrics.get("ic_wcor_sz")),
                "ir": _safe_float(metrics.get("ir_wcor_sz")),
                "mono": _safe_float((summary.get("structure_metrics") or {}).get("group_monotonicity")),
                "ls_nav": _safe_float((summary.get("structure_metrics") or {}).get("ls_nav_return")),
                "coverage": _safe_float(metrics.get("coverage")),
                "stock_num": _safe_float(metrics.get("stock_mean_stock_num_sz")),
            }
        )

    if sota_summary is not None:
        metrics = sota_summary.get("metrics", {})
        rows.append(
            {
                "rank": "SOTA",
                "factor": sota_summary.get("source_factor_name") or sota_summary.get("factor_name", "historical_best"),
                "runtime_factor": sota_summary.get("factor_name", ""),
                "annual_ls": _safe_float(metrics.get("annual_long_short_ret_sz")),
                "sharpe_ls": _safe_float(metrics.get("sharp_long_short_sz")),
                "ic": _safe_float(metrics.get("ic_wcor_sz")),
                "ir": _safe_float(metrics.get("ir_wcor_sz")),
                "mono": _safe_float((sota_summary.get("structure_metrics") or {}).get("group_monotonicity")),
                "ls_nav": _safe_float((sota_summary.get("structure_metrics") or {}).get("ls_nav_return")),
                "coverage": _safe_float(metrics.get("coverage")),
                "stock_num": _safe_float(metrics.get("stock_mean_stock_num_sz")),
            }
        )

    if not rows:
        return "No intraday factor summaries were available."
    return pd.DataFrame(rows).to_string(index=False)


def _extract_trace_sota(trace: Trace) -> tuple[dict[str, Any] | None, str | None]:
    best_summary = None
    best_hypothesis = None

    for hist_hypothesis, hist_exp, hist_feedback in trace.hist:
        if not getattr(hist_feedback, "decision", False):
            continue
        result = getattr(hist_exp, "result", {}) or {}
        summaries = result.get("factor_summaries", []) if isinstance(result, dict) else []
        for summary in summaries:
            summary.setdefault("structure_metrics", _compute_structure_metrics(summary))
            rank_key = _summary_rank_key(summary)
            best_rank_key = _summary_rank_key(best_summary) if best_summary is not None else tuple([float("-inf")] * len(rank_key))
            if rank_key > best_rank_key:
                best_summary = summary
                best_hypothesis = hist_hypothesis.hypothesis

    return best_summary, best_hypothesis


def _process_intraday_results(
    current_summaries: list[dict[str, Any]],
    sota_summary: dict[str, Any] | None,
    sota_hypothesis: str | None,
) -> str:
    if not current_summaries:
        return "No intraday factor summaries were available for analysis."

    best_current = current_summaries[0]
    sections = [
        "Current round ranking:",
        _build_comparison_table(current_summaries, sota_summary),
        "",
        f"Best current factor: {best_current.get('source_factor_name') or best_current.get('factor_name', 'unknown')}",
        _summary_metrics_table(best_current),
        "",
        "Best current structure diagnostics:",
        _summary_structure_table(best_current),
    ]

    if sota_summary is not None:
        sections.extend(
            [
                "",
                f"Historical SOTA factor: {sota_summary.get('source_factor_name') or sota_summary.get('factor_name', 'unknown')}",
                _summary_metrics_table(sota_summary),
                "Historical SOTA structure diagnostics:",
                _summary_structure_table(sota_summary),
                f"Historical SOTA hypothesis: {sota_hypothesis or 'N/A'}",
            ]
        )
    else:
        sections.extend(["", "Historical SOTA factor: None available yet."])

    return "\n".join(sections)


def _build_complexity_feedback(task_detail: dict[str, Any]) -> str | None:
    factor_expr = task_detail.get("factor_expression", "")
    if not factor_expr:
        return None

    warnings = []
    try:
        from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
        from quantaalpha.factors.coder.factor_ast import calculate_symbol_length, count_base_features

        symbol_length = calculate_symbol_length(factor_expr)
        symbol_length_threshold = getattr(FACTOR_COSTEER_SETTINGS, "symbol_length_threshold", 300)
        if symbol_length > symbol_length_threshold:
            warnings.append(
                f"Symbol Length (SL) Check Failed: symbol length {symbol_length} exceeds threshold {symbol_length_threshold}."
            )

        num_base_features = count_base_features(factor_expr)
        base_features_threshold = getattr(FACTOR_COSTEER_SETTINGS, "base_features_threshold", 6)
        if num_base_features > base_features_threshold:
            warnings.append(
                f"Base Features Count (ER) Check Failed: base feature count {num_base_features} exceeds threshold {base_features_threshold}."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to compute complexity feedback for intraday factor: {exc}")
        return None

    if warnings:
        return "\n".join(warnings)
    return None


def _enrich_task_details(exp: Experiment, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_details = [task.get_task_information_and_implementation_result() for task in exp.sub_tasks]
    summary_by_source_name = {
        summary.get("source_factor_name") or summary.get("factor_name"): summary
        for summary in summaries
    }
    for idx, task_detail in enumerate(task_details):
        task_name = task_detail.get("factor_name")
        summary = summary_by_source_name.get(task_name, summaries[idx] if idx < len(summaries) else {})
        task_detail["runtime_factor_name"] = summary.get("factor_name", "")
        task_detail["source_factor_name"] = summary.get("source_factor_name", task_detail.get("factor_name", ""))
        task_detail["intraday_metrics"] = summary.get("metrics", {})
        task_detail["intraday_metrics_table"] = _summary_metrics_table(summary) if summary else "N/A"
        task_detail["intraday_structure_table"] = _summary_structure_table(summary) if summary else "N/A"
        task_detail["analysis_output_dir"] = summary.get("result_paths", {}).get("output_dir", "")
        complexity_feedback = _build_complexity_feedback(task_detail)
        if complexity_feedback:
            task_detail["complexity_feedback"] = complexity_feedback
    return task_details


def _should_replace_best(current_summary: dict[str, Any], sota_summary: dict[str, Any] | None) -> bool:
    current_metrics = current_summary.get("metrics", {})
    current_structure = current_summary.get("structure_metrics", {})
    current_primary = (
        (_safe_float(current_metrics.get("annual_long_short_ret_sz")) or float("-inf")) > 0,
        (_safe_float(current_metrics.get("sharp_long_short_sz")) or float("-inf")) > 0,
        (_safe_float(current_metrics.get("ic_wcor_sz")) or float("-inf")) > 0,
        (_safe_float(current_metrics.get("ir_wcor_sz")) or float("-inf")) > 0,
        (_safe_float(current_structure.get("group_monotonicity")) or float("-inf")) > 0,
    )
    if not all(current_primary[:3]):
        return False
    if sota_summary is None:
        return True
    return _summary_rank_key(current_summary) > _summary_rank_key(sota_summary)


def _fallback_feedback(
    hypothesis: Hypothesis,
    current_summaries: list[dict[str, Any]],
    sota_summary: dict[str, Any] | None,
) -> HypothesisFeedback:
    if not current_summaries:
        return HypothesisFeedback(
            observations="No intraday factor summaries were produced, so the hypothesis could not be validated.",
            hypothesis_evaluation=f"Target hypothesis: {hypothesis.hypothesis}",
            new_hypothesis="Refocus on simpler intraday price-volume reversals with higher coverage and stable cross-sectional sample counts.",
            reason="The evaluation stage returned no usable summaries, so the next iteration should prioritize robustness and observability first.",
            decision=False,
        )

    best_current = current_summaries[0]
    current_metrics = best_current.get("metrics", {})
    current_name = best_current.get("source_factor_name") or best_current.get("factor_name", "unknown")
    long_short = _safe_float(current_metrics.get("annual_long_short_ret_sz")) or 0.0
    ic = _safe_float(current_metrics.get("ic_wcor_sz")) or 0.0
    sharpe = _safe_float(current_metrics.get("sharp_long_short_sz")) or 0.0

    observations = (
        f"Best current intraday factor is {current_name} with annual_long_short_ret_sz={long_short:.6f}, "
        f"sharp_long_short_sz={sharpe:.6f}, ic_wcor_sz={ic:.6f}."
    )

    if sota_summary is not None:
        sota_name = sota_summary.get("source_factor_name") or sota_summary.get("factor_name", "historical_best")
        observations += f" Historical best reference is {sota_name}."

    decision = _should_replace_best(best_current, sota_summary)
    evaluation = (
        "The hypothesis is supported only when the best candidate shows positive long-short return, positive "
        "long-short Sharpe, positive IC, and acceptable group/curve structure; otherwise the construction still needs refinement."
    )
    new_hypothesis = (
        "Within the same short-window reversal theme, emphasize high-coverage price-volume exhaustion signals "
        "that keep positive long-short return and IC while simplifying the expression and stabilizing sample breadth."
    )
    reason = (
        "Fallback feedback was used because LLM feedback could not be parsed. The next step should keep the core "
        "reversal idea but refine normalization, ranking, and window choice using the strongest current candidate."
    )

    return HypothesisFeedback(
        observations=observations,
        hypothesis_evaluation=evaluation,
        new_hypothesis=new_hypothesis,
        reason=reason,
        decision=decision,
    )


class IntradayHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    """Full intraday feedback adapter with SOTA comparison and actionable iteration guidance."""

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        logger.info("Generating intraday feedback...")

        result = getattr(exp, "result", {}) or {}
        summaries = result.get("factor_summaries", []) if isinstance(result, dict) else []
        for summary in summaries:
            summary["structure_metrics"] = _compute_structure_metrics(summary)
        current_summaries = sorted(summaries, key=_summary_rank_key, reverse=True)
        sota_summary, sota_hypothesis = _extract_trace_sota(trace)
        task_details = _enrich_task_details(exp, current_summaries)
        combined_result = _process_intraday_results(current_summaries, sota_summary, sota_hypothesis)

        if not current_summaries:
            return _fallback_feedback(hypothesis, current_summaries, sota_summary)

        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(INTRADAY_FEEDBACK_PROMPTS["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(INTRADAY_FEEDBACK_PROMPTS["factor_feedback_generation"]["user"])
            .render(
                hypothesis_text=hypothesis.hypothesis,
                task_details=task_details,
                combined_result=combined_result,
            )
        )

        response_json = None
        last_error = None
        for attempt in range(MAX_JSON_PARSE_RETRIES):
            try:
                response = APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=usr_prompt,
                    system_prompt=sys_prompt,
                    json_mode=True,
                )
                response_json = robust_json_parse(response)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    f"[Intraday] JSON parse failed (attempt {attempt + 1}/{MAX_JSON_PARSE_RETRIES}): {exc}"
                )
                if attempt < MAX_JSON_PARSE_RETRIES - 1:
                    logger.info("[Intraday] Re-requesting LLM feedback...")

        if response_json is None:
            logger.error(f"[Intraday] JSON parse still failed after {MAX_JSON_PARSE_RETRIES} attempts: {last_error}")
            return _fallback_feedback(hypothesis, current_summaries, sota_summary)

        return HypothesisFeedback(
            observations=response_json.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json.get("Replace Best Result", "no"))
            and _should_replace_best(current_summaries[0], sota_summary),
        )
