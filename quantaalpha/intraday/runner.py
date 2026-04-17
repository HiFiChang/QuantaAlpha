"""
Intraday factor runner.

This runner replaces the default Qlib-based backtest stage with a bridge into a
custom intraday evaluation pipeline.
"""

from __future__ import annotations

import os
import time
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from quantaalpha.core.developer import Developer
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.log import logger
from quantaalpha.factors.experiment import QlibFactorExperiment


class IntradayFactorRunner(Developer[QlibFactorExperiment]):
    """
    Evaluate generated factors with an external intraday framework.

    Expected workflow:
    1. Read each workspace `result.h5`
    2. Convert it into the user's factor storage format
    3. Invoke the local intraday evaluator (e.g. ruogu + MultiSecAna)
    4. Attach normalized metrics back onto `exp.result`

    This implementation executes the full external evaluation path.
    """

    def __init__(self, scen, *args, **kwargs):
        super().__init__(scen)
        self.default_factor_prefix = os.environ.get("INTRADAY_FACTOR_PREFIX", "qa_intra")
        self.upload_chunk_size = int(os.environ.get("INTRADAY_UPLOAD_CHUNK_SIZE", "500"))

    def _build_runtime_factor_name(self, factor_name: str) -> str:
        prefix = self.default_factor_prefix
        clean_name = re.sub(r"[^A-Za-z0-9_]+", "_", factor_name).strip("_") or "factor"
        ts_suffix = str(int(time.time()))
        max_total_len = 30
        reserved = len(prefix) + len(ts_suffix) + 2  # two underscores
        available = max(1, max_total_len - reserved)
        return f"{prefix}_{clean_name[:available]}_{ts_suffix}"

    def _load_workspace_result(self, workspace) -> pd.Series | None:
        result_path = Path(workspace.workspace_path) / "result.h5"
        if not result_path.exists():
            logger.warning(f"result.h5 not found: {result_path}")
            return None

        result = pd.read_hdf(result_path)
        if isinstance(result, pd.DataFrame):
            if result.shape[1] == 1:
                result = result.iloc[:, 0]
            else:
                result = result.iloc[:, 0]

        if not isinstance(result.index, pd.MultiIndex) or "datetime" not in result.index.names:
            logger.warning(f"Unexpected intraday factor index format: {result.index.names}")
            return None

        if "instrument" not in result.index.names:
            logger.warning(f"Unexpected intraday factor index format: {result.index.names}")
            return None

        return result.astype(float)

    def _series_to_upload_df(self, series: pd.Series, factor_name: str) -> pd.DataFrame:
        df = series.rename("factor_value").reset_index()
        rename_map = {}
        if "datetime" in df.columns:
            rename_map["datetime"] = "date_time"
        if "instrument" in df.columns:
            rename_map["instrument"] = "code"
        df = df.rename(columns=rename_map)
        if "date_time" not in df.columns or "code" not in df.columns:
            raise ValueError(f"Upload dataframe missing required columns: {df.columns.tolist()}")
        df["date"] = pd.to_datetime(df["date_time"]).dt.normalize()
        df["factor_name"] = factor_name
        return df[["date_time", "code", "factor_value", "date", "factor_name"]]

    def _evaluate_uploaded_factor(
        self,
        factor_name: str,
        upload_df: pd.DataFrame,
        workspace_path: Path,
    ) -> dict[str, Any]:
        """
        Hook for the user's local intraday evaluation framework.

        The full ruogu/MultiSecAna integration is intentionally isolated here so
        the rest of the QuantaAlpha loop can stay unchanged.
        """
        coverage = float(upload_df["factor_value"].notna().mean()) if len(upload_df) > 0 else 0.0

        import ruogu as rg
        from ruogu import Factor
        from ruogu.analysis_v2.analysis_config import configure

        import sys
        fanalysis_root = os.environ.get("INTRADAY_FANALYSIS_ROOT", "")
        if fanalysis_root:
            fanalysis_path = Path(fanalysis_root).resolve()
            candidate_paths = []
            if fanalysis_path.name == "fanalysis_intra":
                candidate_paths.extend([fanalysis_path.parent, fanalysis_path])
            else:
                candidate_paths.extend(
                    [
                        fanalysis_path,
                        fanalysis_path / "fanalysis_intra",
                    ]
                )
            for candidate in candidate_paths:
                if not candidate.exists():
                    continue
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.append(candidate_str)
        from fanalysis_intra.MultiSecAna import MultiSecAna

        token = os.environ.get("RUOGU_TOKEN")
        if token:
            rg.set_token(token)

        analysis_start = os.environ.get("INTRADAY_ANALYSIS_START")
        analysis_end = os.environ.get("INTRADAY_ANALYSIS_END")
        slice_start = os.environ.get("INTRADAY_SLICE_START")
        slice_end = os.environ.get("INTRADAY_SLICE_END")
        config_name = os.environ.get("INTRADAY_CONFIG_NAME", "config")
        group_num = int(os.environ.get("INTRADAY_GROUP_NUM", "10"))

        if not all([analysis_start, analysis_end, slice_start, slice_end]):
            raise RuntimeError(
                "Missing required intraday runner env vars: "
                "INTRADAY_ANALYSIS_START/END and INTRADAY_SLICE_START/END"
            )

        category = os.environ.get("INTRADAY_FACTOR_CATEGORY", "日内")
        description = os.environ.get("INTRADAY_FACTOR_DESCRIPTION", "QuantaAlpha intraday factor")
        runtime_factor_name = self._build_runtime_factor_name(factor_name)

        factor = Factor(runtime_factor_name)
        factor.create(category, description)

        for start_idx in range(0, len(upload_df), self.upload_chunk_size):
            part = upload_df.iloc[start_idx:start_idx + self.upload_chunk_size].copy()
            factor.upload_df(part.drop(columns=["factor_name"]))

        conf = configure(config_name).set_slicing_date(
            analysis_start,
            analysis_end,
            slice_start,
            slice_end,
        )
        msa = MultiSecAna(fac=Factor(runtime_factor_name), n=group_num, config=conf)
        msa.submit_analysis_intraday()
        result_dfs, analysis_data = msa.get_analysis_data()

        output_root = Path(os.environ.get("DATA_RESULTS_DIR", "data/results")) / "intraday_eval_outputs"
        output_root.mkdir(parents=True, exist_ok=True)
        factor_output_dir = output_root / runtime_factor_name
        factor_output_dir.mkdir(parents=True, exist_ok=True)

        with (factor_output_dir / "analysis_data.json").open("w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=float)

        saved_paths = {}
        for name, df in result_dfs.items():
            csv_path = factor_output_dir / f"{name}.csv"
            df.to_csv(csv_path, encoding="utf-8")
            saved_paths[name] = str(csv_path)

        metrics = {
            "coverage": coverage,
        }
        if isinstance(analysis_data, dict):
            for key, value in analysis_data.items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    metrics[key] = float(value)

        return {
            "metrics": metrics,
            "analysis_data": analysis_data,
            "result_paths": {
                "workspace_path": str(workspace_path),
                "output_dir": str(factor_output_dir),
                **saved_paths,
            },
            "factor_name": runtime_factor_name,
            "status": "ok",
        }

    def develop(self, exp: QlibFactorExperiment, use_local: bool = True) -> QlibFactorExperiment:  # noqa: ARG002
        summaries: list[dict[str, Any]] = []

        for idx, workspace in enumerate(exp.sub_workspace_list):
            if workspace is None:
                continue

            series = self._load_workspace_result(workspace)
            if series is None or series.empty:
                continue

            factor_name = getattr(workspace.target_task, "factor_name", f"{self.default_factor_prefix}_{idx}")
            upload_df = self._series_to_upload_df(series, factor_name)
            summary = self._evaluate_uploaded_factor(
                factor_name=factor_name,
                upload_df=upload_df,
                workspace_path=Path(workspace.workspace_path),
            )
            summaries.append(summary)

        if not summaries:
            raise FactorEmptyError("No valid intraday factor outputs were available for evaluation.")

        metrics = {}
        for summary in summaries:
            factor_name = summary.get("factor_name", "unknown")
            for metric_name, metric_value in (summary.get("metrics") or {}).items():
                metrics[f"{factor_name}.{metric_name}"] = metric_value

        exp.result = {
            "metrics": metrics,
            "factor_summaries": summaries,
            "evaluation_mode": "intraday",
            "use_local": use_local,
            "evaluated_at": time.time(),
        }
        return exp
