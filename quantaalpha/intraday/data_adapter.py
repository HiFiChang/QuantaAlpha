"""
Intraday data adapter for QuantaAlpha.

This module extracts real intraday market data from ClickHouse and converts it
into the standardized panel format required by the intraday factor template:

- HDF5 file name: `intraday_pv.h5`
- key: `data`
- index: MultiIndex(datetime, instrument)
- columns: dollar-prefixed feature names such as `$open`, `$close`, ...

The first production target is `stock_base.m1`, which provides the cleanest and
lightest-weight entry point for running the intraday workflow end-to-end.
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


@dataclass
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str


def _run_clickhouse_query(query: str, cfg: ClickHouseConfig) -> pd.DataFrame:
    cmd = [
        "clickhouse-client",
        "-h",
        cfg.host,
        "--port",
        str(cfg.port),
        "-u",
        cfg.user,
        "--password",
        cfg.password,
        "--format",
        "CSVWithNames",
        "--query",
        query,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or f"clickhouse-client exited with status {exc.returncode}"
        raise RuntimeError(f"ClickHouse query failed: {details}\nQuery:\n{query.strip()}") from exc
    if not result.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(result.stdout), dtype=str)


def _fetch_trade_dates(start_date: str, end_date: str, cfg: ClickHouseConfig) -> list[str]:
    query = f"""
    SELECT toString(date) AS trade_date
    FROM stock_base.trade_dates
    WHERE date BETWEEN toDate('{start_date}') AND toDate('{end_date}')
    ORDER BY date
    """
    df = _run_clickhouse_query(query, cfg)
    if df.empty:
        return []
    return df["trade_date"].astype(str).tolist()


def _fetch_m1_for_date(date_str: str, cfg: ClickHouseConfig) -> pd.DataFrame:
    query = f"""
    SELECT
        toString(date_time) AS datetime,
        code AS instrument,
        open AS open,
        close AS close,
        high AS high,
        low AS low,
        volume AS volume,
        amount AS money
    FROM stock_base.m1
    WHERE date = toDate('{date_str}')
      AND time_int >= Tit('09:30:00')
      AND time_int <= Tit('15:00:00')
    ORDER BY date_time, instrument
    """
    return _run_clickhouse_query(query, cfg)


def _normalize_m1_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df

    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str).str.zfill(6)

    numeric_cols = ["open", "close", "high", "low", "volume", "money"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(
        columns={
            "open": "$open",
            "close": "$close",
            "high": "$high",
            "low": "$low",
            "volume": "$volume",
            "money": "$money",
        }
    )

    # Derived fields commonly used in expression-based mining.
    safe_volume = df["$volume"].replace(0, np.nan).astype(float)
    df["$vwap"] = (df["$money"].astype(float) / safe_volume).astype(float)
    df = df.sort_values(["instrument", "datetime"])
    df["$return"] = df.groupby("instrument")["$close"].pct_change()

    df = df.set_index(["datetime", "instrument"]).sort_index()
    return df


def build_intraday_panel(
    start_date: str,
    end_date: str,
    cfg: ClickHouseConfig,
    output_path: str | Path,
) -> Path:
    trade_dates = _fetch_trade_dates(start_date, end_date, cfg)
    if not trade_dates:
        raise ValueError(f"No open trade dates found between {start_date} and {end_date}")

    frames: list[pd.DataFrame] = []
    for date_str in trade_dates:
        raw_df = _fetch_m1_for_date(date_str, cfg)
        if raw_df.empty:
            continue
        panel_df = _normalize_m1_panel(raw_df)
        if not panel_df.empty:
            frames.append(panel_df)

    if not frames:
        raise ValueError(f"No intraday data extracted between {start_date} and {end_date}")

    data = pd.concat(frames).sort_index()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_hdf(output_path, key="data")
    return output_path


def _default_output_path(debug: bool = False) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_folder = (
        project_root / "git_ignore_folder" / "factor_implementation_source_data_debug"
        if debug
        else project_root / "git_ignore_folder" / "factor_implementation_source_data"
    )
    return data_folder / "intraday_pv.h5"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract intraday m1 data into QuantaAlpha panel format")
    parser.add_argument("--start-date", required=True, help="inclusive start date, e.g. 2024-09-02")
    parser.add_argument("--end-date", required=True, help="inclusive end date, e.g. 2024-09-06")
    parser.add_argument("--host", default=os.environ.get("CLICKHOUSE_HOST", ""))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLICKHOUSE_PORT", "9000")))
    parser.add_argument("--user", default=os.environ.get("CLICKHOUSE_USER", ""))
    parser.add_argument("--password", default=os.environ.get("CLICKHOUSE_PASSWORD", ""))
    parser.add_argument("--output", default=None, help="target HDF5 path; defaults to intraday_pv.h5 in QuantaAlpha data folder")
    parser.add_argument("--debug-output", action="store_true", help="write to factor_implementation_source_data_debug by default")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    missing = [name for name, value in {
        "host": args.host,
        "user": args.user,
        "password": args.password,
    }.items() if not value]
    if missing:
        raise SystemExit(f"Missing ClickHouse arguments: {', '.join(missing)}")

    cfg = ClickHouseConfig(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
    )
    output_path = Path(args.output) if args.output else _default_output_path(debug=args.debug_output)
    saved_path = build_intraday_panel(
        start_date=args.start_date,
        end_date=args.end_date,
        cfg=cfg,
        output_path=output_path,
    )
    print(f"INTRADAY_PANEL_SAVED: {saved_path}")


if __name__ == "__main__":
    main()
