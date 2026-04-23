"""
Intraday execution mode helpers.

The intraday pipeline currently supports two execution styles:

- ``local_panel``: read a prebuilt local ``intraday_pv.h5`` panel and compute
  factor values locally.
- ``ruogu_sql``: fetch raw bars from remote services at execution time and
  compute factor values without depending on a prebuilt local panel.
"""

from __future__ import annotations

import os


INTRADAY_EXECUTION_MODE_LOCAL_PANEL = "local_panel"
INTRADAY_EXECUTION_MODE_RUOGU_SQL = "ruogu_sql"
DEFAULT_INTRADAY_EXECUTION_MODE = INTRADAY_EXECUTION_MODE_LOCAL_PANEL

VALID_INTRADAY_EXECUTION_MODES = {
    INTRADAY_EXECUTION_MODE_LOCAL_PANEL,
    INTRADAY_EXECUTION_MODE_RUOGU_SQL,
}


def get_intraday_execution_mode() -> str:
    value = os.environ.get("INTRADAY_EXECUTION_MODE", DEFAULT_INTRADAY_EXECUTION_MODE).strip().lower()
    if value in VALID_INTRADAY_EXECUTION_MODES:
        return value
    return DEFAULT_INTRADAY_EXECUTION_MODE

