"""
Intraday coder selector.

The intraday pipeline supports two execution styles:

- local_panel: keep the current expression-template path that reads
  ``intraday_pv.h5`` locally.
- ruogu_sql: switch to the full code-generation path so the agent can implement
  a restricted expression, then render a deterministic template that compiles
  the expression into ClickHouse SQL and writes ``result.h5`` locally.
"""

from __future__ import annotations

from quantaalpha.factors.qlib_coder import QlibFactorParser
from quantaalpha.intraday.mode import (
    INTRADAY_EXECUTION_MODE_LOCAL_PANEL,
    INTRADAY_EXECUTION_MODE_RUOGU_SQL,
    get_intraday_execution_mode,
)


class IntradayFactorCoder:
    def __new__(cls, scen, *args, **kwargs):
        mode = get_intraday_execution_mode()
        if mode == INTRADAY_EXECUTION_MODE_RUOGU_SQL:
            return QlibFactorParser(scen, *args, **kwargs)
        if mode == INTRADAY_EXECUTION_MODE_LOCAL_PANEL:
            return QlibFactorParser(scen, *args, **kwargs)
        return QlibFactorParser(scen, *args, **kwargs)
