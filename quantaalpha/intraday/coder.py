"""
Intraday coder selector.

The intraday pipeline supports two execution styles:

- local_panel: keep the current expression-template path that reads
  ``intraday_pv.h5`` locally.
- ruogu_sql: switch to the full code-generation path so the agent can implement
  remote-query / remote-compute code directly, while still writing ``result.h5``
  locally for downstream compatibility.
"""

from __future__ import annotations

from quantaalpha.factors.qlib_coder import QlibFactorCoSTEER, QlibFactorParser
from quantaalpha.intraday.mode import (
    INTRADAY_EXECUTION_MODE_LOCAL_PANEL,
    INTRADAY_EXECUTION_MODE_RUOGU_SQL,
    get_intraday_execution_mode,
)


class IntradayFactorCoder:
    def __new__(cls, scen, *args, **kwargs):
        mode = get_intraday_execution_mode()
        if mode == INTRADAY_EXECUTION_MODE_RUOGU_SQL:
            return QlibFactorCoSTEER(scen, *args, **kwargs)
        if mode == INTRADAY_EXECUTION_MODE_LOCAL_PANEL:
            return QlibFactorParser(scen, *args, **kwargs)
        return QlibFactorParser(scen, *args, **kwargs)
