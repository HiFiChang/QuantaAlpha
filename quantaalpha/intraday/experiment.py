"""
Intraday scenario definitions for QuantaAlpha.

The current daily scenario is heavily tied to Qlib and next-day prediction.
This scenario narrows the task to intraday alpha discovery on user-provided
panel data and custom evaluation pipelines.
"""

from pathlib import Path

from quantaalpha.core.scenario import Scenario
from quantaalpha.core.experiment import Task
from quantaalpha.intraday.mode import (
    INTRADAY_EXECUTION_MODE_RUOGU_SQL,
    get_intraday_execution_mode,
)


RUOGU_SQL_REFERENCE = (Path(__file__).parent / "coder" / "ruogu_sql_reference.py").read_text(encoding="utf-8")


class IntradayAlphaAgentScenario(Scenario):
    """Scenario description used by prompts and coder/evaluator components."""

    def __init__(self, use_local: bool = True, *args, **kwargs):
        self.use_local = use_local
        self.execution_mode = get_intraday_execution_mode()
        self._background = """
You are mining intraday equity alpha factors on bar data rather than daily
factors. The downstream evaluation focuses on short intraday slices, group
returns, IC/RankIC, monotonicity, and robustness. Rolling windows are bar
counts. Avoid future leakage, cross-slice leakage, and overly expensive
expressions.
""".strip()
        if self.execution_mode == INTRADAY_EXECUTION_MODE_RUOGU_SQL:
            self._source_data_hypothesis = """
The implementation stage can fetch remote intraday bars at execution time, so
factor expression design should focus on the logical fields that are reliably
available after remote loading.

Confirmed base sources under `stock_base`:
- `m1`: minute bar OHLCV/amount
- `tk`: snapshot top-of-book depth and aggregated trading stats
- `zb`: trade-level amount flow with aggressor-side markers

Allowed logical fields and derived concepts for hypothesis / expression design:
- Bar price-volume:
- $open: bar open price
- $close: bar close price
- $high: bar high price
- $low: bar low price
- $volume: bar traded share volume
- $money: bar traded amount / turnover
- $vwap: bar volume-weighted average price, typically `$money / $volume`
- $return: bar close-to-close return
- Snapshot / liquidity:
  - bid1, ask1: best bid / ask prices
  - bidv1, askv1: best bid / ask queue sizes
  - total_volume: cumulative traded volume up to the snapshot
  - total_value: cumulative traded amount up to the snapshot
- Transaction / flow:
  - trade_price: per-trade execution price from `zb.price`
  - trade_volume: per-trade size from `zb.volume`
  - trade_flag: marker from `zb.trade_flag`
  - ctype: merged event category from `zb.ctype`
- Standard derived metrics that may be implemented from the whitelist above:
  - spread: `ask1 - bid1`
  - relative_spread: normalized top-of-book spread
  - depth_imbalance_1: top-level bid-vs-ask queue imbalance
  - active_buy_amount / active_sell_amount: `trade_price * trade_volume` grouped by `trade_flag`
  - raw_flow_diff: `active_buy_amount - active_sell_amount`
  - trade_imbalance_amount: normalized buy-vs-sell amount imbalance

Remote execution resources exist, but expression design should stay at the
factor-definition level. Do not assume any fields outside the whitelist above.
In particular, do not invent undeclared large-order / small-order / multi-level
order-book fields unless they can be computed from the allowed raw columns.
""".strip()
            self._source_data = f"""
The execution environment does not rely on a prebuilt local panel. Instead,
your implementation should fetch the required intraday bars from remote
services at execution time.

Reliable remote resources:
- ruogu Python package is available
- `rg.command_get_df(sql)` can execute SQL remotely
- `trade_cal` is available for trade-date expansion

Use a deliberately small schema. Prefer these three sources only:

1. `stock_base.m1`
- `code`: stock code
- `date_time`: bar timestamp
- `date`: trade date
- `time_int`: exchange time encoded as integer
- `open`: bar open price
- `close`: bar close price
- `high`: bar high price
- `low`: bar low price
- `volume`: traded share volume in the bar
- `amount`: traded turnover / traded amount in the bar

Logical mapping from bar data:
- $open  -> open
- $close -> close
- $high  -> high
- $low   -> low
- $volume -> volume
- $money -> amount
- $vwap  -> amount / volume
- $return -> compute in SQL with window functions when needed

2. `stock_base.tk` (snapshot / top-of-book only)
- `code`: stock code
- `date_time`: snapshot timestamp
- `date`: trade date
- `time_int`: exchange time encoded as integer
- `total_volume`: cumulative traded volume up to snapshot
- `total_value`: cumulative traded amount up to snapshot
- `bid1`: best bid price
- `ask1`: best ask price
- `bidv1`: best bid queue size
- `askv1`: best ask queue size

3. `stock_base.zb` (trade flow only)
- `code`: stock code
- `date`: trade date
- `date_time`: event timestamp
- `time_int`: exchange time encoded as integer
- `price`: transaction price
- `volume`: order quantity / transaction quantity
- `trade_flag`: trade status / inner-outer side marker. Official values:
  - `B`: Shanghai outer trade / active buy
  - `S`: Shanghai inner trade / active sell
  - `N`: Shanghai unknown
  - `F`: Shenzhen trade
  - `4`: Shenzhen cancel
- `ctype`: merged event category. Official values:
  - `1`: add limit order
  - `2`: add market order
  - `3`: add own-side-best order
  - `4`: cancel
  - `5`: trade

Allowed standard derived metrics from the whitelist above:
- `spread = ask1 - bid1`: raw top-of-book spread
- `relative_spread = (ask1 - bid1) / ((ask1 + bid1) / 2 + 1e-8)`: normalized spread
- `depth_imbalance_1 = (bidv1 - askv1) / (bidv1 + askv1 + 1e-8)`: best-level queue imbalance
- `active_buy_amount / active_sell_amount`: transaction amount grouped by `trade_flag`
- `raw_flow_diff = active_buy_amount - active_sell_amount`: raw aggressor-flow difference
- `trade_imbalance_amount = (active_buy_amount - active_sell_amount) / (active_buy_amount + active_sell_amount + 1e-8)`: normalized aggressor-flow imbalance

Hard schema rule:
- Use only the tables and columns explicitly listed above.
- Do not use undeclared fields such as multi-level depth, order ids, large/small
  order buckets, or shorthand aliases that are not listed above.

If you need `$return`, compute it in SQL when possible by using window
functions over `(code order by date_time)`. Python should only do orchestration
and output formatting unless SQL support is clearly impossible.

Reference pattern adapted from the user's ruogu pipeline:

```python
import pandas as pd
import ruogu as rg
from ruogu import trade_cal

analysis_start = os.environ["INTRADAY_ANALYSIS_START"]
analysis_end = os.environ["INTRADAY_ANALYSIS_END"]
calc_start = trade_cal.pre_date(analysis_start)
calc_dates = trade_cal.range_date_list(calc_start, analysis_end, check_open=False)

frames = []
for date in calc_dates:
    date_str = str(date).split()[0]
    sql = f\"\"\"
    select
        toString(date_time) as datetime,
        code as instrument,
        factor_value
    from stock_base.m1
    where date = toDate('{{date_str}}')
      and time_int >= Tit('09:30:00')
      and time_int <= Tit('15:00:00')
    order by date_time, code
    \"\"\"
    df = rg.command_get_df(sql)
    frames.append(df)

result = pd.concat(frames, axis=0)
result["datetime"] = pd.to_datetime(result["datetime"])
result["instrument"] = result["instrument"].astype(str).str.zfill(6)
result = result.set_index(["datetime", "instrument"]).sort_index()["factor_value"]
result.to_hdf("result.h5", key="data")
```

Important adaptation:
- Use the pipeline style to fetch remote data and compute factor values.
- But do NOT call `Factor.create(...)` or `upload_df(...)` during implementation.
- In QuantaAlpha, implementation should stop after writing `result.h5`.
- The SQL query should emit `factor_value` directly. Do not fetch a raw panel and
  then compute the main rolling/ranking/zscore logic in pandas unless that is
  impossible to express in SQL.
- Python should be limited to: date expansion, per-date query dispatch,
  concatenation, type normalization, MultiIndex alignment, and writing
  `result.h5`.

A stronger reference skeleton is provided below. Follow its structure unless you
have a clear reason not to:

```python
{RUOGU_SQL_REFERENCE}
```
""".strip()
            self._interface = """
Implement factors as executable Python code that:
1. determines the required date range from environment variables such as
   `INTRADAY_ANALYSIS_START` and `INTRADAY_ANALYSIS_END`
2. expands the start date to include the previous trade date
3. queries remote intraday bars with SQL
4. computes the factor mainly inside SQL and returns a `factor_value` column
5. writes the final factor values to `result.h5`

The final `result.h5` must contain a pandas Series or a single-column DataFrame
indexed by MultiIndex `(datetime, instrument)`, so the downstream runner can
stay compatible.

Do not create or upload the ruogu factor during factor implementation; runner
will still handle upload and evaluation later.

Recommended implementation shape:
- import `ruogu as rg`, `pandas as pd`, and `trade_cal`
- expand `INTRADAY_ANALYSIS_START` to its previous trade date
- fetch each trade day with SQL that already emits `factor_value`
- concatenate all days
- convert to a MultiIndex `(datetime, instrument)`
- save to `result.h5`

Hard constraint:
- In `ruogu_sql` mode, do not pull a full OHLCV panel back to Python and then
  perform the main factor computation with pandas rolling / rank / zscore /
  groupby logic. Those calculations should be pushed into SQL by default.
""".strip()
            self._output_format = """
Return factor values as a pandas Series or a single-column DataFrame indexed by
(datetime, instrument). The values should be aligned to the fetched remote bar
panel and stored to `result.h5` using key `data`.
""".strip()
            self._simulator = """
The final evaluation is not Qlib. Generated factor values will be uploaded by
the intraday runner into a custom factor store and then evaluated with
MultiSecAna. Your implementation stage should only produce correct factor
values and store them in `result.h5`.
""".strip()
            self._rich_style_description = """
Prefer concise, interpretable intraday price-volume implementations. Keep the
code robust: query only the needed date range, avoid scanning unnecessary data,
and ensure the final output keeps the exact `(datetime, instrument)` alignment
required by the downstream runner. Prefer SQL window functions and SQL-side
factor construction over local pandas transforms.
""".strip()
        else:
            self._source_data_hypothesis = """
The execution environment provides `intraday_pv.h5`, a pandas panel indexed by
`(datetime, instrument)`. In the current stage, assume the reliable base fields
are only:
- $open
- $close
- $high
- $low
- $volume
- $vwap
- $money
- $return

Do not assume order-book, Level-2, bid/ask ladder, or tick-only fields unless
they are explicitly listed in the data.
""".strip()
            self._source_data = """
The execution environment provides `intraday_pv.h5`, a pandas panel indexed by
`(datetime, instrument)`. In the current stage, assume the reliable base fields
are only:
- $open
- $close
- $high
- $low
- $volume
- $vwap
- $money
- $return

Do not assume order-book, Level-2, bid/ask ladder, or tick-only fields unless
they are explicitly listed in the data.
""".strip()
            self._interface = """
Implement factors as executable Python code that reads `intraday_pv.h5`,
computes one factor expression, and writes the result to `result.h5`.
The output must align to the input panel index and use MultiIndex
(datetime, instrument).
""".strip()
            self._output_format = """
Return factor values as a pandas Series or a single-column DataFrame indexed by
(datetime, instrument). The factor must be safe for later upload into a custom
intraday evaluation pipeline.
""".strip()
            self._simulator = """
The final evaluation is not Qlib. Generated factor values will be passed to a
custom intraday runner that uploads them into a local factor store and then
evaluates them with an intraday analytics engine.
""".strip()
            self._rich_style_description = """
Prefer concise, interpretable intraday price-volume expressions built from the
available OHLCV-style fields. Start from short windows and simple
cross-sectional or time-series transforms instead of rich microstructure
stories that require unavailable inputs.
""".strip()

    @property
    def background(self) -> str:
        return self._background

    def get_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data

    def get_hypothesis_source_data_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        return self._source_data_hypothesis

    def get_planning_context_desc(self, task: Task | None = None) -> str:  # noqa: ARG002
        if self.execution_mode == INTRADAY_EXECUTION_MODE_RUOGU_SQL:
            return """
Current intraday planning must stay within the confirmed `stock_base` sources:
- `m1`: minute OHLCV / amount bars
- `tk`: best bid / ask prices and best bid / ask queue sizes
- `zb`: trade price, trade volume, `trade_flag`, and `ctype`

Directions may use:
- intraday price-volume dynamics
- order-flow imbalance and signed trade flow
- top-of-book depth imbalance, spread, relative spread
- short-window reversal and liquidity replenishment

Do not propose directions that require external macro or cross-asset datasets
such as index futures, FX, commodities, news, sentiment feeds, or any tables
outside the whitelist above.
""".strip()

        return """
Current intraday planning must stay within the local panel fields only:
- $open
- $close
- $high
- $low
- $volume
- $vwap
- $money
- $return

Do not propose directions that rely on order-book, Level-2, tick, transaction,
macro, cross-asset, news, or sentiment data that are not present in this local
panel.
""".strip()

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def rich_style_description(self) -> str:
        return self._rich_style_description

    def get_scenario_all_desc(
        self,
        task: Task | None = None,  # noqa: ARG002
        filtered_tag: str | None = None,  # noqa: ARG002
        simple_background: bool | None = None,  # noqa: ARG002
    ) -> str:
        if filtered_tag == "hypothesis_and_experiment":
            sections = [
                f"Background:\n{self.background}",
                f"Source Data:\n{self.get_hypothesis_source_data_desc(task)}",
                f"Output Format:\n{self.output_format}",
                f"Style:\n{self.rich_style_description}",
            ]
        else:
            sections = [
                f"Background:\n{self.background}",
                f"Source Data:\n{self.source_data}",
                f"Interface:\n{self.interface}",
                f"Output Format:\n{self.output_format}",
                f"Simulator:\n{self.simulator}",
                f"Style:\n{self.rich_style_description}",
            ]
        return "\n\n".join(sections)
