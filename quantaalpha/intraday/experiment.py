"""
Intraday scenario definitions for QuantaAlpha.

The current daily scenario is heavily tied to Qlib and next-day prediction.
This scenario narrows the task to intraday alpha discovery on user-provided
panel data and custom evaluation pipelines.
"""

from quantaalpha.core.scenario import Scenario
from quantaalpha.core.experiment import Task
from quantaalpha.intraday.mode import (
    INTRADAY_EXECUTION_MODE_RUOGU_SQL,
    get_intraday_execution_mode,
)


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
The intraday remote path is expression-first. The model should design the
factor idea and write a compact factor DSL expression; the backend handles data
loading, alignment, output formatting, and evaluation.

Supported semantic source families:
- `bar`: minute price-volume behavior
- `order_book`: minute-aligned order-book liquidity

Supported semantic variables:
- Price-volume: $open, $close, $high, $low, $volume, $money, $vwap, $return
- Order-book liquidity: $bid1...$bid10, $ask1...$ask10, $bidv1...$bidv10,
  $askv1...$askv10, $last, $ztprice, $dtprice, $num_trades, $total_volume,
  $total_value, $spread, $relative_spread, $depth_imbalance_1

Design at the factor-definition level. Do not mention backend query languages,
Python, table names, joins, event filters, upload steps, or implementation
details. Do not invent large-order buckets, order IDs, external datasets, or
other undeclared microstructure semantics.
""".strip()
            self._source_data = f"""
The execution environment is hidden behind an intraday factor DSL. The model
does not need to know how data is queried or aligned. Return only factor
definitions expressed through the supported semantic variables and operators.

Semantic source families:
- `bar`: minute price-volume behavior
- `order_book`: minute-aligned liquidity and queue state

Semantic variables:
- Price-volume: $open, $close, $high, $low, $volume, $money, $vwap, $return
- Order-book liquidity: $bid1...$bid10, $ask1...$ask10, $bidv1...$bidv10,
  $askv1...$askv10, $last, $ztprice, $dtprice, $num_trades, $total_volume,
  $total_value, $spread, $relative_spread, $depth_imbalance_1

Use only these semantic variables in the executable expression. The backend
will translate them to the required remote data operations.
""".strip()
            self._interface = """
Define factors as intraday DSL expressions. The model should provide the
economic mechanism, semantic variables, bar-count windows, and one compact
`expression_summary`. The backend generates executable code and performs data
access, alignment, output formatting, upload, and evaluation.

Do not write Python code, backend query code, table names, joins, database
filters, or backend orchestration details.
""".strip()
            self._output_format = """
Return a factor task whose `expression_summary` is a compact DSL expression.
The backend will produce aligned factor values for evaluation.
""".strip()
            self._simulator = """
The final evaluation is handled by the intraday backend. The model only needs
to preserve the factor mechanism and provide an executable DSL expression.
""".strip()
            self._rich_style_description = """
Prefer concise, interpretable intraday factor expressions. Use semantic
price-volume and order-book liquidity variables directly; leave data access
and optimization to the backend.
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
Current intraday planning must stay within the supported semantic DSL:
- `bar`: minute price-volume behavior
- `order_book`: minute-aligned liquidity and queue state

Directions may use:
- intraday price-volume dynamics
- spread, relative spread, queue imbalance, and liquidity replenishment
- short-window reversal and liquidity replenishment

Do not propose directions that require external macro or cross-asset datasets
such as index futures, FX, commodities, news, sentiment feeds, undeclared
variables, or backend-specific data sources.
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
