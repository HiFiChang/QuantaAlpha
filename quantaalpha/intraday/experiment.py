"""
Intraday scenario definitions for QuantaAlpha.

The current daily scenario is heavily tied to Qlib and next-day prediction.
This scenario narrows the task to intraday alpha discovery on user-provided
panel data and custom evaluation pipelines.
"""

from quantaalpha.core.scenario import Scenario
from quantaalpha.core.experiment import Task


class IntradayAlphaAgentScenario(Scenario):
    """Scenario description used by prompts and coder/evaluator components."""

    def __init__(self, use_local: bool = True, *args, **kwargs):
        self.use_local = use_local
        self._background = """
You are mining intraday equity alpha factors on bar data rather than daily
factors. The downstream evaluation focuses on short intraday slices, group
returns, IC/RankIC, monotonicity, and robustness. Rolling windows are bar
counts. Avoid future leakage, cross-slice leakage, and overly expensive
expressions.
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
        sections = [
            f"Background:\n{self.background}",
            f"Source Data:\n{self.source_data}",
            f"Interface:\n{self.interface}",
            f"Output Format:\n{self.output_format}",
            f"Simulator:\n{self.simulator}",
            f"Style:\n{self.rich_style_description}",
        ]
        return "\n\n".join(sections)
