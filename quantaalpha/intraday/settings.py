"""
Intraday pipeline settings.

These settings mirror the daily factor-mining settings, but point the loop to
intraday-specific scenario, runner, and feedback components.
"""

from quantaalpha.pipeline.settings import BasePropSetting
from quantaalpha.core.conf import ExtendedSettingsConfigDict


class IntradayFactorBasePropSetting(BasePropSetting):
    """Main experiment setting for intraday factor mining."""

    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.intraday.experiment.IntradayAlphaAgentScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.AlphaAgentHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.AlphaAgentHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.qlib_coder.QlibFactorParser"
    runner: str = "quantaalpha.intraday.runner.IntradayFactorRunner"
    summarizer: str = "quantaalpha.intraday.feedback.IntradayHypothesisExperiment2Feedback"
    evolving_n: int = 5


INTRADAY_FACTOR_PROP_SETTING = IntradayFactorBasePropSetting()
