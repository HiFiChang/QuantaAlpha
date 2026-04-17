"""
Intraday extensions for QuantaAlpha.

This package adapts the factor-mining workflow to:
- intraday panel data
- custom execution templates
- custom evaluation runners
"""

from quantaalpha.intraday.settings import INTRADAY_FACTOR_PROP_SETTING

__all__ = ["INTRADAY_FACTOR_PROP_SETTING"]
