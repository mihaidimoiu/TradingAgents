"""Per-indicator fault tolerance in ``get_indicators``: one failed indicator
must not drop the rest of the batch or abort the analyst turn.

For core categories (`technical_indicators`), the router re-raises terminal
errors when every configured vendor fails (e.g. a transient network outage),
which is not a `ValueError`. Catching only `ValueError` let `ToolNode` re-raise
and abort the entire analyst turn; returning the failure as text instead matches
#1593.
"""

from __future__ import annotations

from unittest import mock

import pytest

from tradingagents.agents import tools

TRADE_DATE = "2026-08-14"


@pytest.mark.unit
def test_a_failed_indicator_does_not_sink_its_batch():
    with mock.patch.object(tools, "route_to_vendor", side_effect=_flaky_router):
        out = tools.get_indicators.func(
            symbol="AAPL", indicator="rsi, macd", curr_date=TRADE_DATE, look_back_days=5, trade_date=TRADE_DATE,
        )
    assert "rsi: network down" in out
    assert "macd-data" in out


@pytest.mark.unit
def test_a_failed_indicator_returns_text_instead_of_raising():
    with mock.patch.object(tools, "route_to_vendor", side_effect=RuntimeError("network down")):
        out = tools.get_indicators.func(
            symbol="AAPL", indicator="rsi", curr_date=TRADE_DATE, look_back_days=5, trade_date=TRADE_DATE,
        )
    assert "network down" in out


@pytest.mark.unit
def test_unknown_indicator_still_reports_the_vocabulary():
    with mock.patch.object(tools, "route_to_vendor", side_effect=_flaky_router):
        out = tools.get_indicators.func(
            symbol="AAPL", indicator="rsii", curr_date=TRADE_DATE, look_back_days=5, trade_date=TRADE_DATE,
        )
    assert "not supported" in out


def _flaky_router(method, symbol, indicator, curr_date, look_back_days):
    if indicator == "rsi":
        raise RuntimeError("network down")
    if indicator == "rsii":
        raise ValueError("Indicator rsii is not supported. Please choose from: [...]")
    return "macd-data"
