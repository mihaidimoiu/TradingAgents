"""Earnings context: keyed on the report date, the snapshot parts only for today. No network."""

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.yahoo.earnings as earnings

DATES = pd.DataFrame(
    {"EPS Estimate": [1.98, 1.89, 1.94], "Reported EPS": [None, 2.02, 2.01], "Surprise(%)": [None, 6.74, 3.46]},
    index=pd.DatetimeIndex(["2026-10-29 16:00", "2026-07-30 16:00", "2026-04-30 16:00"]).tz_localize("America/New_York"),
)


def test_a_report_published_after_the_day_is_not_in_its_history():
    """The quarter ended in June; its report came on July 30. On July 29 it is not known yet."""
    rows = earnings.surprises(DATES, pd.Timestamp("2026-07-29"))
    assert any("2026-04-30" in row for row in rows) and not any("2026-07-30" in row for row in rows)
    assert any("2026-07-30" in row and "6.74%" in row for row in earnings.surprises(DATES, pd.Timestamp("2026-07-30")))


@pytest.mark.parametrize(("day", "sessions", "near"), [("2026-10-23", 4, True), ("2026-09-26", 23, False)])
def test_the_next_report_is_flagged_inside_the_window(day, sessions, near):
    lines = earnings.next_report(DATES, pd.Timestamp(day))
    assert lines[0] == "- Date: 2026-10-29 (after the close)"
    assert lines[1] == f"- Trading days until: {sessions}"
    assert ("YES" in lines[2]) is near


def test_a_past_date_withholds_the_live_snapshot(monkeypatch):
    class Ticker:
        earnings_dates = DATES

        @property
        def earnings_estimate(self):
            raise AssertionError("a live consensus served into a past date")

    monkeypatch.setattr(earnings.yf, "Ticker", lambda symbol: Ticker())
    monkeypatch.setattr(earnings, "get_current_date", lambda: "2026-09-26")
    text = earnings.get_earnings_context("AAPL", "2026-08-01")
    assert "withheld" in text and "Next earnings report" not in text
    assert "2026-07-30" in text and "2026-10-29" not in text
