"""An ETF as a fund: holdings, costs and concentration; withheld for a past date. No network."""

import pandas as pd
import pytest

import tradingagents.dataflows.vendors.yahoo.funds as funds


def test_concentration_of_the_listed_holdings():
    measured = funds.concentration([0.08, 0.07, 0.05, float("nan")])
    assert measured["largest"] == 0.08
    assert measured["top_10"] == pytest.approx(0.20)
    assert measured["herfindahl"] == pytest.approx(0.08**2 + 0.07**2 + 0.05**2)


class Funds:
    fund_overview = {"categoryName": "Large Blend", "family": "Vanguard", "legalType": "Exchange Traded Fund"}
    description = "Tracks the S&P 500."
    fund_operations = pd.DataFrame(
        {"VOO": [0.0003, 0.02], "Category Average": [0.0072, 0.9461]},
        index=["Annual Report Expense Ratio", "Annual Holdings Turnover"],
    )
    asset_classes = {"stockPosition": 0.9987, "cashPosition": 0.0007, "bondPosition": 0.0}
    sector_weightings = {"technology": 0.3872, "energy": 0.0348}
    top_holdings = pd.DataFrame({"Name": ["NVIDIA Corp", "Apple Inc"], "Holding Percent": [0.0808, 0.0703]},
                                index=pd.Index(["NVDA", "AAPL"], name="Symbol"))


def test_today_reads_as_a_fund(monkeypatch):
    monkeypatch.setattr(funds.yf, "Ticker", lambda symbol: type("T", (), {"funds_data": Funds()})())
    monkeypatch.setattr(funds, "get_current_date", lambda: "2026-09-26")
    text = funds.get_fund_profile("VOO", "2026-09-26")
    assert "Expense ratio: 0.03% / 0.72%" in text
    assert "| NVDA | NVIDIA Corp | 8.08% |" in text and "Top 2 together: 15.11%" in text
    assert "bond" not in text  # a zero weight is not listed


def test_a_past_date_is_withheld(monkeypatch):
    def refuse(symbol):
        raise AssertionError("a live fund snapshot fetched for a past date")

    monkeypatch.setattr(funds.yf, "Ticker", refuse)
    monkeypatch.setattr(funds, "get_current_date", lambda: "2026-09-26")
    assert "Withheld for 2026-01-10" in funds.get_fund_profile("SPY", "2026-01-10")
