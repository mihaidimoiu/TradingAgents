"""yfinance news must survive a nested article whose publisher fields are null.

Upstream #1458 / PR #1468: yfinance can return ``content.provider`` present but
null, and ``dict.get`` falls back only when a key is absent, so one such
article raised AttributeError and lost every article of the fetch.
"""
from datetime import UTC, datetime

import pytest

import tradingagents.dataflows.vendors.yahoo.news as ynews


def _nested(provider_value, title="Soybean futures slide"):
    return {
        "content": {
            "title": title,
            "summary": "Chicago soybean futures fell for a third session.",
            "provider": provider_value,
            "canonicalUrl": {"url": "https://example.com/soybean"},
            "pubDate": "2026-09-23T10:00:00Z",
        }
    }


@pytest.mark.unit
def test_null_provider_is_treated_as_missing():
    assert ynews._extract_article_data(_nested(None))["publisher"] == "Unknown"


@pytest.mark.unit
def test_null_display_name_falls_back_to_unknown():
    assert ynews._extract_article_data(_nested({"displayName": None}))["publisher"] == "Unknown"


@pytest.mark.unit
def test_null_provider_does_not_fail_the_whole_fetch(monkeypatch):
    class FakeTicker:
        def __init__(self, *a, **k):
            pass

        def get_news(self, count=20):
            return [
                _nested(None, title="NULL PROVIDER"),
                _nested({"displayName": "Reuters"}, title="GOOD ARTICLE"),
            ]

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    out = ynews.get_news_yfinance("ZS", "2026-09-20", "2026-09-26")

    assert "GOOD ARTICLE" in out
    assert "NULL PROVIDER" in out
    assert "(source: None)" not in out
    assert "(source: Unknown)" in out


@pytest.mark.unit
def test_healthy_provider_is_unchanged():
    data = ynews._extract_article_data(_nested({"displayName": "Bloomberg"}))
    assert data["publisher"] == "Bloomberg"
    assert data["title"] == "Soybean futures slide"
    assert data["link"] == "https://example.com/soybean"
    assert data["pub_date"] == datetime(2026, 9, 23, 10, 0, tzinfo=UTC)
