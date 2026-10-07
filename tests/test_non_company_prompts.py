"""An instrument that is not a company (a currency pair, a metal, a fund) is not debated as one."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


class _LLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return AIMessage("argument")


def _state(asset_type: str) -> dict:
    return {
        "asset_type": asset_type, "instrument_context": "EURUSD", "market_report": "M", "sentiment_report": "S",
        "news_report": "N", "fundamentals_report": "",
        "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "", "current_response": "",
                                    "count": 0},
    }


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_any_type_but_stock_is_debated_as_an_asset_not_a_company(factory):
    llm = _LLM()
    factory(llm)(_state("asset"))
    assert "Asset fundamentals report (may be unavailable: it is not a company)" in llm.prompts[0]
    assert "Company fundamentals report" not in llm.prompts[0]
    llm = _LLM()
    factory(llm)(_state("stock"))
    assert "Company fundamentals report" in llm.prompts[0]


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_a_non_companys_key_points_are_its_drivers_not_a_business(factory):
    company_words = ("market opportunities", "branding", "competitors", "innovation", "revenue")
    llm = _LLM()
    factory(llm)(_state("asset"))
    asset = llm.prompts[0]
    assert "Flows and Positioning" in asset and "macro drivers" in asset and "supply" in asset
    assert not any(word in asset for word in company_words)
    llm = _LLM()
    factory(llm)(_state("stock"))
    assert any(word in llm.prompts[0] for word in company_words) and "Flows and Positioning" not in llm.prompts[0]
