"""The two judges read the caller's evidence index, not only the debate about it."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.trading_graph import TradingAgentsGraph


class _LLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        return AIMessage("Rating: Hold\n\nnothing to do")

    def with_structured_output(self, *args, **kwargs):
        raise NotImplementedError


def _state(digest: str) -> dict:
    return {
        "company_of_interest": "AAPL", "trade_date": "2026-08-14", "asset_type": "stock",
        "instrument_context": "", "investment_plan": "P", "trader_investment_plan": "T", "past_context": "",
        "portfolio_context": "", "evidence_digest": digest,
        "investment_debate_state": {"history": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
    }


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_research_manager, create_portfolio_manager])
def test_each_judge_reads_the_index_when_given_one(factory):
    llm = _LLM()
    factory(llm)(_state("[NEWS-001] 2026-08-13, reuters: margins fell"))
    assert "Evidence Index" in llm.prompts[0] and "[NEWS-001] 2026-08-13, reuters: margins fell" in llm.prompts[0]


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_research_manager, create_portfolio_manager])
def test_no_index_adds_no_section(factory):
    llm = _LLM()
    factory(llm)(_state(""))
    assert "Evidence Index" not in llm.prompts[0]


@pytest.mark.unit
def test_the_run_state_carries_the_callers_index():
    graph = object.__new__(TradingAgentsGraph)
    graph.propagator = Propagator()
    graph.resolve_instrument_context = lambda t, a="stock", d=None: ""
    assert graph.create_run_state("AAPL", "2026-08-14")["evidence_digest"] == ""
    graph.evidence_digest = lambda ticker, trade_date: f"[TECH-001] {ticker} {trade_date}"
    assert graph.create_run_state("AAPL", "2026-08-14")["evidence_digest"] == "[TECH-001] AAPL 2026-08-14"
