"""What the agents are actually told.

Three problems the audit found: one analyst's brief reached the model as a Python
tuple, every analyst was asked for a trade call that nothing reads, and a report
that was never produced was presented as an empty labelled section, which invites
the next agent to fill it in from nothing.
"""

from __future__ import annotations

import importlib

import pytest

ANALYSTS = ["market_analyst", "sentiment_analyst", "news_analyst", "macro_analyst", "fundamentals_analyst"]

REPORT_READERS = [
    ("tradingagents.agents.researchers.bull_researcher", "create_bull_researcher"),
    ("tradingagents.agents.researchers.bear_researcher", "create_bear_researcher"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
]


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_brief_is_text_not_a_python_object(name):
    """A trailing comma made one brief a tuple, so the model was handed its repr
    (quotes, parens and all) instead of the instruction."""
    import ast
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    tree = ast.parse(inspect.getsource(mod))
    briefs = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and getattr(node.targets[0], "id", "") == "system_message"]
    assert briefs, f"{name} has no system_message"
    for brief in briefs:
        assert not isinstance(brief, ast.Tuple), "the brief is a tuple, not text"


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_is_not_asked_for_a_trade_call_nothing_reads(name):
    """The stop signal is never consumed, and asking for it makes an analyst
    open with a direction that then travels as evidence."""
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    assert "FINAL TRANSACTION PROPOSAL" not in inspect.getsource(mod)


def _read_prompt(module: str, factory: str, **reports) -> str:
    """The prompt a report reader is given over `reports`, as text."""
    from langchain_core.messages import AIMessage

    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else str(prompt))
            return AIMessage("argument")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError

    state = {
        "company_of_interest": "NVDA", "trade_date": "2026-08-14", "asset_type": "stock",
        "instrument_context": "", "portfolio_context": "", "past_context": "",
        "market_report": "", "sentiment_report": "", "news_report": "",
        "fundamentals_report": "", "investment_plan": "P", "trader_investment_plan": "T",
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                    "current_response": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": ""},
        **reports,
    }
    getattr(importlib.import_module(module), factory)(_LLM())(state)
    return " ".join(seen)


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", REPORT_READERS)
def test_a_report_that_was_never_produced_says_so(module, factory):
    """`--analysts market` leaves the other reports empty; presenting them as blank
    sections invites the model to invent the contents."""
    prompt = _read_prompt(module, factory, market_report="RSI 61, price 178.", macro_report="")

    assert "not part of this run" in prompt or "not available" in prompt, prompt[:400]
    assert "No macro report in this run" in prompt
    assert "RSI 61" in prompt  # the report that does exist is still passed through


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", REPORT_READERS)
def test_every_reader_weighs_the_macro_report(module, factory):
    """A state from before the macro analyst has no such key; it reads as absent, not as an error."""
    prompt = _read_prompt(module, factory, macro_report="Fed on hold, curve steepening.")
    assert "macroeconomic report: fed on hold, curve steepening." in prompt.lower()
    assert "No macro report in this run" in _read_prompt(module, factory)
