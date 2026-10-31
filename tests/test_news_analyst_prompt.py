"""Guard the news analyst prompt against tool-signature drift (#1116).

The prompt used to advertise ``get_news(query, ...)``, tricking the LLM into
hallucinating free-text query calls. The instrument now comes from the run's
state, so the model passes only the window.
"""
import inspect

import pytest

import tradingagents.agents.analysts.macro_analyst as ma
import tradingagents.agents.analysts.news_analyst as na
from tradingagents.agents.tools import get_news


@pytest.mark.unit
def test_get_news_takes_the_window_not_a_query():
    arg_names = set(get_news.tool_call_schema.model_json_schema()["properties"])
    assert arg_names == {"start_date", "end_date"}


@pytest.mark.unit
def test_news_prompt_matches_get_news_signature():
    src = inspect.getsource(na)
    assert "get_news(start_date, end_date)" in src
    assert "get_news(query" not in src


@pytest.mark.unit
def test_the_macro_data_is_the_macro_analysts_alone():
    """The news analyst reads company and world news; FRED and the prediction
    markets, with their guidance, moved to the macro analyst."""
    assert {t.name for t in na.TOOLS} == {"get_news", "get_global_news", "calculate"}
    news = inspect.getsource(na)
    assert "get_macro_indicators" not in news and "get_prediction_markets" not in news

    assert {t.name for t in ma.TOOLS} == {"get_macro_indicators", "get_prediction_markets", "calculate"}
    macro = inspect.getsource(ma)
    for guidance in ("get_macro_indicators(indicator, curr_date, look_back_days)",
                     "get_prediction_markets(topic, limit)", "'ecb_deposit_rate'", "'aud_usd' for .AX",
                     "look_back_days=1095", 'get_prompt_extra("macro_analyst")'):
        assert guidance in macro, guidance
