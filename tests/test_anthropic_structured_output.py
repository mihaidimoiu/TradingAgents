"""Claude answers structured calls through its own JSON-schema output, not a tool call.

Opus 5.5, Sonnet 5.5 and Fable 5.1 refuse a forced tool call, and with thinking
on no Claude model takes one: LangChain's default then sends the schema as an
optional tool, the model may answer in prose, and the agent pays for a second,
free-text call.
"""

import pytest
from pydantic import BaseModel

from tradingagents.agents.structured import bind_structured
from tradingagents.llm_clients.anthropic_client import AnthropicClient
from tradingagents.llm_clients.fallback import CreditFallback, Exhausted


class Verdict(BaseModel):
    rating: str


def _request_kwargs(structured) -> dict:
    return structured.first.kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5"],
)
def test_claude_binds_the_schema_as_its_output_format(model):
    llm = AnthropicClient(model=model, api_key="x").get_llm()

    kwargs = _request_kwargs(bind_structured(llm, Verdict, "Trader"))

    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in kwargs


@pytest.mark.unit
def test_an_explicit_method_is_still_honoured():
    llm = AnthropicClient(model="claude-opus-4-8", api_key="x").get_llm()

    kwargs = _request_kwargs(llm.with_structured_output(Verdict, method="function_calling"))

    assert "tools" in kwargs


@pytest.mark.unit
def test_claude_behind_a_credit_fallback_keeps_its_output_format():
    claude = AnthropicClient(model="claude-opus-5-5", api_key="x").get_llm()
    backup = AnthropicClient(model="claude-opus-4-8", api_key="x").get_llm()
    llm = CreditFallback(claude, backup, "anthropic", "anthropic", Exhausted())

    structured = bind_structured(llm, Verdict, "Trader")

    assert _request_kwargs(structured.primary)["output_config"]["format"]["type"] == "json_schema"
