"""A failed call goes down the chain of providers as `classify` reads it; the last one's failure names them all."""

import copy

import pytest
from langchain_core.runnables import Runnable

from tradingagents.llm_clients.fallback import (
    ChainExhausted,
    Exhausted,
    ProviderChain,
    credit_only,
    out_of_credit,
)


class Model(Runnable):
    def __init__(self, name, error=None, tools=()):
        self.name, self.error, self.tools, self.calls = name, error, tools, 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return f"{self.name}:{input}"

    def bind_tools(self, tools):
        return Model(self.name, self.error, tools)


NO_CREDIT = RuntimeError("Error code: 400 - {'type': 'invalid_request_error', "
                         "'message': 'Your credit balance is too low to access the Anthropic API.'}")
DOWN = RuntimeError("Error code: 529 - {'type': 'overloaded_error'}")
UNPARSED = ValueError("Invalid json output")


def reading(error):
    """As a caller classifies: an empty account, an outage, a Stop (a KeyError here), or one call."""
    if out_of_credit(error):
        return "account"
    if "529" in str(error):
        return "down"
    return None if isinstance(error, KeyError) else "call"


def chain(*models, exhausted=None, classify=reading):
    return ProviderChain([(m.name, f"{m.name}/model", m) for m in models], exhausted or Exhausted(), classify)


@pytest.mark.unit
def test_an_empty_account_is_out_of_the_run_and_told_once():
    told = []
    exhausted = Exhausted(lambda *heard: told.append(heard))
    claude, gpt = Model("anthropic", NO_CREDIT), Model("openai")
    llm = chain(claude, gpt, exhausted=exhausted)

    assert llm.invoke("a") == "openai:a" and llm.invoke("b") == "openai:b"
    assert claude.calls == 1  # not asked again once known to be empty
    assert told == [("anthropic", NO_CREDIT, "openai/model", True)]
    assert exhausted.providers == {"anthropic"}


@pytest.mark.unit
def test_a_chain_goes_on_until_one_answers_and_one_call_moves_alone():
    told = []
    exhausted = Exhausted(lambda *heard: told.append(heard[0::2]))
    unparsed, down, last = Model("openai", UNPARSED), Model("anthropic", DOWN), Model("google")
    llm = chain(unparsed, down, last, exhausted=exhausted)

    assert llm.invoke("a") == "google:a"
    # The unparsed reply moved its call alone; the outage took its provider out of the run.
    assert told == [("openai", "anthropic/model"), ("anthropic", "google/model")] and exhausted.providers == {"anthropic"}
    llm.invoke("b")
    assert (unparsed.calls, down.calls) == (2, 1)


@pytest.mark.unit
def test_the_last_failure_names_every_providers_and_a_stop_is_never_handed_on():
    llm = chain(Model("openai", NO_CREDIT), Model("anthropic", DOWN))
    with pytest.raises(ChainExhausted) as failed:
        llm.invoke("a")
    assert [provider for provider, _ in failed.value.failures] == ["openai", "anthropic"]
    assert str(failed.value).startswith("openai: Error code: 400")
    # Both out now: the next call fails at once, naming both, and asks neither.
    with pytest.raises(ChainExhausted, match="^openai: .*; anthropic: "):
        llm.invoke("b")

    stopped = Model("openai", KeyError("stop"))
    with pytest.raises(KeyError):
        chain(stopped, Model("anthropic")).invoke("a")


@pytest.mark.unit
def test_by_default_only_an_empty_account_moves_on():
    rate_limited = RuntimeError("Error code: 429 - {'code': 'rate_limit_exceeded'}")
    with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
        chain(Model("openai", rate_limited), Model("anthropic"), classify=credit_only).invoke("a")
    # A chain of one is its model: its error, as it was.
    with pytest.raises(RuntimeError, match="credit balance"):
        chain(Model("anthropic", NO_CREDIT)).invoke("a")


@pytest.mark.unit
def test_a_bound_chain_binds_every_link_the_same_way_and_shares_who_is_out():
    exhausted = Exhausted()
    llm = chain(Model("openai", RuntimeError("insufficient_quota")), Model("anthropic"), exhausted=exhausted)

    bound = llm.bind_tools(["get_news"])

    assert isinstance(bound, ProviderChain) and [m.tools for _, _, m in bound.links] == [["get_news"]] * 2
    assert bound.invoke("a") == "anthropic:a"
    assert "openai" in exhausted.providers  # shared: the unbound client skips it too


@pytest.mark.unit
def test_an_instance_not_yet_initialised_has_no_attributes_rather_than_recursing():
    """copy and pickle build the instance before its fields: __getattr__ read its links and re-entered."""
    bare = ProviderChain.__new__(ProviderChain)
    with pytest.raises(AttributeError):
        _ = bare.model_name
    assert copy.copy(chain(Model("anthropic"), Model("openai"))).invoke("a") == "anthropic:a"


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
    "You exceeded your current quota, please check your plan and billing details.",
    "Your credit balance is too low to access the Anthropic API.",
])
def test_each_providers_empty_account_is_recognised(text):
    assert out_of_credit(RuntimeError(text)) and credit_only(RuntimeError(text)) == "account"
