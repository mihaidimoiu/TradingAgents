"""A provider out of credit hands its calls to the other provider's model; nothing else does."""

import pytest
from langchain_core.runnables import Runnable

from tradingagents.llm_clients.fallback import CreditFallback, Exhausted, out_of_credit


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


@pytest.mark.unit
def test_no_credit_switches_once_and_the_provider_is_skipped_after():
    told = []
    exhausted = Exhausted(told.append)
    primary, backup = Model("claude", NO_CREDIT), Model("gpt")
    llm = CreditFallback(primary, backup, "anthropic", "openai/gpt", exhausted)

    assert llm.invoke("a") == "gpt:a"
    assert llm.invoke("b") == "gpt:b"
    assert primary.calls == 1  # not asked again once known to be empty
    assert told == ["anthropic has no credit left; its roles answer on openai/gpt"]


@pytest.mark.unit
def test_an_instance_not_yet_initialised_has_no_attributes_rather_than_recursing():
    """copy and pickle build the instance before its fields: __getattr__ read self.primary and re-entered."""
    import copy

    bare = CreditFallback.__new__(CreditFallback)
    with pytest.raises(AttributeError):
        _ = bare.model_name
    llm = CreditFallback(Model("claude"), Model("gpt"), "anthropic", "openai/gpt", Exhausted())
    assert copy.copy(llm).invoke("a") == "claude:a"


@pytest.mark.unit
def test_any_other_error_still_fails_the_call():
    rate_limited = RuntimeError("Error code: 429 - {'code': 'rate_limit_exceeded'}")
    llm = CreditFallback(Model("gpt", rate_limited), Model("claude"), "openai", "anthropic/claude", Exhausted())

    with pytest.raises(RuntimeError, match="rate_limit_exceeded"):
        llm.invoke("a")


@pytest.mark.unit
def test_a_bound_model_falls_back_to_the_backup_bound_the_same_way():
    exhausted = Exhausted()
    llm = CreditFallback(Model("gpt", RuntimeError("insufficient_quota")), Model("claude"), "openai",
                         "anthropic/claude", exhausted)

    bound = llm.bind_tools(["get_news"])

    assert isinstance(bound, CreditFallback) and bound.backup.tools == ["get_news"]
    assert bound.invoke("a") == "claude:a"
    assert "openai" in exhausted.providers  # shared: the unbound client skips it too


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
    "You exceeded your current quota, please check your plan and billing details.",
    "Your credit balance is too low to access the Anthropic API.",
])
def test_each_providers_empty_account_is_recognised(text):
    assert out_of_credit(RuntimeError(text))
