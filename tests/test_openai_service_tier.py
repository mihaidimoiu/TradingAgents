"""OpenAI's ``service_tier`` is set per run through the config, or from the environment (#1487).

A run that does not need fast answers can take the cheaper Flex tier. Unset,
no tier is sent and OpenAI uses the project's default.
"""

import pytest

from tradingagents import default_config as default_config_module
from tradingagents.llm_clients import factory
from tradingagents.llm_clients.openai_client import OpenAIClient


@pytest.mark.unit
def test_service_tier_env_override(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_OPENAI_SERVICE_TIER", "flex")
    assert default_config_module.build_default_config()["openai_service_tier"] == "flex"


@pytest.mark.unit
def test_service_tier_defaults_to_none(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_OPENAI_SERVICE_TIER", raising=False)
    assert default_config_module.build_default_config()["openai_service_tier"] is None


@pytest.mark.unit
def test_build_llm_kwargs_forwards_service_tier_for_openai():
    kwargs = factory.build_llm_kwargs({"llm_provider": "openai", "openai_service_tier": "flex"})
    assert kwargs["service_tier"] == "flex"


@pytest.mark.unit
def test_build_llm_kwargs_omits_unset_service_tier():
    kwargs = factory.build_llm_kwargs({"llm_provider": "openai", "openai_service_tier": None})
    assert "service_tier" not in kwargs


@pytest.mark.unit
def test_build_llm_kwargs_omits_service_tier_for_other_providers():
    kwargs = factory.build_llm_kwargs({"llm_provider": "google", "openai_service_tier": "flex"})
    assert "service_tier" not in kwargs


@pytest.mark.unit
def test_openai_client_reaches_chat_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAIClient("gpt-6-luna", provider="openai", service_tier="flex").get_llm()
    assert llm.service_tier == "flex"


@pytest.mark.unit
def test_openai_client_sends_no_tier_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert OpenAIClient("gpt-6-luna", provider="openai").get_llm().service_tier is None
