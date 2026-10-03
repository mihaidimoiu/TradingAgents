"""A model per role: the debate can be argued across providers."""

import pytest

import tradingagents.graph.trading_graph as trading_graph
from tradingagents.graph.setup import DEEP_ROLES, ROLES, GraphSetup


def test_a_role_left_out_keeps_the_quick_or_deep_model():
    setup = GraphSetup("quick", "deep", conditional_logic=None, max_tool_rounds=5, role_llms={"bear": "claude"})
    assert setup.llm_for("bear") == "claude"
    assert setup.llm_for("bull") == "quick"
    assert all(setup.llm_for(role) == "deep" for role in DEEP_ROLES)
    assert set(ROLES) >= {"market", "social", "news", "fundamentals", "bull", "bear",
                          "research_manager", "portfolio_manager"}


def test_one_client_per_provider_and_model_with_that_providers_options(monkeypatch):
    made = []

    class Client:
        def __init__(self, provider, model, **kwargs):
            made.append((provider, model, kwargs))
            self.name = f"{provider}/{model}"

        def get_llm(self):
            return self.name

    monkeypatch.setattr(trading_graph, "create_llm_client",
                        lambda provider, model, base_url=None, **kw: Client(provider, model, base_url=base_url, **kw))
    graph = trading_graph.TradingAgentsGraph.__new__(trading_graph.TradingAgentsGraph)
    graph.callbacks = []
    graph.config = {
        "llm_provider": "openai", "backend_url": "https://proxy.example", "openai_reasoning_effort": "high",
        "anthropic_effort": "medium",
        "role_llms": {
            "bull": {"provider": "openai", "model": "gpt-6-luna"},
            "bear": {"provider": "anthropic", "model": "claude-sonnet-5"},
            "trader": {"provider": "anthropic", "model": "claude-sonnet-5"},
        },
    }
    chosen = graph._role_llms()

    assert chosen == {"bull": "openai/gpt-6-luna", "bear": "anthropic/claude-sonnet-5",
                      "trader": "anthropic/claude-sonnet-5"}
    assert len(made) == 2  # the bear and the trader share one client
    anthropic = next(kw for provider, _, kw in made if provider == "anthropic")
    assert anthropic.get("effort") == "medium" and "reasoning_effort" not in anthropic
    assert anthropic["base_url"] is None  # the endpoint is the default provider's only


def test_double_analysts_run_each_analyst_on_both_models_and_hand_the_debate_both_reports():
    from langchain_core.messages import AIMessage

    from tradingagents.graph.analyst_execution import build_analyst_execution_plan
    from tradingagents.graph.setup import _analyst_graph, _paired

    spec = next(s for s in build_analyst_execution_plan(["news"]).specs if s.key == "news")

    def reader(name):
        def agent(state):
            return {"messages": [AIMessage(content="done")], spec.report_key: f"{name} says quiet week"}
        return _analyst_graph(spec, agent, max_tool_rounds=5)

    both = _paired(spec, ("gpt-6-luna", reader("A")), ("claude-sonnet-5", reader("B")))({"messages": []})
    report = both[spec.report_key]
    assert "## News analysis by gpt-6-luna\n\nA says quiet week" in report
    assert "## News analysis by claude-sonnet-5\n\nB says quiet week" in report


def test_both_models_of_a_double_analyst_read_the_runs_own_config():
    """The pool threads started with an empty context: the run's config was lost, and
    get_config served whatever another run had left process-wide."""
    from langchain_core.messages import AIMessage

    from tradingagents.dataflows.config import get_config, run_config
    from tradingagents.graph.analyst_execution import build_analyst_execution_plan
    from tradingagents.graph.setup import _analyst_graph, _paired

    spec = next(s for s in build_analyst_execution_plan(["news"]).specs if s.key == "news")

    def reader(name):
        def agent(state):
            language = get_config().get("output_language")
            return {"messages": [AIMessage(content="done")], spec.report_key: f"{name} in {language}"}
        return _analyst_graph(spec, agent, max_tool_rounds=5)

    with run_config({"output_language": "Romanian"}):
        report = _paired(spec, ("a", reader("A")), ("b", reader("B")))({"messages": []})[spec.report_key]
    assert "A in Romanian" in report and "B in Romanian" in report


def test_tool_nodes_missing_an_analyst_with_tools_is_refused_not_served_live():
    """The missing analyst fell back to the live vendors, beside the others' frozen data."""
    from langgraph.prebuilt import ToolNode

    setup = GraphSetup("quick", "deep", conditional_logic=None, max_tool_rounds=5,
                       tool_nodes={"news": ToolNode([])})
    with pytest.raises(ValueError, match="no node for the market analyst"):
        setup.setup_graph(["market", "news"])


def test_an_unknown_role_is_refused_rather_than_ignored():
    graph = trading_graph.TradingAgentsGraph.__new__(trading_graph.TradingAgentsGraph)
    graph.callbacks, graph.config = [], {"llm_provider": "openai",
                                         "role_llms": {"judge": {"provider": "openai", "model": "x"}}}
    with pytest.raises(ValueError, match="unknown roles"):
        graph._role_llms()
