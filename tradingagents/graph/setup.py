import logging
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.analysts.sentiment_analyst import (
    SourceFetcher,
    fetch_sentiment_sources,
)
from tradingagents.agents.analysts.turn import WRAP_UP
from tradingagents.agents.state import AgentState

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

logger = logging.getLogger(__name__)

# Every role a model can be chosen for (``config["role_llms"]``), and the model
# it falls back to: the analysts by their plan key.
QUICK_ROLES = ("market", "social", "news", "fundamentals", "bull", "bear", "trader",
               "aggressive", "conservative", "neutral")
DEEP_ROLES = ("research_manager", "portfolio_manager")
ROLES = QUICK_ROLES + DEEP_ROLES

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


# A provider that did not answer: a timeout, a dropped connection, a rate limit,
# a 5xx. By name, so no provider SDK is imported here. Anything else (a bug, a
# caller's cancellation) still ends the run.
PROVIDER_FAILURES = frozenset({
    "APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
    "ServiceUnavailableError", "TimeoutException", "ReadTimeout", "ConnectTimeout",
})


def _provider_failure(error: BaseException) -> bool:
    return isinstance(error, (TimeoutError, ConnectionError)) or any(
        cls.__name__ in PROVIDER_FAILURES for cls in type(error).__mro__
    )


MISSING_REPORT = "analyst's report is missing: the model did not answer"


def report_missing(report: str) -> bool:
    """Whether `report` is the note a failed analyst leaves instead of a report."""
    return MISSING_REPORT in (report or "")[:200]


def _resilient(spec, agent):
    """The analyst, with a provider failure ending only its own report.

    The analysts are independent; one that times out used to end the whole
    run after the others had finished. Its report now says it is missing and
    why, and the debate goes on with the rest.
    """
    def run(state, *args, **kwargs):
        try:
            return agent(state, *args, **kwargs)
        except Exception as error:
            if not _provider_failure(error):
                raise
            note = (f"The {spec.key} {MISSING_REPORT} "
                    f"({type(error).__name__}: {str(error)[:200]}). Weigh the other reports without it; "
                    "do not infer what it would have said.")
            return {"messages": [AIMessage(content=note)], spec.report_key: note}

    return run


def model_label(llm: Any) -> str:
    """The model id a chat client was built for, for a reader of the report."""
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", None) or type(llm).__name__)


def _paired(spec, first: tuple[str, Any], second: tuple[str, Any]):
    """One analyst run on two models at once; the debate reads both reports, each under its model.

    Two independent readings of the same evidence: where they disagree is the
    debate's to weigh, not averaged away here.
    """
    def run(state, config: RunnableConfig | None = None) -> dict:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # A pool thread starts with an empty context: without the caller's,
            # the run's config (its vendors, prompt_extra, language) is lost and
            # get_config falls back to whatever another run left process-wide.
            # One copy each: a context cannot be entered by two threads at once.
            answers = [(label, pool.submit(copy_context().run, graph.invoke, state, config))
                       for label, graph in (first, second)]
            readings = [(label, future.result()[spec.report_key]) for label, future in answers]
        return {spec.report_key: "\n\n".join(
            f"## {spec.key.capitalize()} analysis by {label}\n\n{report}" for label, report in readings)}

    return run


def _tools_or_done(state) -> str:
    """Route an analyst's turn: run its tool calls, or finish with its report."""
    return "tools" if state["messages"][-1].tool_calls else END


def _analyst_graph(spec, agent, max_tool_rounds: int, tools: ToolNode | None = None):
    """One analyst as a graph of its own: the model and its tools, on a private message history.

    It returns only its report, so analysts running side by side never write the
    same key, and its tool calls never reach the other analysts' messages. After
    ``max_tool_rounds`` rounds of tool calls it is told to write its report, and
    that turn ends it whatever it answers, so a model that keeps calling tools
    cannot run the graph into its recursion limit (#1420). `tools` is a caller's
    node in place of the default (the `tool_nodes` hook).
    """
    output = TypedDict(f"{spec.key.capitalize()}Report", {spec.report_key: str})
    graph = StateGraph(AgentState, output_schema=output)
    graph.add_node("agent", agent)
    graph.add_edge(START, "agent")
    if not spec.tools:
        graph.add_edge("agent", END)
        return graph.compile()

    def calls(messages):
        return [call["name"] for m in messages for call in (getattr(m, "tool_calls", None) or [])]

    def rounds(messages) -> int:
        return sum(1 for m in messages if getattr(m, "tool_calls", None))

    def more_or_wrap_up(state) -> str:
        return "wrap_up" if rounds(state["messages"]) >= max_tool_rounds else "agent"

    def wrap_up(state):
        repeated = ", ".join(f"{name} x{n}" for name, n in Counter(calls(state["messages"])).most_common())
        logger.warning("%s used its %d tool rounds (%s); asking for its report",
                       spec.agent_node, max_tool_rounds, repeated)
        return agent({**state, "messages": [*state["messages"], HumanMessage(WRAP_UP)]})

    graph.add_node("tools", tools or ToolNode(list(spec.tools)))
    graph.add_node("wrap_up", wrap_up)
    graph.add_conditional_edges("agent", _tools_or_done, ["tools", END])
    graph.add_conditional_edges("tools", more_or_wrap_up, ["agent", "wrap_up"])
    graph.add_edge("wrap_up", END)
    return graph.compile()


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        conditional_logic: ConditionalLogic,
        max_tool_rounds: int,
        sentiment_sources: SourceFetcher = fetch_sentiment_sources,
        tool_nodes: Mapping[str, ToolNode] | None = None,
        role_llms: Mapping[str, Any] | None = None,
        second_analyst_llm: Any | None = None,
    ):
        """Initialize with required components.

        ``role_llms`` gives a role (``ROLES``) its own model; any other role
        keeps the quick or deep one. ``second_analyst_llm`` runs every analyst
        a second time on that model, and the debate reads both reports.
        """
        self.sentiment_sources = sentiment_sources
        # None: every analyst calls the live vendors. A mapping must cover every
        # analyst that has tools: one missing would quietly fetch live data
        # beside the others' frozen pack.
        self.tool_nodes = tool_nodes
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.conditional_logic = conditional_logic
        self.max_tool_rounds = max_tool_rounds
        self.role_llms = dict(role_llms or {})
        self.second_analyst_llm = second_analyst_llm

    def llm_for(self, role: str) -> Any:
        """The model that plays `role`."""
        if role in self.role_llms:
            return self.role_llms[role]
        return self.deep_thinking_llm if role in DEEP_ROLES else self.quick_thinking_llm

    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Sentiment analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": create_market_analyst,
            "social": lambda llm: create_sentiment_analyst(llm, sources=self.sentiment_sources),
            "news": create_news_analyst,
            "fundamentals": create_fundamentals_analyst,
        }

        bull_researcher_node = create_bull_researcher(self.llm_for("bull"))
        bear_researcher_node = create_bear_researcher(self.llm_for("bear"))
        research_manager_node = create_research_manager(self.llm_for("research_manager"))
        trader_node = create_trader(self.llm_for("trader"))

        aggressive_analyst = create_aggressive_debator(self.llm_for("aggressive"))
        neutral_analyst = create_neutral_debator(self.llm_for("neutral"))
        conservative_analyst = create_conservative_debator(self.llm_for("conservative"))
        portfolio_manager_node = create_portfolio_manager(self.llm_for("portfolio_manager"))

        workflow = StateGraph(AgentState)

        for spec in plan.specs:
            if self.tool_nodes is not None and spec.tools and spec.key not in self.tool_nodes:
                raise ValueError(f"tool_nodes has no node for the {spec.key} analyst, whose tools "
                                 "would otherwise call the live vendors")

            def analyst(llm, spec=spec):
                return _analyst_graph(spec, _resilient(spec, analyst_factories[spec.key](llm)),
                                      self.max_tool_rounds,
                                      None if self.tool_nodes is None else self.tool_nodes.get(spec.key))

            first = self.llm_for(spec.key)
            if self.second_analyst_llm is None:
                workflow.add_node(spec.agent_node, analyst(first))
            else:
                workflow.add_node(spec.agent_node, _paired(
                    spec, (model_label(first), analyst(first)),
                    (model_label(self.second_analyst_llm), analyst(self.second_analyst_llm))))

        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # The analysts work at the same time; the research debate starts once
        # every one of them has filed its report.
        analysts = [spec.agent_node for spec in plan.specs]
        for node in analysts:
            workflow.add_edge(START, node)
        workflow.add_edge(analysts, "Bull Researcher")

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
