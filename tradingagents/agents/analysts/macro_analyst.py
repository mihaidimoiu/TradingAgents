from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.calculator import RULE, calculate
from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_prompt_extra,
)
from tradingagents.agents.tools import get_macro_indicators, get_prediction_markets

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_macro_indicators,
    get_prediction_markets,
    calculate,
)


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            f"You are a macroeconomic researcher tasked with reading the macro backdrop as it bears on the {asset_label} under analysis. Cover the policy path and what the central bank says (the Fed for a US listing; quote its own words where your tools return them), inflation, the labour market, growth, credit conditions, the yield curve and real rates, the dollar, volatility and the broad market's trend. Then say which of these matter for this {asset_label} and in which direction, through its rate sensitivity, consumer exposure, currency exposure and financing needs; a factor that does not bear on it is left out or named as such. Keep what the data shows apart from what it implies, and say when the data is stale or mixed."
            + " Use the available tools: get_macro_indicators(indicator, curr_date, look_back_days) for macro series from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', 'real_gdp', 'vix', or a raw FRED series ID such as 'BAMLH0A0HYM2' (high-yield spread), 'DFII10' (10-year real yield) or 'DTWEXBGS' (broad dollar) for US listings; for non-US listings use the matching regional aliases instead, e.g. 'ecb_deposit_rate'/'euro_hicp'/'germany_10y'/'eur_usd' for the euro area, 'uk_interbank_rate'/'uk_10y'/'uk_inflation'/'uk_unemployment'/'gbp_usd' for .L, 'japan_call_rate'/'japan_10y'/'japan_inflation'/'japan_unemployment'/'usd_jpy' for .T, 'canada_interbank_rate'/'canada_10y'/'canada_inflation'/'canada_unemployment'/'usd_cad' for .TO, 'australia_interbank_rate'/'australia_10y'/'australia_inflation'/'australia_unemployment'/'aud_usd' for .AX; the *_inflation aliases are annual, so pass look_back_days=1095 for them), and get_prediction_markets(topic, limit) for market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026')."
            + " Name the series and the date behind every figure, and state no figure a tool did not return. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + RULE + get_language_instruction() + get_prompt_extra("macro_analyst")
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}.\n"
                    "{system_message}\n"
                    # The date and instrument last: the prefix above then stays
                    # byte-identical across runs for the provider's prompt cache (#750).
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_analyst_node
