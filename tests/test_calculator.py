"""The agents' calculator: exact arithmetic over what the model writes, offered for a bounded number of rounds."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents import calculator
from tradingagents.agents.calculator import calculate, evaluate, invoke_calculating, with_calculations


@pytest.mark.unit
@pytest.mark.parametrize(("expression", "value"), [
    ("27.38e9 * (0.0669 - 0.0485)", 503_792_000.0),
    ("pct(1.35, 27.38)", 1.35 / 27.38 * 100),
    ("change(100, 112.5)", 12.5),
    ("cagr(50, 80.5255, 5)", 10.0),
    ("round(2 / 3, 2) + -1", -0.33),
    ("sqrt(16) + log(e) + max(1, 2, 3)", 8.0),
])
def test_arithmetic_is_exact_and_percentages_go_through_their_functions(expression, value):
    assert evaluate(expression) == pytest.approx(value)


@pytest.mark.unit
@pytest.mark.parametrize(("expression", "why"), [
    ("__import__('os').system('ls')", "not arithmetic"),
    ("(1).real", "not arithmetic"),
    ("12% * 3", "not an expression"),
    ("1 / 0", "division"),
    ("10.0 ** 400", "too large"),
    ("10.0 ** 300 * 10.0 ** 300", "no real, finite value"),
    ("(-8) ** (1 / 3)", "no real, finite value"),
    ("True + 1", "not arithmetic"),
    ("1" + " + 1" * 300, "longer than"),
])
def test_anything_but_arithmetic_is_refused_with_why(expression, why):
    with pytest.raises(ValueError, match=why):
        evaluate(expression)


@pytest.mark.unit
def test_one_call_answers_every_expression_on_its_own_line_and_says_what_failed():
    said = calculate.invoke({"expressions": ["2 + 2", "1 / 0"]})
    assert said.splitlines() == ["2 + 2 = 4", "1 / 0: error: division by zero"]


def _model(*replies: AIMessage):
    """Answers in turn when offered the calculator; the last turn, offered none, writes the answer."""
    llm = MagicMock()
    llm.bind_tools.return_value.invoke.side_effect = list(replies)
    llm.invoke.return_value = AIMessage(content="answered at the limit")
    return llm


def _sums(n: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "calculate", "id": f"c{n}", "args": {"expressions": ["6 * 7"]}}])


@pytest.mark.unit
def test_a_one_call_agent_reads_its_results_and_answers():
    llm = _model(_sums(1), AIMessage(content="42 it is"))
    assert invoke_calculating(llm, "the prompt").content == "42 it is"
    _, history = llm.bind_tools.return_value.invoke.call_args_list
    assert history.args[0][-1].content == "6 * 7 = 42"
    assert "never work out a number in your head" in history.args[0][0].content


@pytest.mark.unit
def test_calculator_rounds_are_capped_and_the_last_turn_offers_no_tool():
    llm = _model(*(_sums(n) for n in range(calculator.ROUNDS)))
    assert invoke_calculating(llm, "the prompt").content == "answered at the limit"
    written = llm.invoke.call_args.args[0]
    assert not any(getattr(m, "tool_calls", None) for m in written)
    assert isinstance(written[-1], HumanMessage) and "rounds are spent" in written[-1].content


@pytest.mark.unit
def test_an_answer_in_a_schema_gets_its_results_in_its_prompt_and_none_unchanged():
    assert with_calculations(_model(_sums(1), AIMessage(content="DONE")), "decide") == (
        "decide\n\nYour calculator results; quote these figures exactly:\n6 * 7 = 42")
    assert with_calculations(_model(AIMessage(content="DONE")), "decide") == "decide"
    stray = AIMessage(content="", tool_calls=[{"name": "get_news", "id": "s", "args": {}}])
    assert with_calculations(_model(stray, AIMessage(content="DONE")), "decide") == "decide"


@pytest.mark.unit
def test_a_model_that_binds_no_tools_answers_as_before():
    llm = MagicMock()
    llm.bind_tools.side_effect = NotImplementedError("no tools")
    llm.invoke.return_value = AIMessage(content="plain")
    assert invoke_calculating(llm, "the prompt").content == "plain"
    assert llm.invoke.call_args.args[0] == "the prompt"
    assert with_calculations(llm, "decide") == "decide"


@pytest.mark.unit
def test_a_call_to_another_tool_or_without_expressions_is_told_so():
    reply = AIMessage(content="", tool_calls=[{"name": "get_news", "id": "a", "args": {}},
                                              {"name": "calculate", "id": "b", "args": {}}])
    assert [m.content for m in calculator.answers(reply)] == [
        "no tool named get_news: only calculate is offered here",
        "give `expressions`, a list of arithmetic expressions"]
