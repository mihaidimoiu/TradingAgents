"""A calculator the agents call for arithmetic, so no figure they state is worked out in the model's head.

Pure arithmetic over what the model writes: it reads no data, so a run is as reproducible with it as
without. An analyst that already calls tools is offered it beside them (its ``TOOLS``); an agent that
answers in one call gets up to ``ROUNDS`` rounds of it first (``invoke_calculating``); one that answers
in a schema makes its calculations before the structured call (``with_calculations``), since that call
offers no tool but the schema.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from tradingagents.agents.analysts.turn import as_text

logger = logging.getLogger(__name__)

# Calculator rounds an agent answering in one call may take before it must answer: each resends its prompt.
ROUNDS = 3
MAX_EXPRESSIONS = 40
MAX_CHARS = 400

RULE = (
    "\n\nArithmetic: never work out a number in your head. Every sum, difference, product, ratio, "
    "percentage, percentage change, growth rate or compounding you state goes through the `calculate` "
    "tool: put all of a turn's expressions in one call, then quote its results exactly."
)
WRAP_UP = "Your calculator rounds are spent. Write your answer now from the results above."
# Before an answer in a schema, which offers no tool: the calculations first, then a bare acknowledgement.
FIRST = (
    "\n\nBefore answering, make every calculation your answer will need with `calculate`, all in one call. "
    "Once none is left, reply with the single word DONE; do not write the answer yet."
)


def _pct(part: float, whole: float) -> float:
    return part / whole * 100


def _change(old: float, new: float) -> float:
    return (new - old) / old * 100


def _cagr(start: float, end: float, years: float) -> float:
    return ((end / start) ** (1 / years) - 1) * 100


def _round(value: float, digits: float = 0) -> float:
    return round(value, int(digits))


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs, "min": min, "max": max, "round": _round, "sqrt": math.sqrt, "log": math.log,
    "log10": math.log10, "exp": math.exp, "pct": _pct, "change": _change, "cagr": _cagr,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
# No % operator: "12%" would read as the start of a modulo; percentages go through pct and change.
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate(expression: str) -> float:
    """The value of one arithmetic expression; ValueError naming what is not arithmetic or has no value."""
    if len(expression) > MAX_CHARS:
        raise ValueError(f"longer than {MAX_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"not an expression ({error.msg})") from None
    try:
        value = _value(tree.body)
    except (ArithmeticError, TypeError) as error:
        raise ValueError(str(error) or type(error).__name__) from None
    if isinstance(value, complex) or not math.isfinite(value):
        raise ValueError("no real, finite value")
    return value


def _value(node: ast.AST) -> float:
    # Every number as a float: an integer power would otherwise grow without bound.
    match node:
        case ast.Constant(value=bool()):
            pass
        case ast.Constant(value=int() | float() as number):
            return float(number)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINARY:
            return _BINARY[type(op)](_value(left), _value(right))
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return _UNARY[type(op)](_value(operand))
        case ast.Name(id=name) if name in CONSTANTS:
            return CONSTANTS[name]
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in FUNCTIONS:
            return float(FUNCTIONS[name](*(_value(arg) for arg in args)))
    raise ValueError(f"not arithmetic: {ast.unparse(node)}")


@tool
def calculate(expressions: list[str]) -> str:
    """Evaluate arithmetic exactly: several expressions at once, one result line each.

    Write plain numbers, with no thousands separators, % signs or units. Operators: + - * / ** and
    brackets. Functions: abs, min, max, round(x, digits), sqrt, log (natural), log10, exp; the constants
    pi and e; pct(part, whole) = part as a percent of whole; change(old, new) = percent change from old to
    new; cagr(start, end, years) = compound yearly growth, in percent.
    """
    lines = []
    for expression in expressions[:MAX_EXPRESSIONS]:
        try:
            lines.append(f"{expression} = {evaluate(expression):.15g}")
        except ValueError as error:
            lines.append(f"{expression}: error: {error}")
    if len(expressions) > MAX_EXPRESSIONS:
        lines.append(f"only the first {MAX_EXPRESSIONS} expressions were evaluated; send the rest in another call")
    return "\n".join(lines)


def answers(reply: AIMessage) -> list[ToolMessage]:
    """The calculator's answer to each call in `reply`; a call to any other tool, or with no expressions, told so."""
    found = []
    for call in reply.tool_calls:
        if call["name"] != calculate.name:
            said = f"no tool named {call['name']}: only {calculate.name} is offered here"
        elif not isinstance(expressions := call["args"].get("expressions"), list):
            said = "give `expressions`, a list of arithmetic expressions"
        else:
            said = calculate.invoke({"expressions": [str(e) for e in expressions]})
        found.append(ToolMessage(content=said, tool_call_id=call["id"], name=call["name"]))
    return found


def _messages(prompt: Any, said: str) -> list:
    """`prompt` (a string or a message list) as messages, `said` added at its end."""
    if isinstance(prompt, str):
        return [HumanMessage(content=prompt + said)]
    return [*prompt, HumanMessage(content=said.strip())]


def _offered(llm: Any) -> Any | None:
    """`llm` offering the calculator; None for a provider that binds no tools (some local models)."""
    try:
        return llm.bind_tools([calculate])
    except (NotImplementedError, AttributeError) as error:
        logger.warning("the model binds no tools (%s): it answers without the calculator", error)
        return None


def invoke_calculating(llm: Any, prompt: Any) -> BaseMessage:
    """`llm`'s answer to `prompt` (a string or messages), after up to ``ROUNDS`` rounds of calculator calls."""
    if (bound := _offered(llm)) is None:
        return llm.invoke(prompt)
    messages = _messages(prompt, RULE)
    for _ in range(ROUNDS):
        reply = bound.invoke(messages)
        if not reply.tool_calls:
            return reply
        messages = [*messages, reply, *answers(reply)]
    logger.info("calculator rounds spent (%d); asking for the answer", ROUNDS)
    # A model offered a tool may call one whatever it is told: the last turn offers none, and its earlier
    # calls reach it as text, since a provider may refuse tool calls in a history when no tool is bound.
    return llm.invoke(as_text([*messages, HumanMessage(content=WRAP_UP)]))


def with_calculations(llm: Any, prompt: Any) -> Any:
    """`prompt` with the calculator's results appended: an answer in a schema, which offers no tool, quotes
    them. `prompt` unchanged when the model makes none."""
    if (bound := _offered(llm)) is None:
        return prompt
    messages = _messages(prompt, RULE + FIRST)
    results: list[str] = []
    for _ in range(ROUNDS):
        reply = bound.invoke(messages)
        if not reply.tool_calls:
            break
        said = answers(reply)
        # Only the calculator's sums are quoted; a call it told off is for the model alone.
        results += [str(message.content) for call, message in zip(reply.tool_calls, said, strict=True)
                    if call["name"] == calculate.name and isinstance(call["args"].get("expressions"), list)]
        messages = [*messages, reply, *said]
    if not results:
        return prompt
    block = "Your calculator results; quote these figures exactly:\n" + "\n".join(results)
    return f"{prompt}\n\n{block}" if isinstance(prompt, str) else [*prompt, HumanMessage(content=block)]
