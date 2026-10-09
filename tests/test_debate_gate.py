"""The optional Jev debate gate (#1404): early stop only on a confident decision-ready
transcript, never on failure, never before both sides have spoken once, and never
consulted unless the config enables it and a TypeSafe key is set.
"""

from unittest.mock import patch

import pytest

from tradingagents.agents.debate_gate import DebateGateVerdict, jev_debate_gate
from tradingagents.agents.post_screen import TypeSafeError
from tradingagents.default_config import build_default_config
from tradingagents.graph.conditional_logic import ConditionalLogic

ENABLED = {"jev_debate_gate": True}


def _state(count: int, current: str = "Bull researcher: x") -> dict:
    return {
        "investment_debate_state": {
            "count": count,
            "current_response": current,
            "bull_history": "bull argument",
            "bear_history": "bear argument",
            "history": "bull argument bear argument",
            "judge_decision": "",
        }
    }


def _debate(count: int = 2, bull: str = "b", bear: str = "r") -> dict:
    return {"count": count, "bull_history": bull, "bear_history": bear}


@pytest.mark.unit
def test_gate_true_ends_debate_after_a_complete_round():
    logic = ConditionalLogic(max_debate_rounds=3, debate_gate=lambda s: True)
    assert logic.should_continue_debate(_state(2)) == "Research Manager"


@pytest.mark.unit
def test_gate_none_keeps_alternation():
    logic = ConditionalLogic(max_debate_rounds=3, debate_gate=lambda s: None)
    assert logic.should_continue_debate(_state(2)) == "Bear Researcher"


@pytest.mark.unit
def test_round_cap_wins_over_gate():
    logic = ConditionalLogic(max_debate_rounds=1, debate_gate=lambda s: None)
    assert logic.should_continue_debate(_state(2)) == "Research Manager"


@pytest.mark.unit
def test_gate_not_consulted_mid_round():
    calls = []

    def gate(state) -> bool | None:
        calls.append(state["count"])
        return True

    logic = ConditionalLogic(max_debate_rounds=3, debate_gate=gate)
    assert logic.should_continue_debate(_state(1)) == "Bear Researcher"
    assert calls == []


@pytest.mark.unit
def test_gate_false_keeps_alternation():
    logic = ConditionalLogic(max_debate_rounds=3, debate_gate=lambda s: False)
    assert logic.should_continue_debate(_state(2)) == "Bear Researcher"


@pytest.mark.unit
def test_factory_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_debate_gate(ENABLED) is None


@pytest.mark.unit
def test_factory_disabled_returns_none_even_with_a_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert jev_debate_gate({"jev_debate_gate": False}) is None
    assert jev_debate_gate(build_default_config()) is None


@pytest.mark.unit
def test_factory_gate_maps_answers_to_bool(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def fake_system_one(state, questions):
        assert "decision_ready" in questions
        assert state["rounds_completed"] == 1
        return {"decision_ready": {"noul": 0.9}}

    with patch("tradingagents.agents.debate_gate.system_one", fake_system_one):
        gate = jev_debate_gate(ENABLED)
        assert gate is not None
        assert gate(_debate()) is True


@pytest.mark.unit
def test_factory_gate_degrades_on_error(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    with patch("tradingagents.agents.debate_gate.system_one", side_effect=TypeSafeError("HTTP 500")):
        assert jev_debate_gate(ENABLED)(_debate()) is None


@pytest.mark.unit
def test_env_override_registers(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_JEV_DEBATE_GATE", "true")
    assert build_default_config()["jev_debate_gate"] is True


@pytest.mark.unit
def test_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_JEV_DEBATE_GATE", raising=False)
    assert build_default_config()["jev_debate_gate"] is False


@pytest.mark.unit
def test_each_judgement_is_reported(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    verdicts: list[DebateGateVerdict] = []
    answers = iter([{"decision_ready": {"noul": 0.4}}, {"decision_ready": {"noul": 0.8}}])

    with patch("tradingagents.agents.debate_gate.system_one", lambda state, questions: next(answers)):
        gate = jev_debate_gate(ENABLED, on_verdict=verdicts.append)
        assert gate(_debate(2)) is False
        assert gate(_debate(4)) is True

    assert verdicts == [
        DebateGateVerdict(round=1, score=0.4, stop=False),
        DebateGateVerdict(round=2, score=0.8, stop=True),
    ]


@pytest.mark.unit
def test_a_failed_judgement_is_reported_without_a_score(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    verdicts: list[DebateGateVerdict] = []

    with patch("tradingagents.agents.debate_gate.system_one", side_effect=TypeSafeError("HTTP 500")):
        jev_debate_gate(ENABLED, on_verdict=verdicts.append)(_debate())

    assert verdicts == [DebateGateVerdict(round=1, score=None, stop=False, failure="HTTP 500")]


@pytest.mark.unit
def test_only_the_tail_of_each_side_leaves_the_machine(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    sent = {}

    def fake_system_one(state, questions):
        sent.update(state)
        return {"decision_ready": {"noul": 0.1}}

    with patch("tradingagents.agents.debate_gate.system_one", fake_system_one):
        jev_debate_gate(ENABLED)(_debate(bull="B" * 10_000 + "end", bear="short"))

    assert set(sent) == {"rounds_completed", "bull_argument", "bear_argument"}
    assert len(sent["bull_argument"]) == 4000 and sent["bull_argument"].endswith("end")
    assert sent["bear_argument"] == "short"
