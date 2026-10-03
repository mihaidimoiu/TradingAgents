"""One analyst whose model does not answer must not end the run."""

import pytest

from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.setup import _resilient, report_missing


class APITimeoutError(Exception):
    """Named as the OpenAI and Anthropic SDKs name theirs."""


def spec(key="fundamentals"):
    return next(s for s in build_analyst_execution_plan([key]).specs if s.key == key)


def test_a_timed_out_analyst_files_a_missing_report_instead_of_ending_the_run():
    def slow(state):
        raise APITimeoutError("Request timed out.")

    found = _resilient(spec(), slow)({"messages": []})
    report = found[spec().report_key]
    assert "fundamentals analyst's report is missing" in report and "APITimeoutError" in report
    assert not found["messages"][-1].tool_calls  # routes to END, not back to the tools
    # How a caller tells the note from a report, rather than storing it as one.
    assert report_missing(report) and not report_missing("Revenue grew 12% [FUND-001].")


def test_anything_but_a_provider_failure_still_ends_the_run():
    class Cancelled(RuntimeError):
        pass

    def stopped(state):
        raise Cancelled("stopped by the user")

    with pytest.raises(Cancelled):
        _resilient(spec(), stopped)({"messages": []})


def test_an_answer_passes_through_untouched():
    answer = {"messages": [], "fundamentals_report": "ok"}
    assert _resilient(spec(), lambda state: answer)({"messages": []}) is answer
