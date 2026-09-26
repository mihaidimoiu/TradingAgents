"""Earnings calendar context from yfinance: the next report, consensus, surprises.

After upstream #835, with its history keyed on the report date rather than the
quarter end: a quarter that ended before ``curr_date`` can be reported after
it. The next date, the consensus and the analyst counts are present-day
snapshots with no historical vintage, so a past ``curr_date`` gets the
surprise history alone (#1300's rule).
"""

from typing import Annotated

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import yf_retry

# A report this many trading days ahead or fewer is flagged: the position would
# hold through a gap no stop in daily ATRs was sized for.
NEAR_EARNINGS_DAYS = 5


def _number(value, whole: bool = False) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{int(value)}" if whole else f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _percent(value) -> str:
    try:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _day(stamp: pd.Timestamp) -> pd.Timestamp:
    """The report's calendar day where it was announced (yfinance: New York)."""
    return (stamp.tz_localize(None) if stamp.tzinfo else stamp).normalize()


def _reports(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    return frame[frame.index.notna()].sort_index()


def next_report(dates: pd.DataFrame | None, day: pd.Timestamp) -> list[str]:
    """The first report on or after `day`, and whether it falls inside the flag window."""
    upcoming = [stamp for stamp in _reports(dates).index if _day(stamp) >= day]
    if not upcoming:
        return ["- No earnings date scheduled."]
    stamp = upcoming[0]
    timing = "" if stamp.hour == stamp.minute == 0 else (
        " (after the close)" if stamp.hour >= 16 else " (before the open)" if stamp.hour < 10 else "")
    sessions = max(0, len(pd.bdate_range(day, _day(stamp))) - 1)
    near = sessions <= NEAR_EARNINGS_DAYS
    return [f"- Date: {stamp:%Y-%m-%d}{timing}", f"- Trading days until: {sessions}",
            f"- Near earnings (within {NEAR_EARNINGS_DAYS} trading days): "
            + ("YES, a position would hold through the report" if near else "no")]


def surprises(dates: pd.DataFrame | None, day: pd.Timestamp, count: int = 4) -> list[str]:
    """The last `count` reports published on or before `day`."""
    frame = _reports(dates)
    published = pd.Series([_day(stamp) <= day for stamp in frame.index], index=frame.index, dtype=bool)
    done = frame[published & frame["Reported EPS"].notna()]
    if done.empty:
        return [f"- (no report on or before {day:%Y-%m-%d})"]
    # yfinance's Surprise(%) is already in percent.
    return ["| Reported | EPS estimate | EPS actual | Surprise |", "|---|---|---|---|"] + [
        f"| {stamp:%Y-%m-%d} | {_number(row['EPS Estimate'])} | {_number(row['Reported EPS'])} "
        f"| {_number(row['Surprise(%)'])}% |"
        for stamp, row in done.sort_index(ascending=False).head(count).iterrows()
    ]


def _money(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        size = abs(float(value))
        for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if size >= scale:
                return f"{float(value) / scale:,.2f}{suffix}"
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _estimates(frame: pd.DataFrame | None, label: str, amount=_number) -> list[str]:
    if frame is None or frame.empty:
        return [f"### {label}", "- (unavailable)"]
    return [f"### {label}", "| Period | Avg | Low | High | Analysts | YoY growth |", "|---|---|---|---|---|---|"] + [
        f"| {period} | {amount(row.get('avg'))} | {amount(row.get('low'))} | {amount(row.get('high'))} "
        f"| {_number(row.get('numberOfAnalysts'), whole=True)} | {_percent(row.get('growth'))} |"
        for period, row in frame.iterrows()
    ]


def _ratings(summary: pd.DataFrame | None) -> list[str]:
    if summary is None or summary.empty:
        return ["- (unavailable)"]
    return ["| Period | Strong buy | Buy | Hold | Sell | Strong sell |", "|---|---|---|---|---|---|"] + [
        f"| {row.get('period', '?')} | {_number(row.get('strongBuy'), True)} | {_number(row.get('buy'), True)} "
        f"| {_number(row.get('hold'), True)} | {_number(row.get('sell'), True)} "
        f"| {_number(row.get('strongSell'), True)} |"
        for _, row in summary.iterrows()
    ]


def _fetch(read):
    """One yfinance surface; None on failure, so it costs that section and not the others."""
    try:
        return yf_retry(read)
    except Exception:
        return None


def get_earnings_context(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
) -> str:
    """The next earnings report and whether it is near, consensus, surprise history and analyst counts."""
    canonical = normalize_symbol(ticker)
    today = get_current_date()
    day = pd.Timestamp(curr_date or today)
    live = (curr_date or today) >= today
    stock = yf.Ticker(canonical)
    dates = _fetch(lambda: stock.earnings_dates)

    sections = [f"# Earnings context for {canonical} (as of {day:%Y-%m-%d})", ""]
    if live:
        sections += ["## Next earnings report", *next_report(dates, day), "",
                     "## Consensus estimates (upcoming periods)",
                     *_estimates(_fetch(lambda: stock.earnings_estimate), "EPS"),
                     *_estimates(_fetch(lambda: stock.revenue_estimate), "Revenue", _money), ""]
    else:
        sections += ["The next report date, consensus estimates and analyst counts are withheld: "
                     f"yfinance serves only today's, which postdate {day:%Y-%m-%d}.", ""]
    sections += [f"## Earnings surprises (reported on or before {day:%Y-%m-%d})", *surprises(dates, day)]
    if live:
        sections += ["", "## Analyst recommendations (counts)",
                     *_ratings(_fetch(lambda: stock.recommendations_summary))]
    return "\n".join(sections)
