"""Earnings calendar context from yfinance: the next report, consensus, surprises.

After upstream #835, with its history keyed on the report date rather than the
quarter end: a quarter that ended before ``as_of_date`` can be reported after
it. The next date, the consensus and the analyst counts are present-day
snapshots with no historical vintage, so a past ``as_of_date`` gets the
surprise history alone (#1300's rule).
"""

from typing import Annotated

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.common import yf_retry

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


def _before_close(stamp: pd.Timestamp, day: pd.Timestamp) -> bool:
    """Announced before the close of `day`; a report with no time counts from the next day."""
    announced = _day(stamp)
    timed = not stamp.hour == stamp.minute == 0
    return announced < day or (announced == day and timed and stamp.hour < 16)


def surprises(dates: pd.DataFrame | None, day: pd.Timestamp, count: int = 4) -> list[str]:
    """The last `count` reports published before the close of `day`."""
    frame = _reports(dates)
    published = pd.Series([_before_close(stamp, day) for stamp in frame.index], index=frame.index, dtype=bool)
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


def _change(now, then) -> float | None:
    try:
        return float(now) / float(then) - 1
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _targets(found: dict | None) -> list[str]:
    if not found or found.get("mean") is None:
        return ["- (unavailable)"]
    return [f"- Current price: {_number(found.get('current'))}",
            f"- Low {_number(found.get('low'))} · median {_number(found.get('median'))} · "
            f"mean {_number(found.get('mean'))} · high {_number(found.get('high'))}",
            f"- Mean target vs current price: {_percent(_change(found['mean'], found.get('current')))}"]


def _trend(frame: pd.DataFrame | None) -> list[str]:
    """How the consensus EPS moved: a rising estimate is the analysts revising up."""
    if frame is None or frame.empty:
        return ["- (unavailable)"]
    columns = ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo")
    rows = [f"| {period} | " + " | ".join(_number(row.get(column)) for column in columns)
            + f" | {_percent(_change(row.get('current'), row.get('90daysAgo')))} |"
            for period, row in frame.iterrows()]
    return ["| Period | Now | 7 days ago | 30 days ago | 60 days ago | 90 days ago | Change over 90 days |",
            "|---|---|---|---|---|---|---|", *rows]


def _revisions(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return ["- (unavailable)"]

    def count(row, *names):
        # yfinance spells one column downLast7Days and the rest ...days.
        return _number(next((row.get(name) for name in names if row.get(name) is not None), None), whole=True)

    return ["| Period | Up, 7 days | Down, 7 days | Up, 30 days | Down, 30 days |", "|---|---|---|---|---|"] + [
        f"| {period} | {count(row, 'upLast7days')} | {count(row, 'downLast7days', 'downLast7Days')} "
        f"| {count(row, 'upLast30days')} | {count(row, 'downLast30days')} |"
        for period, row in frame.iterrows()
    ]


def _fetch(read):
    """One yfinance surface; None on failure, so it costs that section and not the others."""
    try:
        return yf_retry(read)
    except Exception:
        return None


def get_earnings_context(
    ticker: Annotated[str, "ticker symbol of the company"],
    as_of_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
) -> str:
    """The next report and whether it is near, consensus and its revisions, price targets, surprises."""
    canonical = normalize_symbol(ticker)
    today = get_current_date()
    day = pd.Timestamp(as_of_date or today)
    live = (as_of_date or today) >= today
    stock = yf.Ticker(canonical)
    dates = _fetch(lambda: stock.earnings_dates)

    sections = [f"# Earnings context for {canonical} (as of {day:%Y-%m-%d})", ""]
    if live:
        sections += ["## Next earnings report", *next_report(dates, day), "",
                     "## Consensus estimates (upcoming periods)",
                     "Periods: 0q this quarter, +1q the next, 0y this fiscal year, +1y the next.",
                     *_estimates(_fetch(lambda: stock.earnings_estimate), "EPS"),
                     *_estimates(_fetch(lambda: stock.revenue_estimate), "Revenue", _money), "",
                     "## EPS estimate trend (consensus over the last 90 days)",
                     *_trend(_fetch(lambda: stock.eps_trend)), "",
                     "## EPS estimate revisions (analysts revising up or down)",
                     *_revisions(_fetch(lambda: stock.eps_revisions)), "",
                     "## Analyst price targets (12 months)",
                     *_targets(_fetch(lambda: stock.analyst_price_targets)), ""]
    else:
        sections += ["The next report date, consensus estimates, their revisions, price targets and analyst "
                     "counts are withheld: "
                     f"yfinance serves only today's, which postdate {day:%Y-%m-%d}.", ""]
    sections += [f"## Earnings surprises (reported on or before {day:%Y-%m-%d})", *surprises(dates, day)]
    if live:
        sections += ["", "## Analyst recommendations (counts)",
                     *_ratings(_fetch(lambda: stock.recommendations_summary))]
    return "\n".join(sections)
