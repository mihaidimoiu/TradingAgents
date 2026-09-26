"""An ETF read as a fund: what it holds and how concentrated, not income statements.

After upstream #819. yfinance's fund data is a present-day snapshot with no
historical vintage, so a past ``curr_date`` is withheld (#1300's rule).
"""

from typing import Annotated

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import yf_retry


def _share(value) -> str:
    try:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def concentration(weights: list[float]) -> dict[str, float]:
    """Largest holding, the top ten together, and the Herfindahl index over the holdings listed."""
    ordered = sorted((w for w in weights if w == w), reverse=True)
    return {"largest": ordered[0] if ordered else 0.0, "top_10": sum(ordered[:10]),
            "herfindahl": sum(w * w for w in ordered)}


def _get(read):
    try:
        return yf_retry(read)
    except Exception:
        return None


def get_fund_profile(
    ticker: Annotated[str, "ETF ticker symbol"],
    curr_date: Annotated[str, "analysis date in YYYY-MM-DD format"] = None,
) -> str:
    """The fund's category, costs, asset mix, sector weights, top holdings and their concentration."""
    canonical = normalize_symbol(ticker)
    today = get_current_date()
    if curr_date and curr_date < today:
        return (f"# Fund profile for {canonical}\n\nWithheld for {curr_date}: yfinance serves the "
                f"fund's holdings and costs only as of today, which postdate {curr_date}.")
    data = _get(lambda: yf.Ticker(canonical).funds_data)
    if data is None:
        return f"# Fund profile for {canonical}\n\n- (unavailable)"

    lines = [f"# Fund profile for {canonical} (as of {today})"]
    overview = _get(lambda: data.fund_overview) or {}
    if overview:
        lines += ["", f"- Category: {overview.get('categoryName', 'n/a')}",
                  f"- Family: {overview.get('family', 'n/a')}", f"- Type: {overview.get('legalType', 'n/a')}"]
    if description := _get(lambda: data.description):
        lines += ["", "## Strategy", description.strip()]

    operations = _get(lambda: data.fund_operations)
    if operations is not None and not operations.empty:
        own, average = operations.iloc[:, 0], operations.iloc[:, -1]
        lines += ["", "## Costs and turnover (fund / category average)"]
        for label, key in (("Expense ratio", "Annual Report Expense Ratio"),
                           ("Holdings turnover", "Annual Holdings Turnover")):
            if key in own.index:
                lines.append(f"- {label}: {_share(own[key])} / {_share(average[key])}")
        if "Total Net Assets" in own.index:
            lines.append(f"- Total net assets: {own['Total Net Assets']:,.2f} (as yfinance reports it)")

    assets = _get(lambda: data.asset_classes) or {}
    if assets:
        lines += ["", "## Asset mix"] + [
            f"- {name.removesuffix('Position')}: {_share(weight)}"
            for name, weight in sorted(assets.items(), key=lambda pair: -(pair[1] or 0)) if weight
        ]
    sectors = _get(lambda: data.sector_weightings) or {}
    if sectors:
        lines += ["", "## Sector weights"] + [
            f"- {name.replace('_', ' ')}: {_share(weight)}"
            for name, weight in sorted(sectors.items(), key=lambda pair: -(pair[1] or 0)) if weight
        ]

    holdings = _get(lambda: data.top_holdings)
    if holdings is not None and not holdings.empty:
        weights = [float(w) for w in holdings["Holding Percent"]]
        measured = concentration(weights)
        lines += ["", f"## Top {len(holdings)} holdings", "| Symbol | Name | Weight |", "|---|---|---|"]
        lines += [f"| {symbol} | {row['Name']} | {_share(row['Holding Percent'])} |"
                  for symbol, row in holdings.iterrows()]
        lines += ["", "## Concentration",
                  f"- Largest holding: {_share(measured['largest'])}",
                  f"- Top {min(10, len(weights))} together: {_share(measured['top_10'])}",
                  f"- Herfindahl index over these holdings: {measured['herfindahl']:.4f} "
                  "(sum of squared weights; 1/N for N equal holdings)"]
    return "\n".join(lines)
