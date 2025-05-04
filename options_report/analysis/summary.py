"""
summary.py
----------
Functions to build and render weekly performance summary tables.
"""
import pandas as pd
import numpy as np

from .metrics import compute_adjusted_sortino_ratio

def generate_weekly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group trades by weekly close date, compute P/L, win rate, Sharpe, Sortino, etc.
    Guarantees that every calendar week from the first trade through “today”
    appears in the table, even if no trades were closed that week.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Stamp each trade with the Monday that starts its "week"
    # ------------------------------------------------------------------
    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda p: p.start_time.date())

    # ------------------------------------------------------------------
    # 2) Build the *complete* list of weeks to display
    #    • start: Monday of the week that contains the first trade
    #    • end  : Monday of the current week (so the report is always up-to-date)
    # ------------------------------------------------------------------
    first_week = df["WEEK"].min()
    last_week  = pd.Timestamp("today").to_period("W").start_time.date()

    all_weeks = (
        pd.period_range(start=first_week, end=last_week, freq="W")
        .map(lambda p: p.start_time.date())
    )

    # ------------------------------------------------------------------
    # 3) Aggregate the real trades
    # ------------------------------------------------------------------
    weekly = df.groupby("WEEK").agg(
        net_profit     = ("NET PROFIT", "sum"),
        num_trades     = ("NET PROFIT", "count"),
        avg_days_held  = ("DAYS_HELD",  "mean"),
        avg_return     = ("RETURN",     "mean"),
        winning_trades = ("NET PROFIT", lambda x: (x > 0).sum())
    )

    # ------------------------------------------------------------------
    # 4) Re-index onto *all_weeks* so gaps appear, then fill sensible
    #    defaults for “no-trade” rows.
    # ------------------------------------------------------------------
    weekly = weekly.reindex(all_weeks)

    # numeric zeros where appropriate
    weekly["net_profit"]     = weekly["net_profit"].fillna(0)
    weekly["num_trades"]     = weekly["num_trades"].fillna(0).astype(int)
    weekly["winning_trades"] = weekly["winning_trades"].fillna(0).astype(int)
    # leave avg_return / ratios NaN for no-trade weeks; they’ll become “—” later

    # ------------------------------------------------------------------
    # 5) Risk metrics (Sharpe & Sortino) — map from the pre-computed series
    # ------------------------------------------------------------------
    sharpe_by_week = df.groupby("WEEK")["RETURN"].apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    )
    sortino_by_week = df.groupby("WEEK")["RETURN"].apply(
        lambda x: compute_adjusted_sortino_ratio(pd.DataFrame({"RETURN": x}))
    )

    weekly["sharpe_ratio"]          = weekly.index.map(sharpe_by_week).astype(float)
    weekly["adjusted_sortino_ratio"] = weekly.index.map(sortino_by_week).astype(float)

    # ------------------------------------------------------------------
    # 6) Win-rate, rounding, and formatting
    # ------------------------------------------------------------------
    weekly["win_rate"] = np.where(
        weekly["num_trades"] > 0,
        (weekly["winning_trades"] / weekly["num_trades"]) * 100,
        np.nan
    )

    # round numeric
    numeric_cols = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric_cols] = weekly[numeric_cols].round(2)

    # tidy up
    weekly = weekly.reset_index().rename(columns={"index": "WEEK"})
    weekly.insert(0, "Week", range(1, len(weekly) + 1))

    # ------------------------------------------------------------------
    # 7) Pretty-print & HTML-safe strings (unchanged from your version,
    #    but with graceful handling of NaNs)
    # ------------------------------------------------------------------
    def fmt_date(d):
        return (
            f'<span style="white-space: nowrap;" '
            f'data-sort="{d.strftime("%Y-%m-%d")}">{d.day} {d.strftime("%b %Y")}</span>'
        )

    weekly["Starts"] = weekly["WEEK"].apply(fmt_date)
    weekly["net_profit"] = weekly["net_profit"].apply(
        lambda x: (
            "$0"                           # show $0 for tradeless weeks
            if x == 0 else
            f"-${abs(int(round(x))):,}"    # negative weeks
            if x < 0 else
            f"${int(round(x)):,}"          # positive weeks
        )
    )
    weekly["win_rate"] = weekly["win_rate"].apply(
        lambda x: "—" if pd.isna(x) else f"{round(x)}%"
    )
    weekly["avg_return"] = weekly["avg_return"].apply(
        lambda x: "—" if pd.isna(x) else round(x, 2)
    )
    weekly["sharpe_ratio"] = weekly["sharpe_ratio"].apply(
        lambda x: "—" if pd.isna(x) else round(x, 2)
    )
    weekly["Adjusted Sortino"] = weekly["adjusted_sortino_ratio"].apply(
        lambda x: "—" if pd.isna(x) else ("inf" if np.isinf(x) else round(x, 2))
    )

    # Rename & reorder
    weekly = weekly.rename(columns={
        "Starts": "Starting",
        "net_profit": "Profit",
        "num_trades": "Trades",
        "avg_return": "Trade Return",
        "win_rate": "Win Rate",
        "sharpe_ratio": "Sharpe",
        "Adjusted Sortino": "Sortino"
    })
    weekly = weekly[["Week", "Starting", "Profit", "Trades", "Win Rate",
                     "Trade Return", "Sharpe", "Sortino"]]

    # Header tool-tips (unchanged)
    tooltips = {
        "Week": "Sequential week number.",
        "Starting": "First trading day of the week.",
        "Trades": "Number of trades closed this week.",
        "Profit": "Change in equity for the week.",
        "Trade Return": "Average return for trades closed this week.",
        "Win Rate": "Ratio of wins to trades for the week.",
        "Sharpe": "Sharpe Ratio: a measure of risk-adjusted return.",
        "Sortino": "Adjusted Sortino Ratio: ignores volatility from gains."
    }
    weekly.rename(columns={
        col: f'<span title="{tooltips[col]}">{col}</span>' for col in weekly.columns
    }, inplace=True)

    return weekly

def generate_weekly_summary_html(df: pd.DataFrame) -> str:
    """
    Render the weekly summary DataFrame to an HTML table.
    """
    table = generate_weekly_summary(df)
    return table.to_html(escape=False, index=False, classes='sortable-table')