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
    Returns a formatted DataFrame with tooltips and HTML-ready column headers.
    """
    df = df.copy()
    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly = df.groupby("WEEK").agg(
        net_profit=("NET PROFIT", "sum"),
        num_trades=("NET PROFIT", "count"),
        avg_days_held=("DAYS_HELD", "mean"),
        avg_return=("RETURN", "mean"),
        winning_trades=("NET PROFIT", lambda x: (x > 0).sum())
    ).reset_index()
    weekly["win_rate"] = (weekly["winning_trades"] / weekly["num_trades"]) * 100
    weekly.drop(columns=["avg_days_held", "winning_trades"], inplace=True)
    # Sharpe by week
    weekly["sharpe_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    ).values
    # Sortino by week
    weekly["adjusted_sortino_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: compute_adjusted_sortino_ratio(pd.DataFrame({"RETURN": x}))
    ).values
    # Round numeric
    numeric = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric] = weekly[numeric].round(2)
    # Adjusted Sortino inf handling
    weekly["Adjusted Sortino"] = weekly["adjusted_sortino_ratio"].apply(
        lambda x: "inf" if np.isnan(x) else x
    )
    # Insert sequential week number
    weekly.insert(0, "Week", range(1, len(weekly) + 1))
    weekly.rename(columns={"WEEK": "Starts"}, inplace=True)
    weekly.drop(columns=["adjusted_sortino_ratio"], inplace=True)
    # Helper formatters
    def fmt_date(d):
        return (
            f'<span style="white-space: nowrap;" '
            f'data-sort="{d.strftime("%Y-%m-%d")}">'  # noqa: W605
            f'{d.day} {d.strftime("%b %Y")}</span>'
        )
    weekly["Starts"] = weekly["Starts"].apply(fmt_date)
    weekly["net_profit"] = weekly["net_profit"].apply(
        lambda x: f"-${abs(int(round(x))):,}" if x < 0 else f"${int(round(x)):,}"
    )
    weekly["win_rate"] = weekly["win_rate"].apply(lambda x: f"{round(x)}%")
    # Rename and reorder
    weekly.rename(columns={
        "Starts": "Starting",
        "net_profit": "Profit",
        "num_trades": "Trades",
        "avg_return": "Trade Return",
        "win_rate": "Win Rate",
        "sharpe_ratio": "Sharpe",
        "Adjusted Sortino": "Sortino"
    }, inplace=True)
    weekly = weekly[["Week", "Starting", "Profit", "Trades", "Win Rate", "Trade Return", "Sharpe", "Sortino"]]
    # Add header tooltips
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
    weekly.rename(columns={col: f'<span title="{tooltips[col]}">{col}</span>' for col in weekly.columns}, inplace=True)
    return weekly

def generate_weekly_summary_html(df: pd.DataFrame) -> str:
    """
    Render the weekly summary DataFrame to an HTML table.
    """
    table = generate_weekly_summary(df)
    return table.to_html(escape=False, index=False, classes='sortable-table')