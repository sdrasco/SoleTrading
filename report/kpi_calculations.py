# report/kpi_calculations.py

import numpy as np
import pandas as pd

def compute_adjusted_sortino_ratio(df, target_return=0):
    """
    Compute an 'Adjusted Sortino' for a DataFrame with a 'RETURN' column.
    """
    if "RETURN" not in df.columns:
        return np.nan
    returns = df["RETURN"].dropna()
    if returns.empty:
        return np.nan

    mean_return = returns.mean()
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.nan
    downside_deviation = (downside_returns**2).mean() ** 0.5
    if downside_deviation == 0:
        return np.nan

    sortino_ratio = (mean_return - target_return) / downside_deviation
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    adjusted_sortino = sortino_ratio * (
        1 + (skewness * sortino_ratio / 6) - (kurtosis * sortino_ratio**2 / 24)
    )
    return adjusted_sortino


def compute_kpis(df):
    """
    Compute a dictionary of various KPI metrics:
      - total_trades, net_profit, avg_trade_return, win_rate, etc.
    Also calculates an equity_curve that can be used for charting.
    """
    total_trades = len(df)
    if "NET PROFIT" not in df.columns or df.empty:
        return {
            "total_trades": total_trades,
            "net_profit": 0,
            "avg_trade_return": 0,
            "win_rate": 0,
            "sharpe_ratio": np.nan,
            "max_drawdown_pct": 0,
            "volatility": 0,
            "equity_curve": df,
            "adjusted_sortino_ratio": np.nan
        }
    
    net_profit = df["NET PROFIT"].sum()

    if "RETURN" in df.columns:
        avg_trade_return = df["RETURN"].mean()
        std_return = df["RETURN"].std()
        sharpe_ratio = (
            df["RETURN"].mean() / std_return if std_return > 0 else np.nan
        )
    else:
        avg_trade_return = 0
        sharpe_ratio = np.nan
    
    win_rate = (df["NET PROFIT"] > 0).mean() * 100

    # For the equity curve
    if "CLOSE DATE" in df.columns:
        df_sorted = df.sort_values("CLOSE DATE").copy()
    else:
        df_sorted = df.copy()
    df_sorted["CUMULATIVE_NET_PROFIT"] = df_sorted["NET PROFIT"].cumsum()

    cumulative = df_sorted["CUMULATIVE_NET_PROFIT"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = abs(drawdown.min() * 100)

    if "RETURN" in df.columns:
        volatility = df["RETURN"].std()
        adjusted_sortino_ratio = compute_adjusted_sortino_ratio(df)
    else:
        volatility = 0
        adjusted_sortino_ratio = np.nan

    return {
        "total_trades": total_trades,
        "net_profit": net_profit,
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility": volatility,
        "equity_curve": df_sorted,
        "adjusted_sortino_ratio": adjusted_sortino_ratio
    }


def generate_weekly_summary(df):
    """
    Summarize trades by week (based on 'CLOSE DATE'),
    returning a DataFrame with columns for net profit, number of trades, etc.
    """
    if df.empty or "CLOSE DATE" not in df.columns:
        return pd.DataFrame()

    if "RETURN" not in df.columns:
        df["RETURN"] = 0.0

    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda r: r.start_time.date())
    
    weekly = df.groupby("WEEK").agg(
        net_profit=("NET PROFIT", "sum"),
        num_trades=("NET PROFIT", "count"),
        avg_days_held=("DAYS_HELD", "mean"),
        avg_return=("RETURN", "mean"),
        winning_trades=("NET PROFIT", lambda x: (x > 0).sum())
    ).reset_index()

    weekly["win_rate"] = (weekly["winning_trades"] / weekly["num_trades"]) * 100

    # Optionally drop columns you don’t need
    weekly.drop(columns=["avg_days_held", "winning_trades"], inplace=True, errors="ignore")

    # Compute weekly Sharpe / Sortino if desired
    # We'll do it by grouping the original DataFrame again
    def weekly_adjusted_sortino(sub):
        return compute_adjusted_sortino_ratio(sub)

    adj_sortino_series = df.groupby("WEEK", group_keys=False).apply(weekly_adjusted_sortino)
    
    weekly["sharpe_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    ).values
    weekly["adjusted_sortino_ratio"] = adj_sortino_series.reindex(weekly["WEEK"]).values

    numeric_cols = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric_cols] = weekly[numeric_cols].round(2)

    # Insert a human-friendly "Week" column
    weekly.insert(0, "Week", range(1, len(weekly) + 1))

    # rename
    weekly.rename(columns={"WEEK": "Starts"}, inplace=True)

    return weekly


def generate_weekly_summary_html(df):
    """
    Shortcut to produce an HTML table from generate_weekly_summary().
    """
    weekly = generate_weekly_summary(df)
    return weekly.to_html(escape=False, index=False, classes='sortable-table')