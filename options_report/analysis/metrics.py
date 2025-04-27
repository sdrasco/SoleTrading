"""
metrics.py
--------
Compute portfolio KPIs and data loading utility.
"""
import pandas as pd
import numpy as np

def load_trades(filename):
    """
    Load the CSV and parse key date columns.  Derive days held, weekday, and trade direction.
    """
    df = pd.read_csv(filename, parse_dates=["OPEN DATE", "CLOSE DATE", "EXPIRATION"])
    # DAYS_HELD
    if "DAYS_HELD" not in df.columns:
        df["DAYS_HELD"] = (df["CLOSE DATE"] - df["OPEN DATE"]).dt.days
    # Weekday at open/close
    df["DAY_OF_WEEK_AT_OPEN"] = df["OPEN DATE"].dt.day_name()
    df["DAY_OF_WEEK_AT_CLOSE"] = df["CLOSE DATE"].dt.day_name()
    # Trade direction from OPTION TYPE
    if "OPTION TYPE" in df.columns:
        df["OPTION TYPE"] = df["OPTION TYPE"].str.strip().str.upper()
        df["TRADE DIRECTION"] = df["OPTION TYPE"].map({
            "C": "Call",
            "P": "Put",
            "CALL": "Call",
            "PUT": "Put",
        })
    return df

def compute_adjusted_sortino_ratio(df, target_return=0):
    """
    Compute the 'Adjusted Sortino Ratio' accounting for skewness and kurtosis.
    """
    returns = df["RETURN"]
    mean_return = returns.mean()
    # Downside deviation
    downside = returns[returns < target_return]
    if downside.empty:
        return np.nan
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev == 0:
        return np.nan
    sortino = (mean_return - target_return) / downside_dev
    skew = returns.skew()
    kurt = returns.kurtosis()
    # Adjust for skewness and kurtosis
    return sortino * (1 + (skew * sortino / 6) - (kurt * sortino**2 / 24))

def compute_kpis(df):
    """
    Compute net profit, return stats, Sharpe/Sortino ratios, drawdown, and equity curve.
    Returns a dict of KPI values and an equity_curve DataFrame for plotting.
    """
    total_trades = len(df)
    net_profit = df["NET PROFIT"].sum()
    avg_trade_return = df["RETURN"].mean()
    win_rate = (df["NET PROFIT"] > 0).mean() * 100
    std_return = df["RETURN"].std()
    sharpe_ratio = (df["RETURN"].mean() / std_return) if std_return > 0 else np.nan
    # Equity curve
    df_sorted = df.sort_values(by="CLOSE DATE").copy()
    df_sorted["CUMULATIVE_NET_PROFIT"] = df_sorted["NET PROFIT"].cumsum()
    # Drawdown
    cumulative = df_sorted["CUMULATIVE_NET_PROFIT"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = abs(drawdown.min() * 100)
    volatility = std_return
    adjusted_sortino = compute_adjusted_sortino_ratio(df)
    return {
        "total_trades": total_trades,
        "net_profit": net_profit,
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility": volatility,
        "equity_curve": df_sorted,
        "adjusted_sortino_ratio": adjusted_sortino,
    }