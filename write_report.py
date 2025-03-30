import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io, base64, datetime
from jinja2 import Template

# ------------------------------
# Helper Functions for Analysis
# ------------------------------

def load_trades(filename):
    """
    Load the CSV and parse the relevant date columns.
    Rename columns and derive additional columns for analysis.
    """
    # Parse relevant date columns
    df = pd.read_csv(
        filename,
        parse_dates=["OPEN DATE", "CLOSE DATE", "EXPIRATION"]
    )
    
    # If DAYS_HELD is not in the file, calculate it
    if "DAYS_HELD" not in df.columns:
        df["DAYS_HELD"] = (df["CLOSE DATE"] - df["OPEN DATE"]).dt.days

    # Create day-of-week columns
    df["DAY_OF_WEEK_AT_OPEN"] = df["OPEN DATE"].dt.day_name()
    df["DAY_OF_WEEK_AT_CLOSE"] = df["CLOSE DATE"].dt.day_name()

    # Derive trade direction from OPTION TYPE
    if "OPTION TYPE" in df.columns:
        df["OPTION TYPE"] = df["OPTION TYPE"].str.strip().str.upper()
        df["TRADE DIRECTION"] = df["OPTION TYPE"].map({
            "C": "CALL", "P": "PUT", "CALL": "CALL", "PUT": "PUT"
        })
    
    return df

def compute_adjusted_sortino_ratio(df, target_return=0):
    returns = df["RETURN"]
    mean_return = returns.mean()
    
    # Calculate downside deviation based on returns below the target
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.nan
    downside_deviation = np.sqrt((downside_returns**2).mean())
    if downside_deviation == 0:
        return np.nan

    sortino_ratio = (mean_return - target_return) / downside_deviation
    
    # Adjust using skewness and kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    adjusted_sortino = sortino_ratio * (
        1 + (skewness * sortino_ratio / 6) - (kurtosis * sortino_ratio**2 / 24)
    )
    return adjusted_sortino

def compute_kpis(df):
    total_trades = len(df)
    net_profit = df["NET PROFIT"].sum()
    avg_trade_return = df["RETURN"].mean()
    win_rate = (df["NET PROFIT"] > 0).mean() * 100
    sharpe_ratio = df["RETURN"].mean() / df["RETURN"].std() if df["RETURN"].std() > 0 else np.nan

    # Sort for equity curve (trade-by-trade cumulative net profit)
    df_sorted = df.sort_values(by="CLOSE DATE").copy()
    df_sorted["CUMULATIVE_NET_PROFIT"] = df_sorted["NET PROFIT"].cumsum()

    # Drawdown calculation
    cumulative = df_sorted["CUMULATIVE_NET_PROFIT"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = drawdown.min() * 100

    volatility = df["RETURN"].std()
    
    adjusted_sortino_ratio = compute_adjusted_sortino_ratio(df)

    return {
        "total_trades": total_trades,
        "net_profit": net_profit,
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility": volatility,
        "equity_curve": df_sorted,  # still used for other KPIs
        "adjusted_sortino_ratio": adjusted_sortino_ratio
    }

def generate_weekly_summary(df):
    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda r: r.start_time.date())
    weekly = df.groupby("WEEK").agg(
        net_profit=("NET PROFIT", "sum"),
        num_trades=("NET PROFIT", "count"),
        avg_days_held=("DAYS_HELD", "mean"),
        avg_return=("RETURN", "mean"),
        winning_trades=("NET PROFIT", lambda x: (x > 0).sum())
    ).reset_index()

    weekly["win_rate"] = (weekly["winning_trades"] / weekly["num_trades"]) * 100
    weekly["sharpe_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    ).values
    
    # Group only on the "RETURN" column to avoid the deprecation warning
    weekly["adjusted_sortino_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: compute_adjusted_sortino_ratio(pd.DataFrame({"RETURN": x}))
    ).values

    numeric_cols = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric_cols] = weekly[numeric_cols].round(2)

    # Explicitly label infinite Sortino Ratios
    weekly["Adjusted Sortino"] = weekly["adjusted_sortino_ratio"].apply(
        lambda x: "inf" if np.isnan(x) else x
    )

    # Add sequential Week # at the far left and rename columns
    weekly.insert(0, "Week #", range(1, len(weekly) + 1))
    weekly.rename(columns={"WEEK": "Starts"}, inplace=True)

    weekly.drop(columns=["adjusted_sortino_ratio"], inplace=True)
    weekly.columns = [col.replace("_", " ").title() for col in weekly.columns]
    return weekly

def generate_equity_curve_plot(trades_df):
    df = trades_df.copy()
    # Convert CLOSE DATE to a proper datetime (only date portion)
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    
    # Group by date: sum NET PROFIT for equity and count trades for volume
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    daily_volume = df.groupby('DATE').size().sort_index()  # number of trades per day

    # Create a complete date range from the first to last date.
    all_dates = pd.date_range(start=daily_profit.index.min(),
                              end=daily_profit.index.max(), freq='D')
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    daily_volume = daily_volume.reindex(all_dates, fill_value=0)
    
    # Compute cumulative equity (carry forward on days with no trades)
    cumulative_equity = daily_profit.cumsum()
    
    # Transform dates to week numbers: 1 + (days since start)/7 (fractional values)
    start_date = cumulative_equity.index[0]
    x_values = 1 + (cumulative_equity.index - start_date).days / 7
    
    # Create the figure and a twin axis for volume
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Adjust z-orders: make ax1 (the equity axis) drawn above ax2.
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Hide ax1 background so ax2 shows through if needed.
    ax2.set_zorder(1)
    
    # Define colors
    muted_blue = "#4a90e2"     # Equity color
    light_grey = "#D3D3D3"     # Volume color

    # Plot the volume bars on the right y-axis (ax2)
    bar_width = 0.1  # adjust bar width as needed
    ax2.bar(x_values, daily_volume.values, width=bar_width, color=light_grey,
            alpha=0.5, zorder=1)
    ax2.set_ylabel("Number of Trades", fontsize=21, color=light_grey)
    ax2.tick_params(axis='y', labelsize=21, colors=light_grey)
    
    # Plot the equity curve on the left y-axis (ax1)
    ax1.plot(x_values, cumulative_equity.values, marker='o', markersize=12,
             linestyle='-', color=muted_blue, zorder=2)
    ax1.set_xlabel("Week", fontsize=21)
    ax1.set_ylabel("Equity ($)", fontsize=21, color=muted_blue)
    ax1.tick_params(axis='x', labelsize=21)
    ax1.tick_params(axis='y', labelsize=21, colors=muted_blue)

    # Set x-ticks at integer week values
    max_week = math.ceil(x_values.max())
    ticks = np.arange(1, max_week + 1)
    ax1.set_xticks(ticks)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
    
def generate_trade_return_histogram(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["RETURN"], bins=20, edgecolor="black")
    plt.xlabel("Trade Return", fontsize=21)
    plt.ylabel("Number of Trades", fontsize=21)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_win_rate_by_symbol_plot(df):
    summary = df.groupby('SYMBOL').agg(
        win_rate=('NET PROFIT', lambda x: (x > 0).mean() * 100),
        num_trades=('NET PROFIT', 'count')
    )
    summary = summary.sort_values(by=['win_rate', 'num_trades'], ascending=[True, False])[::-1]
    labels = [f"{sym} ({n})" for sym, n in zip(summary.index, summary['num_trades'])]
    num_bars = len(labels)
    bar_height = 0.6
    plt.figure(figsize=(8, max(3, 0.4 * num_bars)))
    plt.barh(labels, summary['win_rate'], color='skyblue', edgecolor='black', height=bar_height)
    
    # Increase font sizes for labels and ticks
    plt.ylabel('Symbol (Number of Trades)', fontsize=16)
    plt.xlabel('Win Rate (%)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.xlim(0, 100)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_feature_plots(df, features):
    plots = {}
    df["WIN"] = (df["NET PROFIT"] > 0).astype(int)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for feature in features:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df[feature]):
            df_win = df[df["WIN"] == 1]
            df_loss = df[df["WIN"] == 0]
            plt.hist(df_win[feature].dropna(), bins=20, alpha=0.5, label="Wins")
            plt.hist(df_loss[feature].dropna(), bins=20, alpha=0.5, label="Losses")
            plt.xlabel(feature.replace("_", " "), fontsize=14)
            plt.ylabel("Number of Trades", fontsize=14)
            plt.legend(fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
        else:
            cat_counts = df.groupby([feature, "WIN"]).size().reset_index(name="count")
            pivot_df = cat_counts.pivot(index=feature, columns="WIN", values="count").fillna(0)
            for col in [0, 1]:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
            pivot_df = pivot_df[[0, 1]].astype(int)
            if feature in ["DAY_OF_WEEK_AT_OPEN", "DAY_OF_WEEK_AT_CLOSE"]:
                pivot_df = pivot_df.reindex(weekday_order).dropna(how='all')
            if pivot_df.empty:
                print(f"Skipping plot for '{feature}': No data available after sorting.")
                plt.close()
                continue
            pivot_df.plot(kind="bar", stacked=True, ax=plt.gca())
            plt.xlabel(feature.replace("_", " "), fontsize=14)
            plt.ylabel("Number of Trades", fontsize=14)
            plt.legend(["Losses", "Wins"], fontsize=14, loc="best")
            plt.xticks(fontsize=12, rotation=45, ha='right')
            plt.yticks(fontsize=12)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        plots[feature] = base64.b64encode(buf.getvalue()).decode("utf-8")
    return plots

def render_report(template_file, context, output_file="report.html"):
    with open(template_file, "r") as f:
        template_str = f.read()
    template = Template(template_str)
    report_html = template.render(context)
    with open(output_file, "w") as f:
        f.write(report_html)
    print(f"Report written to {output_file}")

# ------------------------------
# Main Script
# ------------------------------

if __name__ == '__main__':
    trades_df = load_trades("trades.csv")
    kpis = compute_kpis(trades_df)
    weekly_summary = generate_weekly_summary(trades_df)
    weekly_summary_html = weekly_summary.to_html(index=False, classes="dataframe", border=1)
    
    # Now, generate a daily equity curve plot from the trades data
    equity_curve_img = generate_equity_curve_plot(trades_df)
    
    trade_hist_img = generate_trade_return_histogram(trades_df)
    
    open_positions_df = pd.read_csv("unclosed.csv")
    numeric_cols = open_positions_df.select_dtypes(include=[np.number]).columns
    open_positions_df[numeric_cols] = open_positions_df[numeric_cols].round(2)
    open_positions_df.columns = [col.replace("_", " ") for col in open_positions_df.columns]
    open_positions_html = open_positions_df.to_html(index=False, classes="dataframe", border=1)
    
    trades_html_df = trades_df.copy()
    numeric_cols = trades_html_df.select_dtypes(include=[np.number]).columns
    trades_html_df[numeric_cols] = trades_html_df[numeric_cols].round(2)
    trades_html_df.columns = [col.replace("_", " ") for col in trades_html_df.columns]
    individual_trades_html = trades_html_df.to_html(index=False, classes="dataframe", border=1)
    
    candidate_features = []
    if "DAYS_HELD" in trades_df.columns:
        candidate_features.append("DAYS_HELD")
    if "DTE AT OPEN" in trades_df.columns:
        candidate_features.append("DTE AT OPEN")
    if "DAY_OF_WEEK_AT_OPEN" in trades_df.columns:
        candidate_features.append("DAY_OF_WEEK_AT_OPEN")
    if "DAY_OF_WEEK_AT_CLOSE" in trades_df.columns:
        candidate_features.append("DAY_OF_WEEK_AT_CLOSE")
    if "TRADE DIRECTION" in trades_df.columns:
        candidate_features.append("TRADE DIRECTION")
    
    feature_plots = generate_feature_plots(trades_df, candidate_features)
    win_rate_by_symbol_img = generate_win_rate_by_symbol_plot(trades_df)
    
    context = {
        "Start_Date": str(trades_df["CLOSE DATE"].min().date()),
        "End_Date": str(trades_df["CLOSE DATE"].max().date()),
        "Cumulative_Net_Profit": f"{kpis['net_profit']:.2f}",
        "Overall_Win_Rate": f"{kpis['win_rate']:.2f}",
        "Average_Trade_Return": f"{kpis['avg_trade_return']:.4f}",
        "Total_Trades": kpis["total_trades"],
        "Net_Profit": f"{kpis['net_profit']:.2f}",
        "Avg_Trade_Return": f"{kpis['avg_trade_return']*100:.2f}%",
        "Win_Rate": f"{kpis['win_rate']:.2f}",
        "Sharpe_Ratio": f"{kpis['sharpe_ratio']:.2f}",
        "adjusted_sortino": f"{kpis['adjusted_sortino_ratio']:.2f}" if not np.isnan(kpis['adjusted_sortino_ratio']) else "--",
        "Max_Drawdown": f"{kpis['max_drawdown_pct']:.2f}%",
        "Volatility": f"{kpis['volatility']:.2f}",
        "Equity_Curve": equity_curve_img,
        "Trade_Return_Histogram": trade_hist_img,
        "Weekly_Summary": weekly_summary_html,
        "Open_Positions": open_positions_html,
        "Individual_Trades": individual_trades_html,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,  
        "System_Name": "Steve Drasco",
        "Feature_Plots": feature_plots
    }
    
    render_report("template.html", context)