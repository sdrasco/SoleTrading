import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64, datetime
from jinja2 import Template

# ------------------------------
# Helper Functions for Analysis
# ------------------------------

def load_trades(filename):
    df = pd.read_csv(filename, parse_dates=["OPEN DATE", "CLOSE DATE"])
    # If DAYS_HELD is not computed, calculate it from OPEN DATE and CLOSE DATE
    if "DAYS_HELD" not in df.columns:
        df["DAYS_HELD"] = (df["CLOSE DATE"] - df["OPEN DATE"]).dt.days
    return df

def compute_kpis(df):
    total_trades = len(df)
    net_profit = df["NET PROFIT"].sum()
    avg_trade_return = df["RETURN"].mean()
    win_rate = (df["NET PROFIT"] > 0).mean() * 100
    sharpe_ratio = df["RETURN"].mean() / df["RETURN"].std() if df["RETURN"].std() > 0 else np.nan

    # Build an equity curve: sort by CLOSE DATE and cumulatively sum net profit.
    df_sorted = df.sort_values(by="CLOSE DATE").copy()
    df_sorted["CUMULATIVE_NET_PROFIT"] = df_sorted["NET PROFIT"].cumsum()

    # Calculate maximum drawdown from the equity curve.
    cummax = df_sorted["CUMULATIVE_NET_PROFIT"].cummax()
    drawdown = df_sorted["CUMULATIVE_NET_PROFIT"] - cummax
    max_drawdown = drawdown.min()

    volatility = df["RETURN"].std()
    return {
       "total_trades": total_trades,
       "net_profit": net_profit,
       "avg_trade_return": avg_trade_return,
       "win_rate": win_rate,
       "sharpe_ratio": sharpe_ratio,
       "max_drawdown": max_drawdown,
       "volatility": volatility,
       "equity_curve": df_sorted
    }

def generate_weekly_summary(df):
    # Create a week-ending date from the CLOSE DATE.
    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda r: r.end_time.date())
    weekly = df.groupby("WEEK").agg(
         net_profit=("NET PROFIT", "sum"),
         num_trades=("NET PROFIT", "count"),
         avg_days_held=("DAYS_HELD", "mean"),
         avg_return=("RETURN", "mean"),
         winning_trades=("NET PROFIT", lambda x: (x > 0).sum())
    ).reset_index()
    weekly["win_rate"] = (weekly["winning_trades"] / weekly["num_trades"]) * 100
    # Compute weekly Sharpe ratio (using the trade return data grouped by week)
    weekly["sharpe_ratio"] = df.groupby("WEEK")["RETURN"].apply(lambda x: x.mean()/x.std() if x.std()>0 else np.nan).values
    # Round numeric columns to 2 decimals
    numeric_cols = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric_cols] = weekly[numeric_cols].round(2)
    return weekly

def generate_equity_curve_plot(equity_curve_df):
    plt.figure(figsize=(10,6))
    plt.plot(equity_curve_df["CLOSE DATE"], equity_curve_df["CUMULATIVE_NET_PROFIT"], marker="o")
    plt.xlabel("Close Date")
    plt.ylabel("Cumulative Net Profit ($)")
    plt.title("Equity Curve")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_trade_return_histogram(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["RETURN"], bins=20, edgecolor="black")
    plt.xlabel("Trade Return")
    plt.ylabel("Frequency")
    plt.title("Trade Return Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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
    # Load trades data from trades.csv
    trades_df = load_trades("trades.csv")

    # Compute KPIs and the equity curve.
    kpis = compute_kpis(trades_df)

    # Generate weekly performance summary and convert to HTML.
    weekly_summary = generate_weekly_summary(trades_df)
    weekly_summary_html = weekly_summary.to_html(index=False, classes="dataframe", border=1)

    # Generate equity curve and trade return histogram as Base64-encoded images.
    equity_curve_img = generate_equity_curve_plot(kpis["equity_curve"])
    trade_hist_img = generate_trade_return_histogram(trades_df)

    # Load open positions and individual trades as HTML tables.
    open_positions_df = pd.read_csv("unclosed.csv")
    # Round numeric columns in open positions.
    numeric_cols = open_positions_df.select_dtypes(include=[np.number]).columns
    open_positions_df[numeric_cols] = open_positions_df[numeric_cols].round(2)
    # Remove underscores from column names for display.
    open_positions_df.columns = [col.replace("_", " ") for col in open_positions_df.columns]
    open_positions_html = open_positions_df.to_html(index=False, classes="dataframe", border=1)
    
    trades_html_df = trades_df.copy()
    # Round numeric columns in individual trades.
    numeric_cols = trades_html_df.select_dtypes(include=[np.number]).columns
    trades_html_df[numeric_cols] = trades_html_df[numeric_cols].round(2)
    trades_html_df.columns = [col.replace("_", " ") for col in trades_html_df.columns]
    individual_trades_html = trades_html_df.to_html(index=False, classes="dataframe", border=1)

    # Build context for the template using keys without spaces.
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
        "Max_Drawdown": f"{kpis['max_drawdown']:.2f}",
        "Volatility": f"{kpis['volatility']:.2f}",
        "Equity_Curve": equity_curve_img,
        "Trade_Return_Histogram": trade_hist_img,
        "Weekly_Summary": weekly_summary_html,
        "Open_Positions": open_positions_html,
        "Individual_Trades": individual_trades_html,
        "System_Name": "Your System Name Here"
    }

    # Render the report using your template (template.html) and output to report.html.
    render_report("template.html", context)