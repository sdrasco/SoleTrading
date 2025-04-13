import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import io, base64
import datetime
from zoneinfo import ZoneInfo
from jinja2 import Template
from scipy.interpolate import make_interp_spline

# ------------------------------
# Helper Functions for Analysis
# ------------------------------

def load_trades(filename):
    """
    Load the CSV and parse the relevant date columns.
    Rename columns and derive additional columns for analysis.
    """
    df = pd.read_csv(filename, parse_dates=["OPEN DATE", "CLOSE DATE", "EXPIRATION"])
    
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
            "C": "CALL", 
            "P": "PUT", 
            "CALL": "CALL", 
            "PUT": "PUT"
        })
    
    return df

def compute_adjusted_sortino_ratio(df, target_return=0):
    """
    Compute the 'Adjusted Sortino Ratio' which accounts for
    skewness and kurtosis in the returns distribution.
    """
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
    """
    Compute overall KPIs such as net profit, average return,
    Sharpe ratio, max drawdown, volatility, etc.
    """
    total_trades = len(df)
    net_profit = df["NET PROFIT"].sum()
    avg_trade_return = df["RETURN"].mean()
    win_rate = (df["NET PROFIT"] > 0).mean() * 100
    
    std_return = df["RETURN"].std()
    sharpe_ratio = (df["RETURN"].mean() / std_return) if std_return > 0 else np.nan

    # Sort for equity curve (trade-by-trade cumulative net profit)
    df_sorted = df.sort_values(by="CLOSE DATE").copy()
    df_sorted["CUMULATIVE_NET_PROFIT"] = df_sorted["NET PROFIT"].cumsum()

    # Drawdown calculation
    cumulative = df_sorted["CUMULATIVE_NET_PROFIT"]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = abs(drawdown.min() * 100)

    volatility = std_return
    adjusted_sortino_ratio = compute_adjusted_sortino_ratio(df)

    return {
        "total_trades": total_trades,
        "net_profit": net_profit,
        "avg_trade_return": avg_trade_return,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility": volatility,
        "equity_curve": df_sorted,  # for further charts
        "adjusted_sortino_ratio": adjusted_sortino_ratio
    }

def generate_weekly_summary(df):
    """
    Group trades by weekly close date and show aggregated performance metrics per week,
    with improved aesthetics.

    Aesthetic improvements:
      - Date entries now include the year and are formatted as "3 Feb 2025".
        The underlying data-sort attribute remains as the ISO date for proper sorting.
        The inline style prevents line breaks within the date.
      - The Profit column is formatted such that:
            748   -> "$748"
            1,299 -> "$1,299"
            -6,055 -> "-$6,055"
      - Win Rate is rounded to whole percentages.
      - The "Avg Days Held" and "Wins" columns are removed from the final table.
      - Header text now includes hover (tooltip) text.
    """
    # 1. Compute the week’s start date (using the CLOSE DATE)
    df["WEEK"] = df["CLOSE DATE"].dt.to_period("W").apply(lambda r: r.start_time.date())
    
    # 2. Group by week and compute aggregated performance metrics
    weekly = df.groupby("WEEK").agg(
        net_profit=("NET PROFIT", "sum"),
        num_trades=("NET PROFIT", "count"),
        avg_days_held=("DAYS_HELD", "mean"),
        avg_return=("RETURN", "mean"),
        winning_trades=("NET PROFIT", lambda x: (x > 0).sum())
    ).reset_index()
    
    # 3. Compute the win rate (in percent)
    weekly["win_rate"] = (weekly["winning_trades"] / weekly["num_trades"]) * 100
    
    # Drop columns we don't want to display
    weekly.drop(columns=["avg_days_held", "winning_trades"], inplace=True)
    
    # 4. Compute Sharpe Ratio per week
    weekly["sharpe_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else np.nan
    ).values
    
    # 5. Compute Adjusted Sortino Ratio per week
    weekly["adjusted_sortino_ratio"] = df.groupby("WEEK")["RETURN"].apply(
        lambda x: compute_adjusted_sortino_ratio(pd.DataFrame({"RETURN": x}))
    ).values

    # 6. Round all numeric columns (for intermediate calculations)
    numeric_cols = weekly.select_dtypes(include=[np.number]).columns
    weekly[numeric_cols] = weekly[numeric_cols].round(2)
    
    # 7. Replace NaN adjusted sortino values with "inf" for display
    weekly["Adjusted Sortino"] = weekly["adjusted_sortino_ratio"].apply(
        lambda x: "inf" if np.isnan(x) else x
    )
    
    # 8. Add sequential Week # and rename the WEEK column
    weekly.insert(0, "Week", range(1, len(weekly) + 1))
    weekly.rename(columns={"WEEK": "Starts"}, inplace=True)
    weekly.drop(columns=["adjusted_sortino_ratio"], inplace=True)
    
    # 9. Format the date cells: display the full date as "3 Feb 2025", while including a data-sort
    #    attribute (ISO format) for proper sorting, and prevent wrapping using white-space: nowrap.
    def format_date_cell(d):
        return f'<span style="white-space: nowrap;" data-sort="{d.strftime("%Y-%m-%d")}">{d.day} {d.strftime("%b %Y")}</span>'
    weekly["Starts"] = weekly["Starts"].apply(format_date_cell)
    
    # 10. Format the net profit as specified
    def format_profit(x):
        x = int(round(x))
        if x < 0:
            return f"-${abs(x):,}"
        else:
            return f"${x:,}"
    weekly["net_profit"] = weekly["net_profit"].apply(format_profit)
    
    # 11. Format win rate as whole percentages.
    weekly["win_rate"] = weekly["win_rate"].apply(lambda x: f"{round(x)}%")
    
    # 12. Rename columns to final display names.
    rename_map = {
        "Starts": "Starting",
        "net_profit": "Profit",
        "num_trades": "Trades",
        "avg_return": "Trade Return",
        "win_rate": "Win Rate",
        "sharpe_ratio": "Sharpe",
        "Adjusted Sortino": "Sortino"
    }
    weekly.rename(columns=rename_map, inplace=True)
    
    # 13. Reorder the columns as desired.
    desired_order = ["Week", "Starting", "Profit", "Trades", "Win Rate", "Trade Return", "Sharpe", "Sortino"]
    weekly = weekly[desired_order]
    
    # 14. Update header names to include tooltips.
    # Define stub tooltip text for each header; update these as desired.
    header_tooltips = {
        "Week": "Sequential week number.",
        "Starting": "First trading day of the week.",
        "Trades": "Number of trades that closed this week.",
        "Profit": "Change in equity for the week.",
        "Trade Return": "Average return for trades closed this week.",
        "Win Rate": "Ratio of wins to trades for the week.",
        "Sharpe": "Sharpe Ratio: A measure of risk-adjusted return. It is the average return earned in excess of the risk‑free rate per unit of volatility or total risk. A higher Sharpe Ratio indicates that the trade returns are more favorable relative to the risk taken.",
        "Sortino": "Adjusted Sortino Ratio: Like Sharpe, but ignores volatily from gains. It measures the excess return per unit of downside deviation, with a higher value indicating better risk-adjusted performance when adverse movements are considered."
    }
    # Wrap each header text in a span element that includes a title attribute.
    new_headers = {col: f'<span title="{header_tooltips[col]}">{col}</span>' for col in weekly.columns}
    weekly.rename(columns=new_headers, inplace=True)
    
    return weekly

def generate_weekly_summary_html(df):
    """
    Generate the weekly summary as an HTML table.

    Note:
      - escape=False prevents pandas from escaping HTML, so the custom formatting (including
        the header tooltips and date formatting with <span> tags) is rendered correctly.
      - The table is assigned the class 'sortable-table' so that your template's JavaScript 
        can enable interactive sorting.
    """
    weekly = generate_weekly_summary(df)
    return weekly.to_html(escape=False, index=False, classes='sortable-table')

def generate_equity_curve_plot(trades_df):
    """
    Generate a base64-encoded daily equity curve image
    (plus daily trade volume on a secondary axis).
    """
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    
    # Sum NET PROFIT and count trades by day
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    daily_volume = df.groupby('DATE').size().sort_index()

    # Define the date range from the first trade date up to today,
    # ensuring that even days with no trading are included.
    start_date = daily_profit.index.min()
    end_date = pd.to_datetime('today').normalize()  # Extend to today's date
    all_dates = pd.date_range(start_date, end_date, freq='D')
    
    # Reindex to include all dates in the range
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    daily_volume = daily_volume.reindex(all_dates, fill_value=0)
    
    # Cumulative equity
    cumulative_equity = daily_profit.cumsum()
    x_values = 1 + (cumulative_equity.index - cumulative_equity.index[0]).days / 7
    
    # Set up the plot with a secondary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Make sure the trade volume (bar chart) is plotted behind the equity line
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax2.set_zorder(1)
    
    muted_blue = "#4a90e2"
    light_grey = "#D3D3D3"

    # Plot trade volume on the secondary axis (ax2)
    bar_width = 0.1
    ax2.bar(x_values, daily_volume.values, width=bar_width,
            color=light_grey, alpha=0.5, zorder=1)
    ax2.set_ylabel("Number of Trades", fontsize=21, color=light_grey)
    ax2.tick_params(axis='y', labelsize=21, colors=light_grey)
    
    # Plot cumulative equity on the primary axis (ax1)
    ax1.plot(x_values, cumulative_equity.values, marker='o', markersize=12,
             linestyle='-', color=muted_blue, zorder=2)
    ax1.set_xlabel("Week", fontsize=21)
    ax1.set_ylabel("Equity ($)", fontsize=21, color=muted_blue)
    ax1.tick_params(axis='x', labelsize=21)
    ax1.tick_params(axis='y', labelsize=21, colors=muted_blue)

    # Set x-axis ticks based on weeks passed
    max_week = math.ceil(x_values.max())
    ax1.set_xticks(np.arange(1, max_week + 1))
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_basic_equity(trades_df):
    """
    Basic equity curve with a smoother, spline-interpolated line,
    and a vertical gradient fill from the curve downward.
    The gradient is half-opaque near the line, fading to fully transparent at y=0.
    """

    # Prepare the data
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    start_date = daily_profit.index.min()
    end_date = pd.to_datetime('today').normalize()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    cumulative_equity = daily_profit.cumsum()

    # Spline interpolation
    x_original = np.arange(len(cumulative_equity))
    y_original = cumulative_equity.values

    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
    spline = make_interp_spline(x_original, y_original, k=3)
    y_smooth = spline(x_smooth)

    fig, ax = plt.subplots(figsize=(20, 6))

    # 1) Plot the main line (fully opaque)
    line_color = "#6BA368"
    ax.plot(x_smooth, y_smooth, color=line_color, linewidth=6)

    # Convert line_color to RGBA, then reduce alpha to 0.5 for the gradient's "darkest" part
    rgba_full = to_rgba(line_color)  # e.g. (r, g, b, 1.0)
    rgba_half = (rgba_full[0], rgba_full[1], rgba_full[2], 0.5)

    # 2) Create a 2-step colormap: half-opaque at the top, fully transparent at the bottom
    gradient_cmap = LinearSegmentedColormap.from_list(
        "gradient_cmap",
        [
            (0, rgba_half),         # near the line => half opacity
            (1, (1, 1, 1, 0))       # bottom => fully transparent
        ]
    )

    # Build a gradient array from 0..1 in the vertical direction
    height = 300
    width = 300
    grad_array = np.linspace(0, 1, height).reshape(-1, 1)
    grad_array = np.repeat(grad_array, width, axis=1)

    # Define bounding box for the gradient
    xmin, xmax = x_smooth.min(), x_smooth.max()
    ymin, ymax = 0, y_smooth.max()  # assumes no negative dips
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax * 1.05)

    # Show the gradient
    img = ax.imshow(
        grad_array,
        extent=[xmin, xmax, ymin, ymax],
        cmap=gradient_cmap,
        origin='lower',   # row 0 is bottom => fraction=0 => transparent
        aspect='auto',
        alpha=1.0,
        zorder=-1         # behind the line
    )

    # 3) Clip the gradient to the polygon under the curve (above y=0)
    poly_vertices = [(xv, yv) for xv, yv in zip(x_smooth, y_smooth)]
    poly_vertices += [(xv, 0) for xv in reversed(x_smooth)]
    codes = [mpath.Path.LINETO] * len(poly_vertices)
    codes[0] = mpath.Path.MOVETO
    clip_path = mpath.Path(poly_vertices, codes)
    patch = mpatches.PathPatch(clip_path, transform=ax.transData)
    img.set_clip_path(patch)

    # 4) Axis labels, ticks, etc.
    ax.set_ylabel("Profit ($)", fontsize=24, color="black")
    ax.tick_params(axis='y', labelcolor="black", labelsize=20)
    ax.tick_params(axis='x', labelsize=20, colors="black")

    num_ticks = 6
    xticks = np.linspace(x_original.min(), x_original.max(), num_ticks).astype(int)
    xticks = np.clip(xticks, 0, len(cumulative_equity) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [cumulative_equity.index[i].strftime('%d %b') for i in xticks],
        rotation=45, ha='right'
    )

    plt.tight_layout()

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_trade_return_histogram(df):
    """
    Generate a histogram of trade returns (base64 encoded).
    """
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
    """
    Generate a horizontal bar plot showing Win Rate by Symbol
    (sorted ascending, then reversed).
    """
    summary = df.groupby('SYMBOL').agg(
        win_rate=('NET PROFIT', lambda x: (x > 0).mean() * 100),
        num_trades=('NET PROFIT', 'count')
    )
    # Sort ascending by win_rate, then descending by num_trades, then reverse
    summary = summary.sort_values(by=['win_rate', 'num_trades'], ascending=[True, False])[::-1]
    labels = [f"{sym} ({n})" for sym, n in zip(summary.index, summary['num_trades'])]
    
    num_bars = len(labels)
    bar_height = 0.6
    plt.figure(figsize=(8, max(3, 0.4 * num_bars)))
    plt.barh(labels, summary['win_rate'], color='skyblue', edgecolor='black', height=bar_height)
    
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
    """
    Plot histograms or bar charts of features,
    separated by Win vs. Loss.
    """
    plots = {}
    df["WIN"] = (df["NET PROFIT"] > 0).astype(int)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for feature in features:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(8, 6))
        if pd.api.types.is_numeric_dtype(df[feature]):
            # Numeric feature => histogram of Wins vs Losses
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
            # Categorical feature => stacked bar of Wins vs Losses
            cat_counts = df.groupby([feature, "WIN"]).size().reset_index(name="count")
            pivot_df = cat_counts.pivot(index=feature, columns="WIN", values="count").fillna(0)

            # Ensure both 0 & 1 columns exist
            for col in [0, 1]:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
            pivot_df = pivot_df[[0, 1]].astype(int)

            # Reorder if it's a day-of-week feature
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

def render_report(template_file, context, output_file="docs/report.html"):
    """
    Use Jinja2 to render the final HTML report from a template.
    """
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
    # 1) Load trades data and compute KPIs
    trades_df = load_trades("data/cleaned/trades.csv")
    kpis = compute_kpis(trades_df)

    # 2) Generate the weekly summary (advanced) as HTML
    weekly_summary_html_pro = generate_weekly_summary_html(trades_df)

    # 3) Generate charts for Pro and Basic
    #    -- Pro gets the old equity chart with volume
    equity_curve_img_pro = generate_equity_curve_plot(trades_df)
    #    -- Basic gets the new spline chart in green
    equity_curve_img_basic = generate_basic_equity(trades_df)

    trade_hist_img = generate_trade_return_histogram(trades_df)

    candidate_features = [
        "DAYS_HELD",
        "DTE AT OPEN",
        "DAY_OF_WEEK_AT_OPEN",
        "DAY_OF_WEEK_AT_CLOSE",
        "TRADE DIRECTION"
    ]
    candidate_features = [col for col in candidate_features if col in trades_df.columns]
    feature_plots = generate_feature_plots(trades_df, candidate_features)
    win_rate_by_symbol_img = generate_win_rate_by_symbol_plot(trades_df)

    # 4) Process unclosed positions
    open_positions_df = pd.read_csv("data/cleaned/unclosed.csv")
    numeric_cols = open_positions_df.select_dtypes(include=[np.number]).columns
    open_positions_df[numeric_cols] = open_positions_df[numeric_cols].round(2)
    open_positions_df.columns = [col.replace("_", " ") for col in open_positions_df.columns]
    open_positions_html = open_positions_df.to_html(
        index=False, classes="dataframe sortable-table", border=1
    )

    # 5) Generate individual trades table (pro)
    trades_html_df = trades_df.copy()
    numeric_cols = trades_html_df.select_dtypes(include=[np.number]).columns
    trades_html_df[numeric_cols] = trades_html_df[numeric_cols].round(2)
    trades_html_df.columns = [col.replace("_", " ") for col in trades_html_df.columns]
    individual_trades_html_pro = trades_html_df.to_html(
        index=False, classes="dataframe sortable-table", border=1
    )

    # 6) Format net profit for the pro context
    net_profit = kpis["net_profit"]
    if net_profit < 0:
        net_profit_str = f"-${abs(net_profit):,.0f}"
    else:
        net_profit_str = f"${net_profit:,.0f}"

    # 7) Generate a "Report Generated" timestamp (UK time).
    try:
        tz_uk = ZoneInfo("Europe/London")
        now_uk = datetime.datetime.now(tz_uk)
    except Exception:
        tz_uk = ZoneInfo("UTC")
        now_uk = datetime.datetime.now(tz_uk)

    # IMPORTANT: Produce exactly 3 comma-separated parts, so the template sees
    # splitted[0] -> time + timezone
    # splitted[1] -> weekday
    # splitted[2] -> month day year
    report_generated_str = (
        f"{now_uk.strftime('%I:%M %p %Z')}, "
        f"{now_uk.strftime('%A')}, "
        f"{now_uk.strftime('%B %d %Y')}"
    )

    # 8) Create a human-friendly reporting period
    start_date = trades_df["CLOSE DATE"].min()
    end_date = trades_df["CLOSE DATE"].max()

    if start_date.year == end_date.year:
        reporting_period_str = (
            f"{start_date.strftime('%B')} {start_date.day} "
            f"to {end_date.strftime('%B')} {end_date.day}, {end_date.year}"
        )
    else:
        reporting_period_str = (
            f"{start_date.strftime('%B')} {start_date.day}, {start_date.year} "
            f"to {end_date.strftime('%B')} {end_date.day}, {end_date.year}"
        )

    # ------------------------------------------------------------------
    # PRO REPORT CONTEXT (full detail)
    # ------------------------------------------------------------------
    context_pro = {
        "Report_Generated": report_generated_str,
        "Reporting_Period": reporting_period_str,
        "Total_Trades": kpis["total_trades"],
        "Net_Profit": net_profit_str,
        "Avg_Trade_Return": f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate": f"{kpis['win_rate']:.0f}%",
        "Sharpe_Ratio": f"{kpis['sharpe_ratio']:.2f}",
        "adjusted_sortino": (
            f"{kpis['adjusted_sortino_ratio']:.2f}"
            if not np.isnan(kpis['adjusted_sortino_ratio']) else "--"
        ),
        "Max_Drawdown": f"{kpis['max_drawdown_pct']:.0f}%",
        "Volatility": f"{kpis['volatility']:.2f}",
        # Charts (use the pro version of the equity curve)
        "Equity_Curve": equity_curve_img_pro,
        "Trade_Return_Histogram": trade_hist_img,
        "Feature_Plots": feature_plots,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,
        # Tables
        "Weekly_Summary": weekly_summary_html_pro,
        "Open_Positions": open_positions_html,
        "Individual_Trades": individual_trades_html_pro,
        # Footer
        "System_Name": "Sdrike Systems"
    }

    # ------------------------------------------------------------------
    # BASIC REPORT CONTEXT (stripped down)
    # ------------------------------------------------------------------
    # 1) Generate a simpler weekly summary
    weekly_basic = generate_weekly_summary(trades_df)
    weekly_basic.drop(columns=[col for col in weekly_basic.columns
                               if ("Sharpe" in col or "Sortino" in col)],
                      inplace=True, errors="ignore")
    weekly_summary_html_basic = weekly_basic.to_html(
        escape=False, index=False, classes='sortable-table'
    )

    # 2) Create a copy of trades for the basic table
    basic_trades_df = trades_df.copy()
    if "TRADE DIRECTION" in basic_trades_df.columns:
        basic_trades_df["TRADE DIRECTION"] = basic_trades_df["TRADE DIRECTION"].replace({
            "CALL": "Betting in favor",
            "PUT": "Betting against"
        })
    columns_to_remove = ["SHARPE","SORTINO","ADJUSTED SORTINO RATIO","VOLATILITY",
                         "MAX DRAWDOWN","DTE AT OPEN"]
    for col in columns_to_remove:
        if col in basic_trades_df.columns:
            basic_trades_df.drop(columns=[col], inplace=True)

    numeric_cols = basic_trades_df.select_dtypes(include=[np.number]).columns
    basic_trades_df[numeric_cols] = basic_trades_df[numeric_cols].round(2)
    basic_trades_df.columns = [col.replace("_", " ") for col in basic_trades_df.columns]
    individual_trades_html_basic = basic_trades_df.to_html(
        index=False, classes="dataframe sortable-table", border=1
    )

    # 3) Create a simpler context that uses the new basic chart
    context_basic = {
        "Report_Generated": report_generated_str,     # 3-part version
        "Reporting_Period": reporting_period_str,
        "Total_Trades": kpis["total_trades"],
        "Net_Profit": net_profit_str,
        "Avg_Trade_Return": f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate": f"{kpis['win_rate']:.0f}%",
        "Sharpe_Ratio": "",
        "adjusted_sortino": "",
        "Max_Drawdown": "",
        "Volatility": "",
        # Charts: use the basic equity curve here
        "Equity_Curve": equity_curve_img_basic,
        "Trade_Return_Histogram": trade_hist_img,
        "Feature_Plots": feature_plots,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,
        # Tables
        "Weekly_Summary": weekly_summary_html_basic,
        "Open_Positions": open_positions_html,
        "Individual_Trades": individual_trades_html_basic,
        "System_Name": "Sdrike Systems"
    }

    # 4) Render both versions
    render_report("docs/template_pro.html", context_pro, "docs/pro.html")
    render_report("docs/template_basic.html", context_basic, "docs/basic.html")