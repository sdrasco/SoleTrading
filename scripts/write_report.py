#!/usr/bin/env python3
"""
CLI wrapper: generate HTML report from cleaned CSVs.
"""
# ensure matplotlib can write cache to a local, writable dir
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.getcwd(), '.matplotlib_cache'))
# force non-interactive backend
import matplotlib
matplotlib.use('Agg')
import os, sys
# ensure repo root is on PYTHONPATH so options_report can be imported
_SCRIPT_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
sys.path.insert(0, _REPO_ROOT)
import pandas as pd
import numpy as np
import datetime
from zoneinfo import ZoneInfo

from options_report.analysis.metrics import load_trades, compute_kpis
from options_report.analysis.summary import generate_weekly_summary, generate_weekly_summary_html
from options_report.analysis.plots import (
    generate_equity_curve_plot,
    generate_basic_equity_wide,
    generate_basic_equity_square,
    generate_trade_return_histogram,
    generate_feature_plots,
    generate_win_rate_by_symbol_plot,
)
from options_report.report.renderer import render_report

# Helper: format dates as HTML cells with data-sort attribute
def format_date_cell(d):
    if pd.isnull(d):
        return ''
    return (
        f'<span style="white-space: nowrap;" data-sort="{d.strftime("%Y-%m-%d")}">'
        f"{d.day} {d.strftime('%b %Y')}"  # noqa: W605
        '</span>'
    )

def main():
    # 1) Load trades and compute KPIs
    trades_df = load_trades("data/cleaned/trades.csv")
    kpis = compute_kpis(trades_df)

    # 2) PRO context
    weekly_summary_html_pro = generate_weekly_summary_html(trades_df)
    equity_curve_img_pro = generate_equity_curve_plot(trades_df)
    trade_hist_img = generate_trade_return_histogram(trades_df)
    candidate_features = [
        "DAYS_HELD", "DTE AT OPEN",
        "DAY_OF_WEEK_AT_OPEN", "DAY_OF_WEEK_AT_CLOSE",
        "TRADE DIRECTION"
    ]
    feature_plots = generate_feature_plots(trades_df, candidate_features)
    win_rate_by_symbol_img = generate_win_rate_by_symbol_plot(trades_df)

    # 3) Process open positions from raw unclosed.csv
    open_df = pd.read_csv("data/cleaned/unclosed.csv")
    # Parse dates and compute DTE
    open_df["DATE"] = pd.to_datetime(open_df["DATE"], errors="coerce")
    open_df["EXPIRATION_DATE"] = pd.to_datetime(open_df["EXPIRATION_DATE"], errors="coerce")
    open_df["DTE"] = (open_df["EXPIRATION_DATE"] - pd.to_datetime("today")).dt.days
    # Round numeric
    num_cols = open_df.select_dtypes(include=[np.number]).columns
    open_df[num_cols] = open_df[num_cols].round(2)
    # Map position
    open_df["Position"] = open_df["ACTIVITY"].map({
        "Bought To Open": "Long",
        "Sold To Open": "Short"
    }).fillna("Unknown")
    # Rename columns for display
    open_df = open_df.rename(columns={
        "OPT_SYMBOL": "Symbol",
        "OPTION_TYPE": "Type",
        "STRIKE": "Strike",
        "EXPIRATION_DATE": "Expiration Date",
        "DATE": "Open Date",
        "DTE": "DTE",
        "QTY": "Qty",
        "PRICE": "Premium",
        "AMOUNT": "Cost",
        "NATIVE_CCY": "Ccy"
    })
    # Select and reorder
    open_cols = [
        "Symbol", "Position", "Type", "Strike",
        "Expiration Date", "Open Date", "DTE",
        "Qty", "Premium", "Cost", "Ccy"
    ]
    open_df = open_df[[c for c in open_cols if c in open_df.columns]]
    # Format date cells
    def fmt_date(d):
        if pd.isnull(d):
            return ""
        return f'<span style="white-space: nowrap;" data-sort="{d.strftime("%Y-%m-%d")}">{d.day} {d.strftime("%b %Y")}</span>'
    for col in ["Open Date", "Expiration Date"]:
        if col in open_df.columns:
            open_df[col] = open_df[col].apply(fmt_date)
    open_html = open_df.to_html(index=False, classes="dataframe sortable-table", escape=False)

    # 4) Individual Trades PRO
    trades_html_df = trades_df.copy()
    num_cols = trades_html_df.select_dtypes(include=[np.number]).columns
    trades_html_df[num_cols] = trades_html_df[num_cols].round(2)
    for col in ["OPEN DATE", "CLOSE DATE", "EXPIRATION"]:
        if col in trades_html_df.columns:
            trades_html_df[col] = trades_html_df[col].apply(format_date_cell)

    def rename_and_reorder_trades(df):
        mapping = {
            "SYMBOL": "Symbol",
            "POSITION": "Position",
            "OPTION TYPE": "Type",
            "STRIKE PRICE": "Strike",
            "EXPIRATION": "Expiration Date",
            "OPEN DATE": "Open Date",
            "DTE AT OPEN": "Days Held",
            "CLOSE DATE": "Close Date",
            "QTY": "Qty",
            "OPEN PRICE": "Premium (Open)",
            "CLOSE PRICE": "Premium (Close)",
            "OPEN AMOUNT": "Cost (Open)",
            "CLOSE AMOUNT": "Cost (Close)",
            "COMMISSION TOTAL": "Commission Total",
            "FEES TOTAL": "Fees Total",
            "NET PROFIT": "Profit",
            "RETURN": "Return",
            "NATIVE_CCY": "Ccy"
        }
        df = df.rename(columns=mapping)
        order = [v for v in mapping.values() if v in df.columns]
        df = df[order]
        tooltips = {
            "Symbol": "Underlying ticker.",
            "Position": "Long or Short.",
            "Type": "Put or Call.",
            "Strike": "Option strike price.",
            "Expiration Date": "Expiration date.",
            "Open Date": "Date opened.",
            "Days Held": "Days held.",
            "Close Date": "Date closed.",
            "Qty": "Number of contracts.",
            "Premium (Open)": "Open price per share.",
            "Premium (Close)": "Close price per share.",
            "Cost (Open)": "Cost at open.",
            "Cost (Close)": "Cost at close.",
            "Commission Total": "Total commission.",
            "Fees Total": "Total fees.",
            "Profit": "Net P/L.",
            "Return": "Return fraction.",
            "Ccy": "Currency."
        }
        hdrs = {col: f'<span title="{tooltips.get(col,"")}">{col}</span>' for col in df.columns}
        return df.rename(columns=hdrs)

    ind_html = rename_and_reorder_trades(trades_html_df).to_html(
        index=False, classes="dataframe sortable-table", escape=False
    )

    # 5) Timestamps & period
    try:
        tz = ZoneInfo("Europe/London")
        now = datetime.datetime.now(tz)
    except Exception:
        now = datetime.datetime.now()
    report_generated = f"{now.strftime('%-I:%M %p %Z')}, {now.strftime('%A')}, {now.strftime('%d %B %Y')}"
    start = trades_df["CLOSE DATE"].min()
    end = pd.to_datetime("today").normalize()
    reporting_period = f"{start.strftime('%d %B %Y')} to {end.strftime('%d %B %Y')}"

    net = kpis["net_profit"]
    net_str = f"-${abs(net):,.0f}" if net < 0 else f"${net:,.0f}"

    context_pro = {
        "Report_Generated": report_generated,
        "Reporting_Period": reporting_period,
        "Total_Trades": kpis["total_trades"],
        "Net_Profit": net_str,
        "Avg_Trade_Return": f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate": f"{kpis['win_rate']:.0f}%",
        "Sharpe_Ratio": f"{kpis['sharpe_ratio']:.2f}",
        "adjusted_sortino": f"{kpis['adjusted_sortino_ratio']:.2f}" if not np.isnan(kpis["adjusted_sortino_ratio"]) else "--",
        "Max_Drawdown": f"-${kpis['max_drawdown']:,.0f}",
        "Volatility": f"{kpis['volatility']:.2f}",
        "Equity_Curve": equity_curve_img_pro,
        "Trade_Return_Histogram": trade_hist_img,
        "Feature_Plots": feature_plots,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,
        "Weekly_Summary": weekly_summary_html_pro,
        "Open_Positions": open_html,
        "Individual_Trades": ind_html,
        "System_Name": "Sdrike"
    }
    render_report("docs/template_pro.html", context_pro, "docs/pro.html")

    # 6) BASIC context
    weekly_basic = generate_weekly_summary(trades_df)
    for drop in ["Sharpe", "Sortino"]:
        if drop in weekly_basic.columns:
            weekly_basic.drop(columns=[drop], inplace=True)
    weekly_summary_html_basic = weekly_basic.to_html(
        escape=False, index=False, classes="sortable-table"
    )
    basic_trades = trades_df.copy()
    if "TRADE DIRECTION" in basic_trades.columns:
        basic_trades["TRADE DIRECTION"] = basic_trades["TRADE DIRECTION"].replace(
            {"CALL": "Betting in favor", "PUT": "Betting against"}
        )
    drop_cols = ["SHARPE", "SORTINO", "ADJUSTED SORTINO RATIO", "VOLATILITY", "MAX DRAWDOWN", "DTE AT OPEN"]
    for col in drop_cols:
        if col in basic_trades.columns:
            basic_trades.drop(columns=[col], inplace=True)
    numc = basic_trades.select_dtypes(include=[np.number]).columns
    basic_trades[numc] = basic_trades[numc].round(2)
    basic_trades.columns = [c.replace("_", " ") for c in basic_trades.columns]
    individual_trades_html_basic = basic_trades.to_html(
        index=False, classes="dataframe sortable-table", escape=False
    )

    eq_wide = generate_basic_equity_wide(trades_df)
    eq_sq = generate_basic_equity_square(trades_df)

    context_basic = {
        "Report_Generated": report_generated,
        "Reporting_Period": reporting_period,
        "Total_Trades": kpis["total_trades"],
        "Net_Profit": net_str,
        "Avg_Trade_Return": f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate": f"{kpis['win_rate']:.0f}%",
        "Sharpe_Ratio": "",
        "adjusted_sortino": "",
        "Max_Drawdown": "",
        "Volatility": "",
        "Equity_Curve_Wide": eq_wide,
        "Equity_Curve_Square": eq_sq,
        "Trade_Return_Histogram": trade_hist_img,
        "Feature_Plots": feature_plots,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,
        "Weekly_Summary": weekly_summary_html_basic,
        "Open_Positions": open_html,
        "Individual_Trades": individual_trades_html_basic,
        "System_Name": "Sdrike"
    }
    render_report("docs/template_basic.html", context_basic, "docs/basic.html")

if __name__ == '__main__':
    main()