#!/usr/bin/env python3
# report/main_report.py

"""
main_report.py

Run this file to generate HTML reports from your cleaned trading data.

Usage:
    python main_report.py
"""

import sys
import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

from report.data_loading import load_and_combine_all_trades
from report.kpi_calculations import compute_kpis, generate_weekly_summary_html
from report.visualization import (
    generate_equity_curve_plot,
    generate_trade_return_histogram,
    generate_basic_equity_wide,
    generate_basic_equity_square,
    generate_feature_plots,
    generate_win_rate_by_symbol_plot
)
from report.report_rendering import render_report, format_date_cell


def main():
    # 1) Load combined dataset (Options + FX)
    combined_df = load_and_combine_all_trades(
        "data/cleaned/trades.csv",
        "data/cleaned/fx_trades.csv"
    )

    # 2) Compute KPIs
    kpis = compute_kpis(combined_df)

    # 3) Weekly summary HTML
    weekly_summary_html_pro = generate_weekly_summary_html(combined_df)

    # 4) Generate charts
    equity_curve_img_pro = generate_equity_curve_plot(combined_df)
    trade_hist_img       = generate_trade_return_histogram(combined_df)
    feature_plots        = generate_feature_plots(
        combined_df, 
        features=["DAYS_HELD", "DTE AT OPEN", "DAY_OF_WEEK_AT_OPEN", "DAY_OF_WEEK_AT_CLOSE", "TRADE DIRECTION"]
    )
    win_rate_by_symbol_img = generate_win_rate_by_symbol_plot(combined_df)

    # 5) Load open positions from unclosed.csv
    open_positions_df = pd.read_csv("data/cleaned/unclosed.csv")
    open_positions_html = build_open_positions_html(open_positions_df)

    # 6) Build separate tables for completed FX or non-FX
    fx_completed_html, individual_trades_html_pro = build_fx_and_nonfx_tables(combined_df)

    # 7) Build contexts for different template versions
    net_profit = kpis["net_profit"]
    net_profit_str = (
        f"-${abs(net_profit):,.0f}" if net_profit < 0 else f"${net_profit:,.0f}"
    )

    tz_uk = ZoneInfo("Europe/London")
    now_uk = datetime.datetime.now(tz_uk)
    report_generated_str = (
        f"{now_uk.strftime('%-I:%M %p %Z')}, "
        f"{now_uk.strftime('%A')}, "
        f"{now_uk.strftime('%d %B %Y')}"
    )

    start_date = combined_df["CLOSE DATE"].min()
    end_date = pd.to_datetime('today').normalize()
    reporting_period_str = (
        f"{start_date.strftime('%d %B %Y')} to {end_date.strftime('%d %B %Y')}"
        if not pd.isnull(start_date) else ""
    )

    context_pro = {
        "Report_Generated":         report_generated_str,
        "Reporting_Period":         reporting_period_str,
        "Total_Trades":             kpis["total_trades"],
        "Net_Profit":               net_profit_str,
        "Avg_Trade_Return":         f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate":                 f"{kpis['win_rate']:.0f}%",
        "Sharpe_Ratio":             f"{kpis['sharpe_ratio']:.2f}",
        "adjusted_sortino":         f"{kpis['adjusted_sortino_ratio']:.2f}" if not np.isnan(kpis['adjusted_sortino_ratio']) else "--",
        "Max_Drawdown":             f"{kpis['max_drawdown_pct']:.0f}%",
        "Volatility":               f"{kpis['volatility']:.2f}",

        "Equity_Curve":             equity_curve_img_pro,
        "Trade_Return_Histogram":   trade_hist_img,
        "Feature_Plots":            feature_plots,
        "Win_Rate_By_Symbol":       win_rate_by_symbol_img,

        "Weekly_Summary":           weekly_summary_html_pro,
        "Open_Positions":           open_positions_html,

        "Fx_Completed_Trades":      fx_completed_html,
        "Individual_Trades":        individual_trades_html_pro,

        "System_Name":              "Sdrike"
    }

    # Optionally, build a "basic" context with fewer metrics
    equity_curve_img_basic_wide   = generate_basic_equity_wide(combined_df)
    equity_curve_img_basic_square = generate_basic_equity_square(combined_df)

    context_basic = {
        "Report_Generated":  report_generated_str,
        "Reporting_Period":  reporting_period_str,
        "Total_Trades":      kpis["total_trades"],
        "Net_Profit":        net_profit_str,
        "Avg_Trade_Return":  f"{kpis['avg_trade_return']*100:.0f}%",
        "Win_Rate":          f"{kpis['win_rate']:.0f}%",
        "Equity_Curve_Wide":   equity_curve_img_basic_wide,
        "Equity_Curve_Square": equity_curve_img_basic_square,
        "Trade_Return_Histogram": trade_hist_img,
        "Feature_Plots":     feature_plots,
        "Win_Rate_By_Symbol": win_rate_by_symbol_img,
        "Weekly_Summary":    generate_weekly_summary_html(combined_df),
        "Open_Positions":    open_positions_html,
        "Individual_Trades": build_basic_trades_html(combined_df),
        "System_Name": "Sdrike Systems"
    }

    # 8) Render final HTML
    render_report("docs/template_pro.html",   context_pro,   "docs/pro.html")
    render_report("docs/template_basic.html", context_basic, "docs/basic.html")

    print("All reports generated!")


def build_open_positions_html(open_positions_df):
    """
    Create an HTML table of open positions, with any desired transformations.
    """
    if open_positions_df.empty:
        return "<p>No open positions</p>"

    # 1) Parse date columns that actually exist in unclosed.csv
    if "OPEN_DATE" in open_positions_df.columns:
        open_positions_df["OPEN_DATE"] = pd.to_datetime(open_positions_df["OPEN_DATE"], errors="coerce")
    if "Expiration" in open_positions_df.columns:
        open_positions_df["Expiration"] = pd.to_datetime(open_positions_df["Expiration"], errors="coerce")

    # 2) Compute DTE (days to expiration)
    if "Expiration" in open_positions_df.columns:
        open_positions_df["DTE"] = (open_positions_df["Expiration"] - pd.to_datetime('today')).dt.days

    # 3) Round numeric columns
    numeric_cols = open_positions_df.select_dtypes(include=[np.number]).columns
    open_positions_df[numeric_cols] = open_positions_df[numeric_cols].round(2)

    # 4) Rename columns to user-friendly versions
    #    Our CSV has underscore-based columns: OPEN_DATE, OPEN_PRICE, OPEN_AMOUNT, DTE_AT_OPEN
    rename_map = {
        "POSITION":       "Position",
        "Option Symbol":  "Symbol",
        "Option Type":    "Type",
        "Strike Price":   "Strike",
        "Expiration":     "Expiration Date",
        "OPEN_DATE":      "Open Date",     # underscore -> space
        "DTE_AT_OPEN":    "DTE (Open)",    
        "Quantity":       "Qty",
        "OPEN_PRICE":     "Premium",       # underscore -> space
        "OPEN_AMOUNT":    "Cost",
        "DTE":            "DTE (Now)"
    }
    open_positions_df.rename(columns=rename_map, inplace=True, errors="ignore")

    # 5) Reorder columns in a user-friendly way
    desired_order = [
        "Symbol", "Position", "Type", "Strike",
        "Expiration Date", "Open Date", 
        "DTE (Open)", "DTE (Now)", 
        "Qty", "Premium", "Cost"
    ]
    existing = [c for c in desired_order if c in open_positions_df.columns]
    open_positions_df = open_positions_df[existing]

    # 6) Format the date columns for HTML
    for col in ["Open Date","Expiration Date"]:
        if col in open_positions_df.columns:
            open_positions_df[col] = open_positions_df[col].apply(format_date_cell)

    # 7) Convert to HTML
    return open_positions_df.to_html(
        index=False,
        classes="dataframe sortable-table",
        border=1,
        escape=False
    )


def build_fx_and_nonfx_tables(combined_df):
    """
    Build separate HTML tables for FX vs. non-FX trades.
    """
    fx_trades_df = combined_df[combined_df["POSITION"] == "FX"].copy()
    if fx_trades_df.empty:
        fx_completed_html = "<p>No completed FX trades.</p>"
    else:
        rename_map = {
            "PAIR":       "Ticker",
            "SYMBOL":     "Ticker",
            "ROUND_TRIP": "Round Trip",
            "NET PROFIT": "Profit",
        }
        fx_trades_df.rename(columns=rename_map, inplace=True, errors="ignore")

        for dcol in ["OPEN DATE","CLOSE DATE"]:
            if dcol in fx_trades_df.columns:
                fx_trades_df[dcol] = pd.to_datetime(fx_trades_df[dcol], errors="coerce")
                fx_trades_df[dcol] = fx_trades_df[dcol].apply(format_date_cell)

        keep_cols = ["Ticker","Round Trip","OPEN DATE","CLOSE DATE","Profit"]
        existing = [c for c in keep_cols if c in fx_trades_df.columns]
        fx_trades_df = fx_trades_df[existing]

        if "Profit" in fx_trades_df.columns:
            fx_trades_df["Profit"] = fx_trades_df["Profit"].round(2)

        fx_completed_html = fx_trades_df.to_html(
            index=False, classes="dataframe sortable-table", border=1, escape=False
        )

    # Non‑FX trades
    nonfx_df = combined_df[combined_df["POSITION"] != "FX"].copy()
    for date_col in ["OPEN DATE","CLOSE DATE","EXPIRATION"]:
        if date_col in nonfx_df.columns:
            nonfx_df[date_col] = pd.to_datetime(nonfx_df[date_col], errors="coerce")
            nonfx_df[date_col] = nonfx_df[date_col].apply(format_date_cell)

    numeric_cols = nonfx_df.select_dtypes(include=[np.number]).columns
    nonfx_df[numeric_cols] = nonfx_df[numeric_cols].round(2)
    nonfx_df.columns = [col.replace("_"," ") for col in nonfx_df.columns]

    rename_map = {
        "POSITION":       "Position",
        "SYMBOL":         "Symbol",
        "OPTION TYPE":    "Type",
        "STRIKE PRICE":   "Strike",
        "EXPIRATION":     "Expiration Date",
        "OPEN DATE":      "Open Date",
        "CLOSE DATE":     "Close Date",
        "DTE AT OPEN":    "DTE (Open)",
        "DAYS HELD":      "Days Held",
        "QTY":            "Qty",
        "OPEN PRICE":     "Premium (Open)",
        "CLOSE PRICE":    "Premium (Close)",
        "NET PROFIT":     "Profit",
        "RETURN":         "Return"
    }
    nonfx_df.rename(columns=rename_map, inplace=True, errors="ignore")

    desired_order = [
        "Symbol","Position","Type","Strike",
        "Open Date","Close Date","Expiration Date",
        "Days Held","Qty","Premium (Open)","Premium (Close)",
        "Profit","Return"
    ]
    existing = [col for col in desired_order if col in nonfx_df.columns]
    nonfx_df = nonfx_df[existing]

    individual_trades_html_pro = nonfx_df.to_html(
        index=False, classes="dataframe sortable-table", border=1, escape=False
    )

    return fx_completed_html, individual_trades_html_pro


def build_basic_trades_html(df):
    """
    A simpler version of the trades table, removing advanced columns.
    """
    basic_df = df.copy()
    numeric_cols = basic_df.select_dtypes(include=[np.number]).columns
    basic_df[numeric_cols] = basic_df[numeric_cols].round(2)
    basic_df.columns = [col.replace("_"," ") for col in basic_df.columns]

    # remove advanced columns if you want
    columns_to_remove = [
        "SHARPE","SORTINO","ADJUSTED SORTINO RATIO","VOLATILITY",
        "MAX DRAWDOWN","DTE AT OPEN"
    ]
    for col in columns_to_remove:
        if col in basic_df.columns:
            basic_df.drop(columns=[col], inplace=True)

    # Format date columns
    for date_col in ["OPEN DATE", "CLOSE DATE", "EXPIRATION"]:
        if date_col in basic_df.columns:
            basic_df[date_col] = pd.to_datetime(basic_df[date_col], errors="coerce")
            basic_df[date_col] = basic_df[date_col].apply(format_date_cell)

    return basic_df.to_html(
        index=False, classes="dataframe sortable-table", border=1, escape=False
    )


if __name__ == "__main__":
    main()