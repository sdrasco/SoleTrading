# report/visualization.py

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import io
import base64
import math
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

def generate_equity_curve_plot(trades_df):
    """
    Example that plots daily net profit over time + cumulative equity,
    returning a base64-encoded PNG.
    """
    if trades_df.empty or "CLOSE DATE" not in trades_df.columns:
        return ""
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    if daily_profit.empty:
        return ""
    
    # fill in missing dates
    start_date = daily_profit.index.min()
    end_date = pd.to_datetime('today').normalize()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    
    cumulative_equity = daily_profit.cumsum()
    x_values = 1 + (cumulative_equity.index - cumulative_equity.index[0]).days / 7
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, cumulative_equity.values, marker='o', markersize=10, color="#6BA368")
    ax.set_xlabel("Week", fontsize=15)
    ax.set_ylabel("Equity ($)", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)

    max_week = math.ceil(x_values.max())
    ax.set_xticks(np.arange(1, max_week + 1))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_trade_return_histogram(df):
    """
    Plots a histogram of the 'RETURN' column.
    """
    if df.empty or "RETURN" not in df.columns:
        return ""
    plt.figure(figsize=(10,6))
    plt.hist(df["RETURN"].dropna(), bins=20, edgecolor="black", color="#6BA368")
    plt.xlabel("Trade Return")
    plt.ylabel("Number of Trades")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_basic_equity_custom(trades_df, fig_width, fig_height):
    """
    Another approach with a smoothed curve, gradient fill, etc.
    """
    if trades_df.empty or "CLOSE DATE" not in trades_df.columns:
        return ""
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    if daily_profit.empty:
        return ""
    
    start_date = daily_profit.index.min()
    end_date = pd.to_datetime('today').normalize()
    all_dates = pd.date_range(start_date, end_date, freq='D')
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    cumulative_equity = daily_profit.cumsum()

    x_original = np.arange(len(cumulative_equity))
    y_original = cumulative_equity.values
    if len(x_original) < 2:
        return ""

    x_smooth = np.linspace(x_original.min(), x_original.max(), 300)
    spline = make_interp_spline(x_original, y_original, k=3)
    y_smooth = spline(x_smooth)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    line_color = "#6BA368"
    ax.plot(x_smooth, y_smooth, color=line_color, linewidth=6)

    # gradient fill ...
    rgba_full = to_rgba(line_color)
    rgba_half = (rgba_full[0], rgba_full[1], rgba_full[2], 0.5)
    gradient_cmap = LinearSegmentedColormap.from_list(
        "gradient_cmap",
        [(0, rgba_half), (1, (1, 1, 1, 0))]
    )

    height, width = 300, 300
    grad_array = np.linspace(0, 1, height).reshape(-1, 1)
    grad_array = np.repeat(grad_array, width, axis=1)

    xmin, xmax = x_smooth.min(), x_smooth.max()
    ymin, ymax = 0, y_smooth.max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax * 1.05)

    img = ax.imshow(
        grad_array,
        extent=[xmin, xmax, ymin, ymax],
        cmap=gradient_cmap,
        origin='lower',
        aspect='auto',
        alpha=1.0,
        zorder=-1
    )

    # Clip path ...
    poly_vertices = [(xv, yv) for xv, yv in zip(x_smooth, y_smooth)]
    poly_vertices += [(xv, 0) for xv in reversed(x_smooth)]
    codes = [mpath.Path.LINETO] * len(poly_vertices)
    codes[0] = mpath.Path.MOVETO
    clip_path = mpath.Path(poly_vertices, codes)
    patch = mpatches.PathPatch(clip_path, transform=ax.transData)
    img.set_clip_path(patch)

    ax.set_ylabel("Profit ($)")
    num_ticks = 6
    xticks = np.linspace(x_original.min(), x_original.max(), num_ticks).astype(int)
    xticks = np.clip(xticks, 0, len(cumulative_equity) - 1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([cumulative_equity.index[i].strftime('%d %b') for i in xticks],
                       rotation=45, ha='right')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_basic_equity_wide(trades_df):
    return generate_basic_equity_custom(trades_df, 20, 6)


def generate_basic_equity_square(trades_df):
    return generate_basic_equity_custom(trades_df, 8, 8)


def generate_win_rate_by_symbol_plot(df):
    """
    Generate a horizontal bar plot for Win Rate by Symbol.
    """
    if "SYMBOL" not in df.columns and "PAIR" not in df.columns:
        return ""

    # unify to "SYMBOL"
    if "SYMBOL" not in df.columns:
        df["SYMBOL"] = df.get("PAIR", df.get("Ticker", "Unknown"))

    if df.empty or "NET PROFIT" not in df.columns:
        return ""

    summary = df.groupby('SYMBOL').agg(
        win_rate=('NET PROFIT', lambda x: (x > 0).mean() * 100),
        num_trades=('NET PROFIT', 'count')
    )
    if summary.empty:
        return ""

    summary = summary.sort_values(by=['win_rate','num_trades'], ascending=[True,False])[::-1]
    labels = [f"{sym} ({n})" for sym, n in zip(summary.index, summary['num_trades'])]

    plt.figure(figsize=(8, max(3, 0.4 * len(labels))))
    plt.barh(labels, summary['win_rate'], color="#6BA368", edgecolor="black")
    plt.xlim(0, 100)
    plt.xlabel("Win Rate (%)")
    plt.ylabel("Symbol (Number of Trades)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_feature_plots(df, features):
    """
    For each feature in the list, produce a histogram (if numeric) 
    or a stacked bar (if categorical) showing wins vs. losses.
    """
    plots = {}
    if "NET PROFIT" not in df.columns:
        return plots

    df["WIN"] = (df["NET PROFIT"] > 0).astype(int)
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    win_color = "#6BA368"
    loss_color = "#777777"

    for feature in features:
        if feature not in df.columns:
            continue
        plt.figure(figsize=(8, 6))

        if pd.api.types.is_numeric_dtype(df[feature]):
            df_loss = df[df["WIN"] == 0]
            df_win = df[df["WIN"] == 1]
            plt.hist(df_loss[feature].dropna(), bins=20, alpha=0.5, color=loss_color, label="Losses")
            plt.hist(df_win[feature].dropna(), bins=20, alpha=0.5,  color=win_color,  label="Wins")
            plt.xlabel(feature, fontsize=14)
            plt.ylabel("Number of Trades", fontsize=14)
            plt.legend(fontsize=14)
        else:
            cat_counts = df.groupby([feature, "WIN"]).size().reset_index(name="count")
            pivot_df = cat_counts.pivot(index=feature, columns="WIN", values="count").fillna(0)
            for col in [0, 1]:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
            if feature in ["DAY_OF_WEEK_AT_OPEN","DAY_OF_WEEK_AT_CLOSE"]:
                pivot_df = pivot_df.reindex(weekday_order).dropna(how='all')
            if pivot_df.empty:
                plt.close()
                continue
            pivot_df.plot(
                kind="bar", stacked=True, ax=plt.gca(),
                color=[loss_color, win_color]
            )
            plt.xlabel(feature, fontsize=14)
            plt.ylabel("Number of Trades", fontsize=14)
            plt.xticks(rotation=45, ha='right')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        plots[feature] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return plots