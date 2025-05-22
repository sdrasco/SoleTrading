"""
plots.py
-------
Create charts (equity curves, histograms, categorical breakdowns) as base64 PNGs.
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import io, base64
from scipy.interpolate import make_interp_spline
from .config import (
    FIGURE_SIZE,
    FONT_LABEL,
    FONT_TICK,
    LEGEND_FONT,
    LINE_WIDTH,
    MARKER_SIZE,
    VOLUME_BAR_WIDTH,
    VOLUME_ALPHA,
    # chart-specific overrides
    WIN_RATE_WIDTH,
    WIN_RATE_MIN_HEIGHT,
    WIN_RATE_HEIGHT_PER_ITEM,
    BASIC_EQUITY_WIDE_SIZE,
    BASIC_EQUITY_SQUARE_SIZE,
)
import matplotlib.dates as mdates

def generate_equity_curve_plot(trades_df: pd.DataFrame) -> str:
    """
    Base64 PNG of weekly-scaled equity curve + daily volume bars.
    """
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    daily_profit = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    daily_volume = df.groupby('DATE').size().sort_index()
    start = daily_profit.index.min()
    end = pd.to_datetime('today').normalize()
    all_dates = pd.date_range(start, end, freq='D')
    daily_profit = daily_profit.reindex(all_dates, fill_value=0)
    daily_volume = daily_volume.reindex(all_dates, fill_value=0)
    cumulative = daily_profit.cumsum()
    x_vals = 1 + (cumulative.index - cumulative.index[0]).days / 7
    fig, ax1 = plt.subplots(figsize=FIGURE_SIZE)
    ax2 = ax1.twinx()
    ax1.set_zorder(2); ax1.patch.set_visible(False)
    ax2.set_zorder(1)
    green = '#6BA368'; grey = '#777777'
    # volume bars (slim width)
    ax2.bar(x_vals, daily_volume.values, width=VOLUME_BAR_WIDTH, color=grey, alpha=VOLUME_ALPHA)
    ax2.set_ylabel('Number of Trades', fontsize=FONT_LABEL, color=grey)
    ax2.tick_params(axis='y', labelsize=FONT_TICK, colors=grey)
    # equity curve line (enhanced markers/line)
    ax1.plot(
        x_vals, cumulative.values,
        marker='o', markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH, linestyle='-', color=green
    )
    # axis labels (double size)
    ax1.set_xlabel('Week', fontsize=FONT_LABEL)
    ax1.set_ylabel('Equity ($)', fontsize=FONT_LABEL, color=green)
    # tick labels
    ax1.tick_params(axis='y', labelsize=FONT_TICK, colors=green)
    ax1.tick_params(axis='x', labelsize=FONT_TICK)
    max_w = math.ceil(x_vals.max())
    ax1.set_xticks(np.arange(1, max_w + 1))
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _generate_basic_equity_custom(trades_df: pd.DataFrame, fig_width: int, fig_height: int) -> str:
    """
    Internal: smooth equity line with gradient fill.
    """
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'].dt.date)
    daily = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    start = daily.index.min(); end = pd.to_datetime('today').normalize()
    all_dates = pd.date_range(start, end, freq='D')
    daily = daily.reindex(all_dates, fill_value=0)
    cumulative = daily.cumsum()
    # prepare data for smoothing
    x_orig = np.arange(len(cumulative))
    y_orig = cumulative.values
    x_smooth = np.linspace(x_orig.min(), x_orig.max(), 300)
    spline = make_interp_spline(x_orig, y_orig, k=3)
    y_smooth = spline(x_smooth)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    col = '#6BA368'
    ax.plot(x_smooth, y_smooth, color=col, linewidth=LINE_WIDTH)

    # gradient shading under the curve
    rgba_full = to_rgba(col)
    rgba_half = (rgba_full[0], rgba_full[1], rgba_full[2], 0.5)
    cmap = LinearSegmentedColormap.from_list('grad', [
        (0, rgba_half),
        (1, (1, 1, 1, 0)),
    ])
    height = 300
    width = 300
    grad = np.linspace(0, 1, height).reshape(-1, 1)
    grad = np.repeat(grad, width, axis=1)

    xmin, xmax = x_smooth.min(), x_smooth.max()
    ymin = min(0, y_smooth.min())
    ymax = y_smooth.max()
    # set axis limits with a small margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin * 1.05 if ymin < 0 else ymin, ymax * 1.05)

    # draw gradient image behind the plot
    img = ax.imshow(
        grad,
        extent=[xmin, xmax, ymin, ymax],
        cmap=cmap,
        origin='lower',
        aspect='auto',
        alpha=1.0,
        zorder=-1,
    )

    # clip the gradient to the curve
    verts = [(xv, yv) for xv, yv in zip(x_smooth, y_smooth)]
    verts += [(xv, 0) for xv in reversed(x_smooth)]
    codes = [mpath.Path.LINETO] * len(verts)
    codes[0] = mpath.Path.MOVETO
    clip_path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(clip_path, transform=ax.transData)
    img.set_clip_path(patch)

    # format axes
    ax.set_ylabel('Profit ($)', fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)

    # x-axis date ticks: map numeric positions back to dates
    dates = cumulative.index
    num_ticks = 6
    xticks = np.linspace(x_orig.min(), x_orig.max(), num_ticks).astype(int)
    xticks = np.clip(xticks, x_orig.min(), x_orig.max())
    tick_dates = [dates[i] for i in xticks]
    tick_labels = [d.strftime('%d %b') for d in tick_dates]
    ax.set_xticks(xticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=FONT_TICK)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _generate_basic_equity_date(trades_df: pd.DataFrame, fig_width: int, fig_height: int) -> str:
    """
    Plot daily cumulative equity by actual date, with date-formatted x-axis.
    """
    df = trades_df.copy()
    df['DATE'] = pd.to_datetime(df['CLOSE DATE'])
    daily = df.groupby('DATE')['NET PROFIT'].sum().sort_index()
    cumulative = daily.cumsum()
    dates = cumulative.index
    values = cumulative.values

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    col = '#6BA368'
    # line + fill
    ax.plot(dates, values, color=col, linewidth=LINE_WIDTH)
    ax.fill_between(dates, values, 0, color=col, alpha=0.3)

    # date axis formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter('%d %b')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')

    # axis labels and ticks
    ax.set_ylabel('Profit ($)', fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)

    # y-axis limit +5%, allowing negative equity
    if len(values) > 0:
        ymin = min(0, values.min())
        ax.set_ylim(ymin * 1.05 if ymin < 0 else ymin, values.max() * 1.05)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_basic_equity_wide(trades_df: pd.DataFrame) -> str:
    """Wide aspect ratio basic equity curve (smoothed with gradient shading)."""
    return _generate_basic_equity_custom(
        trades_df,
        fig_width=BASIC_EQUITY_WIDE_SIZE[0],
        fig_height=BASIC_EQUITY_WIDE_SIZE[1]
    )

def generate_basic_equity_square(trades_df: pd.DataFrame) -> str:
    """Square aspect ratio basic equity curve (smoothed with gradient shading)."""
    return _generate_basic_equity_custom(
        trades_df,
        fig_width=BASIC_EQUITY_SQUARE_SIZE[0],
        fig_height=BASIC_EQUITY_SQUARE_SIZE[1]
    )

def generate_trade_return_histogram(df: pd.DataFrame) -> str:
    """Histogram of individual trade returns."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(df['RETURN'], bins=20, color='#6BA368', edgecolor='black')
    # use equity chart font sizes
    ax.set_xlabel('Trade Return', fontsize=FONT_LABEL)
    ax.set_ylabel('Number of Trades', fontsize=FONT_LABEL)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout(pad=3)
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_win_rate_by_symbol_plot(df: pd.DataFrame) -> str:
    """Horizontal bar chart of win rate by underlying symbol."""
    summary = df.groupby('SYMBOL').agg(
        win_rate=('NET PROFIT', lambda x: (x > 0).mean() * 100),
        num_trades=('NET PROFIT', 'count')
    )
    summary = summary.sort_values(['win_rate', 'num_trades'], ascending=[True, False])[::-1]
    labels = [f"{sym} ({n})" for sym,n in zip(summary.index, summary['num_trades'])]
    # compute dynamic height by number of symbols
    height = max(WIN_RATE_MIN_HEIGHT, WIN_RATE_HEIGHT_PER_ITEM * len(labels))
    fig, ax = plt.subplots(figsize=(WIN_RATE_WIDTH, height))
    ax.barh(labels, summary['win_rate'], color='#6BA368', edgecolor='black', height=0.6)
    # use equity chart font sizes
    ax.set_xlabel('Win Rate (%)', fontsize=FONT_LABEL)
    ax.set_ylabel('Symbol (Number of Trades)', fontsize=FONT_LABEL)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    plt.tight_layout(pad=3)
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_feature_plots(df: pd.DataFrame, features: list) -> dict:
    """Plots of win/loss by numeric or categorical features."""
    plots = {}
    df = df.copy()
    df['WIN'] = (df['NET PROFIT'] > 0).astype(int)
    win_col, loss_col = '#6BA368', '#777777'
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    for feature in features:
        if feature not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[df['WIN']==0][feature].plot(
                kind='hist', bins=20, alpha=0.5,
                color=loss_col, ax=ax
            )
            df[df['WIN']==1][feature].plot(
                kind='hist', bins=20, alpha=0.5,
                color=win_col, ax=ax
            )
            # use equity chart font sizes
            ax.set_xlabel(feature.replace('_',' '), fontsize=FONT_LABEL)
            ax.set_ylabel('Number of Trades', fontsize=FONT_LABEL)
            ax.legend(['Losses','Wins'], fontsize=LEGEND_FONT)
            ax.tick_params(axis='x', labelsize=FONT_TICK)
            ax.tick_params(axis='y', labelsize=FONT_TICK)
        else:
            counts = df.groupby([feature,'WIN']).size().unstack(fill_value=0)
            if feature in ['DAY_OF_WEEK_AT_OPEN','DAY_OF_WEEK_AT_CLOSE']:
                counts = counts.reindex(weekday_order).dropna(how='all')
            counts.plot(
                kind='bar', stacked=True,
                color=[loss_col, win_col], ax=ax
            )
            # use equity chart font sizes
            ax.set_xlabel(feature.replace('_',' '), fontsize=FONT_LABEL)
            ax.set_ylabel('Number of Trades', fontsize=FONT_LABEL)
            ax.legend(['Losses','Wins'], fontsize=LEGEND_FONT)
            ax.tick_params(axis='x', labelsize=FONT_TICK, rotation=45)
            ax.tick_params(axis='y', labelsize=FONT_TICK)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
        plots[feature] = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plots