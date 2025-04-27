"""
options_report.analysis: metrics, summary tables, and chart plotting functions.
"""
from .metrics import (
    load_trades,
    compute_adjusted_sortino_ratio,
    compute_kpis,
)
from .summary import (
    generate_weekly_summary,
    generate_weekly_summary_html,
)
from .plots import (
    generate_equity_curve_plot,
    generate_basic_equity_wide,
    generate_basic_equity_square,
    generate_trade_return_histogram,
    generate_win_rate_by_symbol_plot,
    generate_feature_plots,
)