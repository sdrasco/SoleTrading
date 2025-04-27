"""
config.py
--------
Centralized chart styling parameters for all plots.
"""
# Figure size for main charts (width, height) in inches
FIGURE_SIZE = (14, 10)

# Font sizes
FONT_LABEL = 36    # axis labels
FONT_TICK = 32     # tick labels
LEGEND_FONT = 32   # legend text

# Line and marker styles
LINE_WIDTH = 8     # line width for plots
MARKER_SIZE = 16   # marker size for plots

# Bar plot settings
VOLUME_BAR_WIDTH = 0.1   # width for volume bars
VOLUME_ALPHA = 0.6       # transparency for volume bars
 
# --- Chart-specific exceptions ---
# Win-rate by symbol chart: width fixed, height scales by number of symbols
WIN_RATE_WIDTH = FIGURE_SIZE[0]
# Minimum height in inches
WIN_RATE_MIN_HEIGHT = FIGURE_SIZE[1]
# Height per symbol (inches)
WIN_RATE_HEIGHT_PER_ITEM = 0.75

# Basic report equity chart sizes (wide and square) override default
BASIC_EQUITY_WIDE_SIZE = (2*FIGURE_SIZE[0], FIGURE_SIZE[1])
BASIC_EQUITY_SQUARE_SIZE = (FIGURE_SIZE[0], FIGURE_SIZE[1])