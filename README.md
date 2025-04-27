# Options Trading Report Generator

A Python toolkit to parse, process, and visualize options trading activity from Ally Invest and Interactive Brokers.

Features:
- Parse raw Ally and IB activity data into standardized CSVs
- Compute key performance metrics (net profit, win rate, Sharpe, Sortino, drawdown, volatility)
- Generate visualizations: equity curves with smoothing and gradient shading, trade return histograms, win-rate by symbol, and feature breakdown plots
- Output both Basic and Pro HTML reports with customizable Jinja2 templates

Requirements:
- Python 3.8 or newer
- Install dependencies via `pip install -r requirements.txt`

Directory Structure:
```
data/          # Input and output data files
  Ally/        # Raw Ally activity (activity.txt)
  IB/          # Raw IB CSV files
  cleaned/     # Generated CSVs: transactions.csv, trades.csv, unclosed.csv, unopened.csv
options_report/  # Core library: parsing, processing, analysis, and plotting
scripts/       # CLI wrappers: process_activity.py, write_report.py
docs/          # Jinja2 templates and generated HTML reports (basic.html, pro.html)
requirements.txt
LICENSE
README.md
```

Usage:
1. Prepare your data:
   - Place Ally activity in `data/Ally/activity.txt` (tab-delimited copy-paste)
   - Place Interactive Brokers CSV files in `data/IB/`
2. Clean and merge data:
   ```bash
   python process_activity.py
   ```
   Outputs standardized CSV files in `data/cleaned/`.
3. Generate HTML reports:
   ```bash
   python write_report.py
   ```
   - `docs/basic.html`: Basic report (equity curves, summary tables, trade list)
   - `docs/pro.html`: Pro report (detailed KPIs and additional plots)

Customization:
- Modify the Jinja2 templates in `docs/template_basic.html` or `docs/template_pro.html` to change report layout or styling.

Contributing:
Contributions welcome! Please open issues or submit pull requests.

License:
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
