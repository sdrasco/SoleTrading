"""
process_activity.py
-------------------
Orchestrates parsing raw Ally+IB files and writing cleaned CSVs.
"""
import sys
from pathlib import Path

from options_report.parsers.ally import AllyParser
from options_report.parsers.ib import scan_ib_file, IBParser
from options_report.processing.transactions import (
    build_transactions,
    read_transactions,
    merge_simultaneous,
    match_trades,
)

def process_all(ally_file: str | Path, ib_dir: str | Path, out_dir: str | Path):  # noqa: C901
    """
    Read Ally activity and IB CSVs, normalize, match opens/closes, and write:
      trades.csv, unopened.csv, unclosed.csv into out_dir.
    """
    ally_path = Path(ally_file)
    if not ally_path.is_file():
        print(f"Error: Ally file not found at {ally_path}")
        sys.exit(1)

    ib_path = Path(ib_dir)
    ib_files = sorted(ib_path.glob('*.csv'))
    if not ib_files:
        print(f"Warning: No IB CSV files found in {ib_path}, proceeding with Ally only.")

    # Build list of (filepath, parser) pairs
    broker_files = [(ally_path, AllyParser())]
    for csv_file in ib_files:
        fx_lookup, exch_lookup = scan_ib_file(csv_file)
        broker_files.append((csv_file, IBParser(fx_lookup, exch_lookup)))

    # 1) Build and normalize transactions
    print("Building transactions...")
    df_raw = build_transactions(broker_files)
    df = read_transactions(df_raw)
    df = merge_simultaneous(df)

    # 2) Match opens/closes
    trades_df, unopened_df, unclosed_df = match_trades(df)

    # Ensure output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 3) Write out CSVs
    trades_file = out_path / 'trades.csv'
    trades_df.to_csv(trades_file, index=False)
    print(f"✔ trades.csv written to {trades_file}")

    if not unopened_df.empty:
        unopened_file = out_path / 'unopened.csv'
        unopened_df.to_csv(unopened_file, index=False)
        print(f"✔ unopened.csv written to {unopened_file}")

    if not unclosed_df.empty:
        unclosed_file = out_path / 'unclosed.csv'
        unclosed_df.to_csv(unclosed_file, index=False)
        print(f"✔ unclosed.csv written to {unclosed_file}")