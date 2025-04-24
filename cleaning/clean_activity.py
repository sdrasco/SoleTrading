#!/usr/bin/env python3
"""
clean_activity.py

Usage:
    python clean_activity.py

This script consolidates and cleans trading activity from:
  - Ally (./data/Ally/activity.txt)
  - IB   (./data/IB/*.csv)

It produces final CSV outputs in ./data/cleaned:
  - trades.csv       (all closed option trades)
  - unclosed.csv     (option positions still open)
  - fx_trades.csv    (closed FX trades)
  - fx_unclosed.csv  (open FX positions)
"""

import sys
import csv
import re
from pathlib import Path

# Import our custom modules
from cleaning.broker_parsers import AllyParser, IBParser
from cleaning.options_pipeline import process_options  # updated to skip double-clean
from cleaning.fx_pipeline import process_fx
from cleaning.data_helpers import parse_currency, parse_mixed_date  # existing helpers

def parse_broker_files(broker_files):
    """
    Given a list of (filepath, parser) tuples, parse each file line-by-line.
    Split resulting rows into two lists: option_rows and fx_rows.
    """
    option_rows = []
    fx_rows = []

    for (filepath, parser) in broker_files:
        print(f"Parsing {filepath} ...")
        delim = "\t" if filepath.endswith(".txt") else ","
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            for row in reader:
                parsed = parser.parse_row(row)
                if parsed is None:
                    continue
                if parsed["asset"] == "OPT":
                    option_rows.append(parsed)
                elif parsed["asset"] == "FX":
                    fx_rows.append(parsed)

    return option_rows, fx_rows


def clean_option_rows(option_rows):
    """
    Perform the same data-cleaning logic the old 'load_opt()' had:

      1) Discard 'Cash Movement'
      2) Convert 'Expired' -> 'Sold To Close'
      3) Convert numeric columns
      4) Parse 'date' as datetime
      5) Take abs of 'qty'
      6) Attempt to extract symbol/expiration/strike/type from 'description'
      7) Drop any row where 'OPT_SYMBOL' wasn't parsed successfully
      8) Sort by date & an activity priority (open legs before close legs)
    """

    # 1) Discard 'Cash Movement'
    filtered = [r for r in option_rows if r["activity"].lower() != "cash movement"]

    # 2) Convert "Expired" -> "Sold To Close"
    for r in filtered:
        if r["activity"].lower() == "expired":
            r["activity"] = "Sold To Close"

    # 3) Convert numeric columns
    for r in filtered:
        r["price"] = parse_currency(r["price"])
        r["commission"] = parse_currency(r["commission"])
        r["fees"] = parse_currency(r["fees"])
        r["amount"] = parse_currency(r["amount"])

    # 4) parse 'date' as datetime
    for r in filtered:
        r["date"] = parse_mixed_date(r["date"])

    # 5) abs of 'qty'
    for r in filtered:
        try:
            qf = float(str(r["qty"]).replace(",", ""))
            r["qty"] = abs(qf)
        except ValueError:
            r["qty"] = 0.0

    # 6) Parse out "OPT_SYMBOL", "EXPIRATION_DATE", "STRIKE", "OPTION_TYPE" from description
    def parse_option_description(desc):
        pattern = (
            r"^(?P<sym>\S+)\s+(?P<month>\w+)\s+(?P<day>\d{1,2})\s+"
            r"(?P<year>\d{4})\s+(?P<strike>\d+(\.\d+)?)\s+(?P<type>Call|Put)$"
        )
        m = re.match(pattern, desc)
        if not m:
            return None, None, None, None
        try:
            exp_str = f"{m['month']} {int(m['day']):02d} {m['year']}"
            exp_dt = parse_mixed_date(exp_str)
        except:
            exp_dt = None
        try:
            strike_f = float(m["strike"])
        except:
            strike_f = None
        return (m["sym"], exp_dt, strike_f, m["type"])

    for r in filtered:
        sym_val, exp_dt, strike_val, typ_val = parse_option_description(r["description"])
        r["OPT_SYMBOL"] = sym_val
        r["EXPIRATION_DATE"] = exp_dt
        r["STRIKE"] = strike_val
        r["OPTION_TYPE"] = typ_val

    # 7) Drop rows missing 'OPT_SYMBOL'
    final = [r for r in filtered if r["OPT_SYMBOL"]]

    # 8) Sort by date + open-before-close
    def activity_priority(act):
        mapping = {
            "bought to open": 0,
            "sold to open": 0,
            "bought to close": 1,
            "sold to close": 1,
        }
        return mapping.get(act.lower(), 99)

    final.sort(key=lambda x: (x["date"], activity_priority(x["activity"])))
    return final


def main():
    # 1) Verify inputs
    ally_file = Path("./data/Ally/activity.txt")
    ib_dir = Path("./data/IB")

    if not ally_file.is_file():
        sys.exit(f"ERROR: Missing Ally file: {ally_file}")

    ib_csvs = list(ib_dir.glob("*.csv"))
    if not ib_csvs:
        print("WARNING: No IB CSV files found in", ib_dir)

    # 2) Build list of (filepath, parser)
    broker_files = [(str(ally_file), AllyParser())]
    for csv_file in ib_csvs:
        broker_files.append((str(csv_file), IBParser()))

    # 3) Parse → separate into option_rows / fx_rows
    option_rows, fx_rows = parse_broker_files(broker_files)

    # Clean the option rows thoroughly here
    option_rows = clean_option_rows(option_rows)

    # 4) Output directory
    out_dir = Path("./data/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) Now process with the pipeline
    #    (which expects pre-cleaned rows)
    opt_trades_df, opt_unclosed_df = process_options(option_rows)

    opt_trades_csv = out_dir / "trades.csv"
    unclosed_csv = out_dir / "unclosed.csv"
    opt_trades_df.to_csv(opt_trades_csv, index=False)
    print("Option trades →", opt_trades_csv)

    # Write unclosed options
    if not opt_unclosed_df.empty:
        opt_unclosed_df["POSITION"] = opt_unclosed_df["activity"].map(
            {"Bought To Open": "Long", "Sold To Open": "Short"}
        ).fillna("UNK")

        opt_unclosed_df["DTE_AT_OPEN"] = (
            opt_unclosed_df["EXPIRATION_DATE"] - opt_unclosed_df["date"]
        ).dt.days

        opt_unclosed_df.rename(
            columns={
                "date": "OPEN_DATE",
                "price": "OPEN_PRICE",
                "amount": "OPEN_AMOUNT",
                "OPT_SYMBOL": "Option Symbol",
                "EXPIRATION_DATE": "Expiration",
                "STRIKE": "Strike Price",
                "OPTION_TYPE": "Option Type",
                "qty": "Quantity",
            },
            inplace=True,
        )

        keep_cols = [
            "account", "POSITION", "Option Symbol", "Expiration", "Strike Price",
            "Option Type", "OPEN_DATE", "DTE_AT_OPEN", "Quantity",
            "OPEN_PRICE", "OPEN_AMOUNT",
        ]
        opt_unclosed_df[keep_cols].to_csv(unclosed_csv, index=False)
    else:
        opt_unclosed_df.to_csv(unclosed_csv, index=False)

    print("Option unclosed →", unclosed_csv)

    # 6) FX
    fx_trades_csv = out_dir / "fx_trades.csv"
    fx_unclosed_csv = out_dir / "fx_unclosed.csv"
    if fx_rows:
        fx_trades_df, fx_unclosed_df = process_fx(fx_rows)
        fx_trades_df.to_csv(fx_trades_csv, index=False)
        fx_unclosed_df.to_csv(fx_unclosed_csv, index=False)
        print("FX trades →", fx_trades_csv)
        print("FX unclosed →", fx_unclosed_csv)
    else:
        import pandas as pd
        pd.DataFrame().to_csv(fx_trades_csv, index=False)
        pd.DataFrame().to_csv(fx_unclosed_csv, index=False)
        print("No FX activity detected.")


if __name__ == "__main__":
    main()