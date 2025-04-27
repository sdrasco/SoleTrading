#!/usr/bin/env python3

from __future__ import annotations

import sys
import csv
import glob
import pandas as pd
import re
from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

######################################################################
# Helper functions
######################################################################

def parse_ib_timestamp(ts: str) -> datetime:
    """Convert IB timestamp string → datetime object (second precision)."""
    return datetime.strptime(ts.strip(), "%Y-%m-%d, %H:%M:%S")


def fx_rate_from_autofx_row(row: list) -> Optional[tuple[str, datetime, float]]:
    """Return (ccy, timestamp, usd_per_ccy) if *row* is an AutoFX line."""
    if len(row) < 17:
        return None
    if row[0:3] != ["Trades", "Data", "Order"] or "Forex" not in row[3] or row[16] != "AFx":
        return None

    pair      = row[5].strip()           # e.g. "USD.HKD" or "EUR.USD"
    price_str = row[8].replace(",", "")
    ts        = parse_ib_timestamp(row[6])

    try:
        px = float(price_str)
    except ValueError:
        return None

    if "." not in pair:
        return None
    base, counter = pair.split(".")
    if base == "USD":
        ccy, usd_per_ccy = counter, 1.0 / px
    elif counter == "USD":
        ccy, usd_per_ccy = base, px
    else:
        return None
    return ccy.upper(), ts, usd_per_ccy


def scan_ib_file(csv_path: Path):
    fx, exch = {}, {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            out = fx_rate_from_autofx_row(row)
            if out:
                fx[(out[0], out[1])] = out[2]
            if row and row[0] == "Financial Instrument Information" and len(row) > 8:
                symbol  = row[4].strip(); exchange = row[7].strip().upper()
                if symbol:
                    exch[symbol] = exchange
    return fx, exch


def process_statements(pattern: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "trades.csv"
    with csv_out.open("w", newline="") as fout:
        wr = csv.writer(fout)
        wr.writerow(["ACCOUNT","DATE","ACTIVITY","QTY","SYMBOL","PRICE","COMMISSION","FEES","AMOUNT","NATIVE_CCY","EXCHANGE"])

        for path in Path().glob(pattern):
            if path.suffix.lower() != ".csv":
                continue  # skip Ally here — no change needed
            fx_lookup, exch_lookup = scan_ib_file(path)
            parser = IBParser(fx_lookup, exch_lookup)
            with path.open(newline="") as f:
                rdr = csv.reader(f)
                for row in rdr:
                    parsed = parser.parse_row(row)
                    if not parsed:
                        continue
                    wr.writerow([
                        parsed[k] for k in ("account","date","activity","qty","symbol","price","commission","fees","amount","native_ccy","exchange")
                    ])
    print("✔ trades.csv written to", csv_out)


##################################################################
# 1) Base broker parser classes
##################################################################

class BaseBrokerParser(ABC):
    @abstractmethod
    def parse_row(self, row: list):
        """
        Must return a dict:
          {
            'account': ...,
            'date': ...,
            'activity': ...,
            'qty': ...,
            'symbol': ...,
            'description': ...,
            'price': ...,
            'commission': ...,
            'fees': ...,
            'amount': ...
          }
        or None if the row doesn't parse or isn't relevant.
        """
        pass


class AllyParser(BaseBrokerParser):
    """
    Parses rows from a tab-delimited ally.txt file.
    Each row is like:
      [
        '03/28/2025', 'Sold To Close', '-1', 'QQQ Put',
        'QQQ Apr 04 2025 475.00 Put', '$10.04', '$0.50', '$0.07', '$1,003.43'
      ]
    We require at least 9 columns.
    """

    def parse_row(self, row: list):
        # Skip empty or short rows
        if len(row) < 9:
            return None

        date, activity, qty, sym_or_type, description, price, commission, fees, amount = row[:9]

        # For "Cash Movement," deduce symbol differently
        if activity == "Cash Movement":
            if "FULLYPAID LENDING REBATE" in description:
                symbol = sym_or_type
            else:
                symbol = ""
        else:
            symbol = sym_or_type.split()[0]

        return {
            'account': 'Ally',
            'date': date,
            'activity': activity,
            'qty': qty,
            'symbol': symbol,
            'description': description,
            'price': price,
            'commission': commission,
            'fees': fees,
            'amount': amount,
            'native_ccy' : 'USD',
            'exchange'   : 'ALLY'
        }


class IBParser(BaseBrokerParser):
    """
    Parse IB option-trade rows, convert non-USD fills to USD,
    and label each fill with its original currency + exchange.
    """

    REQUIRED = 17

    def __init__(self,
                 fx_lookup: Dict[tuple[str, datetime], float],
                 exch_lookup: Dict[str, str]):
        self.fx_lookup   = fx_lookup     # (CCY, timestamp) → USD/CCY
        self.exch_lookup = exch_lookup   # option symbol → exchange

    # ------------------------------------------------------------------
    def parse_row(self, row: list):
        # ---------- filter for rows we care about ---------------------
        if len(row) < self.REQUIRED or row[0:3] != ['Trades', 'Data', 'Order']:
            return None
        if 'Options' not in row[3]:
            return None                    # skip equities / forex lines

        # ---------- raw fields ----------------------------------------
        ccy_orig    = row[4].strip().upper()        # save BEFORE conversion
        symbol_full = row[5].strip()                # “CAT 17APR25 250 P”
        ts          = parse_ib_timestamp(row[6])
        qty         = int(row[7].replace(',', ''))
        t_price     = float(row[8] or 0)
        proceeds    = float(row[10] or 0)
        commission  = float(row[11] or 0)
        code        = row[16].split(';')[0]

        # ---------- activity mapping ---------------------------------
        if code == 'O':
            activity = 'Bought To Open' if qty > 0 else 'Sold To Open'
        elif code == 'C':
            activity = 'Bought To Close' if qty > 0 else 'Sold To Close'
        else:
            return None                      # ignore assignments/exercises

        # ---------- FX conversion ------------------------------------
        if ccy_orig != 'USD':
            rate = self._rate(ccy_orig, ts)
            if rate:                         # convert price/proceeds/comm → USD
                t_price    *= rate
                proceeds   *= rate
                commission *= rate

        # ---------- description & ticker -----------------------------
        try:
            ticker, raw_exp, raw_strike, pc = symbol_full.split()
            month_map = dict(JAN='Jan', FEB='Feb', MAR='Mar', APR='Apr',
                              MAY='May', JUN='Jun', JUL='Jul', AUG='Aug',
                              SEP='Sep', OCT='Oct', NOV='Nov', DEC='Dec')
            day  = int(raw_exp[:2])
            mon  = month_map[raw_exp[2:5].upper()]
            yr   = f"20{raw_exp[5:]}"
            expiry = f"{mon} {day:02d} {yr}"
            strike = f"{float(raw_strike):.2f}"
            opt_type = 'Put' if pc.upper() == 'P' else 'Call'
            description = f"{ticker} {expiry} {strike} {opt_type}"
        except Exception:
            ticker, description = symbol_full.split()[0], symbol_full

        # ---------- exchange lookup ----------------------------------
        exchange = self.exch_lookup.get(symbol_full, 'UNKNOWN')

        # ---------- return record ------------------------------------
        return {
            'account'     : 'IB',
            'date'        : ts.strftime('%Y-%m-%d, %H:%M:%S'),
            'activity'    : activity,
            'qty'         : str(qty),
            'symbol'      : ticker,
            'description' : description,
            'price'       : f'{t_price:.4f}',
            'commission'  : f'{commission:.2f}',
            'fees'        : '0',
            'amount'      : f'{proceeds:.2f}',
            'native_ccy'  : ccy_orig,        # ← new flag
            'exchange'    : exchange,
        }

    # ------------------------------------------------------------------
    def _rate(self, ccy: str, ts: datetime):
        """
        Return the first AutoFX rate for *ccy* at or after *ts* (≤48 h).
        """
        candidate, best_delta = None, timedelta(days=2)

        for (cur, fx_ts), rate in self.fx_lookup.items():
            if cur != ccy or fx_ts < ts:
                continue
            delta = fx_ts - ts
            if delta < best_delta:
                best_delta, candidate = delta, rate
                if delta.total_seconds() == 0:   # exact second match
                    break
        return candidate


##################################################################
# 2) parse_currency, parse_mixed_date
##################################################################

def parse_currency(val):
    return float(str(val).replace('$', '').replace(',', ''))

def parse_mixed_date(str_date):
    """
    Tries to parse:
      Ally: "MM/DD/YYYY"
      IB:   "YYYY-MM-DD, HH:MM:SS"
    If both fail, fallback to auto-parse.
    """
    try:
        return pd.to_datetime(str_date, format='%m/%d/%Y')
    except ValueError:
        pass

    try:
        return pd.to_datetime(str_date, format='%Y-%m-%d, %H:%M:%S')
    except ValueError:
        pass

    return pd.to_datetime(str_date)  # fallback guess

##################################################################
# 3) build_transactions
##################################################################

def build_transactions(broker_files, transactions_file=None):
    """
    Read every broker file, pass each row through its parser, and return a
    consolidated pandas DataFrame with columns:

        ACCOUNT, DATE, ACTIVITY, QTY, SYMBOL, DESCRIPTION,
        PRICE, COMMISSION, FEES, AMOUNT, CURRENCY, EXCHANGE
    
    If transactions_file is provided, also write the CSV to that file.
    """
    columns = [
        "ACCOUNT", "DATE", "ACTIVITY", "QTY", "SYMBOL", "DESCRIPTION",
        "PRICE", "COMMISSION", "FEES", "AMOUNT", "NATIVE_CCY", "EXCHANGE"
    ]

    rows = []
    for file_path, parser in broker_files:
        print(f"Parsing '{file_path}' with {parser.__class__.__name__}...")

        # pick delimiter based on extension
        if file_path.endswith('.txt'):
            delimiter = '\t'  # Ally
        elif file_path.endswith('.csv'):
            delimiter = ','   # IB
        else:
            print(f"Warning: Unknown file extension for {file_path}, skipping...")
            continue

        with open(file_path, newline='') as f:
            for row in csv.reader(f, delimiter=delimiter):
                parsed = parser.parse_row(row)
                if not parsed:
                    continue

                # make all keys uppercase so they match the header names
                parsed_uc = {k.upper(): v for k, v in parsed.items()}
                rows.append([parsed_uc.get(col, "") for col in columns])

    df = pd.DataFrame(rows, columns=columns)

    if transactions_file:
        df.to_csv(transactions_file, index=False)
        print("✔ transactions.csv written to", transactions_file)

    return df

##################################################################
# 4) parse_description, read_transactions, merges, matching, etc.
##################################################################

def parse_description(row):
    pattern = r'^(?P<symbol>\S+) (?P<date>\w+ \d{2} \d{4}) (?P<strike>\d+\.\d+) (?P<type>Call|Put)$'
    match = re.match(pattern, row['DESCRIPTION'])
    if match:
        row['OPT_SYMBOL'] = match.group('symbol')
        row['EXPIRATION_DATE'] = pd.to_datetime(match.group('date'))
        row['STRIKE'] = float(match.group('strike'))
        row['OPTION_TYPE'] = match.group('type')
    else:
        row['OPT_SYMBOL'] = None
        row['EXPIRATION_DATE'] = None
        row['STRIKE'] = None
        row['OPTION_TYPE'] = None
    return row

def read_transactions(file_or_df):
    # Accept either a filename or a DataFrame
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        df = pd.read_csv(file_or_df, thousands=',')
    # Filter out "Cash Movement" just like before
    df = df[df['ACTIVITY'] != 'Cash Movement'].copy()

    currency_cols = ['PRICE', 'COMMISSION', 'FEES', 'AMOUNT']
    for col in currency_cols:
        df[col] = df[col].apply(parse_currency)

    df['QTY'] = pd.to_numeric(df['QTY'])
    df['QTY'] = df['QTY'].abs()

    df['DATE'] = df['DATE'].apply(parse_mixed_date)

    # Some brokers might show expired options as "Expired" activity
    df.loc[df['ACTIVITY'] == 'Expired', 'ACTIVITY'] = 'Sold To Close'

    df = df.apply(parse_description, axis=1)
    df.dropna(subset=['OPT_SYMBOL'], inplace=True)

    df['ACTIVITY PRIORITY'] = df['ACTIVITY'].map({
        'Bought To Open': 0,
        'Sold To Open': 0,
        'Sold To Close': 1,
        'Bought To Close': 1
    })

    df.sort_values(by=['DATE', 'ACTIVITY PRIORITY'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def merge_simultaneous(df):
    """
    Merges trades that have identical date/activity/option identity,
    but might come in separate lines.  Weighted price is re-calculated.
    """
    merged = []
    i = 0
    while i < len(df):
        while (
            i + 1 < len(df)
            and df.at[i, 'DATE'] == df.at[i + 1, 'DATE']
            and df.at[i, 'ACTIVITY'] == df.at[i + 1, 'ACTIVITY']
            and df.at[i, 'ACCOUNT'] == df.at[i + 1, 'ACCOUNT']  # must also be same account
            and df.at[i, 'OPT_SYMBOL'] == df.at[i + 1, 'OPT_SYMBOL']
            and df.at[i, 'STRIKE'] == df.at[i + 1, 'STRIKE']
            and df.at[i, 'EXPIRATION_DATE'] == df.at[i + 1, 'EXPIRATION_DATE']
            and df.at[i, 'OPTION_TYPE'] == df.at[i + 1, 'OPTION_TYPE']
        ):
            price1 = df.at[i, 'PRICE']
            price2 = df.at[i + 1, 'PRICE']
            weighted_price = (price1 * df.at[i, 'QTY'] + price2 * df.at[i + 1, 'QTY']) / (df.at[i, 'QTY'] + df.at[i + 1, 'QTY'])
            df.at[i + 1, 'QTY'] += df.at[i, 'QTY']
            df.at[i + 1, 'PRICE'] = round(weighted_price, 2)
            df.at[i + 1, 'COMMISSION'] += df.at[i, 'COMMISSION']
            df.at[i + 1, 'FEES'] += df.at[i, 'FEES']
            df.at[i + 1, 'AMOUNT'] += df.at[i, 'AMOUNT']
            i += 1
        merged.append(df.iloc[i])
        i += 1
    return pd.DataFrame(merged)

def format_strike(strike):
    if strike is None:
        return None
    if float(strike).is_integer():
        return int(strike)
    return round(strike, 2)

def match_trades(df):
    """
    FIFO-match open/close option fills within the same ACCOUNT.
    Returns (trades_df, unopened_df, unclosed_df).
    """
    open_positions = {}
    trades, unopened = [], []

    for _, row in df.iterrows():
        key = (
            row['ACCOUNT'],
            row['OPT_SYMBOL'],
            row['EXPIRATION_DATE'],
            row['STRIKE'],
            row['OPTION_TYPE']
        )

        # ---------- OPEN side ----------------------------------------
        if row['ACTIVITY'] in ('Bought To Open', 'Sold To Open'):
            open_positions.setdefault(key, deque())
            row['_original_qty'] = row['QTY']      # keep for pro-rating
            open_positions[key].append(row.copy())
            continue

        # ---------- CLOSE side ---------------------------------------
        if row['ACTIVITY'] not in ('Sold To Close', 'Bought To Close'):
            continue

        qty_to_match = row['QTY']
        close_per_ct = row['AMOUNT'] / row['QTY']
        close_comm_pc = row['COMMISSION'] / row['QTY']
        close_fees_pc = row['FEES'] / row['QTY']
        close_price   = round(row['PRICE'], 2)

        if key not in open_positions or not open_positions[key]:
            unopened.append(row.copy())
            continue

        while qty_to_match > 0 and open_positions[key]:
            open_row = open_positions[key][0]
            matched_qty = min(qty_to_match, open_row['QTY'])

            orig_qty = open_row['_original_qty']
            open_per_ct   = open_row['AMOUNT'] / orig_qty
            open_comm_pc  = open_row['COMMISSION'] / orig_qty
            open_fees_pc  = open_row['FEES'] / orig_qty
            open_price    = round(open_row['PRICE'], 2)

            open_amt_matched  = round(open_per_ct  * matched_qty, 2)
            close_amt_matched = round(close_per_ct * matched_qty, 2)
            net_profit        = round(close_amt_matched + open_amt_matched, 2)

            comm_total = round((open_comm_pc + close_comm_pc) * matched_qty, 2)
            fees_total = round((open_fees_pc  + close_fees_pc)  * matched_qty, 2)

            trade_return = 0.0
            if open_amt_matched:
                trade_return = round(net_profit / abs(open_amt_matched), 4)

            # ---------------------- output record --------------------
            trade_record = {
                'SYMBOL'          : row['SYMBOL'],
                'OPTION TYPE'     : open_row['OPTION_TYPE'],
                'STRIKE PRICE'    : format_strike(open_row['STRIKE']),
                'EXPIRATION'      : open_row['EXPIRATION_DATE'],
                'OPEN DATE'       : open_row['DATE'],
                'DTE AT OPEN'     : (open_row['EXPIRATION_DATE'] - open_row['DATE']).days,
                'CLOSE DATE'      : row['DATE'],
                'QTY'             : matched_qty,
                'OPEN PRICE'      : open_price,
                'CLOSE PRICE'     : close_price,
                'OPEN AMOUNT'     : open_amt_matched,
                'CLOSE AMOUNT'    : close_amt_matched,
                'COMMISSION TOTAL': comm_total,
                'FEES TOTAL'      : fees_total,
                'NET PROFIT'      : net_profit,
                'RETURN'          : trade_return,
                'ACCOUNT'         : open_row['ACCOUNT'],
                'NATIVE_CCY'      : open_row['NATIVE_CCY'],   # ← new column
            }

            # LONG vs SHORT position
            trade_record['POSITION'] = (
                'Long'  if open_row['ACTIVITY'] == 'Bought To Open' else
                'Short' if open_row['ACTIVITY'] == 'Sold To Open'   else
                'Unknown'
            )

            trades.append(trade_record)

            # update deque / counters
            open_row['QTY'] -= matched_qty
            qty_to_match    -= matched_qty
            if open_row['QTY'] == 0:
                open_positions[key].popleft()

    unclosed = [pos for q in open_positions.values() for pos in q]
    return (
        pd.DataFrame(trades),
        pd.DataFrame(unopened),
        pd.DataFrame(unclosed)
    )

def verify_consistency(df, trades_df, unopened_df, unclosed_df):
    """
    Optional debugging method to check whether
    matched QTY sums line up with original open/close QTY sums.
    """
    original_open = df[df['ACTIVITY'].str.contains('Open')].groupby(
        ['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_open'})

    original_close = df[df['ACTIVITY'].str.contains('Close')].groupby(
        ['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_close'})

    trades_group = trades_df.groupby(
        ['ACCOUNT', 'SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'matched_qty'})

    unclosed_group = unclosed_df.groupby(
        ['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unclosed_qty'})

    unopened_group = unopened_df.groupby(
        ['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unopened_qty'})

    open_check = original_open.merge(
        trades_group,
        left_on=['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        right_on=['ACCOUNT', 'SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE'],
        how='left'
    )
    open_check = open_check.merge(
        unclosed_group,
        on=['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        how='left'
    )
    open_check['matched_qty'] = open_check['matched_qty'].fillna(0)
    open_check['unclosed_qty'] = open_check['unclosed_qty'].fillna(0)
    open_check['total_open_calc'] = open_check['matched_qty'] + open_check['unclosed_qty']
    open_check['open_diff'] = open_check['total_open'] - open_check['total_open_calc']

    close_check = original_close.merge(
        trades_group,
        left_on=['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        right_on=['ACCOUNT', 'SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE'],
        how='left'
    )
    close_check = close_check.merge(
        unopened_group,
        on=['ACCOUNT', 'OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        how='left'
    )
    close_check['matched_qty'] = close_check['matched_qty'].fillna(0)
    close_check['unopened_qty'] = close_check['unopened_qty'].fillna(0)
    close_check['total_close_calc'] = close_check['matched_qty'] + close_check['unopened_qty']
    close_check['close_diff'] = close_check['total_close'] - close_check['total_close_calc']

    # Filter to only the rows where there's a mismatch
    open_problems = open_check[open_check['open_diff'].abs() >= 1e-6]
    close_problems = close_check[close_check['close_diff'].abs() >= 1e-6]

    print("\nOpen events consistency check:")
    if open_problems.empty:
        print("  All open event quantities are consistent.")
    else:
        print(f"  Found {len(open_problems)} open event mismatch rows. Showing all:\n")
        # Temporarily override Pandas display options so we don't see '...'
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None,
                               'display.width', 1000):
            print(open_problems)

    print("\nClose events consistency check:")
    if close_problems.empty:
        print("  All close event quantities are consistent.")
    else:
        print(f"  Found {len(close_problems)} close event mismatch rows. Showing all:\n")
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None,
                               'display.width', 1000):
            print(close_problems)


##################################################################
# 5) Main block  –  updated for FX conversion + exchange tagging
##################################################################

if __name__ == '__main__':
    # ----------------------------------------------------------------
    # 1) Locate the Ally activity file
    # ----------------------------------------------------------------
    ally_path = Path('./data/Ally/activity.txt')
    if not ally_path.is_file():
        print(f"Error: Ally file not found at {ally_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2) Locate all IB CSV files
    # ----------------------------------------------------------------
    ib_dir = Path('./data/IB')
    ib_csv_paths = list(ib_dir.glob('*.csv'))
    if not ib_csv_paths:
        print(f"Warning: No IB csv files found in {ib_dir}.")
        # we still continue; you might be running Ally-only

    # ----------------------------------------------------------------
    # 3) Build the [(filepath, parser)] list
    #     – AllyParser unchanged
    #     – each IBParser now needs its own fx / exchange lookup table
    # ----------------------------------------------------------------
    broker_files = [(str(ally_path), AllyParser())]

    for csv_file in ib_csv_paths:
        fx_lookup, exch_lookup = scan_ib_file(csv_file)      # <-- new helper
        broker_files.append((str(csv_file), IBParser(fx_lookup, exch_lookup)))

    # ----------------------------------------------------------------
    # 4) Output directory for cleaned files
    # ----------------------------------------------------------------
    out_dir = Path('./data/cleaned')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 5) Build and parse transactions without intermediate file
    # ----------------------------------------------------------------
    print("\nBuilding transactions in memory...")
    for fpath, parser in broker_files:
        print(" –", fpath, "via", parser.__class__.__name__)
    df_raw = build_transactions(broker_files)
    df = read_transactions(df_raw)
    df = merge_simultaneous(df)
    trades_df, unopened_df, unclosed_df = match_trades(df)

    # ----------------------------------------------------------------
    # 6) Write final results
    # ----------------------------------------------------------------
    # Write final trades; skip saving unmatched (unopened) closes
    trades_df.to_csv(out_dir / 'trades.csv', index=False)

    # ----- post-processing for unclosed_df (same as your original) -----
    unclosed_df_original = unclosed_df.copy()
    if not unclosed_df.empty:
        unclosed_df['POSITION'] = unclosed_df['ACTIVITY'].map({
            'Bought To Open': 'Long',
            'Sold To Open'  : 'Short'
        }).fillna('UNKNOWN')

        unclosed_df['DTE AT OPEN'] = (
            unclosed_df['EXPIRATION_DATE'] - unclosed_df['DATE']
        ).dt.days

        unclosed_df.rename(columns={
            'DATE'           : 'OPEN DATE',
            'PRICE'          : 'OPEN PRICE',
            'AMOUNT'         : 'OPEN AMOUNT',
            'OPT_SYMBOL'     : 'Option Symbol',
            'EXPIRATION_DATE': 'Expiration',
            'STRIKE'         : 'Strike Price',
            'OPTION_TYPE'    : 'Option Type',
            'QTY'            : 'Quantity',
            'NATIVE_CCY'     : 'Ccy'
        }, inplace=True)

        unclosed_df = unclosed_df[
            [
                'ACCOUNT', 'POSITION', 'Option Symbol', 'Expiration',
                'Strike Price', 'Option Type', 'OPEN DATE',
                'DTE AT OPEN', 'Quantity', 'Ccy', 'OPEN PRICE', 'OPEN AMOUNT'
            ]
        ]

    unclosed_df.to_csv(out_dir / 'unclosed.csv', index=False)

    print("\nProcessed trades written to:", out_dir / 'trades.csv')
    print("Unclosed positions written to:", out_dir / 'unclosed.csv')

    # ----------------------------------------------------------------
    # 7) Optional consistency checks
    # ----------------------------------------------------------------
    verify_consistency(df, trades_df, unopened_df, unclosed_df_original)