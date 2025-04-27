"""
transactions.py
-------------
Core routines to build, normalize, merge, and match option transactions.
"""
import csv
import re
from collections import deque
from datetime import datetime, timedelta

import pandas as pd

def parse_currency(val):  # noqa: C901
    """Strip currency formatting and convert to float."""
    return float(str(val).replace('$', '').replace(',', ''))

def parse_mixed_date(str_date):  # noqa: C901
    """
    Parse dates in either Ally or IB formats:
      Ally: 'MM/DD/YYYY'
      IB:   'YYYY-MM-DD, HH:MM:SS'
    Falls back to pandas automated parsing.
    """
    try:
        return pd.to_datetime(str_date, format='%m/%d/%Y')
    except (ValueError, TypeError):
        pass
    try:
        return pd.to_datetime(str_date, format='%Y-%m-%d, %H:%M:%S')
    except (ValueError, TypeError):
        pass
    return pd.to_datetime(str_date)

def parse_description(row):
    """
    Extract OPT_SYMBOL, EXPIRATION_DATE, STRIKE, OPTION_TYPE from DESCRIPTION.
    """
    pattern = (
        r'^(?P<symbol>\S+) (?P<date>\w+ \d{2} \d{4}) '
        r'(?P<strike>\d+\.\d+) (?P<type>Call|Put)$'
    )
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

def build_transactions(broker_files, transactions_file=None):
    """
    Read each raw broker file via its parser, return a DataFrame.
    If transactions_file is given, also write CSV to that path.
    """
    columns = [
        'ACCOUNT', 'DATE', 'ACTIVITY', 'QTY', 'SYMBOL', 'DESCRIPTION',
        'PRICE', 'COMMISSION', 'FEES', 'AMOUNT', 'NATIVE_CCY', 'EXCHANGE'
    ]
    rows = []
    for file_path, parser in broker_files:
        # pick delimiter
        if str(file_path).endswith('.txt'):
            delimiter = '\t'
        elif str(file_path).endswith('.csv'):
            delimiter = ','
        else:
            # skip unknown
            continue
        with open(file_path, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                parsed = parser.parse_row(row)
                if not parsed:
                    continue
                # uppercase keys for consistency
                parsed_uc = {k.upper(): v for k, v in parsed.items()}
                rows.append([parsed_uc.get(col, '') for col in columns])
    df = pd.DataFrame(rows, columns=columns)
    if transactions_file:
        df.to_csv(transactions_file, index=False)
    return df

def read_transactions(file_or_df):
    """
    Normalize raw transactions DataFrame:
      - parse currencies
      - normalize dates
      - split option descriptions
      - drop non-option rows
      - sort by date & priority
    """
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        df = pd.read_csv(file_or_df, thousands=',')
    # drop cash movements
    df = df[df['ACTIVITY'] != 'Cash Movement'].copy()
    # parse numeric columns
    for col in ['PRICE', 'COMMISSION', 'FEES', 'AMOUNT']:
        df[col] = df[col].apply(parse_currency)
    df['QTY'] = pd.to_numeric(df['QTY']).abs()
    # parse dates
    df['DATE'] = df['DATE'].apply(parse_mixed_date)
    # normalize 'Expired' → Sold To Close
    df.loc[df['ACTIVITY'] == 'Expired', 'ACTIVITY'] = 'Sold To Close'
    # extract option fields
    df = df.apply(parse_description, axis=1)
    df.dropna(subset=['OPT_SYMBOL'], inplace=True)
    # sort by date and open/close priority
    df['ACTIVITY PRIORITY'] = df['ACTIVITY'].map({
        'Bought To Open': 0,
        'Sold To Open':   0,
        'Sold To Close':  1,
        'Bought To Close':1
    })
    df.sort_values(by=['DATE', 'ACTIVITY PRIORITY'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def merge_simultaneous(df):
    """
    Merge rows on identical date/account/option if they occurred simultaneously,
    recomputing weighted price, summing commission, fees, amount.
    """
    merged = []
    i = 0
    while i < len(df):  # noqa: WPS122
        # collapse any contiguous matching rows
        while (
            i + 1 < len(df)
            and df.at[i, 'DATE'] == df.at[i + 1, 'DATE']
            and df.at[i, 'ACTIVITY'] == df.at[i + 1, 'ACTIVITY']
            and df.at[i, 'ACCOUNT'] == df.at[i + 1, 'ACCOUNT']
            and df.at[i, 'OPT_SYMBOL'] == df.at[i + 1, 'OPT_SYMBOL']
            and df.at[i, 'STRIKE'] == df.at[i + 1, 'STRIKE']
            and df.at[i, 'EXPIRATION_DATE'] == df.at[i + 1, 'EXPIRATION_DATE']
            and df.at[i, 'OPTION_TYPE'] == df.at[i + 1, 'OPTION_TYPE']
        ):
            # weighted price
            total_qty = df.at[i, 'QTY'] + df.at[i + 1, 'QTY']
            weighted_price = (
                df.at[i, 'PRICE'] * df.at[i, 'QTY']
                + df.at[i + 1, 'PRICE'] * df.at[i + 1, 'QTY']
            ) / total_qty
            df.at[i + 1, 'QTY'] = total_qty
            df.at[i + 1, 'PRICE'] = round(weighted_price, 2)
            df.at[i + 1, 'COMMISSION'] += df.at[i, 'COMMISSION']
            df.at[i + 1, 'FEES'] += df.at[i, 'FEES']
            df.at[i + 1, 'AMOUNT'] += df.at[i, 'AMOUNT']
            i += 1
        merged.append(df.iloc[i])
        i += 1
    return pd.DataFrame(merged)

def format_strike(strike):
    """Return strike as integer if whole, else float with 2 decimals."""
    if strike is None:
        return None
    f = float(strike)
    return int(f) if f.is_integer() else round(f, 2)

def match_trades(df):
    """
    FIFO-match opens/closes for each option per account.
    Returns (trades_df, unopened_df, unclosed_df).
    """
    open_positions = {}
    trades = []
    unopened = []
    # iterate through sorted rows
    for _, row in df.iterrows():
        key = (
            row['ACCOUNT'], row['OPT_SYMBOL'], row['EXPIRATION_DATE'],
            row['STRIKE'], row['OPTION_TYPE']
        )
        # OPEN side
        if row['ACTIVITY'] in ('Bought To Open', 'Sold To Open'):
            open_positions.setdefault(key, deque())
            row['_original_qty'] = row['QTY']
            open_positions[key].append(row.copy())
            continue
        # skip non-close
        if row['ACTIVITY'] not in ('Sold To Close', 'Bought To Close'):
            continue
        qty_to_match = row['QTY']
        close_per_ct = row['AMOUNT'] / row['QTY']
        close_comm_pc = row['COMMISSION'] / row['QTY']
        close_fees_pc = row['FEES'] / row['QTY']
        close_price = round(row['PRICE'], 2)
        if key not in open_positions or not open_positions[key]:
            unopened.append(row.copy())
            continue
        while qty_to_match > 0 and open_positions[key]:
            open_row = open_positions[key][0]
            matched_qty = min(qty_to_match, open_row['QTY'])
            orig_qty = open_row['_original_qty']
            open_per_ct = open_row['AMOUNT'] / orig_qty
            open_comm_pc = open_row['COMMISSION'] / orig_qty
            open_fees_pc = open_row['FEES'] / orig_qty
            open_price = round(open_row['PRICE'], 2)
            open_amt = round(open_per_ct * matched_qty, 2)
            close_amt = round(close_per_ct * matched_qty, 2)
            net_profit = round(close_amt + open_amt, 2)
            comm_total = round((open_comm_pc + close_comm_pc) * matched_qty, 2)
            fees_total = round((open_fees_pc + close_fees_pc) * matched_qty, 2)
            trade_return = (
                round(net_profit / abs(open_amt), 4)
                if open_amt else 0.0
            )
            rec = {
                'SYMBOL': row['SYMBOL'],
                'OPTION TYPE': open_row['OPTION_TYPE'],
                'STRIKE PRICE': format_strike(open_row['STRIKE']),
                'EXPIRATION': open_row['EXPIRATION_DATE'],
                'OPEN DATE': open_row['DATE'],
                'DTE AT OPEN': (open_row['EXPIRATION_DATE'] - open_row['DATE']).days,
                'CLOSE DATE': row['DATE'],
                'QTY': matched_qty,
                'OPEN PRICE': open_price,
                'CLOSE PRICE': close_price,
                'OPEN AMOUNT': open_amt,
                'CLOSE AMOUNT': close_amt,
                'COMMISSION TOTAL': comm_total,
                'FEES TOTAL': fees_total,
                'NET PROFIT': net_profit,
                'RETURN': trade_return,
                'ACCOUNT': open_row['ACCOUNT'],
                'NATIVE_CCY': open_row['NATIVE_CCY'],
                'POSITION': (
                    'Long' if open_row['ACTIVITY'] == 'Bought To Open' else
                    'Short' if open_row['ACTIVITY'] == 'Sold To Open' else
                    'Unknown'
                )
            }
            trades.append(rec)
            # update counters
            open_row['QTY'] -= matched_qty
            qty_to_match -= matched_qty
            if open_row['QTY'] == 0:
                open_positions[key].popleft()
    # any leftover opens are unclosed
    unclosed = [pos for queue in open_positions.values() for pos in queue]
    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)