#!/usr/bin/env python3

import sys
import csv
import glob
import pandas as pd
import re
from collections import deque
from abc import ABC, abstractmethod
from pathlib import Path


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
            'amount': amount
        }


class IBParser(BaseBrokerParser):
    """
    Parses rows from a comma-delimited IB.csv file.
    Each row might look like:
      [
        'Trades', 'Data', 'Order', 'Equity and Index Options', 'USD',
        'CAT 17APR25 250 P', '2025-04-04, 10:53:04', '1', '3.92', '2.955',
        '-392', '-1.05725', '393.05725', '0', '0', '-96.5', 'O'
      ]
    We require at least 17 columns. Only rows with [0]=='Trades', [1]=='Data', [2]=='Order' are relevant.
    """

    def parse_row(self, row: list):
        if len(row) < 17:
            return None

        # Check the first three columns
        if row[0] != 'Trades' or row[1] != 'Data' or row[2] != 'Order':
            return None

        symbol_full  = row[5]   # e.g. "CAT 17APR25 250 P"
        date_time    = row[6]   # e.g. "2025-04-04, 10:53:04"
        qty_str      = row[7]   # e.g. "1" or "-1"
        t_price_str  = row[8]   # e.g. "3.92"
        proceeds_str = row[10]  # e.g. "-392"
        comm_fee_str = row[11]  # e.g. "-1.05725"
        code         = row[16]  # e.g. "O" or "C" or "O;P" etc.

        # Convert quantity
        try:
            quantity = int(qty_str)
        except ValueError:
            return None

        # Trim the code so "O;P" => "O", "C;something" => "C"
        code_core = code.split(';')[0].strip()

        # Determine activity from code + sign of quantity
        if code_core == 'O':
            if quantity > 0:
                activity = 'Bought To Open'
            else:
                activity = 'Sold To Open'
        elif code_core == 'C':
            if quantity > 0:
                activity = 'Bought To Close'
            else:
                activity = 'Sold To Close'
        else:
            # Unrecognized code
            return None

        # We'll treat comm_fee_str as commission, fees=0
        commission_str = comm_fee_str
        fees_str = '0'

        # 'amount' from proceeds
        amount_str = proceeds_str
        # 'price' from T. Price
        price_str = t_price_str

        # Parse "CAT 17APR25 250 P" -> "CAT Apr 17 2025 250.00 Put"
        parts = symbol_full.split()
        if len(parts) != 4:
            return None

        ticker = parts[0]
        raw_exp = parts[1]      # "17APR25"
        raw_strike = parts[2]   # "250"
        raw_putcall = parts[3]  # "P" or "C"

        month_map = {
            'JAN':'Jan','FEB':'Feb','MAR':'Mar','APR':'Apr','MAY':'May','JUN':'Jun',
            'JUL':'Jul','AUG':'Aug','SEP':'Sep','OCT':'Oct','NOV':'Nov','DEC':'Dec'
        }
        day_str = raw_exp[:2]         # "17"
        mon_str = raw_exp[2:5].upper()# "APR"
        yr_str  = raw_exp[5:]         # "25" => "2025"

        if mon_str not in month_map:
            return None

        month_str = month_map[mon_str]
        year_full = f"20{yr_str}"
        # zero-pad day
        expiry_str = f"{month_str} {int(day_str):02d} {year_full}"

        try:
            strike_float = float(raw_strike)
        except ValueError:
            return None
        strike_formatted = f"{strike_float:.2f}"

        if raw_putcall.upper() == 'P':
            option_type = 'Put'
        else:
            option_type = 'Call'

        description_str = f"{ticker} {expiry_str} {strike_formatted} {option_type}"

        return {
            'account': 'IB',
            'date': date_time,
            'activity': activity,
            'qty': qty_str,
            'symbol': ticker,
            'description': description_str,
            'price': price_str,
            'commission': commission_str,
            'fees': fees_str,
            'amount': amount_str
        }


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

def build_transactions(broker_files, transactions_file='transactions.csv'):
    """
    For each file in broker_files, read it with the correct delimiter, pass each row to the parser,
    and write out a single combined CSV with columns:
       ACCOUNT, DATE, ACTIVITY, QTY, SYMBOL, DESCRIPTION, PRICE, COMMISSION, FEES, AMOUNT
    """
    columns = [
        "ACCOUNT", "DATE", "ACTIVITY", "QTY", "SYMBOL", "DESCRIPTION",
        "PRICE", "COMMISSION", "FEES", "AMOUNT"
    ]

    with open(transactions_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)

        for file_path, parser in broker_files:
            print(f"Parsing '{file_path}' with {parser.__class__.__name__}...")

            # Decide delimiter by file extension
            if file_path.endswith('.txt'):
                # Ally
                with open(file_path, 'r', newline='') as f:
                    rowreader = csv.reader(f, delimiter='\t')
                    for row in rowreader:
                        parsed = parser.parse_row(row)
                        if not parsed:
                            continue
                        writer.writerow([
                            parsed['account'],
                            parsed['date'],
                            parsed['activity'],
                            parsed['qty'],
                            parsed['symbol'],
                            parsed['description'],
                            parsed['price'],
                            parsed['commission'],
                            parsed['fees'],
                            parsed['amount']
                        ])

            elif file_path.endswith('.csv'):
                # IB
                with open(file_path, 'r', newline='') as f:
                    rowreader = csv.reader(f)
                    for row in rowreader:
                        parsed = parser.parse_row(row)
                        if not parsed:
                            continue
                        writer.writerow([
                            parsed['account'],
                            parsed['date'],
                            parsed['activity'],
                            parsed['qty'],
                            parsed['symbol'],
                            parsed['description'],
                            parsed['price'],
                            parsed['commission'],
                            parsed['fees'],
                            parsed['amount']
                        ])

            else:
                print(f"Warning: Unknown file extension for {file_path}, skipping...")

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

def read_transactions(file):
    df = pd.read_csv(file, thousands=',')
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
    Matches open/close trades on a FIFO basis, but only within the same ACCOUNT.
    """
    open_positions = {}
    trades = []
    unopened = []

    for _, row in df.iterrows():
        # Add ACCOUNT to the key so that cross-account matches don't happen:
        key = (
            row['ACCOUNT'],
            row['OPT_SYMBOL'],
            row['EXPIRATION_DATE'],
            row['STRIKE'],
            row['OPTION_TYPE']
        )

        if row['ACTIVITY'] in ['Bought To Open', 'Sold To Open']:
            if key not in open_positions:
                open_positions[key] = deque()
            row['_original_qty'] = row['QTY']
            open_positions[key].append(row.copy())

        elif row['ACTIVITY'] in ['Sold To Close', 'Bought To Close']:
            qty_to_match = row['QTY']
            close_per_contract = row['AMOUNT'] / row['QTY']
            close_commission_per_contract = row['COMMISSION'] / row['QTY']
            close_fees_per_contract = row['FEES'] / row['QTY']
            close_price = round(row['PRICE'], 2)

            # If there's nothing in open_positions for this key, it's "unopened"
            if key not in open_positions or not open_positions[key]:
                unopened.append(row.copy())
                continue

            while qty_to_match > 0 and open_positions[key]:
                open_row = open_positions[key][0]
                matched_qty = min(qty_to_match, open_row['QTY'])
                original_qty = open_row['_original_qty']
                open_per_contract = open_row['AMOUNT'] / original_qty
                open_commission_per_contract = open_row['COMMISSION'] / original_qty
                open_fees_per_contract = open_row['FEES'] / original_qty
                open_price = round(open_row['PRICE'], 2)

                open_amount_matched = round(open_per_contract * matched_qty, 2)
                close_amount_matched = round(close_per_contract * matched_qty, 2)
                net_profit = round(close_amount_matched + open_amount_matched, 2)

                commission_total = round((open_commission_per_contract + close_commission_per_contract) * matched_qty, 2)
                fees_total = round((open_fees_per_contract + close_fees_per_contract) * matched_qty, 2)

                trade_return = 0.0
                if open_amount_matched != 0:
                    trade_return = round(net_profit / abs(open_amount_matched), 4)

                trade_record = {
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
                    'OPEN AMOUNT': open_amount_matched,
                    'CLOSE AMOUNT': close_amount_matched,
                    'COMMISSION TOTAL': commission_total,
                    'FEES TOTAL': fees_total,
                    'NET PROFIT': net_profit,
                    'RETURN': trade_return,
                    'ACCOUNT': open_row['ACCOUNT'],  # or row['ACCOUNT'], they should be the same
                }
                # Determine if the open was Bought or Sold
                open_activity = open_row['ACTIVITY']
                if open_activity == 'Bought To Open':
                    position_side = 'Long'
                elif open_activity == 'Sold To Open':
                    position_side = 'Short'
                else:
                    # Should only be Open activities, but just in case:
                    position_side = 'Unknown'

                trade_record['POSITION'] = position_side
                trades.append(trade_record)

                open_row['QTY'] -= matched_qty
                qty_to_match -= matched_qty
                if open_row['QTY'] == 0:
                    open_positions[key].popleft()

    unclosed = [pos for positions in open_positions.values() for pos in positions]
    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)

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
# 5) Main block
##################################################################

if __name__ == '__main__':
    # ----------------------------------------------------------------
    # 1) Find ally's single activity file:  ./data/Ally/activity.txt
    # ----------------------------------------------------------------
    ally_path = Path('./data/Ally/activity.txt')
    if not ally_path.is_file():
        print(f"Error: Ally file not found at {ally_path}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2) Find all IB CSV files in ./data/IB/*.csv
    # ----------------------------------------------------------------
    ib_dir = Path('./data/IB')
    ib_csv_paths = list(ib_dir.glob('*.csv'))
    if not ib_csv_paths:
        print(f"Warning: No IB csv files found in {ib_dir}. "
              "If you expected IB files, please check your directory.")

    # ----------------------------------------------------------------
    # 3) Build up our list of (filepath, parser) for the aggregator
    # ----------------------------------------------------------------
    broker_files = []
    broker_files.append((str(ally_path), AllyParser()))
    for csv_file in ib_csv_paths:
        broker_files.append((str(csv_file), IBParser()))

    # ----------------------------------------------------------------
    # 4) Where to write out the cleaned files? => ./data/cleaned/
    # ----------------------------------------------------------------
    out_dir = Path('./data/cleaned')
    out_dir.mkdir(parents=True, exist_ok=True)  # Make sure it exists

    transactions_file = out_dir / 'transactions.csv'

    print(f"Building transactions from these sources into {transactions_file}:")
    for fpath, parser in broker_files:
        print(" -", fpath, "via", parser.__class__.__name__)

    # Create a single consolidated CSV
    build_transactions(broker_files, transactions_file=str(transactions_file))

# ----------------------------------------------------------------
# 5) Read, parse, and merge trades
# ----------------------------------------------------------------
df = read_transactions(str(transactions_file))
df = merge_simultaneous(df)
trades_df, unopened_df, unclosed_df = match_trades(df)

# ----------------------------------------------------------------
# 6) Write final results to ./data/cleaned/
# ----------------------------------------------------------------
trades_df.to_csv(out_dir / 'trades.csv', index=False)
unopened_df.to_csv(out_dir / 'unopened.csv', index=False)

unclosed_df_original = unclosed_df.copy()
if not unclosed_df.empty:
    # 1) Add 'POSITION' = LONG or SHORT, based on the original ACTIVITY
    #    (We keep ACTIVITY temporarily just to create this column.)
    #    unclosed_df has the same columns as the open_row in match_trades()
    #    so it includes 'ACTIVITY'.
    unclosed_df['POSITION'] = unclosed_df['ACTIVITY'].map({
        'Bought To Open': 'Long',
        'Sold To Open': 'Short'
    }).fillna('UNKNOWN')  # Fallback if we ever see an unexpected open activity

    # 2) Calculate DTE (Days to Expiration) at open
    unclosed_df['DTE AT OPEN'] = (unclosed_df['EXPIRATION_DATE'] - unclosed_df['DATE']).dt.days

    # 3) Rename columns for clarity
    unclosed_df.rename(columns={
        'DATE': 'OPEN DATE',
        'PRICE': 'OPEN PRICE',
        'AMOUNT': 'OPEN AMOUNT',
        'OPT_SYMBOL': 'Option Symbol',
        'EXPIRATION_DATE': 'Expiration',
        'STRIKE': 'Strike Price',
        'OPTION_TYPE': 'Option Type',
        'QTY': 'Quantity'
    }, inplace=True)

    # 4) Re-select columns in the final unclosed.csv output
    #    We do NOT keep ACTIVITY, COMMISSION, FEES, etc.
    unclosed_df = unclosed_df[
        [
            'ACCOUNT',
            'POSITION',         # show LONG or SHORT
            'Option Symbol',
            'Expiration',
            'Strike Price',
            'Option Type',
            'OPEN DATE',
            'DTE AT OPEN',
            'Quantity',
            'OPEN PRICE',
            'OPEN AMOUNT'
        ]
    ]

unclosed_df.to_csv(out_dir / 'unclosed.csv', index=False)

print("\nProcessed trades written to:", out_dir / 'trades.csv')
print("Unopened positions written to:", out_dir / 'unopened.csv')
print("Unclosed positions (concise) written to:", out_dir / 'unclosed.csv')

# ----------------------------------------------------------------
# 7) Optional consistency checks
# ----------------------------------------------------------------
verify_consistency(df, trades_df, unopened_df, unclosed_df_original)
