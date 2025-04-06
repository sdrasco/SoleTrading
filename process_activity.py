# process_activity.py

import sys
import csv
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
    def parse_line(self, line: str):
        """
        Must return a dict:
          {
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
        or None if the line doesn't parse or isn't relevant.
        """
        pass


class AllyParser(BaseBrokerParser):
    """
    Parses lines from an ally.txt file (tab-delimited).
    """

    def parse_line(self, line: str):
        if not line.strip():
            return None
        parts = line.strip().split('\t')
        if len(parts) < 9:
            return None

        date, activity, qty, sym_or_type, description, price, commission, fees, amount = parts[:9]

        if activity == "Cash Movement":
            symbol = sym_or_type if "FULLYPAID LENDING REBATE" in description else ""
        else:
            symbol = sym_or_type.split()[0]

        return {
            'date': date,              # e.g. "03/28/2025"
            'activity': activity,      # e.g. "Bought To Open"
            'qty': qty,                # e.g. "1" or "-1"
            'symbol': symbol,          # e.g. "TSLA"
            'description': description,# e.g. "TSLA Apr 04 2025 475.00 Put"
            'price': price,            # e.g. "$10.04"
            'commission': commission,  # e.g. "$0.50"
            'fees': fees,              # e.g. "$0.07"
            'amount': amount           # e.g. "$1,003.43"
        }


class IBParser(BaseBrokerParser):
    """
    Parses lines from an IB.txt file (tab-delimited), matching your 'Trades Data Order' lines.
    """

    def parse_line(self, line: str):
        columns = line.strip().split('\t')
        if len(columns) < 17:
            return None

        # The user-supplied data: only lines with columns[1] == 'Data' and columns[2] == 'Order' matter.
        if columns[1] != 'Data' or columns[2] != 'Order':
            return None

        symbol_full  = columns[5]   # e.g. "CAT 17APR25 250 P"
        date_time    = columns[6]   # e.g. "2025-04-04, 10:53:04"
        qty_str      = columns[7]   # e.g. "1" or "-1"
        t_price_str  = columns[8]   # e.g. "3.92"
        proceeds_str = columns[10]  # e.g. "-392"
        comm_fee_str = columns[11]  # e.g. "-1.05725"
        code         = columns[16]  # e.g. "O" or "C"

        try:
            quantity = int(qty_str)
        except ValueError:
            return None

        # Map code + sign of qty to "Bought To Open", "Sold To Close", etc.
        if code == 'O':
            if quantity > 0:
                activity = 'Bought To Open'
            else:
                activity = 'Sold To Open'
        elif code == 'C':
            if quantity > 0:
                activity = 'Bought To Close'
            else:
                activity = 'Sold To Close'
        else:
            return None

        # We'll treat 'comm_fee_str' as total commission, and set fees to "0"
        commission_str = comm_fee_str
        fees_str = '0'

        # The total amount is 'proceeds' from IB
        amount_str = proceeds_str
        # The 'price' is the "Trade Price"
        price_str = t_price_str

        # Next, parse "CAT 17APR25 250 P" into a friendlier description
        parts = symbol_full.split()
        if len(parts) != 4:
            return None

        ticker = parts[0]          # e.g. "CAT"
        raw_exp = parts[1]         # e.g. "17APR25"
        raw_strike = parts[2]      # e.g. "250"
        raw_putcall = parts[3]     # e.g. "P"

        # Convert "17APR25" => "Apr 17 2025"
        month_map = {
            'JAN':'Jan','FEB':'Feb','MAR':'Mar','APR':'Apr','MAY':'May','JUN':'Jun',
            'JUL':'Jul','AUG':'Aug','SEP':'Sep','OCT':'Oct','NOV':'Nov','DEC':'Dec'
        }
        day_str = raw_exp[:2]       # "17"
        mon_str = raw_exp[2:5].upper()  # "APR"
        yr_str  = raw_exp[5:]       # "25" => 2025
        if mon_str not in month_map:
            return None

        month_str = month_map[mon_str]
        year_full = f"20{yr_str}"
        expiry_str = f"{month_str} {int(day_str)} {year_full}"

        # Convert strike to float
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
            'date': date_time,      # e.g. "2025-04-04, 10:53:04"
            'activity': activity,   # e.g. "Bought To Open"
            'qty': qty_str,         # e.g. "-1"
            'symbol': ticker,       # e.g. "CAT"
            'description': description_str,    # e.g. "CAT Apr 17 2025 250.00 Put"
            'price': price_str,               # e.g. "3.92"
            'commission': commission_str,      # e.g. "-1.05725"
            'fees': fees_str,                 # e.g. "0"
            'amount': amount_str              # e.g. "-392"
        }

##################################################################
# 2) A function to parse both date formats
##################################################################

def parse_currency(val):
    return float(str(val).replace('$', '').replace(',', ''))

def parse_mixed_date(str_date):
    """
    Tries to parse a date from either:
      Ally format: "MM/DD/YYYY"
      IB format:   "YYYY-MM-DD, HH:MM:SS"
    """
    # Attempt Ally (e.g. "03/28/2025")
    try:
        return pd.to_datetime(str_date, format='%m/%d/%Y')
    except ValueError:
        pass

    # Attempt IB (e.g. "2025-04-04, 10:53:04")
    try:
        return pd.to_datetime(str_date, format='%Y-%m-%d, %H:%M:%S')
    except ValueError:
        pass

    # If neither format works, let pandas guess (or raise an error)
    return pd.to_datetime(str_date)  # fallback guess

##################################################################
# 3) Building transactions and reading them
##################################################################

def build_transactions(broker_files, transactions_file='transactions.csv'):
    """
    Reads each file with its appropriate parser, then writes out a
    single CSV containing columns:
       DATE, ACTIVITY, QTY, SYMBOL, DESCRIPTION, PRICE, COMMISSION, FEES, AMOUNT
    """
    columns = ["DATE", "ACTIVITY", "QTY", "SYMBOL", "DESCRIPTION", "PRICE", "COMMISSION", "FEES", "AMOUNT"]

    with open(transactions_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)

        for activity_file, parser in broker_files:
            print(f"Parsing file '{activity_file}' with {parser.__class__.__name__}...")
            with open(activity_file, 'r') as infile:
                for line in infile:
                    parsed = parser.parse_line(line)
                    if not parsed:
                        continue
                    writer.writerow([
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

def parse_description(row):
    """
    After we unify everything in transactions.csv, this function
    attempts to parse the 'DESCRIPTION' column if it looks like:
      "CAT Apr 17 2025 250.00 Put"
    extracting:
      row['OPT_SYMBOL'] = 'CAT'
      row['EXPIRATION_DATE'] = 2025-04-17
      row['STRIKE'] = 250.00
      row['OPTION_TYPE'] = 'Put'
    """
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
    # Exclude 'Cash Movement' rows
    df = df[df['ACTIVITY'] != 'Cash Movement'].copy()

    # Convert money columns
    currency_cols = ['PRICE', 'COMMISSION', 'FEES', 'AMOUNT']
    for col in currency_cols:
        df[col] = df[col].apply(parse_currency)

    # Convert quantity to numeric and take absolute value
    df['QTY'] = pd.to_numeric(df['QTY'])
    df['QTY'] = df['QTY'].abs()

    # Convert the 'DATE' column using our mixed-date parser
    df['DATE'] = df['DATE'].apply(parse_mixed_date)

    # Treat "Expired" as "Sold To Close"
    df.loc[df['ACTIVITY'] == 'Expired', 'ACTIVITY'] = 'Sold To Close'

    # Attempt to parse the option description
    df = df.apply(parse_description, axis=1)
    df.dropna(subset=['OPT_SYMBOL'], inplace=True)

    # Assign an order priority for sorting
    #    Open trades (BTO, STO) => priority 0
    #    Close trades (STC, BTC) => priority 1
    # so that when we have same-day trades, the open is listed first.
    df['ACTIVITY PRIORITY'] = df['ACTIVITY'].map({
        'Bought To Open': 0,
        'Sold To Open': 0,
        'Sold To Close': 1,
        'Bought To Close': 1
    })

    # Sort by date, then activity priority
    df.sort_values(by=['DATE', 'ACTIVITY PRIORITY'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

##################################################################
# 4) Matching and consistency checks
##################################################################

def merge_simultaneous(df):
    """
    If you have multiple lines for the same day, same trade,
    merges them into a single line with a weighted average price, etc.
    """
    merged = []
    i = 0
    while i < len(df):
        while (
            i + 1 < len(df)
            and df.at[i, 'DATE'] == df.at[i + 1, 'DATE']
            and df.at[i, 'ACTIVITY'] == df.at[i + 1, 'ACTIVITY']
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
    if strike.is_integer():
        return int(strike)
    return round(strike, 2)

def match_trades(df):
    open_positions = {}
    trades = []
    unopened = []

    for _, row in df.iterrows():
        key = (row['OPT_SYMBOL'], row['EXPIRATION_DATE'], row['STRIKE'], row['OPTION_TYPE'])
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
                }
                trades.append(trade_record)

                open_row['QTY'] -= matched_qty
                qty_to_match -= matched_qty
                if open_row['QTY'] == 0:
                    open_positions[key].popleft()

    unclosed = [pos for positions in open_positions.values() for pos in positions]
    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)

def verify_consistency(df, trades_df, unopened_df, unclosed_df):
    original_open = df[df['ACTIVITY'].str.contains('Open')].groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_open'})

    original_close = df[df['ACTIVITY'].str.contains('Close')].groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_close'})

    trades_group = trades_df.groupby(
        ['SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'matched_qty'})

    unclosed_group = unclosed_df.groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unclosed_qty'})

    unopened_group = unopened_df.groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unopened_qty'})

    open_check = original_open.merge(
        trades_group,
        left_on=['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        right_on=['SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE'],
        how='left'
    )
    open_check = open_check.merge(
        unclosed_group,
        on=['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        how='left'
    )
    open_check['matched_qty'] = open_check['matched_qty'].fillna(0)
    open_check['unclosed_qty'] = open_check['unclosed_qty'].fillna(0)
    open_check['total_open_calc'] = open_check['matched_qty'] + open_check['unclosed_qty']
    open_check['open_diff'] = open_check['total_open'] - open_check['total_open_calc']

    close_check = original_close.merge(
        trades_group,
        left_on=['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        right_on=['SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE'],
        how='left'
    )
    close_check = close_check.merge(
        unopened_group,
        on=['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE'],
        how='left'
    )
    close_check['matched_qty'] = close_check['matched_qty'].fillna(0)
    close_check['unopened_qty'] = close_check['unopened_qty'].fillna(0)
    close_check['total_close_calc'] = close_check['matched_qty'] + close_check['unopened_qty']
    close_check['close_diff'] = close_check['total_close'] - close_check['total_close_calc']

    print("\nOpen events consistency check:")
    if (open_check['open_diff'].abs() < 1e-6).all():
        print("  All open event quantities are consistent.")
    else:
        print(open_check[['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE',
                          'total_open', 'total_open_calc', 'open_diff']])

    print("\nClose events consistency check:")
    if (close_check['close_diff'].abs() < 1e-6).all():
        print("  All close event quantities are consistent.")
    else:
        print(close_check[['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE',
                           'total_close', 'total_close_calc', 'close_diff']])

##################################################################
# 5) Main block
##################################################################

if __name__ == '__main__':
    # We'll look for ally.txt and IB.txt in the current directory.
    ally_path = Path('ally.txt')
    ib_path = Path('IB.txt')
    broker_files = []

    # If ally.txt is present, parse with AllyParser
    if ally_path.is_file():
        broker_files.append((str(ally_path), AllyParser()))

    # If IB.txt is present, parse with IBParser
    if ib_path.is_file():
        broker_files.append((str(ib_path), IBParser()))

    if not broker_files:
        print("Error: Neither 'ally.txt' nor 'IB.txt' found in the current directory.")
        sys.exit(1)

    # Combine everything into transactions.csv
    transactions_file = 'transactions.csv'
    print(f"Building transactions from these files into {transactions_file}:")
    for bf in broker_files:
        print(" -", bf[0])

    build_transactions(broker_files, transactions_file)

    # Read, analyze, match trades, produce final CSVs
    df = read_transactions(transactions_file)
    df = merge_simultaneous(df)
    trades_df, unopened_df, unclosed_df = match_trades(df)

    trades_df.to_csv('trades.csv', index=False)
    unopened_df.to_csv('unopened.csv', index=False)

    unclosed_df_original = unclosed_df.copy()
    if not unclosed_df.empty:
        unclosed_df['DTE AT OPEN'] = (unclosed_df['EXPIRATION_DATE'] - unclosed_df['DATE']).dt.days
        unclosed_df.rename(columns={
            'DATE': 'OPEN DATE',
            'PRICE': 'OPEN PRICE',
            'AMOUNT': 'OPEN AMOUNT'
        }, inplace=True)
        unclosed_df = unclosed_df[['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE',
                                   'OPEN DATE', 'DTE AT OPEN', 'QTY', 'OPEN PRICE', 'OPEN AMOUNT']]
        unclosed_df.rename(columns={
            'OPT_SYMBOL': 'Option Symbol',
            'EXPIRATION_DATE': 'Expiration',
            'STRIKE': 'Strike Price',
            'OPTION_TYPE': 'Option Type',
            'QTY': 'Quantity'
        }, inplace=True)

    unclosed_df.to_csv('unclosed.csv', index=False)

    print("\nProcessed trades written to trades.csv")
    print("Unopened positions written to unopened.csv")
    print("Unclosed positions (concise) written to unclosed.csv")

    verify_consistency(df, trades_df, unopened_df, unclosed_df_original)