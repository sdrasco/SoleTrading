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
    def parse_row(self, row: list):
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
        code         = row[16]  # e.g. "O" or "C"

        # Convert quantity
        try:
            quantity = int(qty_str)
        except ValueError:
            return None

        # Determine activity from code + sign of quantity
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
       DATE, ACTIVITY, QTY, SYMBOL, DESCRIPTION, PRICE, COMMISSION, FEES, AMOUNT
    """
    columns = ["DATE", "ACTIVITY", "QTY", "SYMBOL", "DESCRIPTION", "PRICE", "COMMISSION", "FEES", "AMOUNT"]

    with open(transactions_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)

        for file_path, parser in broker_files:
            print(f"Parsing '{file_path}' with {parser.__class__.__name__}...")

            # Decide delimiter by file extension
            if file_path.endswith('.txt'):
                # Ally
                with open(file_path, 'r', newline='') as f:
                    # Parse as a tab-delimited file
                    rowreader = csv.reader(f, delimiter='\t')
                    for row in rowreader:
                        parsed = parser.parse_row(row)
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

            elif file_path.endswith('.csv'):
                # IB
                with open(file_path, 'r', newline='') as f:
                    # Parse as a comma-delimited file
                    rowreader = csv.reader(f)
                    for row in rowreader:
                        parsed = parser.parse_row(row)
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
    df = df[df['ACTIVITY'] != 'Cash Movement'].copy()

    currency_cols = ['PRICE', 'COMMISSION', 'FEES', 'AMOUNT']
    for col in currency_cols:
        df[col] = df[col].apply(parse_currency)

    df['QTY'] = pd.to_numeric(df['QTY'])
    df['QTY'] = df['QTY'].abs()

    df['DATE'] = df['DATE'].apply(parse_mixed_date)

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
    ally_path = Path('ally.txt')
    ib_csv_path = Path('IB.csv')
    broker_files = []

    # If ally.txt is present
    if ally_path.is_file():
        broker_files.append((str(ally_path), AllyParser()))

    # If IB.csv is present
    if ib_csv_path.is_file():
        broker_files.append((str(ib_csv_path), IBParser()))

    if not broker_files:
        print("Error: neither ally.txt nor IB.csv found.")
        sys.exit(1)

    transactions_file = 'transactions.csv'
    print(f"Building transactions from the following source(s) into {transactions_file}:")
    for fpath, parser in broker_files:
        print(" -", fpath, "via", parser.__class__.__name__)

    build_transactions(broker_files, transactions_file)

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