import pandas as pd
import re
from collections import deque

def parse_currency(val):
    return float(str(val).replace('$', '').replace(',', ''))

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

    # For our processing we want QTY to be positive, regardless of the original sign.
    df['QTY'] = df['QTY'].abs()

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Treat "Expired" as "Sold To Close" at $0
    df.loc[df['ACTIVITY'] == 'Expired', 'ACTIVITY'] = 'Sold To Close'

    df = df.apply(parse_description, axis=1)
    df.dropna(subset=['OPT_SYMBOL'], inplace=True)

    df['ACTIVITY PRIORITY'] = df['ACTIVITY'].map({'Bought To Open': 0, 'Sold To Close': 1})
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
            # Merge the next row into the current row using absolute quantities.
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
    """Format the strike price: no decimals if whole, otherwise two decimals."""
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
        if row['ACTIVITY'] == 'Bought To Open':
            if key not in open_positions:
                open_positions[key] = deque()
            # Save the original quantity (always positive) for prorating purposes.
            row['_original_qty'] = row['QTY']
            open_positions[key].append(row.copy())
        elif row['ACTIVITY'] == 'Sold To Close':
            qty_to_match = row['QTY']
            # Compute per-contract values for the close event.
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
                # Compute per-contract values for the open event using its original quantity.
                original_qty = open_row['_original_qty']
                open_per_contract = open_row['AMOUNT'] / original_qty
                open_commission_per_contract = open_row['COMMISSION'] / original_qty
                open_fees_per_contract = open_row['FEES'] / original_qty
                open_price = round(open_row['PRICE'], 2)
                
                # Prorate the amounts, commissions, and fees by the matched quantity.
                open_amount_matched = round(open_per_contract * matched_qty, 2)
                close_amount_matched = round(close_per_contract * matched_qty, 2)
                net_profit = round(close_amount_matched + open_amount_matched, 2)
                
                commission_total = round((open_commission_per_contract + close_commission_per_contract) * matched_qty, 2)
                fees_total = round((open_fees_per_contract + close_fees_per_contract) * matched_qty, 2)
                
                # Calculate the trade return as net profit divided by the open amount for the matched contracts.
                trade_return = round(net_profit / abs(open_amount_matched), 4) if open_amount_matched != 0 else 0.0
                
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

                # Adjust the remaining quantities.
                open_row['QTY'] -= matched_qty
                qty_to_match -= matched_qty

                if open_row['QTY'] == 0:
                    open_positions[key].popleft()

    unclosed = [pos for positions in open_positions.values() for pos in positions]
    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)

def verify_consistency(df, trades_df, unopened_df, unclosed_df):
    """
    For each option series, this function verifies:
      - The total open quantity from 'Bought To Open' events equals the sum of matched open quantities
        in trades_df plus the unmatched open quantity in unclosed_df.
      - The total close quantity from 'Sold To Close' events equals the sum of matched close quantities
        in trades_df plus the unmatched close quantity in unopened_df.
    """
    # Open events from the original transactions.
    original_open = df[df['ACTIVITY'] == 'Bought To Open'].groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_open'})

    # Close events from the original transactions.
    original_close = df[df['ACTIVITY'] == 'Sold To Close'].groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'total_close'})

    # Matched quantities in trades_df. Note the key names in trades_df differ.
    trades_group = trades_df.groupby(
        ['SYMBOL', 'EXPIRATION', 'STRIKE PRICE', 'OPTION TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'matched_qty'})

    # Unmatched open events.
    unclosed_group = unclosed_df.groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unclosed_qty'})

    # Unmatched close events.
    unopened_group = unopened_df.groupby(
        ['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE']
    )['QTY'].sum().reset_index().rename(columns={'QTY': 'unopened_qty'})

    # Merge for open events.
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

    # Merge for close events.
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

    print("Open events consistency check:")
    if (open_check['open_diff'].abs() < 1e-6).all():
        print("  All open event quantities are consistent.")
    else:
        print(open_check[['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE', 'total_open', 'total_open_calc', 'open_diff']])

    print("Close events consistency check:")
    if (close_check['close_diff'].abs() < 1e-6).all():
        print("  All close event quantities are consistent.")
    else:
        print(close_check[['OPT_SYMBOL', 'EXPIRATION_DATE', 'STRIKE', 'OPTION_TYPE', 'total_close', 'total_close_calc', 'close_diff']])

if __name__ == '__main__':
    df = read_transactions('transactions.csv')
    df = merge_simultaneous(df)
    trades_df, unopened_df, unclosed_df = match_trades(df)

    trades_df.to_csv('trades.csv', index=False)
    unopened_df.to_csv('unopened.csv', index=False)
    unclosed_df.to_csv('unclosed.csv', index=False)

    print('Processed trades written to trades.csv')
    print('Unopened positions written to unopened.csv')
    print('Unclosed positions written to unclosed.csv')

    # Run the consistency check and print the results.
    verify_consistency(df, trades_df, unopened_df, unclosed_df)