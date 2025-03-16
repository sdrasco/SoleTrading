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

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Treat "Expired" as "Sold To Close" at $0
    df.loc[df['ACTIVITY'] == 'Expired', 'ACTIVITY'] = 'Sold To Close'

    df = df.apply(parse_description, axis=1)
    df.dropna(subset=['OPT_SYMBOL'], inplace=True)

    df['ACTIVITY_PRIORITY'] = df['ACTIVITY'].map({'Bought To Open': 0, 'Sold To Close': 1})
    df.sort_values(by=['DATE', 'ACTIVITY_PRIORITY'], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def merge_simultaneous(df):
    merged = []
    i = 0
    while i < len(df):
        current = df.iloc[i].copy()
        while (
            i + 1 < len(df)
            and df.at[i, 'DATE'] == df.at[i + 1, 'DATE']
            and df.at[i, 'ACTIVITY'] == df.at[i + 1, 'ACTIVITY']
            and df.at[i, 'OPT_SYMBOL'] == df.at[i + 1, 'OPT_SYMBOL']
            and df.at[i, 'STRIKE'] == df.at[i + 1, 'STRIKE']
            and df.at[i, 'EXPIRATION_DATE'] == df.at[i + 1, 'EXPIRATION_DATE']
            and df.at[i, 'OPTION_TYPE'] == df.at[i + 1, 'OPTION_TYPE']
        ):
            next_row = df.iloc[i + 1]

            price1 = df.at[i, 'PRICE']
            price2 = df.at[i + 1, 'PRICE']
            weighted_price = (price1 * abs(df.at[i, 'QTY']) + price2 * abs(df.at[i + 1, 'QTY'])) / (abs(df.at[i, 'QTY']) + abs(df.at[i + 1, 'QTY']))

            df.at[i + 1, 'QTY'] += df.at[i, 'QTY']
            df.at[i + 1, 'PRICE'] = round(weighted_price, 2)
            df.at[i + 1, 'COMMISSION'] += df.at[i, 'COMMISSION']
            df.at[i + 1, 'FEES'] += df.at[i, 'FEES']
            df.at[i + 1, 'AMOUNT'] += df.at[i, 'AMOUNT']

            i += 1

        merged.append(df.iloc[i])
        i += 1
    return pd.DataFrame(merged)

def match_trades(df):
    open_positions = {}
    trades = []
    unopened = []

    for _, row in df.iterrows():
        key = (row['OPT_SYMBOL'], row['EXPIRATION_DATE'], row['STRIKE'], row['OPTION_TYPE'])

        if row['ACTIVITY'] == 'Bought To Open':
            if key not in open_positions:
                open_positions[key] = deque()
            open_positions[key].append(row.copy())
        elif row['ACTIVITY'] == 'Sold To Close':
            qty_to_match = abs(row['QTY'])
            if key not in open_positions or not open_positions[key]:
                unopened.append(row.copy())
                continue
            while qty_to_match > 0 and open_positions[key]:
                open_row = open_positions[key][0]
                open_qty = abs(open_row['QTY'])
                matched_qty = min(qty_to_match, open_qty)

                trade_record = {
                    'SYMBOL': row['SYMBOL'],
                    'DESCRIPTION': row['DESCRIPTION'],
                    'OPEN_DATE': open_row['DATE'],
                    'CLOSE_DATE': row['DATE'],
                    'QTY': matched_qty,
                    'OPEN_PRICE': open_row['PRICE'],
                    'CLOSE_PRICE': row['PRICE'],
                    'OPEN_AMOUNT': open_row['AMOUNT'],
                    'CLOSE_AMOUNT': row['AMOUNT'],
                    'COMMISSION_TOTAL': open_row['COMMISSION'] + row['COMMISSION'],
                    'FEES_TOTAL': open_row['FEES'] + row['FEES'],
                    'NET_PROFIT': row['AMOUNT'] + open_row['AMOUNT'],
                }
                trades.append(trade_record)

                open_row['QTY'] -= matched_qty
                qty_to_match -= matched_qty

                if open_row['QTY'] == 0:
                    open_positions[key].popleft()

    unclosed = [pos for positions in open_positions.values() for pos in positions]

    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)

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