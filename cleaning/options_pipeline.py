# cleaning/options_pipeline.py

import re
import pandas as pd
from collections import deque
from cleaning.data_helpers import parse_currency, parse_mixed_date


def dicts_to_option_df(parsed_option_rows):
    """
    Convert a list of dictionaries (parsed option records) into a DataFrame,
    cleaning up columns and data types similarly to the old code.
    """
    df = pd.DataFrame(parsed_option_rows)
    
    # 1) Drop "Cash Movement"
    df = df[df["activity"] != "Cash Movement"].copy()

    # 2) Convert numeric fields
    for col in ["price", "commission", "fees", "amount"]:
        df[col] = df[col].apply(parse_currency)

    # 3) QTY → numeric, absolute
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").abs()

    # 4) date → parsed
    df["date"] = df["date"].apply(parse_mixed_date)

    # 5) "Expired" → "Sold To Close"
    df.loc[df["activity"] == "Expired", "activity"] = "Sold To Close"

    # 6) Extract option details from 'description'
    df = df.apply(_extract_option_details, axis=1)

    # 7) Drop rows where "OPT_SYMBOL" is still NaN → indicates parse failure
    df.dropna(subset=["OPT_SYMBOL"], inplace=True)

    # 8) Add an "activity_priority" so open legs sort before close legs
    df["activity_priority"] = df["activity"].map({
        "Bought To Open": 0,
        "Sold To Open": 0,
        "Sold To Close": 1,
        "Bought To Close": 1
    }).fillna(99)

    # 9) Sort by (date, activity_priority)
    df.sort_values(["date", "activity_priority"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _extract_option_details(row):
    """
    Attempt to parse the 'description' to fill:
      - OPT_SYMBOL
      - EXPIRATION_DATE
      - STRIKE
      - OPTION_TYPE (Call/Put)
    e.g. "SPY Apr 14 2025 400 Call"
    """
    pattern = (
        r'^(?P<sym>\S+)\s+(?P<month>\w+)\s+(?P<day>\d{1,2})\s+'
        r'(?P<year>\d{4})\s+(?P<strike>\d+(\.\d+)?)\s+(?P<type>Call|Put)$'
    )
    match = re.match(pattern, row["description"])
    if match:
        row["OPT_SYMBOL"] = match["sym"]
        row["EXPIRATION_DATE"] = pd.to_datetime(
            f"{match['month']} {match['day']} {match['year']}"
        )
        row["STRIKE"] = float(match["strike"])
        row["OPTION_TYPE"] = match["type"]
    else:
        # If parse fails, these remain NaN => we'll drop them later
        row["OPT_SYMBOL"] = pd.NA
        row["EXPIRATION_DATE"] = pd.NaT
        row["STRIKE"] = pd.NA
        row["OPTION_TYPE"] = pd.NA
    return row


def merge_simultaneous_fills(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine partial fills with the same (date, activity, account, OPT_SYMBOL, STRIKE,
    EXPIRATION_DATE, OPTION_TYPE) into a single row, matching the old logic.
    """
    merged_rows = []
    i = 0
    while i < len(df):
        j = i + 1
        while j < len(df):
            same_time = df.at[i, "date"] == df.at[j, "date"]
            same_activity = df.at[i, "activity"] == df.at[j, "activity"]
            same_account = df.at[i, "account"] == df.at[j, "account"]
            same_symbol = df.at[i, "OPT_SYMBOL"] == df.at[j, "OPT_SYMBOL"]
            same_strike = df.at[i, "STRIKE"] == df.at[j, "STRIKE"]
            same_expiry = df.at[i, "EXPIRATION_DATE"] == df.at[j, "EXPIRATION_DATE"]
            same_type = df.at[i, "OPTION_TYPE"] == df.at[j, "OPTION_TYPE"]

            if (
                same_time and same_activity and same_account and same_symbol
                and same_strike and same_expiry and same_type
            ):
                # Merge row i into row j
                total_qty = df.at[i, "qty"] + df.at[j, "qty"]
                if total_qty != 0:
                    w_avg_price = (
                        df.at[i, "price"] * df.at[i, "qty"]
                        + df.at[j, "price"] * df.at[j, "qty"]
                    ) / total_qty
                else:
                    w_avg_price = 0.0

                df.at[j, "qty"] = total_qty
                df.at[j, "commission"] += df.at[i, "commission"]
                df.at[j, "fees"] += df.at[i, "fees"]
                df.at[j, "amount"] += df.at[i, "amount"]
                df.at[j, "price"] = round(w_avg_price, 4)

                i = j  # effectively skip row i by jumping forward
                break
            else:
                j += 1
        else:
            # No merge => keep row i as is
            merged_rows.append(df.iloc[i])
        i += 1

    return pd.DataFrame(merged_rows).reset_index(drop=True)


def match_option_trades(df: pd.DataFrame):
    """
    FIFO match within each (account, OPT_SYMBOL, EXPIRATION_DATE, STRIKE, OPTION_TYPE).
    Returns (trades_df, unclosed_df).
    """
    open_q = {}
    trades = []

    for _, row in df.iterrows():
        key = (
            row["account"],
            row["OPT_SYMBOL"],
            row["EXPIRATION_DATE"],
            row["STRIKE"],
            row["OPTION_TYPE"],
        )
        activity = row["activity"]
        qty_left = row["qty"]

        if "Open" in activity:
            open_q.setdefault(key, deque()).append(row.copy())
            open_q[key][-1]["_orig_qty"] = qty_left
            continue

        # If it's a Close leg but no open-lots, skip or track as "unopened"
        if key not in open_q or not open_q[key]:
            continue

        # partial or full close
        close_amt_unit = row["amount"] / qty_left if qty_left else 0
        close_comm_unit = row["commission"] / qty_left if qty_left else 0
        close_fees_unit = row["fees"] / qty_left if qty_left else 0

        while qty_left and open_q[key]:
            o = open_q[key][0]
            matched_qty = min(qty_left, o["qty"])

            open_amt_unit = o["amount"] / o["_orig_qty"] if o["_orig_qty"] else 0
            open_comm_unit = o["commission"] / o["_orig_qty"] if o["_orig_qty"] else 0
            open_fees_unit = o["fees"] / o["_orig_qty"] if o["_orig_qty"] else 0

            open_amt = round(open_amt_unit * matched_qty, 2)
            close_amt = round(close_amt_unit * matched_qty, 2)
            net_pl = round(open_amt + close_amt, 2)

            comm_tot = round((open_comm_unit + close_comm_unit) * matched_qty, 2)
            fees_tot = round((open_fees_unit + close_fees_unit) * matched_qty, 2)
            ret = round(net_pl / abs(open_amt), 4) if open_amt else 0.0

            trades.append(dict(
                SYMBOL=row["symbol"],
                ACCOUNT=o["account"],
                POSITION="Long" if o["activity"] == "Bought To Open" else "Short",
                OPTION_TYPE=o["OPTION_TYPE"],
                STRIKE_PRICE=o["STRIKE"],
                EXPIRATION=o["EXPIRATION_DATE"],
                OPEN_DATE=o["date"],
                CLOSE_DATE=row["date"],
                DTE_AT_OPEN=(
                    (o["EXPIRATION_DATE"] - o["date"]).days
                    if pd.notnull(o["EXPIRATION_DATE"]) else None
                ),
                QTY=matched_qty,
                OPEN_PRICE=round(o["price"], 4),
                CLOSE_PRICE=round(row["price"], 4),
                OPEN_AMOUNT=open_amt,
                CLOSE_AMOUNT=close_amt,
                COMMISSION_TOTAL=comm_tot,
                FEES_TOTAL=fees_tot,
                NET_PROFIT=net_pl,
                RETURN=ret,
            ))

            o["qty"] -= matched_qty
            qty_left -= matched_qty
            if o["qty"] == 0:
                open_q[key].popleft()

    # leftover open-lots
    unclosed_rows = []
    for dq in open_q.values():
        unclosed_rows.extend(dq)

    unclosed_df = pd.DataFrame(unclosed_rows)
    trades_df = pd.DataFrame(trades)

    # rename columns in trades_df
    if not trades_df.empty:
        trades_df.rename(columns={
            "OPTION_TYPE": "OPTION TYPE",
            "STRIKE_PRICE": "STRIKE PRICE",
            "OPEN_DATE": "OPEN DATE",
            "CLOSE_DATE": "CLOSE DATE",
            "DTE_AT_OPEN": "DTE AT OPEN",
            "NET_PROFIT": "NET PROFIT",
        }, inplace=True)

    return trades_df, unclosed_df


def process_options(parsed_option_rows):
    """
    1) Convert to DataFrame + clean
    2) merge_simultaneous_fills
    3) match_option_trades
    """
    df_opt = dicts_to_option_df(parsed_option_rows)
    df_opt_merged = merge_simultaneous_fills(df_opt)
    trades_df, unclosed_df = match_option_trades(df_opt_merged)
    return trades_df, unclosed_df