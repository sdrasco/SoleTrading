# process_activity/fx_pipeline.py

import pandas as pd
from collections import deque
from cleaning.data_helpers import parse_currency, parse_mixed_date

def dicts_to_fx_df(parsed_fx_rows):
    """
    Convert a list of FX trade dictionaries to a DataFrame.
    Each dict must at least have:
      'account', 'date', 'activity', 'qty', 'symbol', 'description',
      'price', 'commission', 'fees', 'amount',
      plus 'base_ccy' and 'quote_ccy' for FX lines.

    Returns a sorted DataFrame by date.
    """
    df = pd.DataFrame(parsed_fx_rows)

    # Convert numeric fields
    for col in ["price", "commission", "fees", "amount"]:
        df[col] = df[col].apply(parse_currency)

    # Convert qty to float
    df["qty"] = df["qty"].apply(lambda x: float(str(x).replace(",", "")))

    # Convert date
    df["date"] = df["date"].apply(parse_mixed_date)

    # Sort chronologically
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def match_fx(df: pd.DataFrame):
    """
    FIFO match per (account, symbol).  Symbol is typically "GBP.USD" or "EUR.USD".

    + qty => buy base currency (side=+1)
    - qty => sell base currency (side=-1)

    Steps:
      1) Maintain a queue of "open" lots for each (account, symbol).
      2) When we encounter a trade that is the same side as the last open lot,
         it just adds to that queue entry (or starts a new entry).
      3) If it's the opposite side, we reduce from the earliest open-lot(s),
         generating a closed trade record for each matched portion.

    Returns:
      trades_df: a DataFrame of all closed round‑trips,
                 with columns for net P/L and return.
      unclosed_df: a DataFrame of any leftover open lots.
    """
    opens = {}  # dict of key -> deque of open-lots
    trades = []

    for _, row in df.iterrows():
        key = (row["account"], row["symbol"])
        side = 1 if row["qty"] > 0 else -1  # +1=buy base, -1=sell base
        qty_abs = abs(row["qty"])

        if key not in opens:
            opens[key] = deque()

        # If there's nothing open yet or the last side is the same, we open a new lot
        # or extend the existing side.
        # But for simplicity, let's just store each fill as a separate "open lot".
        # We'll match them as needed.
        # If it's the opposite side, we do the matching logic.
        # This approach ensures FIFO among multiple older opens.

        # Opposite side? => close out with existing opens
        if opens[key] and opens[key][-1].get("SIDE") != side:
            remaining = qty_abs
            # We'll keep matching until we run out of open lots or the entire trade is matched.
            while remaining and opens[key]:
                o = opens[key][0]
                matched_qty = min(remaining, o["QTY_BASE"])

                # Weighted portion of open-lot's amounts
                open_amt_per_unit = o["amount"] / o["QTY_BASE"] if o["QTY_BASE"] else 0
                open_comm_per_unit = o["commission"] / o["QTY_BASE"] if o["QTY_BASE"] else 0

                # Weighted portion of close-lot's amounts
                close_amt_per_unit = row["amount"] / qty_abs if qty_abs else 0
                close_comm_per_unit = row["commission"] / qty_abs if qty_abs else 0

                open_usd = round(open_amt_per_unit * matched_qty, 2)
                close_usd = round(close_amt_per_unit * matched_qty, 2)
                pl = round(close_usd + open_usd, 2)  # net P/L from that portion

                commission_total = round(
                    (open_comm_per_unit + close_comm_per_unit) * matched_qty, 2
                )
                # rate of return
                ret = round(pl / abs(open_usd), 4) if open_usd else 0.0

                # Round-trip string
                if o["SIDE"] == 1:
                    # if open-lot side=+1 => "QUOTE -> BASE -> QUOTE"
                    round_trip = f"{o['quote_ccy']} -> {o['base_ccy']} -> {o['quote_ccy']}"
                else:
                    # if open-lot side=-1 => "BASE -> QUOTE -> BASE"
                    round_trip = f"{o['base_ccy']} -> {o['quote_ccy']} -> {o['base_ccy']}"

                trades.append({
                    "ACCOUNT": row["account"],
                    "PAIR": row["symbol"],
                    "BASE_CCY": row["base_ccy"],
                    "QUOTE_CCY": row["quote_ccy"],
                    "POSITION": ("Long" if side == -1 else "Short"),  # we are unwinding the opposite side
                    "OPEN_DATE": o["date"],
                    "CLOSE_DATE": row["date"],
                    "QTY_BASE": matched_qty,
                    "OPEN_PRICE": o["price"],
                    "CLOSE_PRICE": row["price"],
                    "OPEN_USD": open_usd,
                    "CLOSE_USD": close_usd,
                    "COMMISSION_TOTAL_USD": commission_total,
                    "NET_P_L_USD": pl,
                    "RETURN_PCT": ret,
                    "ROUND_TRIP": round_trip
                })

                # Decrement from the open-lot
                o["QTY_BASE"] -= matched_qty
                # Decrement from the close-lot as well
                remaining -= matched_qty

                if o["QTY_BASE"] == 0:
                    opens[key].popleft()

            # Any leftover from 'remaining' becomes a new open-lot of the opposite side
            if remaining > 0:
                row_copy = row.copy()
                row_copy["SIDE"] = side
                row_copy["QTY_BASE"] = remaining
                # Pro-rate the amounts
                row_copy["amount"] = close_amt_per_unit * remaining
                row_copy["commission"] = close_comm_per_unit * remaining
                opens[key].append(row_copy)

        else:
            # same side => just store this fill
            row_copy = row.copy()
            row_copy["SIDE"] = side
            row_copy["QTY_BASE"] = qty_abs
            opens[key].append(row_copy)

    # Build unclosed-lots DataFrame
    unclosed_rows = []
    for dq in opens.values():
        for d in dq:
            unclosed_rows.append({
                "ACCOUNT": d["account"],
                "PAIR": d["symbol"],
                "BASE_CCY": d["base_ccy"],
                "QUOTE_CCY": d["quote_ccy"],
                "POSITION": ("Long" if d["SIDE"] == 1 else "Short"),
                "OPEN_DATE": d["date"],
                "QTY_BASE": d["QTY_BASE"],
                "OPEN_PRICE": d["price"],
                "OPEN_USD": d["amount"],
            })

    trades_df = pd.DataFrame(trades)
    unclosed_df = pd.DataFrame(unclosed_rows)

    # Sort trades by close date if you like
    trades_df.sort_values("CLOSE_DATE", inplace=True, ignore_index=True)
    unclosed_df.sort_values("OPEN_DATE", inplace=True, ignore_index=True)

    return trades_df, unclosed_df


def process_fx(parsed_fx_rows):
    """
    High-level pipeline function for FX trades:
      1) Convert parsed dictionaries (from IBParser, etc.) to a DataFrame.
      2) Match open-lots vs. close-lots in FIFO order.
      3) Return (fx_trades_df, fx_unclosed_df).
    """
    df_fx = dicts_to_fx_df(parsed_fx_rows)
    fx_trades_df, fx_unclosed_df = match_fx(df_fx)
    return fx_trades_df, fx_unclosed_df