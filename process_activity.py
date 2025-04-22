#!/usr/bin/env python3
"""
process_activity.py  –  consolidate Ally & IB activity

Outputs in ./data/cleaned/
    trades.csv        options (closed)
    unopened.csv      options close‑leg without matching open‑leg
    unclosed.csv      options positions still open
    fx_trades.csv     spot‑FX trades (closed)
    fx_unclosed.csv   spot‑FX open lots
"""
import csv, re, sys
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import pandas as pd


# ────────────────────────────────────────────────────────────────
# 1)  Broker‑specific parsers
# ────────────────────────────────────────────────────────────────
class BaseBrokerParser(ABC):
    @abstractmethod
    def parse_row(self, row: list):
        """Return a dict or None if the row is not relevant."""
        ...


class AllyParser(BaseBrokerParser):
    """Ally’s tab‑delimited `activity.txt`."""

    def parse_row(self, row):
        if len(row) < 9:
            return None
        date, activity, qty, sym_or_type, desc, price, comm, fees, amt = row[:9]
        symbol = sym_or_type.split()[0] if activity != "Cash Movement" else ""
        return dict(
            account="Ally",
            date=date,
            activity=activity,
            qty=qty,
            symbol=symbol,
            description=desc,
            price=price,
            commission=comm,
            fees=fees,
            amount=amt,
            asset="OPT",
        )


class IBParser(BaseBrokerParser):
    """
    Parses both option‑rows and spot‑FX rows from
    `Trades,Data,Order,…` sections of an IB statement.
    """

    MONTH = dict(
        JAN="Jan",
        FEB="Feb",
        MAR="Mar",
        APR="Apr",
        MAY="May",
        JUN="Jun",
        JUL="Jul",
        AUG="Aug",
        SEP="Sep",
        OCT="Oct",
        NOV="Nov",
        DEC="Dec",
    )

    # ---------------- FX rows -------------------------------------------------
    def _parse_fx(self, row):
        # columns: …,PAIR,DATE,  QTY,PRICE,,PROCEEDS,COMM,…
        pair, dt, qty_s, px, proceeds, comm = row[5], row[6], row[7], row[8], row[10], row[11]
        try:
            qty = float(qty_s.replace(",", ""))
        except ValueError:
            return None
        base, quote = pair.split(".")
        side = "Bought" if qty > 0 else "Sold"
        return dict(
            account="IB",
            date=dt,
            activity=f"{side} {base}",
            qty=qty_s,
            symbol=pair,           # e.g. "GBP.USD"
            description=f"FX {pair}",
            price=px,
            commission=comm,
            fees="0",
            amount=proceeds,
            asset="FX",
            base_ccy=base,        # e.g. "GBP"
            quote_ccy=quote,      # e.g. "USD"
        )

    # ---------------- Option rows --------------------------------------------
    def _parse_opt(self, row):
        sym, dt, qty_s, tpx, proceeds, comm, code = (
            row[5],
            row[6],
            row[7],
            row[8],
            row[10],
            row[11],
            row[16],
        )

        try:
            qty = int(qty_s)
        except ValueError:
            return None

        side_flag = code.split(";")[0]
        if side_flag == "O":
            activity = "Bought To Open" if qty > 0 else "Sold To Open"
        elif side_flag == "C":
            activity = "Bought To Close" if qty > 0 else "Sold To Close"
        else:
            return None  # other codes not handled

        try:
            tk, raw_exp, raw_strike, pc = sym.split()
        except ValueError:
            return None  # not a 4‑part option symbol

        day, mon3, yr = raw_exp[:2], raw_exp[2:5], raw_exp[5:]
        if mon3.upper() not in self.MONTH:
            return None
        exp = f"{self.MONTH[mon3.upper()]} {int(day):02d} 20{yr}"
        strike = f"{float(raw_strike):.2f}"
        opt_type = "Put" if pc.upper() == "P" else "Call"
        desc = f"{tk} {exp} {strike} {opt_type}"

        return dict(
            account="IB",
            date=dt,
            activity=activity,
            qty=qty_s,
            symbol=tk,
            description=desc,
            price=tpx,
            commission=comm,
            fees="0",
            amount=proceeds,
            asset="OPT",
        )

    # ---------------- master dispatch ----------------------------------------
    def parse_row(self, row):
        if len(row) < 12 or row[:3] != ["Trades", "Data", "Order"]:
            return None
        return (
            self._parse_fx(row)
            if row[3] == "Forex"
            else self._parse_opt(row)
            if row[3] == "Equity and Index Options"
            else None
        )


# ────────────────────────────────────────────────────────────────
# 2)  Generic helpers
# ────────────────────────────────────────────────────────────────
def parse_currency(x: str) -> float:
    """Convert strings like '$1,234.50' → 1234.5"""
    return float(str(x).replace("$", "").replace(",", ""))


def parse_mixed_date(s: str) -> pd.Timestamp:
    for f in ("%m/%d/%Y", "%Y-%m-%d, %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=f)
        except ValueError:
            pass
    return pd.to_datetime(s)  # fall back to auto‑parse


# ────────────────────────────────────────────────────────────────
# 3)  Stage‑1 : raw → option/FX staging CSVs
# ────────────────────────────────────────────────────────────────
def stage_csvs(brokers, out_opt: Path, out_fx: Path):
    opt_cols = [
        "ACCOUNT",
        "DATE",
        "ACTIVITY",
        "QTY",
        "SYMBOL",
        "DESCRIPTION",
        "PRICE",
        "COMMISSION",
        "FEES",
        "AMOUNT",
    ]
    fx_cols = [
        "ACCOUNT",
        "DATE",
        "ACTIVITY",
        "QTY",
        "PAIR",
        "PRICE",
        "COMMISSION",
        "FEES",
        "AMOUNT",
        "BASE",
        "QUOTE",
    ]

    with open(out_opt, "w", newline="") as fo, open(out_fx, "w", newline="") as ff:
        ow, fw = csv.writer(fo), csv.writer(ff)
        ow.writerow(opt_cols)
        fw.writerow(fx_cols)

        for path, parser in brokers:
            print(f"Parsing {path}")
            delim = "\t" if path.endswith(".txt") else ","
            for row in csv.reader(open(path, newline=""), delimiter=delim):
                p = parser.parse_row(row)
                if not p:
                    continue

                if p["asset"] == "OPT":
                    ow.writerow([p[k.lower()] for k in opt_cols])
                else:  # FX
                    fw.writerow(
                        [
                            p["account"],
                            p["date"],
                            p["activity"],
                            p["qty"],
                            p["symbol"],        # PAIR
                            p["price"],
                            p["commission"],
                            p["fees"],
                            p["amount"],
                            p["base_ccy"],
                            p["quote_ccy"],
                        ]
                    )
    print("Stage‑1 complete.")


# ────────────────────────────────────────────────────────────────
# 4)  Option lane helpers
# ────────────────────────────────────────────────────────────────
def _desc(row):
    m = re.match(
        r"^(?P<s>\S+) (?P<d>\w+ \d{2} \d{4}) (?P<k>\d+\.\d+) (?P<t>Call|Put)$",
        row["DESCRIPTION"],
    )
    if m:
        row["OPT_SYMBOL"] = m["s"]
        row["EXPIRATION_DATE"] = pd.to_datetime(m["d"])
        row["STRIKE"] = float(m["k"])
        row["OPTION_TYPE"] = m["t"]
    return row


def load_opt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, thousands=",")
    df = df[df["ACTIVITY"] != "Cash Movement"].copy()
    for c in ["PRICE", "COMMISSION", "FEES", "AMOUNT"]:
        df[c] = df[c].apply(parse_currency)

    df["QTY"] = pd.to_numeric(df["QTY"]).abs()
    df["DATE"] = df["DATE"].apply(parse_mixed_date)
    df.loc[df["ACTIVITY"] == "Expired", "ACTIVITY"] = "Sold To Close"

    df = df.apply(_desc, axis=1)
    df.dropna(subset=["OPT_SYMBOL"], inplace=True)

    df["ACTIVITY PRIORITY"] = df["ACTIVITY"].map(
        {"Bought To Open": 0, "Sold To Open": 0, "Sold To Close": 1, "Bought To Close": 1}
    )
    df.sort_values(["DATE", "ACTIVITY PRIORITY"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def merge_simultaneous(df: pd.DataFrame) -> pd.DataFrame:
    """Combine fills of the same leg on the same timestamp."""
    merged, i = [], 0
    while i < len(df):
        while (
            i + 1 < len(df)
            and df.at[i, "DATE"] == df.at[i + 1, "DATE"]
            and df.at[i, "ACTIVITY"] == df.at[i + 1, "ACTIVITY"]
            and df.at[i, "ACCOUNT"] == df.at[i + 1, "ACCOUNT"]
            and df.at[i, "OPT_SYMBOL"] == df.at[i + 1, "OPT_SYMBOL"]
            and df.at[i, "STRIKE"] == df.at[i + 1, "STRIKE"]
            and df.at[i, "EXPIRATION_DATE"] == df.at[i + 1, "EXPIRATION_DATE"]
            and df.at[i, "OPTION_TYPE"] == df.at[i + 1, "OPTION_TYPE"]
        ):
            w_avg = (
                df.at[i, "PRICE"] * df.at[i, "QTY"]
                + df.at[i + 1, "PRICE"] * df.at[i + 1, "QTY"]
            ) / (df.at[i, "QTY"] + df.at[i + 1, "QTY"])
            for col in ["QTY", "COMMISSION", "FEES", "AMOUNT"]:
                df.at[i + 1, col] += df.at[i, col]
            df.at[i + 1, "PRICE"] = round(w_avg, 2)
            i += 1
        merged.append(df.iloc[i])
        i += 1
    return pd.DataFrame(merged)


def match_option_trades(df: pd.DataFrame):
    """
    FIFO match within each account/symbol/expiration/strike/type.
    Returns (trades_df, unopened_df, unclosed_df).
    """
    open_q, trades, unopened = {}, [], []

    for _, row in df.iterrows():
        key = (
            row["ACCOUNT"],
            row["OPT_SYMBOL"],
            row["EXPIRATION_DATE"],
            row["STRIKE"],
            row["OPTION_TYPE"],
        )

        if "Open" in row["ACTIVITY"]:
            open_q.setdefault(key, deque()).append(row.copy())
            open_q[key][-1]["_orig_qty"] = row["QTY"]
            continue

        if key not in open_q or not open_q[key]:
            unopened.append(row.copy())
            continue

        qty_left = row["QTY"]
        close_pc = row["AMOUNT"] / row["QTY"]
        close_cpc = row["COMMISSION"] / row["QTY"]
        close_fpc = row["FEES"] / row["QTY"]

        while qty_left and open_q[key]:
            o = open_q[key][0]
            matched = min(qty_left, o["QTY"])
            open_pc = o["AMOUNT"] / o["_orig_qty"]
            open_cpc = o["COMMISSION"] / o["_orig_qty"]
            open_fpc = o["FEES"] / o["_orig_qty"]

            open_amt = round(open_pc * matched, 2)
            close_amt = round(close_pc * matched, 2)
            net = round(close_amt + open_amt, 2)
            comm_tot = round((open_cpc + close_cpc) * matched, 2)
            fees_tot = round((open_fpc + close_fpc) * matched, 2)
            ret = round(net / abs(open_amt), 4) if open_amt else 0.0

            trades.append(
                dict(
                    SYMBOL=row["SYMBOL"],
                    ACCOUNT=o["ACCOUNT"],
                    POSITION="Long" if o["ACTIVITY"] == "Bought To Open" else "Short",
                    OPTION_TYPE=o["OPTION_TYPE"],
                    STRIKE_PRICE=o["STRIKE"],
                    EXPIRATION=o["EXPIRATION_DATE"],
                    OPEN_DATE=o["DATE"],
                    DTE_AT_OPEN=(o["EXPIRATION_DATE"] - o["DATE"]).days,
                    CLOSE_DATE=row["DATE"],
                    QTY=matched,
                    OPEN_PRICE=round(o["PRICE"], 2),
                    CLOSE_PRICE=round(row["PRICE"], 2),
                    OPEN_AMOUNT=open_amt,
                    CLOSE_AMOUNT=close_amt,
                    COMMISSION_TOTAL=comm_tot,
                    FEES_TOTAL=fees_tot,
                    NET_PROFIT=net,
                    RETURN=ret,
                )
            )

            o["QTY"] -= matched
            qty_left -= matched
            if o["QTY"] == 0:
                open_q[key].popleft()

    unclosed = [pos for dq in open_q.values() for pos in dq]
    return pd.DataFrame(trades), pd.DataFrame(unopened), pd.DataFrame(unclosed)


# ────────────────────────────────────────────────────────────────
# 5)  FX lane
# ────────────────────────────────────────────────────────────────
def load_fx(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, thousands=",")
    for c in ["PRICE", "COMMISSION", "FEES", "AMOUNT"]:
        df[c] = df[c].apply(parse_currency)
    df["QTY"] = df["QTY"].apply(lambda x: float(str(x).replace(",", "")))
    df["DATE"] = df["DATE"].apply(parse_mixed_date)
    return df.sort_values("DATE").reset_index(drop=True)


def match_fx(df: pd.DataFrame):
    """
    FIFO match per account/pair.  
    Positive QTY = buy BASE  (side=+1)
    Negative QTY = sell BASE (side=-1)

    ### NEW ROUND_TRIP LOGIC ###
    We'll add a 'ROUND_TRIP' column that shows something like:
      "USD -> GBP -> USD" or "EUR -> JPY -> EUR"
    depending on side = +1 or -1.

    Returns (trades_df, unclosed_df).
    """
    opens, trades = {}, []

    for _, r in df.iterrows():
        key = (r["ACCOUNT"], r["PAIR"])
        side = 1 if r["QTY"] > 0 else -1  # 1 = long base, -1 = short base
        qty = abs(r["QTY"])

        opens.setdefault(key, deque())

        # same‑side → store as new open lot
        if not opens[key] or opens[key][-1]["SIDE"] == side:
            row = r.copy()
            row["SIDE"] = side
            row["QTY_BASE"] = qty
            opens[key].append(row)
            continue

        # opposite‑side → close lot(s)
        remaining = qty
        while remaining and opens[key]:
            o = opens[key][0]
            matched = min(remaining, o["QTY_BASE"])

            open_usd = (o["AMOUNT"] / o["QTY_BASE"]) * matched
            close_usd = (r["AMOUNT"] / qty) * matched
            pl = round(close_usd + open_usd, 2)
            comm_tot = round(
                (o["COMMISSION"] / o["QTY_BASE"] + r["COMMISSION"] / qty) * matched, 2
            )
            ret = round(pl / abs(open_usd), 4) if open_usd else 0.0

            # ### NEW ROUND_TRIP LOGIC ###
            # If side=1 (we're closing a long base), that means
            # open was quote->base->quote => e.g. "USD -> GBP -> USD"
            # If side=-1 (closing a short base), we do base->quote->base
            if o["SIDE"] == 1:
                round_trip = f"{o['QUOTE']} -> {o['BASE']} -> {o['QUOTE']}"
            else:
                round_trip = f"{o['BASE']} -> {o['QUOTE']} -> {o['BASE']}"

            trades.append(
                dict(
                    ACCOUNT=r["ACCOUNT"],
                    PAIR=r["PAIR"],
                    BASE_CCY=r["BASE"],
                    QUOTE_CCY=r["QUOTE"],
                    POSITION="Long" if side == -1 else "Short",  # we’re unwinding
                    OPEN_DATE=o["DATE"],
                    CLOSE_DATE=r["DATE"],
                    QTY_BASE=matched,
                    OPEN_PRICE=o["PRICE"],
                    CLOSE_PRICE=r["PRICE"],
                    OPEN_USD=open_usd,
                    CLOSE_USD=close_usd,
                    COMMISSION_TOTAL_USD=comm_tot,
                    NET_P_L_USD=pl,
                    RETURN_PCT=ret,
                    ROUND_TRIP=round_trip,  # new column
                )
            )

            o["QTY_BASE"] -= matched
            remaining -= matched
            if o["QTY_BASE"] == 0:
                opens[key].popleft()

        # residual (over‑close) becomes new opposite‑side open
        if remaining:
            res = r.copy()
            res["SIDE"] = side
            res["QTY_BASE"] = remaining
            res["AMOUNT"] = (r["AMOUNT"] / qty) * remaining
            res["COMMISSION"] = (r["COMMISSION"] / qty) * remaining
            opens[key].append(res)

    # build unclosed lots DataFrame
    uc_rows = []
    for dq in opens.values():
        for d in dq:
            uc_rows.append(
                dict(
                    ACCOUNT=d["ACCOUNT"],
                    PAIR=d["PAIR"],
                    BASE_CCY=d["BASE"],
                    QUOTE_CCY=d["QUOTE"],
                    POSITION="Long" if d["SIDE"] == 1 else "Short",
                    OPEN_DATE=d["DATE"],
                    QTY_BASE=d["QTY_BASE"],
                    OPEN_PRICE=d["PRICE"],
                    OPEN_USD=d["AMOUNT"],
                )
            )

    return pd.DataFrame(trades), pd.DataFrame(uc_rows)


# ────────────────────────────────────────────────────────────────
# 6)  Main driver
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # locate source files
    ally = Path("./data/Ally/activity.txt")
    ib_csvs = list(Path("./data/IB").glob("*.csv"))

    if not ally.is_file():
        sys.exit("Ally activity.txt missing")

    broker_files = [(str(ally), AllyParser())] + [
        (str(p), IBParser()) for p in ib_csvs
    ]

    # output location
    out_dir = Path("./data/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_opt = out_dir / "transactions_opt.csv"
    raw_fx = out_dir / "transactions_fx.csv"

    # Stage‑1
    stage_csvs(broker_files, raw_opt, raw_fx)

    # ───────── OPTIONS PIPELINE ─────────
    opt_df = merge_simultaneous(load_opt(raw_opt))
    trades_df, unopened_df, unclosed_df = match_option_trades(opt_df)

    trades_df.rename(
        columns={
            "OPTION_TYPE": "OPTION TYPE",
            "STRIKE_PRICE": "STRIKE PRICE",
            "OPEN_DATE": "OPEN DATE",
            "CLOSE_DATE": "CLOSE DATE",
            "DTE_AT_OPEN": "DTE AT OPEN",
            "NET_PROFIT": "NET PROFIT",
        },
        inplace=True,
    )
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    unopened_df.to_csv(out_dir / "unopened.csv", index=False)

    if not unclosed_df.empty:
        unclosed_df["POSITION"] = unclosed_df["ACTIVITY"].map(
            {"Bought To Open": "Long", "Sold To Open": "Short"}
        ).fillna("UNK")
        unclosed_df["DTE AT OPEN"] = (
            unclosed_df["EXPIRATION_DATE"] - unclosed_df["DATE"]
        ).dt.days
        unclosed_df.rename(
            columns={
                "DATE": "OPEN DATE",
                "PRICE": "OPEN PRICE",
                "AMOUNT": "OPEN AMOUNT",
                "OPT_SYMBOL": "Option Symbol",
                "EXPIRATION_DATE": "Expiration",
                "STRIKE": "Strike Price",
                "OPTION_TYPE": "Option Type",
                "QTY": "Quantity",
            },
            inplace=True,
        )
        keep_cols = [
            "ACCOUNT",
            "POSITION",
            "Option Symbol",
            "Expiration",
            "Strike Price",
            "Option Type",
            "OPEN DATE",
            "DTE AT OPEN",
            "Quantity",
            "OPEN PRICE",
            "OPEN AMOUNT",
        ]
        unclosed_df[keep_cols].to_csv(out_dir / "unclosed.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "unclosed.csv", index=False)

    # ───────── FX PIPELINE ──────────────
    if raw_fx.stat().st_size > 80:  # not just the header
        fx_df = load_fx(raw_fx)
        fx_trades_df, fx_unclosed_df = match_fx(fx_df)

        # Save final matched trades (which now includes ROUND_TRIP)
        fx_trades_df.to_csv(out_dir / "fx_trades.csv", index=False)

        # Save unclosed
        fx_unclosed_df.to_csv(out_dir / "fx_unclosed.csv", index=False)

        print("FX trades   →", out_dir / "fx_trades.csv")
        print("FX unclosed →", out_dir / "fx_unclosed.csv")

    else:
        # create empty placeholders so downstream code never breaks
        pd.DataFrame().to_csv(out_dir / "fx_trades.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "fx_unclosed.csv", index=False)
        print("No FX activity detected.")

    print("Option trades →", out_dir / "trades.csv")