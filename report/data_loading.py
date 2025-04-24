# report/data_loading.py

import pandas as pd
import numpy as np

def load_trades(filename):
    """
    Loads the 'trades.csv' (options or stocks).
    """
    df = pd.read_csv(filename, parse_dates=["OPEN DATE", "CLOSE DATE", "EXPIRATION"])

    # If DAYS_HELD is not in the file, calculate it
    if "DAYS_HELD" not in df.columns:
        df["DAYS_HELD"] = (df["CLOSE DATE"] - df["OPEN DATE"]).dt.days

    # Create day-of-week columns
    df["DAY_OF_WEEK_AT_OPEN"] = df["OPEN DATE"].dt.day_name()
    df["DAY_OF_WEEK_AT_CLOSE"] = df["CLOSE DATE"].dt.day_name()

    # Mark position = "OPT" if not otherwise specified
    df["POSITION"] = df.get("POSITION", "OPT")

    # If there's an OPTION TYPE column, create "TRADE DIRECTION"
    if "OPTION TYPE" in df.columns:
        df["OPTION TYPE"] = df["OPTION TYPE"].str.strip().str.upper()
        df["TRADE DIRECTION"] = df["OPTION TYPE"].map({
            "C": "Call", 
            "P": "Put", 
            "CALL": "Call", 
            "PUT": "Put"
        })
    
    return df


def load_fx_trades(filename):
    """
    Loads the 'fx_trades.csv' for matched spot-FX trades.
    """
    fx = pd.read_csv(filename, parse_dates=["OPEN_DATE", "CLOSE_DATE"])
    if "NET_P_L_USD" not in fx.columns or fx.empty:
        return pd.DataFrame()

    fx.rename(
        columns={
            "OPEN_DATE":   "OPEN DATE",
            "CLOSE_DATE":  "CLOSE DATE",
            "NET_P_L_USD": "NET PROFIT",
            "RETURN_PCT":  "RETURN"
        },
        inplace=True
    )

    fx["OPEN DATE"]  = pd.to_datetime(fx["OPEN DATE"],  errors="coerce")
    fx["CLOSE DATE"] = pd.to_datetime(fx["CLOSE DATE"], errors="coerce")

    # unify "PAIR" into "SYMBOL" if needed
    if "PAIR" in fx.columns and "SYMBOL" not in fx.columns:
        fx["SYMBOL"] = fx["PAIR"]

    if "DAYS_HELD" not in fx.columns:
        fx["DAYS_HELD"] = (fx["CLOSE DATE"] - fx["OPEN DATE"]).dt.days

    # Mark as "FX"
    fx["POSITION"] = "FX"
    return fx


def load_and_combine_all_trades(options_file, fx_file):
    """
    Convenience function that loads option trades + FX trades
    and concatenates them into one DataFrame.
    """
    option_trades_df = load_trades(options_file)
    fx_trades_df = load_fx_trades(fx_file)

    combined_df = pd.concat([option_trades_df, fx_trades_df], ignore_index=True)
    combined_df.sort_values("CLOSE DATE", inplace=True)
    return combined_df