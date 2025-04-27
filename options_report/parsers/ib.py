"""
Interactive Brokers (IB) parser with FX lookup and exchange tagging.
"""
import csv
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from .base import BaseBrokerParser

def parse_ib_timestamp(ts: str) -> datetime:
    """Convert IB timestamp string → datetime object (second precision)."""
    return datetime.strptime(ts.strip(), "%Y-%m-%d, %H:%M:%S")

def fx_rate_from_autofx_row(row: list) -> Optional[Tuple[str, datetime, float]]:
    """Return (ccy, timestamp, usd_per_ccy) if *row* is an AutoFX line."""
    if len(row) < 17:
        return None
    if row[0:3] != ["Trades", "Data", "Order"] or "Forex" not in row[3] or row[16] != "AFx":
        return None

    pair      = row[5].strip()
    price_str = row[8].replace(",", "")
    ts        = parse_ib_timestamp(row[6])

    try:
        px = float(price_str)
    except ValueError:
        return None

    if "." not in pair:
        return None
    base, counter = pair.split('.')
    if base == "USD":
        ccy, usd_per_ccy = counter, 1.0 / px
    elif counter == "USD":
        ccy, usd_per_ccy = base, px
    else:
        return None
    return ccy.upper(), ts, usd_per_ccy

def scan_ib_file(csv_path: Path) -> Tuple[Dict[Tuple[str, datetime], float], Dict[str, str]]:
    """Read an IB CSV, extract FX rates and instrument→exchange mapping."""
    fx: Dict[Tuple[str, datetime], float] = {}
    exch: Dict[str, str] = {}
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            out = fx_rate_from_autofx_row(row)
            if out:
                fx[(out[0], out[1])] = out[2]
            if row and row[0] == "Financial Instrument Information" and len(row) > 8:
                symbol   = row[4].strip()
                exchange = row[7].strip().upper()
                if symbol:
                    exch[symbol] = exchange
    return fx, exch

class IBParser(BaseBrokerParser):  # noqa: C901
    """
    Parses IB option-trade rows, converts non-USD fills to USD,
    and labels each fill with its original currency + exchange.
    """
    REQUIRED = 17

    def __init__(
        self,
        fx_lookup: Dict[Tuple[str, datetime], float],
        exch_lookup: Dict[str, str]
    ):
        self.fx_lookup   = fx_lookup
        self.exch_lookup = exch_lookup

    def parse_row(self, row: list):
        # filter for rows we care about
        if len(row) < self.REQUIRED or row[0:3] != ['Trades', 'Data', 'Order']:
            return None
        if 'Options' not in row[3]:
            return None  # skip equities/forex

        # raw fields
        ccy_orig    = row[4].strip().upper()
        symbol_full = row[5].strip()
        ts          = parse_ib_timestamp(row[6])
        qty         = int(row[7].replace(',', ''))
        t_price     = float(row[8] or 0)
        proceeds    = float(row[10] or 0)
        commission  = float(row[11] or 0)
        code        = row[16].split(';')[0]

        # activity mapping
        if code == 'O':
            activity = 'Bought To Open' if qty > 0 else 'Sold To Open'
        elif code == 'C':
            activity = 'Bought To Close' if qty > 0 else 'Sold To Close'
        else:
            return None

        # FX conversion
        if ccy_orig != 'USD':
            rate = self._rate(ccy_orig, ts)
            if rate:
                t_price    *= rate
                proceeds   *= rate
                commission *= rate

        # description & ticker
        try:
            ticker, raw_exp, raw_strike, pc = symbol_full.split()
            month_map = dict(
                JAN='Jan', FEB='Feb', MAR='Mar', APR='Apr',
                MAY='May', JUN='Jun', JUL='Jul', AUG='Aug',
                SEP='Sep', OCT='Oct', NOV='Nov', DEC='Dec'
            )
            day  = int(raw_exp[:2])
            mon  = month_map[raw_exp[2:5].upper()]
            yr   = f"20{raw_exp[5:]}"
            expiry = f"{mon} {day:02d} {yr}"
            strike = f"{float(raw_strike):.2f}"
            opt_type = 'Put' if pc.upper() == 'P' else 'Call'
            description = f"{ticker} {expiry} {strike} {opt_type}"
        except Exception:
            ticker, description = symbol_full.split()[0], symbol_full

        # exchange lookup
        exchange = self.exch_lookup.get(symbol_full, 'UNKNOWN')

        return {
            'account'     : 'IB',
            'date'        : ts.strftime('%Y-%m-%d, %H:%M:%S'),
            'activity'    : activity,
            'qty'         : str(qty),
            'symbol'      : ticker,
            'description' : description,
            'price'       : f'{t_price:.4f}',
            'commission'  : f'{commission:.2f}',
            'fees'        : '0',
            'amount'      : f'{proceeds:.2f}',
            'native_ccy'  : ccy_orig,
            'exchange'    : exchange,
        }

    def _rate(self, ccy: str, ts: datetime) -> Optional[float]:
        """
        Return first FX rate for *ccy* at or after *ts* (within 48h).
        """
        candidate, best_delta = None, timedelta(days=2)
        for (cur, fx_ts), rate in self.fx_lookup.items():
            if cur != ccy or fx_ts < ts:
                continue
            delta = fx_ts - ts
            if delta < best_delta:
                best_delta, candidate = delta, rate
                if delta.total_seconds() == 0:
                    break
        return candidate