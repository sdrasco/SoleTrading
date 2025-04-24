# cleaning/broker_parsers.py

import csv
from abc import ABC, abstractmethod

class BaseBrokerParser(ABC):
    """
    All broker parsers must implement `parse_row(row: list) -> dict|None`.
    The returned dict must at least have:
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
          'amount': ...,
          'asset': 'OPT' or 'FX'
        }
    """
    @abstractmethod
    def parse_row(self, row: list):
        """Return a dict or None if the row is not relevant."""
        ...


class AllyParser(BaseBrokerParser):
    """
    Parses Ally’s tab‑delimited `activity.txt`.
    Lines typically have columns:
        0: date
        1: activity
        2: qty
        3: sym_or_type
        4: description
        5: price
        6: commission
        7: fees
        8: amount
    """

    def parse_row(self, row):
        # Ensure the row has at least 9 fields
        if len(row) < 9:
            return None

        date, activity, qty, sym_or_type, desc, price, comm, fees, amt = row[:9]

        # If it's a "Cash Movement" line, we consider symbol empty
        symbol = sym_or_type.split()[0] if activity != "Cash Movement" else ""

        # Build the dictionary. For Ally, we treat everything as "OPT" for simplicity
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
    Parses rows from an Interactive Brokers CSV statement.
    It handles:
      - Option rows (asset="OPT")
      - Spot‑FX rows (asset="FX")
    
    We look for lines where columns [0,1,2] == ["Trades", "Data", "Order"].
    Then we check row[3]:
        - if it .startswith("Forex"): parse as FX
        - if it .startswith("Equity and Index Options"): parse as Option
    Otherwise, ignore.
    """

    # Mapping from short month code to a 3-letter month
    MONTH = dict(
        JAN="Jan", FEB="Feb", MAR="Mar", APR="Apr", MAY="May", JUN="Jun",
        JUL="Jul", AUG="Aug", SEP="Sep", OCT="Oct", NOV="Nov", DEC="Dec",
    )

    def parse_row(self, row: list):
        # Must have at least 12 columns and match the IB prefix
        if len(row) < 12:
            return None
        if row[0:3] != ["Trades", "Data", "Order"]:
            return None

        category = row[3]
        if category.startswith("Forex"):  # e.g. "Forex - Held with Interactive Brokers..."
            return self._parse_fx(row)
        elif category.startswith("Equity and Index Options"):
            return self._parse_opt(row)
        else:
            return None

    def _parse_fx(self, row):
        """
        Example line might look like:
            Trades,Data,Order,Forex - Held with Interactive Brokers...,HKD,GBP.HKD,2025-04-22, ...
        
        columns: 
          0       1    2      3                                    4    5        6             7       8    9    10        11
          "Trades","Data","Order","Forex - Held ...","HKD","GBP.HKD","2025-04-22","3,024.29","10.37515","", "-31377.4623","-2"
        """

        # Adjust indices as needed:
        pair = row[5]   # e.g. "GBP.HKD"
        dt = row[6]     # e.g. "2025-04-22"
        qty_s = row[7]  # e.g. "3,024.29"
        px = row[8]     # e.g. "10.37515"
        proceeds = row[10]  # e.g. "-31377.4623935"
        comm = row[11]      # e.g. "-2"

        # Convert quantity string to float
        try:
            qty_float = float(qty_s.replace(",", ""))
        except ValueError:
            return None

        # e.g. "GBP.HKD" -> base="GBP", quote="HKD"
        if "." not in pair:
            return None

        base, quote = pair.split(".")
        side = "Bought" if qty_float > 0 else "Sold"

        return dict(
            account="IB",
            date=dt,
            activity=f"{side} {base}",
            qty=qty_s,
            symbol=pair,             # e.g. "GBP.HKD"
            description=f"FX {pair}",
            price=px,
            commission=comm,
            fees="0",
            amount=proceeds,
            asset="FX",
            base_ccy=base,
            quote_ccy=quote,
        )

    def _parse_opt(self, row):
        """
        Example line might look like:
            Trades,Data,Order,Equity and Index Options,...,SPY 14APR23 400 C,2023-04-01,10,1.20,...,1200.00,-5.00,...,O;...
        columns: 
          0      1     2     3                           4   5                    6           7   8     9    10         11     ... 16
                    ...
        """
        sym = row[5]        # e.g. "SPY 14APR23 400 C"
        dt = row[6]         # e.g. "2023-04-01"
        qty_s = row[7]      # e.g. "10"
        tpx = row[8]        # e.g. "1.20"
        proceeds = row[10]  # e.g. "1200.00"
        comm = row[11]      # e.g. "-5.00"

        # We might need to ensure row[16] exists. Let's do a quick check:
        code = row[16] if len(row) > 16 else ""

        # Convert quantity to integer
        try:
            qty_int = int(qty_s.replace(",", ""))
        except ValueError:
            return None

        # Option side_flag: "O" (open) or "C" (close), indicated in row[16], e.g. "O;..."
        side_flag = code.split(";")[0] if code else ""
        if side_flag == "O":
            activity = "Bought To Open" if qty_int > 0 else "Sold To Open"
        elif side_flag == "C":
            activity = "Bought To Close" if qty_int > 0 else "Sold To Close"
        else:
            # Not recognized or not an option line we can parse
            return None

        # e.g. "SPY 14APR23 400 C" -> underlying="SPY", raw_exp="14APR23", raw_strike="400", pc="C"
        try:
            tk, raw_exp, raw_strike, pc = sym.split()
        except ValueError:
            return None

        # raw_exp might be "14APR23"
        day = raw_exp[:2]      # "14"
        mon3 = raw_exp[2:5]    # "APR"
        yr = raw_exp[5:]       # "23"

        # Validate month code
        if mon3.upper() not in self.MONTH:
            return None

        # Convert to e.g. "Apr 14 2023"
        exp = f"{self.MONTH[mon3.upper()]} {int(day):02d} 20{yr}"
        opt_type = "Call" if pc.upper() == "C" else "Put"

        desc = f"{tk} {exp} {raw_strike} {opt_type}"

        return dict(
            account="IB",
            date=dt,
            activity=activity,
            qty=qty_s,
            symbol=tk,             # e.g. "SPY"
            description=desc,      # e.g. "SPY Apr 14 2023 400 Call"
            price=tpx,
            commission=comm,
            fees="0",
            amount=proceeds,
            asset="OPT",
        )