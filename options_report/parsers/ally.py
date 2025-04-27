"""
Ally brokerage parser: parses tab-delimited Ally activity rows.
"""
from .base import BaseBrokerParser

class AllyParser(BaseBrokerParser):
    """
    Parses rows from a tab-delimited Ally activity.txt file.
    Each row is like:
      [
        '03/28/2025', 'Sold To Close', '-1', 'QQQ Put',
        'QQQ Apr 04 2025 475.00 Put', '$10.04', '$0.50', '$0.07', '$1,003.43'
      ]
    Requires at least 9 columns.
    """
    def parse_row(self, row: list):  # noqa: C901
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
            'account': 'Ally',
            'date': date,
            'activity': activity,
            'qty': qty,
            'symbol': symbol,
            'description': description,
            'price': price,
            'commission': commission,
            'fees': fees,
            'amount': amount,
            'native_ccy': 'USD',
            'exchange': 'ALLY'
        }