"""
Base parser abstraction for broker activity rows.
"""
from abc import ABC, abstractmethod

class BaseBrokerParser(ABC):
    @abstractmethod
    def parse_row(self, row: list):  # noqa: U100
        """
        Parse one row of raw broker activity and return a dict with keys:
          account, date, activity, qty, symbol, description,
          price, commission, fees, amount, native_ccy, exchange
        or None if the row is not relevant.
        """
        pass