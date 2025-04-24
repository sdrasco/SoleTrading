# process_activity/data_helpers.py

import pandas as pd

def parse_currency(x: str) -> float:
    """
    Convert strings like '$1,234.50' into a float (e.g., 1234.50).
    Also handles values with commas but no '$', e.g. '1,234.50'.
    Returns 0.0 if the input can't be parsed as a float.
    """
    s = str(x).strip()
    # remove $ and commas
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_mixed_date(s: str) -> pd.Timestamp:
    """
    Attempt to parse a date string using known formats first,
    then fall back to automatic Pandas parsing.
    
    Known formats tried in order:
      1) '%m/%d/%Y'      (e.g. '04/23/2023')
      2) '%Y-%m-%d, %H:%M:%S'  (e.g. '2023-04-23, 14:30:00')
    
    If those fail, tries pd.to_datetime(s) as a fallback.
    
    Returns a Pandas Timestamp. If parsing fails entirely,
    Pandas typically raises an error or returns NaT.
    """
    formats_to_try = [
        "%m/%d/%Y",
        "%Y-%m-%d, %H:%M:%S",
    ]
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            pass
    # fallback
    return pd.to_datetime(s, errors="coerce")