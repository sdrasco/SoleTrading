import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime

# Load CSV
df = pd.read_csv('transactions.csv', thousands=',')

# Sum "Cash Movement" lines before removal
cash_movements = df[df['ACTIVITY'] == 'Cash Movement'].copy()
cash_movements['AMOUNT'] = cash_movements['AMOUNT'].replace(r'[\$,]', '', regex=True).astype(float)
cash_withdrawn = -cash_movements['AMOUNT'].sum()

# Remove "Cash Movement" lines
df = df[df['ACTIVITY'] != 'Cash Movement'].copy()

# Convert dates and amounts
df['DATE'] = pd.to_datetime(df['DATE'])
df['AMOUNT'] = df['AMOUNT'].replace(r'[\$,]', '', regex=True).astype(float)

# Date range
start_date = df['DATE'].min().date()
end_date = df['DATE'].max().date()
total_days = (end_date - start_date).days + 1

# US market trading days
nyse = mcal.get_calendar('NYSE')
nyse_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = nyse_schedule.shape[0]

# Set DATE as index
df.set_index('DATE', inplace=True)

# Weekly summary
weekly_summary = df.resample('W').agg({
    'AMOUNT': ['sum', 'size', lambda x: x[x > 0].sum(), lambda x: x[x < 0].sum()]
}).reset_index()
weekly_summary.columns = ['Week Ending', 'Weekly Revenue', 'Trades', 'Cash Inflow', 'Cash Outflow']

# Weekly trading days
def count_weekly_trading_days(week_end):
    week_start = week_end - pd.Timedelta(days=6)
    nyse = mcal.get_calendar('NYSE')
    nyse_schedule = nyse.schedule(start_date=week_start, end_date=week_end)
    return nyse_schedule.shape[0]

weekly_summary['Trading Days'] = weekly_summary['Week Ending'].apply(count_weekly_trading_days)

# Symbol summary
symbol_summary = df.groupby('SYMBOL').agg(
    Trades=('SYMBOL', 'size'),
    Total_Revenue=('AMOUNT', 'sum'),
    Average_Revenue=('AMOUNT', 'mean'),
    Min_Revenue=('AMOUNT', 'min'),
    Max_Revenue=('AMOUNT', 'max')
).reset_index()

symbol_summary.columns = ['Symbol', 'Trades', 'Total Revenue', 'Average Revenue', 'Min Revenue', 'Max Revenue']

# Dollar formatting
def dollar_format(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"

weekly_summary[['Weekly Revenue', 'Cash Inflow', 'Cash Outflow']] = weekly_summary[
    ['Weekly Revenue', 'Cash Inflow', 'Cash Outflow']
].apply(lambda col: col.map(dollar_format))

symbol_summary[['Total Revenue', 'Average Revenue', 'Min Revenue', 'Max Revenue']] = symbol_summary[[
    'Total Revenue', 'Average Revenue', 'Min Revenue', 'Max Revenue'
]].apply(lambda col: col.map(dollar_format))

# Open Positions Data
open_positions = pd.DataFrame({
    'Symbol': ['BABA', 'JD', 'TSLA', 'QQQ'],
    'Option': ['Call', 'Call', 'Put', 'Put'],
    'Strike': [155, 46, 200, 475],
    'Expiration': ['June 20, 2025', 'March 21, 2025', 'April 11, 2025', 'April 04, 2025'],
    'Contracts': [1, 3, 1, 1],
    'Premium': [11.76, 2.45, 13.01, 12.57],
})

open_positions['Total Cost'] = (open_positions['Contracts'] * open_positions['Premium'] * 100).apply(dollar_format)
open_positions['Premium'] = open_positions['Premium'].apply(dollar_format)


# HTML report
html_report = f'''
<html>
<head>
<title>Transaction Summary Report</title>
<style>
  body {{ font-family: 'Arial', sans-serif; background-color: #f4f7f6; color: #333; padding: 20px; }}
  table {{
    border-collapse: collapse;
    width: 55%;
    margin-bottom: 30px;
    background-color: #ffffff;
    margin-left: auto;
    margin-right: auto;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  }}
  th, td {{ border: 1px solid #ddd; padding: 10px; }}
  th {{ background-color: #8fbcb9; color: white; }}
  tr:nth-child(even) {{ background-color: #f9fbfa; }}
  h1, h2 {{ color: #6b9080; text-align: center; }}
  .summary {{ margin-bottom: 40px; text-align: center; }}
</style>
</head>
<body>
<h1>Transaction Summary Report</h1>

<div class="summary">
<table>
<tr><th>Company</th><td>Sole Trader - Steve Drasco</td></tr>
<tr><th>Reporting Period</th><td>{start_date} to {end_date}</td></tr>
<tr><th>Total Days</th><td>{total_days}</td></tr>
<tr><th>Trading Days</th><td>{trading_days}</td></tr>
<tr><th>Total Revenue</th><td>{dollar_format(df['AMOUNT'].sum())}</td></tr>
<tr><th>Revenue per Trading Day</th><td>{dollar_format(df['AMOUNT'].sum()/trading_days)}</td></tr>
<tr><th>Estimated Annual Revenue</th><td>{dollar_format(df['AMOUNT'].sum()/trading_days*252)}</td></tr>
</table>
</div>

<h2>Weekly Revenue Summary</h2>
{weekly_summary.to_html(index=False)}

<h2>Transaction Summary by Symbol</h2>
{symbol_summary.to_html(index=False)}

<h2>Open Positions</h2>
{open_positions.to_html(index=False)}

</body>
</html>
'''

with open('transaction_summary_report.html', 'w') as f:
    f.write(html_report)

print("HTML report generated as 'transaction_summary_report.html'")
