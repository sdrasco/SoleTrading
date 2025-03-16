import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Read trades data
df = pd.read_csv('trades.csv', parse_dates=['OPEN_DATE', 'CLOSE_DATE'])

# Extract EXPIRATION_DATE, STRIKE, and OPTION_TYPE from DESCRIPTION
exp_pattern = r'(\w+) (\w+) (\d{1,2}) (\d{4}) ([\d.]+) (Call|Put)'
df[['SYMBOL', 'MONTH', 'DAY', 'YEAR', 'STRIKE', 'OPTION_TYPE']] = df['DESCRIPTION'].str.extract(exp_pattern)
df['EXPIRATION_DATE'] = pd.to_datetime(df['MONTH'] + ' ' + df['DAY'] + ' ' + df['YEAR'])
df['STRIKE'] = df['STRIKE'].astype(float)
df.drop(['MONTH', 'DAY', 'YEAR'], axis=1, inplace=True)

# Rename columns
df.rename(columns={'COMMISSION_TOTAL': 'COMMISSION', 'FEES_TOTAL': 'FEES'}, inplace=True)

# Calculate days held
df['Days Held'] = (df['CLOSE_DATE'] - df['OPEN_DATE']).dt.days

# Calculate Days to Expiration (DTE)
df['DTE'] = (df['EXPIRATION_DATE'] - df['CLOSE_DATE']).dt.days

# Calculate Net Profit (%)
df['Net Profit (%)'] = round((df['NET_PROFIT'] / abs(df['OPEN_AMOUNT'])) * 100, 2)

# Simplified DESCRIPTION (e.g., "Put at 395" or "Call at 250.5")
df['DESCRIPTION'] = df.apply(
    lambda row: f"{row['OPTION_TYPE']} at {int(row['STRIKE']) if row['STRIKE'].is_integer() else row['STRIKE']}", 
    axis=1
)

# Weekly summary
weekly_summary = df.set_index('CLOSE_DATE').resample('W').agg({
    'NET_PROFIT': 'sum',
    'QTY': 'count',
    'Days Held': 'mean',
    'Net Profit (%)': 'mean',
})

weekly_summary.rename(columns={
    'NET_PROFIT': 'Net Profit ($)',
    'QTY': 'Number of Trades',
    'Days Held': 'Average Days Held',
    'Net Profit (%)': 'Average Net Profit (%)'
}, inplace=True)

weekly_summary['Winning Trades'] = df[df['NET_PROFIT'] > 0].set_index('CLOSE_DATE').resample('W')['NET_PROFIT'].count()
weekly_summary['Win Rate (%)'] = (weekly_summary['Winning Trades'] / weekly_summary['Number of Trades']) * 100

# Sharpe ratio calculation (annualized weekly)
weekly_returns = df.set_index('CLOSE_DATE').resample('W')['NET_PROFIT'].sum()
weekly_summary['Sharpe Ratio'] = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(52)

# Open positions
open_positions = pd.read_csv('unclosed.csv', parse_dates=['DATE', 'EXPIRATION_DATE'])
open_positions = open_positions[['SYMBOL', 'OPTION_TYPE', 'EXPIRATION_DATE', 'STRIKE', 'DATE', 'QTY', 'PRICE', 'AMOUNT']]
open_positions.rename(columns={'DATE': 'Open Date', 'AMOUNT': 'Cost Basis', 'EXPIRATION_DATE': 'Expiration Date',
                               'SYMBOL': 'Symbol', 'OPTION_TYPE': 'Option Type', 'STRIKE': 'Strike',
                               'QTY': 'Quantity', 'PRICE': 'Price'}, inplace=True)

# Function to encode plot to base64
def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return encoded

# Weekly Net Profit plot
plt.figure(figsize=(6, 3))
weekly_returns.plot(kind='bar', color='#8fbcb9')
plt.title('Weekly Net Profit')
plt.xlabel('Week Ending')
plt.ylabel('Net Profit ($)')
plt.xticks(ticks=range(len(weekly_returns)), labels=[date.strftime('%Y-%m-%d') for date in weekly_returns.index], rotation=45)
net_profit_img = plot_to_base64(plt)

# Trade Duration Distribution plot
plt.figure(figsize=(6, 3))
df['Days Held'].plot.hist(bins=range(0, df['Days Held'].max() + 2), color='#8fbcb9', edgecolor='black')
plt.title('Distribution of Trade Durations')
plt.xlabel('Days Held')
plt.ylabel('Number of Trades')
trade_duration_img = plot_to_base64(plt)

# Generate HTML Report
html_report = f"""
<html>
<head>
<title>Trading Performance Report</title>
<style>
body {{ font-family: Arial; background-color: #f4f7f6; color: #333; padding: 20px; }}
table {{border-collapse: collapse; width: 80%; margin: auto;}}
th {{background-color: #8fbcb9; color: white;}}
td, th {{padding: 5px; text-align: center;}}
tr:nth-child(even) {{background-color: #f9fbfa;}}
h1, h2 {{color: #6b9080; text-align: center;}}
</style>
</head>
<body>
<h1>Trading Performance Report</h1>

<h2>Weekly Performance Summary</h2>
{weekly_summary.to_html(float_format=lambda x: f'{x:,.2f}')}

<h2>Visualizations</h2>
<div style=\"text-align:center;\">
    <img src='data:image/png;base64,{net_profit_img}' style='width:25%;'>
    <img src='data:image/png;base64,{trade_duration_img}' style='width:25%;'>
</div>

<h2>Open Positions</h2>
{open_positions.to_html(index=False)}

<h2>Individual Trades</h2>
{df[['SYMBOL', 'DESCRIPTION', 'OPEN_DATE', 'CLOSE_DATE', 'QTY', 'OPEN_PRICE','CLOSE_PRICE', 'OPEN_AMOUNT', 'CLOSE_AMOUNT', 'COMMISSION',
     'FEES', 'NET_PROFIT', 'Net Profit (%)', 'Days Held', 'DTE']].rename(columns={'SYMBOL':'Symbol','DESCRIPTION':'Description','QTY':'Quantity','NET_PROFIT':'Net Profit ($)'}).to_html(index=False,float_format=lambda x:f'{x:,.2f}')}

</body>
</html>
"""

with open('trading_report.html', 'w') as f:
    f.write(html_report)

print("Trading performance report generated as 'trading_report.html'")