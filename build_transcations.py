import sys
import csv

if len(sys.argv) != 3:
    print("Usage: python build_transactions.py activity.txt transactions.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

columns = ["DATE", "ACTIVITY", "QTY", "SYMBOL", "DESCRIPTION", "PRICE", "COMMISSION", "FEES", "AMOUNT"]

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(columns)

    for line in infile:
        if not line.strip():
            continue

        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue

        date, activity, qty, sym_or_type, description, price, commission, fees, amount = parts[:9]

        if activity == "Cash Movement":
            symbol = sym_or_type if "FULLYPAID LENDING REBATE" in description else ""
        else:
            symbol = sym_or_type.split()[0]

        writer.writerow([date, activity, qty, symbol, description, price, commission, fees, amount])
