# Options Trading Report Generator

This repository contains code that converts options trading activity from two brokerage accounts, [Ally Invest](https://www.ally.com/) and [Interactive Brokers](https://www.interactivebrokers.co.uk/) (IB), into a combined set of CSV files and an optional HTML report.

The code is released under the [MIT License](LICENSE).

---

## Pipeline Overview

This pipeline processes option activity data from **both** Ally and IB, merges them, and generates CSV files with matched trades, open positions, and more. You can then generate an HTML report if desired.

### 1. Organize Your Input Files

**Ally:**  
Place your Ally activity data in `./data/Ally/activity.txt`.  
- From Ally's website, copy the desired rows of activity (skipping any header), and paste them (tabs intact) into `activity.txt`.

**IB:**  
Copy your Interactive Brokers trade files (log in to Portal and select the Performance & Reports > Statements menu, pick activity, then near CSV click download) into `./data/IB/`.  
- Multiple CSV files are allowed; All will be processed.

### 2. Run the Processing Script

```bash
python process_activity.py
```

This script will read both Ally and IB data, combine them, and produce several cleaned CSV files in `./data/cleaned/`:

- **transactions.csv** – A consolidated log of all lines parsed, from both Ally and IB.
- **trades.csv** – All completed trades (open + close matched).
- **unclosed.csv** – Trades that are still open (missing a close event).
- **unopened.csv** – Close events where the open wasn’t found in the given data.

### 3. (Optional) Generate an HTML Report

If you want a simple HTML summary:

```bash
python write_report.py
```

This will produce a file named `report.html` in the repository (or whichever location your script is configured for). It contains an overview of the processed trades and their outcomes.

---

## Notes

### 1. Tab Preservation (Ally Only)
When copying data from the Ally website, ensure your editor preserves tab characters in `activity.txt`. The parser relies on tabs to correctly separate columns.

### 2. Multiple IB CSV Files
The script automatically processes every `.csv` file in `./data/IB/`. If you only have one IB file, that’s fine—put it in the same folder.

### 3. Verification
After processing, check the CSV files in `./data/cleaned/` to confirm the data looks correct before generating a final report.

### 4. Missing Data
If you only have Ally or only have IB data, the script will still run, ignoring the other missing files.

### 5. License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
