# Pipeline Overview

This pipeline processes option activity data and generates a report. Follow these steps:

1. **Copy Activity Data:**
   - From Ally's website, copy the activity of interest.
   - **Note:** Do not copy the header.

2. **Prepare the Activity File:**
   - Open your favorite editor (e.g., `vi` or any editor that preserves tabs).
   - Paste the copied data into a file named `activities.txt`.

3. **Process the Activity Data:**
   - Run the processing script:
     ```bash
     python process_activity.py
     ```
   - This will generate the following CSV files:
     - `transactions.csv` – Contains all lines from `activities.txt` (including those not used in further analysis).
     - `trades.csv` – Contains all completed trades.
     - `unclosed.csv` – Contains all open positions (trades that haven't yet closed).
     - `unopened.csv` – Contains all trades where the opening wasn't contained in the activity file.

4. **Generate the Report:**
   - Run the report generation script:
     ```bash
     python write_report.py
     ```
   - This will produce a simple HTML report: `report.html`.

## Notes

- **Tab Preservation:** Ensure your text editor preserves tab characters when pasting the data.
- **Verification:** You may want to check the CSV files for correctness before generating the final report.