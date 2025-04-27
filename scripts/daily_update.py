#!/usr/bin/env python3
"""
One-step daily update: clean raw files then generate HTML report.
"""
import subprocess
import sys

def main():
    # Step 1: Process raw Ally & IB statements into cleaned CSVs
    ret = subprocess.call([sys.executable, 'scripts/process_activity.py'])
    if ret != 0:
        sys.exit(ret)

    # Step 2: Generate HTML report (via scripts/write_report.py)
    ret = subprocess.call([sys.executable, 'scripts/write_report.py'])
    if ret != 0:
        sys.exit(ret)

if __name__ == '__main__':
    main()