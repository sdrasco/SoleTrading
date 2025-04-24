#!/usr/bin/env python3

import subprocess

def main():
    # 1) Clean the data
    subprocess.run(["python", "-m", "cleaning.clean_activity"], check=True)

    # 2) Generate the report
    subprocess.run(["python", "-m", "report.main_report"], check=True)

if __name__ == "__main__":
    main()
