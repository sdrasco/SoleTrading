#!/usr/bin/env python3
"""
CLI wrapper: clean raw Ally + IB activity into standardized CSVs.
"""
import os, sys
# ensure repo root is on PYTHONPATH so options_report can be imported
_SCRIPT_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
sys.path.insert(0, _REPO_ROOT)
from pathlib import Path
from options_report.processing import process_all

def main():
    ally_file = Path('data/Ally/activity.txt')
    ib_dir = Path('data/IB')
    out_dir = Path('data/cleaned')
    process_all(ally_file, ib_dir, out_dir)

if __name__ == '__main__':
    main()