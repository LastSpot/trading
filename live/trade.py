#!/usr/bin/env python3
"""Compatibility shim -- same as live/run.py with default account paper_r50d.

Prefer:  poetry run python live/run.py --account paper_r50d [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import load_env  # noqa: E402
from run import main as run_main  # noqa: E402


if __name__ == "__main__":
    load_env()
    run_main(sys.argv[1:])
