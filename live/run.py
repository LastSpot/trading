#!/usr/bin/env python3
"""CLI entry for named Alpaca paper/live accounts.

Usage:
  poetry run python live/run.py --account paper_r50d
  poetry run python live/run.py --account paper_dm42 --dry-run
  poetry run python live/run.py --account paper_dm42 --phase reduce
  poetry run python live/run.py --account paper_dm42 --phase increase
  poetry run python live/run.py --account live_r50d --confirm-live

Env (per account; see live/accounts.py and .env.example):
  ALPACA_PAPER_R50D_KEY / ALPACA_PAPER_R50D_SECRET
  ALPACA_PAPER_DM42_KEY / ALPACA_PAPER_DM42_SECRET
  GMAIL_USER / GMAIL_APP_PASSWORD  (optional)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python live/run.py` with sibling imports (package-mode = false).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from accounts import get_account, list_accounts  # noqa: E402
from engine import PHASES, load_env, notify_run_failure, run_account  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--account",
        default="paper_r50d",
        choices=list_accounts(),
        help="named account id from live/accounts.py (default: paper_r50d)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print orders without submitting",
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="required to submit orders on paper=False (real-money) accounts",
    )
    parser.add_argument(
        "--phase",
        default="auto",
        choices=sorted(PHASES),
        help=(
            "cash-rotate accounts only: reduce=sells/exits after signal close; "
            "increase=buys after reduces free cash; auto=increase if a pending "
            "rebalance exists else reduce (ignored for non-rotate accounts)"
        ),
    )
    args = parser.parse_args(argv)

    account = get_account(args.account)
    try:
        run_account(
            account,
            dry_run=args.dry_run,
            confirm_live=args.confirm_live,
            phase=args.phase,
        )
    except Exception:
        notify_run_failure(account.id)
        raise


if __name__ == "__main__":
    load_env()
    main()
