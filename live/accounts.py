"""Named trading accounts: strategy + paper/live mode + credential env vars.

Add a new paper book by registering another Account that points at a strategy
module and its own Alpaca paper keys. Promote to real money by adding a
paper=False account with live credential env names (and a separate CI workflow).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    id: str
    strategy: str
    paper: bool
    key_env: str
    secret_env: str
    email_tag: str
    # Optional legacy env names accepted as fallback (soft cutover for paper_r50d).
    legacy_key_env: str | None = None
    legacy_secret_env: str | None = None

    @property
    def halt_path_name(self) -> str:
        """Per-account halt file under live/halts/<id>."""
        return self.id


ACCOUNTS: dict[str, Account] = {
    "paper_r50d": Account(
        id="paper_r50d",
        strategy="r50d",
        paper=True,
        key_env="ALPACA_PAPER_R50D_KEY",
        secret_env="ALPACA_PAPER_R50D_SECRET",
        email_tag="paper/r50d",
        legacy_key_env="ALPACA_KEY",
        legacy_secret_env="ALPACA_SECRET",
    ),
    # Promote later:
    # "live_r50d": Account(
    #     id="live_r50d",
    #     strategy="r50d",
    #     paper=False,
    #     key_env="ALPACA_LIVE_R50D_KEY",
    #     secret_env="ALPACA_LIVE_R50D_SECRET",
    #     email_tag="live/r50d",
    # ),
}


def get_account(account_id: str) -> Account:
    try:
        return ACCOUNTS[account_id]
    except KeyError as exc:
        known = ", ".join(sorted(ACCOUNTS))
        raise KeyError(f"unknown account {account_id!r}; known: {known}") from exc


def list_accounts() -> list[str]:
    return sorted(ACCOUNTS)


def paper_account_ids() -> list[str]:
    return sorted(a.id for a in ACCOUNTS.values() if a.paper)
