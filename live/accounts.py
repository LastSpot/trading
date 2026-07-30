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
    # Cash-account rotators (e.g. DM42): sell/exit after Friday close, buy next session
    # once proceeds free buying power. Matches research T+1 atomic rebalance.
    cash_rotate: bool = False

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
    "paper_dm42": Account(
        id="paper_dm42",
        strategy="dm42",
        paper=True,
        key_env="ALPACA_PAPER_DM42_KEY",
        secret_env="ALPACA_PAPER_DM42_SECRET",
        email_tag="paper/dm42",
        cash_rotate=True,
    ),
    "paper_r50d_asym": Account(
        id="paper_r50d_asym",
        strategy="r50d_asym",
        paper=True,
        key_env="ALPACA_PAPER_R50D_ASYM_KEY",
        secret_env="ALPACA_PAPER_R50D_ASYM_SECRET",
        email_tag="paper/r50d_asym",
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
