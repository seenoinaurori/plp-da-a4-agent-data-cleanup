"""Generate reproducible raw transaction and agent data for the cleanup lab."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 44
TRANSACTION_COUNT = 1_000
AGENT_COUNT = 40
OUTPUT_DIR = Path(__file__).resolve().parent

COUNTIES = (
    "Nairobi",
    "Mombasa",
    "Kisumu",
    "Nakuru",
    "Uasin Gishu",
    "Kiambu",
    "Machakos",
    "Nyeri",
)


def generate_data(
    seed: int = SEED,
    transaction_count: int = TRANSACTION_COUNT,
    agent_count: int = AGENT_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return deterministic raw transaction and agent tables with quality issues.

    The transactions include missing and malformed values, inconsistent
    transaction-type labels, duplicate identifiers, extreme amounts, and
    references to agents absent from the agent table. The issues are injected
    by sampling positions from the seeded random generator, not by fixed row
    numbers.
    """
    if transaction_count < 1:
        raise ValueError("transaction_count must be at least 1")
    if agent_count < 2:
        raise ValueError("agent_count must be at least 2")

    rng = np.random.default_rng(seed)
    agent_numbers = np.arange(1, agent_count + 1)
    agents = pd.DataFrame(
        {
            "agent_id": [f"AGT-{number:03d}" for number in agent_numbers],
            "agent_name": [
                f"Agent {number:03d}" for number in agent_numbers
            ],
            "county": rng.choice(COUNTIES, size=agent_count),
        }
    )

    transaction_numbers = np.arange(1, transaction_count + 1)
    txn_types = rng.choice(
        ["Deposit", "Withdrawal"], size=transaction_count, p=[0.58, 0.42]
    ).astype(object)
    amounts = np.round(rng.lognormal(mean=7.6, sigma=0.8, size=transaction_count), 2)
    transaction_dates = (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            rng.integers(0, 365, size=transaction_count), unit="D"
        )
    )
    agent_ids = rng.choice(agents["agent_id"].to_numpy(), size=transaction_count)

    transactions = pd.DataFrame(
        {
            "txn_id": [f"TXN-{number:05d}" for number in transaction_numbers],
            "txn_date": transaction_dates.strftime("%Y-%m-%d"),
            "agent_id": agent_ids,
            "txn_type": txn_types,
            "amount": amounts.astype(object),
            "customer_id": [
                f"CUS-{number:05d}"
                for number in rng.integers(1, transaction_count * 2, size=transaction_count)
            ],
        }
    )

    def sampled_positions(fraction: float) -> np.ndarray:
        count = int(round(transaction_count * fraction))
        return rng.choice(transaction_count, size=count, replace=False)

    # Mixed formats and invalid date strings exercise safe date parsing.
    date_positions = sampled_positions(0.06)
    transactions.loc[date_positions[: len(date_positions) // 2], "txn_date"] = (
        transaction_dates[date_positions[: len(date_positions) // 2]].strftime(
            "%d/%m/%Y"
        )
    )
    transactions.loc[date_positions[len(date_positions) // 2 :], "txn_date"] = (
        "not-a-date"
    )

    # Missing and non-numeric amount values exercise auditable imputation.
    missing_amount_positions = sampled_positions(0.05)
    transactions.loc[missing_amount_positions, "amount"] = np.nan
    invalid_amount_positions = sampled_positions(0.02)
    transactions.loc[invalid_amount_positions, "amount"] = "unknown"

    # Invalid values and presentation variants exercise type normalization.
    type_positions = sampled_positions(0.08)
    type_variants = np.resize(
        np.array(["deposit", "WITHDRAWAL", " Deposit ", "cash-in", None], dtype=object),
        len(type_positions),
    )
    rng.shuffle(type_variants)
    transactions.loc[type_positions, "txn_type"] = type_variants

    # Zero, negative, and unusually large amounts exercise validity/outlier checks.
    invalid_amount_value_positions = sampled_positions(0.025)
    transactions.loc[invalid_amount_value_positions[: len(invalid_amount_value_positions) // 2], "amount"] = 0
    transactions.loc[invalid_amount_value_positions[len(invalid_amount_value_positions) // 2 :], "amount"] = -1
    outlier_positions = sampled_positions(0.01)
    transactions.loc[outlier_positions, "amount"] = 250_000

    # Replace a sample of IDs with other generated IDs to create duplicates.
    duplicate_positions = sampled_positions(0.015)
    if len(duplicate_positions) > 0:
        source_positions = rng.integers(
            0, transaction_count, size=len(duplicate_positions)
        )
        transactions.loc[duplicate_positions, "txn_id"] = transactions.loc[
            source_positions, "txn_id"
        ].to_numpy()

    # Keep unmatched references in the transaction data for merge validation.
    unknown_agent_positions = sampled_positions(0.03)
    transactions.loc[unknown_agent_positions, "agent_id"] = [
        f"AGT-X{number:03d}"
        for number in rng.integers(1, transaction_count + 1, size=len(unknown_agent_positions))
    ]

    return transactions, agents


def main() -> None:
    """Write the reproducible raw CSV inputs beside this script."""
    transactions, agents = generate_data()
    transactions_path = OUTPUT_DIR / "transactions_raw.csv"
    agents_path = OUTPUT_DIR / "agents_raw.csv"
    transactions.to_csv(transactions_path, index=False)
    agents.to_csv(agents_path, index=False)
    print(f"Wrote {len(transactions):,} transactions to {transactions_path}")
    print(f"Wrote {len(agents):,} agents to {agents_path}")


if __name__ == "__main__":
    main()
