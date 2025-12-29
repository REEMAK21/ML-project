import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from ml_baseline.config import Paths
from ml_baseline.io import write_tabular, parquet_supported


def make_sample_feature_table(*, root: Path | None = None, n_users: int = 50, seed: int = 42) -> Path:
    # Get paths object
    paths = Paths.from_repo_root() if root is None else Paths(root=root)
    
    # Ensure processed folder exists
    paths.processed.mkdir(parents=True, exist_ok=True)

    # Generate sample data
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user_id": [f"u{i:03d}" for i in range(1, n_users + 1)],
        "country": rng.choice(["US", "CA", "GB"], size=n_users),
        "n_orders": rng.integers(1, 10, size=n_users)
    })
    df["avg_amount"] = rng.normal(10, 3, size=n_users).clip(min=1).round(2)
    df["total_amount"] = (df["n_orders"] * df["avg_amount"]).round(2)
    df["is_high_value"] = (df["total_amount"] >= 80).astype(int)

    # Write CSV
    csv_path = paths.processed / "features.csv"
    write_tabular(df, csv_path)

    # Optional: write Parquet if supported
    if parquet_supported():
        write_tabular(df, paths.processed / "features.parquet")

    return csv_path


if __name__ == "__main__":
    path = make_sample_feature_table()
    print(f"Sample data created at: {path}")
