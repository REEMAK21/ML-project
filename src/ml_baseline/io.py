from pathlib import Path
import pandas as pd

NA = ["", "None", "NA", "N/A", "null"]

def read_orders_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"order_id": "string", "user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )

def read_users_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"user_id": "string"},
        na_values=NA,
        keep_default_na=True,
    )

def parquet_supported() -> bool:
    """
    Check if Parquet support is available (pyarrow or fastparquet installed).
    """
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401
        return True
    except ImportError:
        pass
    return False

def write_tabular(df: pd.DataFrame, path: Path) -> None:
    """
    Write DataFrame to CSV or Parquet depending on file extension.
    If Parquet is requested but not supported, fallback to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".parquet":
        if parquet_supported():
            df.to_parquet(path, index=False)
        else:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f" Parquet not supported. Saved as CSV instead: {csv_path}")
    else:
        df.to_csv(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
