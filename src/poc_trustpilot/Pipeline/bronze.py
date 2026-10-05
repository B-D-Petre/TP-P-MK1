"""Bronze layer: the raw CSV, typed and stored as Parquet. No text changes."""
import pandas as pd

from .config import BRONZE, RAW_CSV


def build_bronze() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV).rename(columns={"Title": "title", "Text": "text"})
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["created_at"] = pd.to_datetime(df["created_date"], utc=True)
    df["language"] = df["language"].replace({"nb": "no"})  # same language (Norwegian Bokmål), two codes
    df = df.drop(columns="created_date")

    BRONZE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE / "reviews.parquet", index=False)
    return df
