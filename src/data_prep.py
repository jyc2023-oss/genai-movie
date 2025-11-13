from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def safe_json_loads(payload: str | float | int | None) -> list[dict[str, Any]]:
    """Parse the TMDB stringified JSON columns."""
    if payload in (None, np.nan):
        return []
    if isinstance(payload, (list, tuple)):
        return payload
    if not isinstance(payload, str):
        return []
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return []


def extract_names(items: list[dict[str, Any]], key: str = "name") -> list[str]:
    return [item.get(key) for item in items if item.get(key)]


def get_top_name(items: list[dict[str, Any]], key: str = "name") -> str | None:
    names = extract_names(items, key=key)
    return names[0] if names else None


def extract_director(crew_items: list[dict[str, Any]]) -> str | None:
    for item in crew_items:
        if item.get("job") == "Director" and item.get("name"):
            return item["name"]
    return None


def load_and_merge(config: ProjectConfig) -> pd.DataFrame:
    """Load the raw CSVs and return a merged DataFrame."""
    movies = pd.read_csv(config.paths.raw_movies)
    credits = pd.read_csv(config.paths.raw_credits)
    credits = credits.rename(columns={"movie_id": "id"})
    merged = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))
    logger.info("Merged dataset shape: %s", merged.shape)
    return merged


def engineer_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implement Phase 2 of the design doc."""
    working = df.copy()

    # Basic numeric hygiene
    numeric_cols = ["budget", "revenue", "popularity", "runtime", "vote_count"]
    for col in numeric_cols:
        working[col] = (
            working[col]
            .replace({0: np.nan})
            .astype(float)
            .fillna(working[col].replace({0: np.nan}).astype(float).median())
        )

    working["budget_log"] = np.log1p(working["budget"])
    working["revenue_log"] = np.log1p(working["revenue"])

    # Release date features
    working["release_date"] = pd.to_datetime(working["release_date"], errors="coerce")
    working["release_year"] = working["release_date"].dt.year.fillna(working["release_date"].dt.year.median())
    working["release_month"] = working["release_date"].dt.month.fillna(working["release_date"].dt.month.median())

    # Parse JSON columns into python objects
    working["genres_parsed"] = working["genres"].apply(safe_json_loads)
    working["keywords_parsed"] = working["keywords"].apply(safe_json_loads)
    working["cast_parsed"] = working["cast"].apply(safe_json_loads)
    working["crew_parsed"] = working["crew"].apply(safe_json_loads)

    working["top_genre"] = working["genres_parsed"].apply(get_top_name)
    working["top_keyword"] = working["keywords_parsed"].apply(get_top_name)
    working["top_actor"] = working["cast_parsed"].apply(get_top_name)
    working["director"] = working["crew_parsed"].apply(extract_director)

    # Fill missing categorical values
    categorical_cols = ["original_language", "status", "top_genre", "top_keyword", "top_actor", "director"]
    for col in categorical_cols:
        working[col] = working[col].fillna("unknown")

    feature_columns = [
        "id",
        "title",
        "budget_log",
        "revenue_log",
        "popularity",
        "runtime",
        "release_year",
        "release_month",
        "original_language",
        "status",
        "top_genre",
        "top_keyword",
        "top_actor",
        "director",
        "vote_average",
    ]
    engineered = working[feature_columns]
    return engineered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 - traditional feature engineering pipeline."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="(Optional) path to a python module that exposes ProjectConfig as `config`.",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()

    merged = load_and_merge(config)
    structured = engineer_structured_features(merged)
    structured.to_parquet(config.paths.structured_features_path, index=False)
    logger.info("Structured features saved to %s", config.paths.structured_features_path)


if __name__ == "__main__":
    main()
