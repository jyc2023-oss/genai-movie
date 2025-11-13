from __future__ import annotations

import argparse
import logging

import pandas as pd

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def fuse_datasets(config: ProjectConfig) -> pd.DataFrame:
    structured = pd.read_parquet(config.paths.structured_features_path)
    poster = pd.read_parquet(config.paths.poster_features_path)
    plot = pd.read_parquet(config.paths.plot_features_path)

    fused = (
        structured.merge(poster, on="id", how="left")
        .merge(plot, on="id", how="left")
    )
    fused["poster_keywords"] = fused["poster_keywords"].fillna("")
    for column in ["plot_novelty", "emotion_complexity"]:
        if column in fused:
            fused[column] = fused[column].fillna(fused[column].median())
        else:
            fused[column] = 0.0

    fused.to_parquet(config.paths.fusion_dataset_path, index=False)
    logger.info("Super feature table saved to %s", config.paths.fusion_dataset_path)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 - feature fusion.")
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()
    fuse_datasets(config)


if __name__ == "__main__":
    main()
