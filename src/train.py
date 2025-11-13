from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


NUMERIC_COLS = [
    "budget_log",
    "revenue_log",
    "popularity",
    "runtime",
    "release_year",
    "release_month",
]

CATEGORICAL_COLS = [
    "original_language",
    "status",
    "top_genre",
    "top_keyword",
    "top_actor",
    "director",
]


def build_preprocessor(include_genai: bool) -> ColumnTransformer:
    transformers = [
        ("numeric", StandardScaler(), NUMERIC_COLS),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_COLS,
        ),
    ]

    if include_genai:
        transformers.append(
            (
                "poster_tfidf",
                TfidfVectorizer(max_features=500, ngram_range=(1, 2)),
                "poster_keywords",
            )
        )
        transformers.append(
            (
                "plot_scores",
                "passthrough",
                ["plot_novelty", "emotion_complexity"],
            )
        )

    return ColumnTransformer(transformers, sparse_threshold=0.7)


def build_model(include_genai: bool) -> Pipeline:
    preprocessor = build_preprocessor(include_genai=include_genai)
    regressor = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", regressor),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[float, Pipeline]:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse, pipeline


def plot_rmse(rmse_baseline: float, rmse_genai: float, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Baseline", "GenAI"], [rmse_baseline, rmse_genai], color=["#6baed6", "#fd8d3c"])
    for bar, value in zip(bars, [rmse_baseline, rmse_genai]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.ylabel("RMSE")
    plt.title("A/B Experiment: Baseline vs GenAI-enhanced Features")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(pipeline: Pipeline, output_path: Path, top_k: int = 25) -> None:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    regressor: XGBRegressor = pipeline.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_
    top_indices = np.argsort(importances)[-top_k:]
    top_features = feature_names[top_indices]
    top_values = importances[top_indices]

    order = np.argsort(top_values)
    plt.figure(figsize=(8, 8))
    plt.barh(range(len(order)), top_values[order], color="#2ca25f")
    plt.yticks(range(len(order)), top_features[order])
    plt.xlabel("Feature importance")
    plt.title("GenAI Model - Top Feature Importances")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 - training + evaluation.")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()

    fused = pd.read_parquet(config.paths.fusion_dataset_path)
    target = fused[config.training.target_column]
    drop_cols = [
        config.training.target_column,
        "id",
        "title",
        "vote_count",
        "raw_response",
    ]
    train_features = fused.drop(columns=[col for col in drop_cols if col in fused.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        train_features,
        target,
        test_size=args.test_size,
        random_state=config.training.random_state,
    )

    baseline_pipeline = build_model(include_genai=False)
    genai_pipeline = build_model(include_genai=True)

    baseline_rmse, baseline_pipeline = evaluate_model(
        baseline_pipeline, X_train, X_test, y_train, y_test
    )
    genai_rmse, genai_pipeline = evaluate_model(
        genai_pipeline, X_train, X_test, y_train, y_test
    )

    logger.info("Baseline RMSE: %.4f", baseline_rmse)
    logger.info("GenAI RMSE: %.4f", genai_rmse)

    metrics = {
        "baseline_rmse": baseline_rmse,
        "genai_rmse": genai_rmse,
        "delta": baseline_rmse - genai_rmse,
    }
    metrics_path = config.paths.reports_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    plot_rmse(
        baseline_rmse,
        genai_rmse,
        config.paths.figures_dir / "rmse_comparison.png",
    )
    plot_feature_importance(
        genai_pipeline,
        config.paths.figures_dir / "feature_importance.png",
    )


if __name__ == "__main__":
    main()
