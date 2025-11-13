from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathsConfig:
    """Centralized filesystem layout."""

    project_root: Path = Path(".")
    raw_movies: Path = Path("tmdb_5000_movies.csv")
    raw_credits: Path = Path("tmdb_5000_credits.csv")
    posters_dir: Path = Path("artifacts/posters")
    datasets_dir: Path = Path("artifacts/datasets")
    poster_features_path: Path = Path("artifacts/datasets/poster_features.parquet")
    plot_features_path: Path = Path("artifacts/datasets/plot_features.parquet")
    structured_features_path: Path = Path("artifacts/datasets/structured_features.parquet")
    fusion_dataset_path: Path = Path("artifacts/datasets/super_features.parquet")
    reports_dir: Path = Path("artifacts/reports")
    figures_dir: Path = Path("artifacts/reports/figures")

    def ensure_directories(self) -> None:
        """Create the directories that will store intermediate artifacts."""
        for path in [
            self.posters_dir,
            self.datasets_dir,
            self.reports_dir,
            self.figures_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class PosterModelConfig:
    """Configuration for the LLaVA feature extractor."""

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    prompt: str = (
        "You are a world-class movie poster critic. "
        "Describe the poster using five concise English keywords that capture "
        "visual style, genre cues, and emotional tone. "
        "Return the keywords as a comma-separated list."
    )
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9
    batch_size: int = 1


@dataclass(slots=True)
class PlotModelConfig:
    """Configuration for the Qwen plot scoring model."""

    model_id: str = "Qwen/Qwen1.5-7B-Chat"
    system_prompt: str = (
        "You are an experienced narrative analyst. "
        "Read the movie plot summary and output strict JSON with the schema "
        '{"plot_novelty": <float 1-10>, "emotion_complexity": <float 1-10>}. '
        "Use one decimal place where appropriate."
    )
    user_prompt_template: str = (
        "Plot summary:\n{overview}\n\nReturn the JSON payload now."
    )
    temperature: float = 0.1
    top_p: float = 0.8
    max_new_tokens: int = 128


@dataclass(slots=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    baseline_features: tuple[str, ...] = (
        "budget_log",
        "revenue_log",
        "popularity",
        "runtime",
        "release_year",
        "release_month",
        "original_language",
        "status",
        "top_genre",
        "director",
    )
    target_column: str = "vote_average"


@dataclass(slots=True)
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    poster_model: PosterModelConfig = field(default_factory=PosterModelConfig)
    plot_model: PlotModelConfig = field(default_factory=PlotModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def prepare(self) -> None:
        self.paths.ensure_directories()
