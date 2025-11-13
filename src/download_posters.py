from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


BASE_URL = "https://image.tmdb.org/t/p"


def build_image_url(poster_path: str, size: str = "w500") -> str:
    return f"{BASE_URL}/{size}{poster_path}"


def download_posters(
    df: pd.DataFrame,
    output_dir: Path,
    image_size: str = "w500",
    timeout: int = 30,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading posters"):
        poster_path = row.get("poster_path")
        movie_id = row.get("id")
        if not poster_path or not isinstance(poster_path, str):
            continue

        url = build_image_url(poster_path, size=image_size)
        destination = output_dir / f"{movie_id}.jpg"
        if destination.exists():
            continue

        response = session.get(url, timeout=timeout)
        if response.status_code == 200:
            destination.write_bytes(response.content)
        else:
            logger.warning("Failed to download %s (%s) -> %s", movie_id, poster_path, response.status_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 - poster downloader.")
    parser.add_argument(
        "--image-size",
        default="w500",
        help="TMDb poster size (w185, w342, w500, original ...).",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()

    movies = pd.read_csv(config.paths.raw_movies)
    download_posters(movies, config.paths.posters_dir, image_size=args.image_size)
    logger.info("Posters available under %s", config.paths.posters_dir.resolve())


if __name__ == "__main__":
    main()
