from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def load_llava(model_id: str) -> tuple[LlavaForConditionalGeneration, AutoProcessor]:
    logger.info("Loading LLaVA model %s (4-bit)", model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_prompt(processor: AutoProcessor, prompt: str) -> str:
    conversation = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Respond with five comma-separated keywords.",
                },
            ],
        },
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def describe_posters(
    df: pd.DataFrame,
    posters_dir: Path,
    prompt: str,
    model: LlavaForConditionalGeneration,
    processor: AutoProcessor,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    template = build_prompt(processor, prompt)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Describing posters"):
        movie_id = row["id"]
        image_path = posters_dir / f"{movie_id}.jpg"
        if not image_path.exists():
            logger.warning("Poster missing for movie %s", movie_id)
            continue

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            images=image,
            text=template,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        output = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        outputs.append({"id": movie_id, "poster_keywords": output.strip()})

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1A - poster keyword extraction via LLaVA.")
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()

    structured = pd.read_parquet(config.paths.structured_features_path)
    model, processor = load_llava(config.poster_model.model_id)
    records = describe_posters(
        structured,
        config.paths.posters_dir,
        config.poster_model.prompt,
        model,
        processor,
        max_new_tokens=config.poster_model.max_new_tokens,
        temperature=config.poster_model.temperature,
        top_p=config.poster_model.top_p,
    )
    df = pd.DataFrame(records)
    df.to_parquet(config.paths.poster_features_path, index=False)
    logger.info("Poster keywords saved to %s", config.paths.poster_features_path)


if __name__ == "__main__":
    main()
