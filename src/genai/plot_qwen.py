from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ProjectConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def load_qwen(model_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading Qwen model %s (4-bit)", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_overviews(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_template: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring plots"):
        overview = row.get("overview") or ""
        movie_id = row["id"]
        user_prompt = user_template.format(overview=overview.strip())
        prompt_text = build_prompt(tokenizer, system_prompt, user_prompt)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        generated_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        ).strip()
        scores = parse_scores(response_text)
        scores["id"] = movie_id
        results.append(scores)

    return results


def parse_scores(text: str) -> dict[str, float]:
    """Gracefully parse the JSON snippet."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %s", text)
        return {"plot_novelty": None, "emotion_complexity": None, "raw_response": text}

    return {
        "plot_novelty": float(data.get("plot_novelty", 0.0)),
        "emotion_complexity": float(data.get("emotion_complexity", 0.0)),
        "raw_response": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B - plot scoring via Qwen.")
    args = parser.parse_args()

    config = ProjectConfig()
    config.prepare()
    merged = pd.read_csv(config.paths.raw_movies)

    model, tokenizer = load_qwen(config.plot_model.model_id)
    records = score_overviews(
        merged,
        model,
        tokenizer,
        config.plot_model.system_prompt,
        config.plot_model.user_prompt_template,
        max_new_tokens=config.plot_model.max_new_tokens,
        temperature=config.plot_model.temperature,
        top_p=config.plot_model.top_p,
    )

    df = pd.DataFrame(records)
    df.to_parquet(config.paths.plot_features_path, index=False)
    logger.info("Plot scores saved to %s", config.paths.plot_features_path)


if __name__ == "__main__":
    main()
