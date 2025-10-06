#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate semantic perturbations for StockNet tweets using the project LLM settings."""

import argparse
import json
import logging
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import openai
import xml.etree.ElementTree as ET


LOGGER = logging.getLogger("semantic_perturbation")


DEFAULT_TONE_DIRECTIVES = [
    "Introduce a tone of cautious optimism that still sounds professional.",
    "Add a restrained sense of wary concern without altering the facts.",
    "Blend in mild analytical enthusiasm appropriate for institutional commentary.",
    "Incorporate a measured skepticism that remains objective and fact-based.",
    "Infuse subtle prudent confidence that mirrors seasoned market analysts."
]

PLACEHOLDER_PATTERN = re.compile(r"\b([A-Z]{2,}_[A-Z0-9]+|URL|AT_USER|DATE_REF|YEAR_X|MONTH_X|DAY_X)\b")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_openai_client(api_cfg: Dict) -> Tuple[openai.OpenAI, str, float]:
    providers = api_cfg.get("providers", {})
    active_provider = api_cfg.get("active_llm_provider", "openai").lower()

    if active_provider not in providers:
        raise ValueError(f"Active provider '{active_provider}' not found in config providers")

    provider_cfg = providers[active_provider]
    api_key = provider_cfg.get("api_key")
    if not api_key:
        raise ValueError(f"API key missing for provider '{active_provider}'")

    base_url = provider_cfg.get("base_url")
    model_name = provider_cfg.get("llm_model_default", "gpt-4o")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    temperature = provider_cfg.get("llm_temperature", 0.3)
    return client, model_name, temperature


def load_prompt_template(template_path: Path) -> Tuple[str, str]:
    tree = ET.parse(str(template_path))
    root = tree.getroot()

    system_node = root.find("SystemMessage")
    user_node = root.find("UserMessage")
    if system_node is None or user_node is None:
        raise ValueError("Template must contain <SystemMessage> and <UserMessage>")

    system_msg = "".join(system_node.itertext()).strip()
    user_msg = "".join(user_node.itertext()).strip()
    return system_msg, user_msg


def ensure_placeholders_preserved(original: str, rewritten: str) -> bool:
    orig_tokens = set(PLACEHOLDER_PATTERN.findall(original))
    if not orig_tokens:
        return True
    rewritten_tokens = set(PLACEHOLDER_PATTERN.findall(rewritten))
    missing = orig_tokens - rewritten_tokens
    if missing:
        LOGGER.debug("Missing placeholders: %s", ", ".join(sorted(missing)))
    return not missing


def length_within_range(original: str, rewritten: str, tolerance: float = 0.2) -> bool:
    orig_len = max(len(original.strip()), 1)
    rewritten_len = len(rewritten.strip())
    lower = orig_len * (1 - tolerance)
    upper = orig_len * (1 + tolerance)
    return lower <= rewritten_len <= upper


def call_llm(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> Dict:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.warning("LLM call failed on attempt %s/%s: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Failed to obtain LLM response after {max_retries} attempts: {last_error}")


def perturb_tweet(
    client: openai.OpenAI,
    model: str,
    system_template: str,
    user_template: str,
    original_tweet: str,
    tone_directive: str,
    temperature: float,
) -> str:
    system_prompt = system_template.replace("[TONE_DIRECTIVE]", tone_directive)
    user_prompt = user_template.replace("[ORIGINAL_TWEET]", original_tweet)

    response = call_llm(client, model, system_prompt, user_prompt, temperature)
    rewritten = response.get("rewritten_tweet", "").strip()
    if not rewritten:
        raise ValueError("LLM response missing 'rewritten_tweet'")

    if not ensure_placeholders_preserved(original_tweet, rewritten):
        LOGGER.warning("LLM response dropped required placeholders; keeping generated text as-is")

    if not length_within_range(original_tweet, rewritten):
        LOGGER.debug(
            "Length outside tolerance (original=%d, rewritten=%d)",
            len(original_tweet.strip()),
            len(rewritten.strip()),
        )

    return rewritten


def iter_stock_tweets(stock_json: Dict) -> Iterable[Tuple[str, str, str]]:
    for day_key, tweets in stock_json.items():
        for tweet_id, tweet_data in tweets.items():
            original_text = tweet_data.get("content", "")
            if not original_text:
                continue
            yield day_key, tweet_id, original_text


def save_json_atomic(path: Path, data: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp_file:
        json.dump(data, tmp_file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_json_if_exists(path: Path, default: Dict) -> Dict:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_completed_tweets(text_json: Dict) -> int:
    total = 0
    for tweets in text_json.values():
        for tweet in tweets.values():
            if tweet.get("content"):
                total += 1
    return total


def update_progress(progress_path: Path, stock: str, processed: int, total: int) -> None:
    progress_data = load_json_if_exists(progress_path, {})
    progress_data[stock] = {
        "processed": processed,
        "total": total,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    save_json_atomic(progress_path, progress_data)


def process_stock(
    stock_dir: Path,
    output_dir: Path,
    perturb_fn,
    tone_options: List[str],
    seed_rng: random.Random,
    progress_path: Path,
) -> None:
    with (stock_dir / "text_data.json").open("r", encoding="utf-8") as f:
        text_json = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_text_path = output_dir / "text_data.json"

    output_text_json = load_json_if_exists(output_text_path, {})

    total_tweets = sum(len(tweets) for tweets in text_json.values())
    processed_so_far = count_completed_tweets(output_text_json)

    LOGGER.info(
        "Processing %s (%d tweets, %d already completed)",
        stock_dir.name,
        total_tweets,
        processed_so_far,
    )

    for day_key, tweets in text_json.items():
        day_output = output_text_json.setdefault(day_key, {})
        for tweet_id, tweet_data in tweets.items():
            existing_entry = day_output.get(tweet_id)
            if existing_entry and existing_entry.get("content"):
                continue

            original_content = tweet_data.get("content", "")
            if not original_content:
                day_output[tweet_id] = tweet_data
                continue

            tone_directive = seed_rng.choice(tone_options)
            rewritten = perturb_fn(original_content, tone_directive)
            day_output[tweet_id] = {**tweet_data, "content": rewritten}

            save_json_atomic(output_text_path, output_text_json)

            processed_so_far += 1
            update_progress(progress_path, stock_dir.name, processed_so_far, total_tweets)

    for filename in ("price_data.json", "labels.json"):
        src = stock_dir / filename
        dst = output_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    LOGGER.info("Completed %s (%d/%d tweets)", stock_dir.name, processed_so_far, total_tweets)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate semantic perturbations for StockNet tweets")
    parser.add_argument("--config", default="config.json", help="Path to project config.json")
    parser.add_argument(
        "--template",
        default="templates/semantic_perturbation_prompt_template.xml",
        help="Path to the semantic perturbation prompt template",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/StockNet/Train",
        help="Root path of the StockNet training split",
    )
    parser.add_argument(
        "--stocks",
        nargs="*",
        default=None,
        help="Subset of stock tickers to perturb (defaults to config data_params.default_stocks)",
    )
    parser.add_argument(
        "--output-root",
        default="data/StockNet_SemPerturb/Train",
        help="Destination root for perturbed dataset",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.json if present",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tone directives")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per tweet")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Base retry delay in seconds")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between LLM calls (seconds)")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    project_root = Path(__file__).resolve().parent.parent
    config_path = (project_root / args.config).resolve()
    template_path = (project_root / args.template).resolve()
    dataset_root = (project_root / args.dataset_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    if not dataset_root.exists():
        LOGGER.error("Dataset root not found: %s", dataset_root)
        return 1

    config = load_config(config_path)

    api_cfg = config.get("api", {})
    client, model_name, cfg_temperature = build_openai_client(api_cfg)
    temperature = args.temperature if args.temperature is not None else cfg_temperature

    system_template, user_template = load_prompt_template(template_path)

    default_stocks = config.get("data_params", {}).get("default_stocks", [])
    target_stocks = args.stocks if args.stocks else default_stocks
    if not target_stocks:
        raise ValueError("No target stocks specified")

    rng = random.Random(args.seed)

    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    if not args.resume and progress_path.exists():
        LOGGER.warning("Removing existing progress log because --resume was not set: %s", progress_path)
        progress_path.unlink()

    def perturb_wrapper(original_text: str, tone_directive: str) -> str:
        retries = args.max_retries
        while retries:
            try:
                rewritten = perturb_tweet(
                    client,
                    model_name,
                    system_template,
                    user_template,
                    original_text,
                    tone_directive,
                    temperature,
                )
                return rewritten
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Retrying tweet due to error: %s", exc)
                retries -= 1
                if retries == 0:
                    raise
                time.sleep(args.retry_delay)
        raise RuntimeError("Unreachable")

    for stock in target_stocks:
        stock_dir = dataset_root / stock
        if not stock_dir.exists():
            LOGGER.warning("Skipping %s (directory not found)", stock)
            continue
        output_dir = output_root / stock

        process_stock(
            stock_dir,
            output_dir,
            perturb_wrapper,
            DEFAULT_TONE_DIRECTIVES,
            rng,
            progress_path,
        )
        time.sleep(args.sleep)

    LOGGER.info("Perturbed dataset saved to %s", output_root)
    if progress_path.exists():
        LOGGER.info("Progress logged at %s", progress_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

