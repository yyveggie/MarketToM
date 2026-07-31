#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config_utils import deep_update
from experiments.robustness import entity_masking

DATASET_ALIASES = {
    "ACL18": "StockNet",
    "StockNet": "StockNet",
    "CMIN-US": "CMIN_US",
    "CMIN_US": "CMIN_US",
    "CMIN-CN": "CMIN_CN",
    "CMIN_CN": "CMIN_CN",
    "StockNet-EntityMasked": "StockNet_EntityMasked",
    "StockNet_EntityMasked": "StockNet_EntityMasked",
    "CMIN-US-EntityMasked": "CMIN_US_EntityMasked",
    "CMIN_US_EntityMasked": "CMIN_US_EntityMasked",
    "CMIN-CN-EntityMasked": "CMIN_CN_EntityMasked",
    "CMIN_CN_EntityMasked": "CMIN_CN_EntityMasked",
    "StockNet-SemPerturb": "StockNet_SemPerturb",
    "StockNet_SemPerturb": "StockNet_SemPerturb",
    "Holdout2025": "unseen_data_aligned",
    "unseen_data_aligned": "unseen_data_aligned",
    "unseen_data": "unseen_data",
    "unseen": "unseen_data",
}
MODEL_ALIASES = {
    "gpt4o": "gpt-4o-2024-11-20",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "qwen": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-72B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
}
TASK_DEFAULTS = {
    "main": {
        "datasets": ["ACL18", "CMIN-US", "CMIN-CN"],
        "splits": ["Test"],
        "presets": ["MarketToM-2nd"],
        "models": ["gpt4o", "qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "train_first": True,
    },
    "ablation": {
        "datasets": ["ACL18", "CMIN-US", "CMIN-CN"],
        "splits": ["Test"],
        "presets": ["LLM-only", "MarketToM-NoCEP", "MarketToM-1st", "MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "train_first": True,
    },
    "tom_counterfactual": {
        "datasets": ["ACL18", "CMIN-US", "CMIN-CN"],
        "splits": ["Test"],
        "presets": ["MarketToM-2nd", "MarketToM-1st", "MarketToM-RoleShuffled"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "train_first": True,
    },
    "ccn_dependency_ablation": {
        "datasets": ["ACL18", "CMIN-US", "CMIN-CN"],
        "splits": ["Test"],
        "presets": ["MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": [
            "full",
            "no_belief_to_intent",
            "no_belief_to_emotion",
            "no_intent_emotion_to_action",
            "shuffled_belief_parents",
        ],
        "train_first": True,
    },
    "prompt_format_sensitivity": {
        "datasets": ["ACL18", "CMIN-US", "CMIN-CN"],
        "splits": ["Test"],
        "presets": ["MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "prompt_formats": ["xml", "markdown", "plain"],
        "train_first": False,
    },
    "robustness": {
        "datasets": ["StockNet-EntityMasked", "StockNet-SemPerturb"],
        "splits": ["Train"],
        "presets": ["MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "prompt_formats": ["xml"],
        "train_first": False,
    },
    "entity_masking": {
        "datasets": [
            "StockNet", "StockNet-EntityMasked",
            "CMIN-US", "CMIN-US-EntityMasked",
            "CMIN-CN", "CMIN-CN-EntityMasked",
        ],
        "splits": ["Test"],
        "presets": ["MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "prompt_formats": ["xml"],
        "train_first": True,
    },
    "semantic_perturbation": {
        "datasets": ["StockNet", "StockNet-SemPerturb"],
        "splits": ["Train"],
        "presets": ["MarketToM-2nd"],
        "models": ["qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "prompt_formats": ["xml"],
        "train_first": False,
    },
    "temporal_holdout": {
        "datasets": ["Holdout2025"],
        "splits": [""],
        "presets": ["MarketToM-2nd"],
        "models": ["gpt4o", "qwen2.5-72b"],
        "ccn_dependency_variants": ["full"],
        "prompt_formats": ["xml"],
        "cep_update_mode": "online",
        "train_first": False,
    }
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched MarketToM main-model experiments")
    parser.add_argument("--task", default="main", help="Batch experiment task")
    parser.add_argument("--config", default="config.json", help="Base config file")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--presets", nargs="+", default=None)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--ccn-dependency-variants", nargs="+", default=None)
    parser.add_argument("--prompt-formats", nargs="+", default=None,
                        choices=["xml", "markdown", "plain"])
    parser.add_argument("--prepare-robustness-data", action="store_true")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--stocks-per-run", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stock-pools-file", default=None)
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--skip-train-first", action="store_true")
    parser.add_argument("--combine-stocks-in-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--cep-update-mode", default=None)
    return parser.parse_args(argv)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def configured_task_defaults(base_config: Dict) -> Dict:
    tasks = copy.deepcopy(TASK_DEFAULTS)
    for task_name, task_config in base_config.get("experiment_tasks", {}).items():
        if not isinstance(task_config, dict):
            continue
        if task_name in tasks:
            deep_update(tasks[task_name], task_config)
        else:
            tasks[task_name] = copy.deepcopy(task_config)
    return tasks


def normalize_dataset_name(name: str) -> str:
    if name not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{name}'. Available aliases: {', '.join(sorted(DATASET_ALIASES))}")
    return DATASET_ALIASES[name]


def normalize_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "default"


def transform_prompt_template(template_text: str, prompt_format: str) -> str:
    if prompt_format == "xml":
        return template_text
    if prompt_format == "markdown":
        text = re.sub(r"<([A-Za-z][A-Za-z0-9_-]*)>", r"\n### \1\n", template_text)
        text = re.sub(r"</([A-Za-z][A-Za-z0-9_-]*)>", "\n", text)
        text = re.sub(r"<([A-Za-z][A-Za-z0-9_-]*)\s*/>", r"\n### \1\nN/A\n", text)
        return text.strip() + "\n"
    text = re.sub(r"<[^>]+>", "\n", template_text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def prepare_prompt_templates(base_config: Dict, batch_name: str, prompt_format: str) -> Dict[str, str]:
    if prompt_format == "xml":
        return {}
    generated_dir = PROJECT_ROOT / "runtime_storage" / "batch_prompt_templates" / batch_name / prompt_format
    generated_dir.mkdir(parents=True, exist_ok=True)
    template_overrides = {}
    for key in ["forward_inference", "expert_action_probability", "backward_inference"]:
        source_value = base_config.get("templates", {}).get(key)
        if not source_value:
            continue
        source_path = resolve_path(source_value)
        target_path = generated_dir / f"{key}_{prompt_format}.txt"
        template_text = source_path.read_text(encoding="utf-8")
        target_path.write_text(transform_prompt_template(template_text, prompt_format), encoding="utf-8")
        template_overrides[key] = str(target_path.relative_to(PROJECT_ROOT))
    return template_overrides


def prepare_robustness_data(datasets=None) -> None:
    entity_masking.derive_all(datasets or ["StockNet"], overwrite=True)


def prepare_holdout_aligned() -> None:
    source_root = PROJECT_ROOT / "data" / "unseen_data"
    target_root = PROJECT_ROOT / "data" / "unseen_data_aligned"
    if target_root.exists():
        return
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    placeholder = "No relevant news on this day"
    for stock_dir in sorted(source_root.iterdir()):
        if not stock_dir.is_dir():
            continue
        price_path = stock_dir / "price_data.json"
        labels_path = stock_dir / "labels.json"
        text_path = stock_dir / "text_data.json"
        if not (price_path.exists() and labels_path.exists() and text_path.exists()):
            continue
        price = json.loads(price_path.read_text(encoding="utf-8"))
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        text = json.loads(text_path.read_text(encoding="utf-8"))
        price_rows = price.get("price_data", [])
        label_rows = labels.get("labels", [])
        base_days = {row["day"] for row in price_rows} & {row["day"] for row in label_rows}
        if not base_days:
            continue
        aligned_price = [row for row in price_rows if row["day"] in base_days]
        aligned_labels = [row for row in label_rows if row["day"] in base_days]
        aligned_text = {}
        for day in base_days:
            day_text = text.get(day)
            aligned_text[day] = day_text if day_text else {"news1": {"content": placeholder}}
        target_stock = target_root / stock_dir.name
        target_stock.mkdir(parents=True, exist_ok=True)
        (target_stock / "price_data.json").write_text(json.dumps({"price_data": aligned_price}, ensure_ascii=False, indent=2), encoding="utf-8")
        (target_stock / "labels.json").write_text(json.dumps({"labels": aligned_labels}, ensure_ascii=False, indent=2), encoding="utf-8")
        (target_stock / "text_data.json").write_text(json.dumps(aligned_text, ensure_ascii=False, indent=2), encoding="utf-8")


def list_available_stocks(dataset_name: str, split: str) -> List[str]:
    split_dir = PROJECT_ROOT / "data" / dataset_name / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    stocks = []
    for item in sorted(split_dir.iterdir()):
        if not item.is_dir():
            continue
        if item.name.endswith("_backup"):
            continue
        required = [item / "text_data.json", item / "price_data.json", item / "labels.json"]
        if all(path.exists() for path in required):
            stocks.append(item.name)
    if not stocks:
        raise ValueError(f"No valid stock folders found in {split_dir}")
    return stocks


def load_stock_pools(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    raw = load_json(resolve_path(path))
    return {normalize_dataset_name(k): list(v) for k, v in raw.items()}


def choose_stocks(dataset_name: str, splits: List[str], seed: int, stocks_per_run: int, pools: Dict[str, List[str]]) -> List[str]:
    if pools.get(dataset_name):
        available = set(pools[dataset_name])
    else:
        available = None
        for split in splits:
            split_stocks = set(list_available_stocks(dataset_name, split))
            available = split_stocks if available is None else available & split_stocks
        available = available or set()
    available = sorted(available)
    if stocks_per_run <= 0 or stocks_per_run >= len(available):
        return available
    rng = random.Random(seed)
    return sorted(rng.sample(available, stocks_per_run))


def stock_digest(stocks: List[str]) -> str:
    joined = ",".join(stocks)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def model_label(base_config: Dict, provider: Optional[str], model: Optional[str]) -> str:
    if model:
        return safe_name(normalize_model_name(model))
    active_provider = provider or base_config.get("api", {}).get("active_llm_provider", "openai")
    provider_cfg = base_config.get("api", {}).get("providers", {}).get(active_provider, {})
    return safe_name(provider_cfg.get("llm_model_default", active_provider))


def build_config(
    base_config: Dict,
    dataset_name: str,
    split: str,
    stocks: List[str],
    prediction_log: Path,
    inference_logs: Path,
    strategy_database: Path,
    provider: Optional[str],
    model: Optional[str],
    max_samples: Optional[int],
    ccn_dependency_variant: str,
    prompt_format: str,
    template_overrides: Dict[str, str],
    cep_update_mode: Optional[str] = None,
) -> Dict:
    config = copy.deepcopy(base_config)
    config.setdefault("directories", {})
    config["directories"]["prediction_log_path"] = str(prediction_log.relative_to(PROJECT_ROOT))
    config["directories"]["inference_logs"] = str(inference_logs.relative_to(PROJECT_ROOT))
    config["directories"]["strategy_database"] = str(strategy_database.relative_to(PROJECT_ROOT))
    config.setdefault("data_params", {})
    config["data_params"]["dataset_name"] = dataset_name
    config["data_params"]["dataset_split"] = split
    config["data_params"]["default_stocks"] = stocks
    config["data_params"].setdefault("default_window_size", 5)
    if cep_update_mode:
        config["data_params"]["cep_update_mode"] = cep_update_mode
    else:
        config["data_params"].setdefault("cep_update_mode", "benchmark")
    config.setdefault("ablation", {})
    config["ablation"]["ccn_dependency_variant"] = ccn_dependency_variant
    config["ablation"]["prompt_format"] = prompt_format
    if template_overrides:
        config.setdefault("templates", {}).update(template_overrides)
    if max_samples is not None:
        config["data_params"]["max_samples"] = max_samples
    if provider:
        config.setdefault("api", {})["active_llm_provider"] = provider
    active_provider = config.get("api", {}).get("active_llm_provider", "openai")
    if model:
        config.setdefault("api", {}).setdefault("providers", {}).setdefault(active_provider, {})["llm_model_default"] = normalize_model_name(model)
    return config


def calculate_metrics(predictions: List[Dict]) -> Dict:
    tp = tn = fp = fn = 0
    for pred in predictions:
        predicted_up = bool(pred.get("predicted_up"))
        label = int(pred.get("label", 0))
        if predicted_up and label == 1:
            tp += 1
        elif not predicted_up and label == 0:
            tn += 1
        elif predicted_up and label == 0:
            fp += 1
        elif not predicted_up and label == 1:
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "n": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calculate_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "n": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def read_prediction_metrics(prediction_log: Path) -> Dict:
    if not prediction_log.exists():
        return calculate_metrics([])
    data = load_json(prediction_log)
    return calculate_metrics(data.get("predictions", []))


def run_command(python_executable: str, config_path: Path, preset: str, quiet: bool, dry_run: bool, log_path: Path) -> int:
    command = [python_executable, "run.py", "--experiment", "single", "--config", str(config_path), "--preset", preset]
    print(" ".join(command))
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if quiet:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def write_summary(summary_path: Path, rows: List[Dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "batch_name", "dataset", "split", "preset", "model", "ccn_dependency_variant", "prompt_format", "seed", "stock_group", "stocks", "returncode",
        "n", "accuracy", "precision", "recall", "f1", "mcc", "tp", "tn", "fp", "fn",
        "prediction_log", "config_path", "run_log",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def metric_std(values: List[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def write_aggregate_summary(summary_path: Path, rows: List[Dict]) -> None:
    groups = {}
    for row in rows:
        if row.get("returncode") not in (0, "0"):
            continue
        key = (row.get("dataset"), row.get("split"), row.get("preset"), row.get("model"), row.get("ccn_dependency_variant"), row.get("prompt_format"))
        groups.setdefault(key, []).append(row)
    fields = [
        "dataset", "split", "preset", "model", "ccn_dependency_variant", "prompt_format", "runs", "total_n",
        "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
        "recall_mean", "recall_std", "f1_mean", "f1_std", "mcc_mean", "mcc_std",
        "pooled_accuracy", "pooled_precision", "pooled_recall", "pooled_f1", "pooled_mcc",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, group_rows in sorted(groups.items()):
            metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1", "mcc"]:
                values = [float(row.get(metric, 0.0)) for row in group_rows]
                metrics[f"{metric}_mean"] = mean(values) if values else 0.0
                metrics[f"{metric}_std"] = metric_std(values)
            pooled = calculate_metrics_from_counts(
                tp=sum(int(row.get("tp", 0)) for row in group_rows),
                tn=sum(int(row.get("tn", 0)) for row in group_rows),
                fp=sum(int(row.get("fp", 0)) for row in group_rows),
                fn=sum(int(row.get("fn", 0)) for row in group_rows),
            )
            writer.writerow({
                "dataset": key[0],
                "split": key[1],
                "preset": key[2],
                "model": key[3],
                "ccn_dependency_variant": key[4],
                "prompt_format": key[5],
                "runs": len(group_rows),
                "total_n": sum(int(row.get("n", 0)) for row in group_rows),
                **metrics,
                "pooled_accuracy": pooled["accuracy"],
                "pooled_precision": pooled["precision"],
                "pooled_recall": pooled["recall"],
                "pooled_f1": pooled["f1"],
                "pooled_mcc": pooled["mcc"],
            })


def write_subset_summary(summary_path: Path, rows: List[Dict]) -> None:
    seed_groups = {}
    for row in rows:
        if row.get("returncode") not in (0, "0"):
            continue
        key = (
            row.get("dataset"), row.get("split"), row.get("preset"), row.get("model"),
            row.get("ccn_dependency_variant"), row.get("prompt_format"), row.get("seed"),
        )
        seed_groups.setdefault(key, []).append(row)

    subset_metrics = {}
    for key, group_rows in seed_groups.items():
        base_key = key[:6]
        pooled = calculate_metrics_from_counts(
            tp=sum(int(row.get("tp", 0)) for row in group_rows),
            tn=sum(int(row.get("tn", 0)) for row in group_rows),
            fp=sum(int(row.get("fp", 0)) for row in group_rows),
            fn=sum(int(row.get("fn", 0)) for row in group_rows),
        )
        subset_metrics.setdefault(base_key, []).append(pooled)

    fields = [
        "dataset", "split", "preset", "model", "ccn_dependency_variant", "prompt_format", "subsets",
        "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
        "recall_mean", "recall_std", "f1_mean", "f1_std", "mcc_mean", "mcc_std",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for base_key, metric_list in sorted(subset_metrics.items()):
            record = {
                "dataset": base_key[0],
                "split": base_key[1],
                "preset": base_key[2],
                "model": base_key[3],
                "ccn_dependency_variant": base_key[4],
                "prompt_format": base_key[5],
                "subsets": len(metric_list),
            }
            for metric in ["accuracy", "precision", "recall", "f1", "mcc"]:
                values = [item[metric] for item in metric_list]
                record[f"{metric}_mean"] = mean(values) if values else 0.0
                record[f"{metric}_std"] = metric_std(values)
            writer.writerow(record)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} ± {std_value:.4f}"


def format_mean_std_tex(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"


def latex_escape(value: str) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write_main_results_table(output_dir: Path, rows: List[Dict]) -> None:
    groups = {}
    for row in rows:
        if row.get("returncode") not in (0, "0"):
            continue
        key = (row.get("dataset"), row.get("model"), row.get("preset"), row.get("ccn_dependency_variant"), row.get("prompt_format"))
        groups.setdefault(key, []).append(row)

    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    csv_path = output_dir / "main_results_table.csv"
    tex_path = output_dir / "main_results_table.tex"
    csv_fields = ["dataset", "model", "preset", "ccn_dependency_variant", "prompt_format", "runs", "seeds", "stock_groups", "total_n", *metrics]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        table_rows = []
        for (dataset, model, preset, ccn_dependency_variant, prompt_format), group_rows in sorted(groups.items()):
            result_row = {
                "dataset": dataset,
                "model": model,
                "preset": preset,
                "ccn_dependency_variant": ccn_dependency_variant,
                "prompt_format": prompt_format,
                "runs": len(group_rows),
                "seeds": len({row.get("seed") for row in group_rows}),
                "stock_groups": len({(row.get("seed"), row.get("stock_group")) for row in group_rows}),
                "total_n": sum(int(row.get("n", 0)) for row in group_rows),
            }
            for metric in metrics:
                values = [float(row.get(metric, 0.0)) for row in group_rows]
                result_row[metric] = format_mean_std(mean(values), metric_std(values)) if values else format_mean_std(0.0, 0.0)
            table_rows.append(result_row)
            writer.writerow(result_row)

    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Main MarketToM results across datasets and LLM backbones.}\n")
        f.write("\\label{tab:main_markettom_batch_results}\n")
        f.write("\\begin{tabular}{lllllrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Model & Preset & CCN Variant & Prompt & ACC & Precision & Recall & F1 & MCC \\\\\n")
        f.write("\\midrule\n")
        for row in table_rows:
            tex_values = {}
            for metric in metrics:
                metric_values = [float(group_row.get(metric, 0.0)) for group_row in groups[(row["dataset"], row["model"], row["preset"], row["ccn_dependency_variant"], row["prompt_format"])]]
                tex_values[metric] = format_mean_std_tex(mean(metric_values), metric_std(metric_values)) if metric_values else format_mean_std_tex(0.0, 0.0)
            f.write(
                f"{latex_escape(row['dataset'])} & {latex_escape(row['model'])} & "
                f"{latex_escape(row['preset'])} & {latex_escape(row['ccn_dependency_variant'])} & "
                f"{latex_escape(row['prompt_format'])} & {tex_values['accuracy']} & "
                f"{tex_values['precision']} & {tex_values['recall']} & {tex_values['f1']} & {tex_values['mcc']} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def write_stock_sampling_plan(output_dir: Path, rows: List[Dict]) -> None:
    seen = set()
    plan_rows = []
    for row in rows:
        key = (row.get("dataset"), row.get("seed"), row.get("stock_group"))
        if key in seen:
            continue
        seen.add(key)
        stocks = [stock for stock in str(row.get("stock_group", "")).split(",") if stock]
        plan_rows.append({
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "stocks_per_run": len(stocks),
            "stocks": ",".join(stocks),
        })

    path = output_dir / "stock_sampling_plan.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "stocks_per_run", "stocks"])
        writer.writeheader()
        for row in sorted(plan_rows, key=lambda item: (str(item["dataset"]), int(item["seed"]))):
            writer.writerow(row)


def run_stage(
    base_config: Dict,
    batch_name: str,
    dataset_name: str,
    split: str,
    preset: str,
    model: Optional[str],
    provider: Optional[str],
    seed: int,
    ccn_dependency_variant: str,
    prompt_format: str,
    template_overrides: Dict[str, str],
    stocks: List[str],
    stock_group: List[str],
    strategy_database: Path,
    python_executable: str,
    max_samples: Optional[int],
    quiet: bool,
    dry_run: bool,
    force: bool,
    cep_update_mode: Optional[str] = None,
) -> Dict:
    model_name = model_label(base_config, provider, model)
    group_id = f"{safe_name(dataset_name)}__{safe_name(preset)}__{model_name}__{safe_name(ccn_dependency_variant)}__{safe_name(prompt_format)}__seed{seed}__{stock_digest(stocks)}"
    run_id = f"{group_id}__{safe_name(split)}"
    output_dir = PROJECT_ROOT / "outputs" / "metrics" / "batch_experiments" / batch_name / run_id
    runtime_dir = PROJECT_ROOT / "runtime_storage" / "batch_experiments" / batch_name / run_id
    prediction_log = output_dir / "prediction_results.json"
    inference_logs = runtime_dir / "inference_logs"
    config_path = PROJECT_ROOT / "runtime_storage" / "batch_configs" / batch_name / f"{run_id}.json"
    log_path = output_dir / "run.log"
    config = build_config(base_config, dataset_name, split, stocks, prediction_log, inference_logs, strategy_database, provider, model, max_samples, ccn_dependency_variant, prompt_format, template_overrides, cep_update_mode)
    if prediction_log.exists() and force:
        prediction_log.unlink()
    save_json(config_path, config)
    returncode = run_command(python_executable, config_path, preset, quiet, dry_run, log_path)
    metrics = read_prediction_metrics(prediction_log)
    row = {
        "batch_name": batch_name,
        "dataset": dataset_name,
        "split": split,
        "preset": preset,
        "model": model_name,
        "ccn_dependency_variant": ccn_dependency_variant,
        "prompt_format": prompt_format,
        "seed": seed,
        "stock_group": ",".join(stock_group),
        "stocks": ",".join(stocks),
        "returncode": returncode,
        "prediction_log": str(prediction_log.relative_to(PROJECT_ROOT)),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "run_log": str(log_path.relative_to(PROJECT_ROOT)),
    }
    row.update(metrics)
    return row


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    base_config = load_json(resolve_path(args.config))
    task_defaults_map = configured_task_defaults(base_config)
    if args.task not in task_defaults_map:
        available = ", ".join(sorted(task_defaults_map))
        raise ValueError(f"Unknown batch task '{args.task}'. Available: {available}")
    task_defaults = task_defaults_map[args.task]
    batch_name = args.batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "metrics" / "batch_experiments" / batch_name
    pools = load_stock_pools(args.stock_pools_file)
    rows = []
    datasets = [normalize_dataset_name(dataset) for dataset in (args.datasets or task_defaults["datasets"])]
    splits = args.splits or task_defaults["splits"]
    presets = args.presets or task_defaults["presets"]
    models = args.models or task_defaults["models"] or [None]
    ccn_dependency_variants = args.ccn_dependency_variants or task_defaults["ccn_dependency_variants"]
    prompt_formats = args.prompt_formats or task_defaults.get("prompt_formats", ["xml"])
    train_first = args.train_first or (task_defaults.get("train_first", False) and not args.skip_train_first)
    cep_update_mode = args.cep_update_mode or task_defaults.get("cep_update_mode")
    if args.prepare_robustness_data or args.task == "robustness":
        prepare_robustness_data()
    if args.task == "entity_masking":
        base_for_masking = sorted({
            name[: -len(entity_masking.MASKED_SUFFIX)]
            for name in datasets
            if name.endswith(entity_masking.MASKED_SUFFIX)
        })
        entity_masking.derive_all(base_for_masking, overwrite=True)
        if not pools:
            pools = {normalize_dataset_name(k): v for k, v in entity_masking.masked_stock_pools().items()}
    if args.task == "semantic_perturbation" and not pools:
        perturb_stocks = list_available_stocks("StockNet_SemPerturb", "Train")
        pools = {"StockNet": perturb_stocks, "StockNet_SemPerturb": perturb_stocks}
    if args.task == "temporal_holdout":
        prepare_holdout_aligned()
    for dataset_name in datasets:
        for seed in args.seeds:
            required_splits = list(dict.fromkeys((["Train"] if train_first else []) + splits))
            stocks = choose_stocks(dataset_name, required_splits, seed, args.stocks_per_run, pools)
            run_stock_sets = [stocks] if args.combine_stocks_in_run else [[stock] for stock in stocks]
            for preset in presets:
                for model in models:
                    model_name = model_label(base_config, args.provider, model)
                    for ccn_dependency_variant in ccn_dependency_variants:
                        for prompt_format in prompt_formats:
                            template_overrides = prepare_prompt_templates(base_config, batch_name, prompt_format)
                            for run_stocks in run_stock_sets:
                                strategy_id = f"{safe_name(dataset_name)}__{safe_name(preset)}__{model_name}__{safe_name(ccn_dependency_variant)}__{safe_name(prompt_format)}__seed{seed}__{stock_digest(run_stocks)}"
                                strategy_database = PROJECT_ROOT / "runtime_storage" / "batch_experiments" / batch_name / strategy_id / "strategy_database"
                                if train_first:
                                    rows.append(run_stage(base_config, batch_name, dataset_name, "Train", preset, model, args.provider, seed, ccn_dependency_variant, prompt_format, template_overrides, run_stocks, stocks, strategy_database, args.python, args.max_samples, args.quiet, args.dry_run, args.force, cep_update_mode))
                                for split in splits:
                                    if train_first and split == "Train":
                                        continue
                                    rows.append(run_stage(base_config, batch_name, dataset_name, split, preset, model, args.provider, seed, ccn_dependency_variant, prompt_format, template_overrides, run_stocks, stocks, strategy_database, args.python, args.max_samples, args.quiet, args.dry_run, args.force, cep_update_mode))
                                summary_path = output_dir / "summary.csv"
                                write_summary(summary_path, rows)
                                write_aggregate_summary(summary_path.with_name("aggregate_summary.csv"), rows)
                                write_subset_summary(summary_path.with_name("subset_summary.csv"), rows)
                                write_stock_sampling_plan(output_dir, rows)
                                write_main_results_table(output_dir, rows)
                                save_json(summary_path.with_suffix(".json"), {"runs": rows})
    final_summary = output_dir / "summary.csv"
    print(f"Batch summary written to: {final_summary}")


if __name__ == "__main__":
    main()
