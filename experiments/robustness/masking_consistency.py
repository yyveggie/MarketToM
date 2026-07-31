import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASKED_SUFFIX = "_EntityMasked"
PAIR_KEYS = ("preset", "model", "ccn_dependency_variant", "prompt_format", "seed", "stocks", "split")
TABLE_FIELDS = [
    "dataset", "pairs", "samples",
    "unmasked_acc", "unmasked_f1", "unmasked_mcc",
    "masked_acc", "masked_f1", "masked_mcc",
    "vector_similarity", "wasserstein_mean", "wasserstein_std",
]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_runs(summary_path):
    data = json.loads(resolve_path(summary_path).read_text(encoding="utf-8"))
    return data.get("runs", [])


def load_predictions(prediction_log):
    path = resolve_path(prediction_log)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for item in data.get("predictions", []):
        if "index" in item:
            records[item["index"]] = item
    return records


def base_name(dataset):
    if dataset.endswith(MASKED_SUFFIX):
        return dataset[: -len(MASKED_SUFFIX)]
    return dataset


def confusion_metrics(pairs):
    tp = tn = fp = fn = 0
    for predicted_up, label in pairs:
        if predicted_up and label == 1:
            tp += 1
        elif not predicted_up and label == 0:
            tn += 1
        elif predicted_up and label == 0:
            fp += 1
        else:
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"accuracy": accuracy, "f1": f1, "mcc": mcc}


def cosine_similarity(left, right):
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def collect_pairs(runs):
    grouped = defaultdict(dict)
    for row in runs:
        if str(row.get("returncode")) != "0":
            continue
        dataset = row.get("dataset", "")
        role = "masked" if dataset.endswith(MASKED_SUFFIX) else "unmasked"
        key = (base_name(dataset),) + tuple(str(row.get(field, "")) for field in PAIR_KEYS)
        grouped[key][role] = row
    return grouped


def evaluate(runs, include_splits):
    grouped = collect_pairs(runs)
    per_dataset = defaultdict(lambda: {
        "labels": [],
        "preds_unmasked": [],
        "preds_masked": [],
        "probs_unmasked": [],
        "probs_masked": [],
        "wasserstein": [],
        "pairs": 0,
        "samples": 0,
    })
    for sides in grouped.values():
        if "masked" not in sides or "unmasked" not in sides:
            continue
        reference = sides["unmasked"]
        if include_splits and reference.get("split", "") not in include_splits:
            continue
        unmasked_records = load_predictions(reference["prediction_log"])
        masked_records = load_predictions(sides["masked"]["prediction_log"])
        common = sorted(set(unmasked_records) & set(masked_records))
        if not common:
            continue
        bucket = per_dataset[base_name(reference["dataset"])]
        group_unmasked = []
        group_masked = []
        for index in common:
            unmasked = unmasked_records[index]
            masked = masked_records[index]
            label = int(unmasked.get("label", 0))
            unmasked_prob = float(unmasked.get("probability", 0.0))
            masked_prob = float(masked.get("probability", 0.0))
            bucket["labels"].append(label)
            bucket["preds_unmasked"].append(bool(unmasked.get("predicted_up")))
            bucket["preds_masked"].append(bool(masked.get("predicted_up")))
            bucket["probs_unmasked"].append(unmasked_prob)
            bucket["probs_masked"].append(masked_prob)
            group_unmasked.append(unmasked_prob)
            group_masked.append(masked_prob)
        if len(group_unmasked) >= 2:
            bucket["wasserstein"].append(float(wasserstein_distance(group_unmasked, group_masked)))
        bucket["pairs"] += 1
        bucket["samples"] += len(common)
    return per_dataset


def build_summary(per_dataset):
    summary = []
    for dataset in sorted(per_dataset.keys()):
        bucket = per_dataset[dataset]
        unmasked = confusion_metrics(list(zip(bucket["preds_unmasked"], bucket["labels"])))
        masked = confusion_metrics(list(zip(bucket["preds_masked"], bucket["labels"])))
        wasserstein_values = bucket["wasserstein"]
        summary.append({
            "dataset": dataset,
            "pairs": bucket["pairs"],
            "samples": bucket["samples"],
            "unmasked_acc": unmasked["accuracy"],
            "unmasked_f1": unmasked["f1"],
            "unmasked_mcc": unmasked["mcc"],
            "masked_acc": masked["accuracy"],
            "masked_f1": masked["f1"],
            "masked_mcc": masked["mcc"],
            "vector_similarity": cosine_similarity(bucket["probs_unmasked"], bucket["probs_masked"]),
            "wasserstein_mean": mean(wasserstein_values) if wasserstein_values else 0.0,
            "wasserstein_std": stdev(wasserstein_values) if len(wasserstein_values) > 1 else 0.0,
        })
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in TABLE_FIELDS})


def write_tex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Consistency of MarketToM Predictions With and Without Entity Masking}",
        "\\begin{tabular}{lllllc}",
        "\\toprule",
        "Dataset & Version & ACC & F1 & MCC & Vec. Sim. / Wass. \\\\",
        "\\midrule",
    ]
    for row in rows:
        consistency = (
            f"{row['vector_similarity']:.4f} / "
            f"{row['wasserstein_mean']:.4f} $\\pm$ {row['wasserstein_std']:.4f}"
        )
        lines.append(
            f"{row['dataset']} & Masked & {row['masked_acc']:.4f} & "
            f"{row['masked_f1']:.4f} & {row['masked_mcc']:.4f} & "
            f"\\multirow{{2}}{{*}}{{{consistency}}} \\\\"
        )
        lines.append(
            f" & Unmasked & {row['unmasked_acc']:.4f} & "
            f"{row['unmasked_f1']:.4f} & {row['unmasked_mcc']:.4f} & \\\\"
        )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Entity-masking prediction consistency metrics")
    parser.add_argument("--summary", default="outputs/metrics/batch_experiments/summary.json")
    parser.add_argument("--output-dir", default="outputs/metrics/entity_masking")
    parser.add_argument("--include-splits", nargs="*", default=["Test"])
    args = parser.parse_args(argv)
    runs = load_runs(args.summary)
    per_dataset = evaluate(runs, set(args.include_splits) if args.include_splits else None)
    summary = build_summary(per_dataset)
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "masking_consistency.csv", summary)
    write_tex(output_dir / "masking_consistency.tex", summary)
    print(f"Entity-masking consistency written to: {output_dir}")
    for row in summary:
        print(
            f"{row['dataset']}: pairs={row['pairs']} samples={row['samples']} "
            f"vec_sim={row['vector_similarity']:.4f} "
            f"wass={row['wasserstein_mean']:.4f}±{row['wasserstein_std']:.4f}"
        )


if __name__ == "__main__":
    main()
