import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERTURB_SUFFIX = "_SemPerturb"
GROUP_KEYS = ("preset", "model", "ccn_dependency_variant", "prompt_format")
METRICS = ("accuracy", "precision", "recall", "f1", "mcc")
NEAR_ZERO = 0.01
CSV_FIELDS = (
    ["dataset", "preset", "model", "ccn_dependency_variant", "prompt_format", "original_n", "perturbed_n"]
    + [f"original_{metric}" for metric in METRICS]
    + [f"perturbed_{metric}" for metric in METRICS]
    + [f"delta_{metric}" for metric in METRICS]
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_runs(summary_path):
    data = json.loads(resolve_path(summary_path).read_text(encoding="utf-8"))
    return data.get("runs", [])


def load_prediction_pairs(prediction_log):
    path = resolve_path(prediction_log)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for item in data.get("predictions", []):
        pairs.append((bool(item.get("predicted_up")), int(item.get("label", 0))))
    return pairs


def base_name(dataset):
    if dataset.endswith(PERTURB_SUFFIX):
        return dataset[: -len(PERTURB_SUFFIX)]
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
    return {"n": total, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def relative_delta(original, perturbed):
    if abs(original) < NEAR_ZERO:
        return None
    return (perturbed - original) / abs(original) * 100.0


def collect(runs, include_splits):
    grouped = defaultdict(lambda: {"original": [], "perturbed": []})
    for row in runs:
        if str(row.get("returncode")) != "0":
            continue
        if include_splits and row.get("split", "") not in include_splits:
            continue
        dataset = row.get("dataset", "")
        role = "perturbed" if dataset.endswith(PERTURB_SUFFIX) else "original"
        key = (base_name(dataset),) + tuple(str(row.get(field, "")) for field in GROUP_KEYS)
        grouped[key][role].extend(load_prediction_pairs(row.get("prediction_log", "")))
    return grouped


def build_summary(grouped):
    summary = []
    for key, sides in sorted(grouped.items()):
        if not sides["original"] or not sides["perturbed"]:
            continue
        original = confusion_metrics(sides["original"])
        perturbed = confusion_metrics(sides["perturbed"])
        row = {
            "dataset": key[0],
            "preset": key[1],
            "model": key[2],
            "ccn_dependency_variant": key[3],
            "prompt_format": key[4],
            "original_n": original["n"],
            "perturbed_n": perturbed["n"],
        }
        for metric in METRICS:
            row[f"original_{metric}"] = original[metric]
            row[f"perturbed_{metric}"] = perturbed[metric]
            row[f"delta_{metric}"] = relative_delta(original[metric], perturbed[metric])
        summary.append(row)
    return summary


def format_delta(value):
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {}
            for field in CSV_FIELDS:
                value = row.get(field)
                if field.startswith("delta_"):
                    record[field] = format_delta(value)
                else:
                    record[field] = value
            writer.writerow(record)


def write_tex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\caption{Stress test results on the ACL18 dataset with semantic perturbations.}",
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Model & Preset & ACC ($\\Delta$) & Precision ($\\Delta$) & Recall ($\\Delta$) & F1 ($\\Delta$) & MCC ($\\Delta$) \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [row["model"], row["preset"]]
        for metric in METRICS:
            cells.append(f"{row[f'perturbed_{metric}']:.4f} ({format_delta(row[f'delta_{metric}'])})")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semantic-perturbation stress-test delta metrics")
    parser.add_argument("--summary", default="outputs/metrics/batch_experiments/summary.json")
    parser.add_argument("--output-dir", default="outputs/metrics/semantic_perturbation")
    parser.add_argument("--include-splits", nargs="*", default=["Train"])
    args = parser.parse_args(argv)
    runs = load_runs(args.summary)
    grouped = collect(runs, set(args.include_splits) if args.include_splits else None)
    summary = build_summary(grouped)
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "perturbation_delta.csv", summary)
    write_tex(output_dir / "perturbation_delta.tex", summary)
    print(f"Semantic-perturbation delta written to: {output_dir}")
    for row in summary:
        print(
            f"{row['dataset']} {row['preset']} {row['model']}: "
            f"ACC {row['perturbed_accuracy']:.4f} ({format_delta(row['delta_accuracy'])}), "
            f"MCC {row['perturbed_mcc']:.4f} ({format_delta(row['delta_mcc'])})"
        )


if __name__ == "__main__":
    main()
