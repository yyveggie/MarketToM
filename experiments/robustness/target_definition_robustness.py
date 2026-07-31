import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_KEYS = ("preset", "model", "ccn_dependency_variant", "prompt_format", "seed")
METRICS = ("accuracy", "precision", "recall", "f1", "mcc")
TARGETS = ("raw", "tc")
DATASET_DISPLAY = {
    "StockNet": "ACL18",
    "ACL18": "ACL18",
    "CMIN_US": "CMIN-US",
    "CMIN-US": "CMIN-US",
    "CMIN_CN": "CMIN-CN",
    "CMIN-CN": "CMIN-CN",
}
DATASET_ORDER = ["StockNet", "CMIN_US", "CMIN_CN"]
TARGET_DISPLAY = {"raw": "Raw direction", "tc": "10 bps adjusted"}
CSV_FIELDS = (
    ["dataset", "preset", "model", "ccn_dependency_variant", "prompt_format", "seeds"]
    + [f"{target}_{metric}_{stat}" for target in TARGETS for metric in METRICS for stat in ("mean", "std")]
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_runs(summary_path):
    data = json.loads(resolve_path(summary_path).read_text(encoding="utf-8"))
    return data.get("runs", [])


def day_number(day):
    match = re.search(r"\d+", str(day))
    return int(match.group()) if match else 0


def load_close(dataset, split, stock):
    path = PROJECT_ROOT / "data" / dataset / split / stock / "price_data.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = sorted(data.get("price_data", []), key=lambda entry: day_number(entry.get("day")))
    return [float(entry["close"]) for entry in entries]


def load_predictions(prediction_log):
    path = resolve_path(prediction_log)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in data.get("predictions", []):
        if "index" in item:
            records.append((int(item["index"]), bool(item.get("predicted_up"))))
    records.sort(key=lambda pair: pair[0])
    return records


def add_to_counts(counts, predicted_up, label):
    if predicted_up and label == 1:
        counts[0] += 1
    elif not predicted_up and label == 0:
        counts[1] += 1
    elif predicted_up and label == 0:
        counts[2] += 1
    else:
        counts[3] += 1


def run_counts(predictions, close, raw_threshold, tc_threshold):
    counts = {target: [0, 0, 0, 0] for target in TARGETS}
    for index, predicted_up in predictions:
        if index < 1 or index >= len(close) or close[index - 1] == 0:
            continue
        realized = close[index] / close[index - 1] - 1.0
        add_to_counts(counts["raw"], predicted_up, 1 if realized > raw_threshold else 0)
        add_to_counts(counts["tc"], predicted_up, 1 if realized > tc_threshold else 0)
    return counts


def metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"n": total, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}


def metric_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def collect(runs, include_splits, raw_threshold, tc_threshold):
    seed_counts = defaultdict(lambda: {target: [0, 0, 0, 0] for target in TARGETS})
    for row in runs:
        if str(row.get("returncode")) != "0":
            continue
        if include_splits and row.get("split", "") not in include_splits:
            continue
        stock = row.get("stocks", "")
        if not stock or "," in stock:
            continue
        close = load_close(row.get("dataset", ""), row.get("split", ""), stock)
        if not close:
            continue
        predictions = load_predictions(row.get("prediction_log", ""))
        if not predictions:
            continue
        counts = run_counts(predictions, close, raw_threshold, tc_threshold)
        seed_key = (row.get("dataset", ""),) + tuple(str(row.get(field, "")) for field in SEED_KEYS)
        for target in TARGETS:
            for position in range(4):
                seed_counts[seed_key][target][position] += counts[target][position]
    return seed_counts


def build_summary(seed_counts):
    grouped = defaultdict(lambda: {target: [] for target in TARGETS})
    for seed_key, target_counts in seed_counts.items():
        base_key = seed_key[:5]
        for target in TARGETS:
            tp, tn, fp, fn = target_counts[target]
            if tp + tn + fp + fn == 0:
                continue
            grouped[base_key][target].append(metrics_from_counts(tp, tn, fp, fn))
    summary = []
    for base_key, target_metrics in sorted(grouped.items()):
        record = {
            "dataset": base_key[0],
            "preset": base_key[1],
            "model": base_key[2],
            "ccn_dependency_variant": base_key[3],
            "prompt_format": base_key[4],
            "seeds": max(len(target_metrics[target]) for target in TARGETS),
        }
        for target in TARGETS:
            for metric in METRICS:
                values = [item[metric] for item in target_metrics[target]]
                record[f"{target}_{metric}_mean"] = mean(values) if values else 0.0
                record[f"{target}_{metric}_std"] = metric_std(values)
        summary.append(record)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {}
            for field in CSV_FIELDS:
                value = row.get(field)
                record[field] = f"{value:.4f}" if isinstance(value, float) else value
            writer.writerow(record)


def select_tex_rows(rows, model_substring, preset):
    filtered = []
    for row in rows:
        if preset and row.get("preset") != preset:
            continue
        if model_substring and model_substring.lower() not in str(row.get("model", "")).lower():
            continue
        filtered.append(row)
    by_dataset = {}
    for row in filtered:
        dataset = row["dataset"]
        current = by_dataset.get(dataset)
        if current is None:
            by_dataset[dataset] = row
            continue
        prefers = (row.get("ccn_dependency_variant") == "full", row.get("prompt_format") == "xml")
        keeps = (current.get("ccn_dependency_variant") == "full", current.get("prompt_format") == "xml")
        if prefers > keeps:
            by_dataset[dataset] = row
    ordered_datasets = [d for d in DATASET_ORDER if d in by_dataset]
    ordered_datasets += [d for d in sorted(by_dataset) if d not in DATASET_ORDER]
    return [by_dataset[d] for d in ordered_datasets]


def format_cell(row, target, metric):
    return f"{row[f'{target}_{metric}_mean']:.4f}$\\pm${row[f'{target}_{metric}_std']:.4f}"


def write_tex(path, rows, model_substring, preset):
    selected = select_tex_rows(rows, model_substring, preset)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[!ht]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.85}",
        "\\caption{Target-definition robustness analysis using MarketToM (Qwen2.5-72B). "
        "The raw-direction rows reproduce the main Qwen2.5-72B setting; the 10 bps-adjusted "
        "rows use $y_{t+1}^{tc}=\\mathbb{I}(r_{t+1}>0.001)$.}",
        "\\label{tab:target_definition_robustness}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\multicolumn{1}{c}{\\textbf{Dataset}} & \\multicolumn{1}{c}{\\textbf{Target definition}} "
        "& \\textbf{ACC} $\\uparrow$ & \\textbf{Precision} $\\uparrow$ & \\textbf{Recall} $\\uparrow$ "
        "& \\textbf{F1 Score} $\\uparrow$ & \\textbf{MCC} $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for position, row in enumerate(selected):
        display = DATASET_DISPLAY.get(row["dataset"], row["dataset"])
        lines.append(f"\\multirow{{2}}{{*}}{{{display}}}")
        for target in TARGETS:
            cells = " & ".join(format_cell(row, target, metric) for metric in METRICS)
            lines.append(f" & {TARGET_DISPLAY[target]} & {cells} \\\\")
        if position != len(selected) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Target-definition (transaction-cost-adjusted) robustness diagnostic")
    parser.add_argument("--summary", default="outputs/metrics/batch_experiments/summary.json")
    parser.add_argument("--output-dir", default="outputs/metrics/target_definition_robustness")
    parser.add_argument("--include-splits", nargs="*", default=["Test"])
    parser.add_argument("--raw-threshold", type=float, default=0.0,
                        help="Return threshold for the raw directional label y=I(r>threshold).")
    parser.add_argument("--tc-threshold", type=float, default=0.001,
                        help="Return threshold for the cost-adjusted label (0.001 = 10 bps).")
    parser.add_argument("--tex-model", default="Qwen2.5-72B",
                        help="Substring filter selecting which model appears in the LaTeX table.")
    parser.add_argument("--tex-preset", default="MarketToM-2nd",
                        help="Preset selected for the LaTeX table.")
    args = parser.parse_args(argv)

    runs = load_runs(args.summary)
    seed_counts = collect(runs, set(args.include_splits) if args.include_splits else None,
                          args.raw_threshold, args.tc_threshold)
    summary = build_summary(seed_counts)
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "target_definition_robustness.csv", summary)
    write_tex(output_dir / "target_definition_robustness.tex", summary, args.tex_model, args.tex_preset)
    print(f"Target-definition robustness written to: {output_dir}")
    for row in summary:
        print(
            f"{row['dataset']} {row['preset']} {row['model']}: "
            f"raw ACC {row['raw_accuracy_mean']:.4f} MCC {row['raw_mcc_mean']:.4f} | "
            f"tc ACC {row['tc_accuracy_mean']:.4f} MCC {row['tc_mcc_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
