#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def load_prediction_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value) -> int:
    return int(float(value))


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def group_rows(rows: List[Dict], keys: Tuple[str, ...]) -> Dict[Tuple, List[Dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in keys), []).append(row)
    return grouped


def pesaran_timmermann(rows: List[Dict]) -> Dict:
    pairs = [(as_int(row["y_pred"]), as_int(row["y_true"])) for row in rows if row.get("y_pred") != "" and row.get("y_true") != ""]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "pt_stat": 0.0, "p_value": 1.0}
    pred_up = sum(pred for pred, _ in pairs) / n
    true_up = sum(true for _, true in pairs) / n
    accuracy = sum(1 for pred, true in pairs if pred == true) / n
    expected = pred_up * true_up + (1 - pred_up) * (1 - true_up)
    variance = max(expected * (1 - expected) / n, 1e-12)
    z_value = (accuracy - expected) / math.sqrt(variance)
    return {"n": n, "accuracy": accuracy, "pt_stat": z_value, "p_value": normal_two_sided_p(z_value)}


def zero_one_loss(row: Dict) -> float:
    return 0.0 if as_int(row["y_pred"]) == as_int(row["y_true"]) else 1.0


def brier_loss(row: Dict) -> float:
    prob = as_float(row.get("y_prob", "nan"))
    if math.isnan(prob):
        prob = float(as_int(row["y_pred"]))
    true = float(as_int(row["y_true"]))
    return (prob - true) ** 2


def dm_test(reference_rows: List[Dict], baseline_rows: List[Dict], loss_name: str) -> Dict:
    reference_by_sample = {row["sample_id"]: row for row in reference_rows if row.get("sample_id")}
    baseline_by_sample = {row["sample_id"]: row for row in baseline_rows if row.get("sample_id")}
    common_ids = sorted(set(reference_by_sample) & set(baseline_by_sample))
    loss_fn = zero_one_loss if loss_name == "zero_one" else brier_loss
    diffs = [loss_fn(baseline_by_sample[sample_id]) - loss_fn(reference_by_sample[sample_id]) for sample_id in common_ids]
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean_loss_diff": 0.0, "dm_stat": 0.0, "p_value": 1.0}
    mean_diff = sum(diffs) / n
    variance = sum((value - mean_diff) ** 2 for value in diffs) / (n - 1)
    dm_stat = mean_diff / math.sqrt(max(variance / n, 1e-12))
    return {"n": n, "mean_loss_diff": mean_diff, "dm_stat": dm_stat, "p_value": normal_two_sided_p(dm_stat)}


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_pt(rows: List[Dict], output_dir: Path) -> None:
    result_rows = []
    for (dataset, model), model_rows in sorted(group_rows(rows, ("dataset", "model")).items()):
        result = pesaran_timmermann(model_rows)
        result_rows.append({
            "dataset": dataset,
            "model": model,
            "n": result["n"],
            "accuracy": f"{result['accuracy']:.6f}",
            "pt_stat": f"{result['pt_stat']:.6f}",
            "p_value": f"{result['p_value']:.6g}",
        })
    write_csv(output_dir / "pesaran_timmermann_results.csv", result_rows, ["dataset", "model", "n", "accuracy", "pt_stat", "p_value"])


def run_dm(rows: List[Dict], output_dir: Path, reference_models: List[str]) -> None:
    pairwise_rows = []
    summary = {}
    grouped = group_rows(rows, ("dataset", "model"))
    for (dataset, reference_model), reference_rows in sorted(grouped.items()):
        if reference_models and reference_model not in reference_models:
            continue
        for (baseline_dataset, baseline_model), baseline_rows in sorted(grouped.items()):
            if baseline_dataset != dataset or baseline_model == reference_model:
                continue
            for loss_name in ["zero_one", "brier"]:
                result = dm_test(reference_rows, baseline_rows, loss_name)
                row = {
                    "dataset": dataset,
                    "reference_model": reference_model,
                    "baseline_model": baseline_model,
                    "loss": loss_name,
                    "n": result["n"],
                    "mean_loss_diff": f"{result['mean_loss_diff']:.6f}",
                    "dm_stat": f"{result['dm_stat']:.6f}",
                    "p_value": f"{result['p_value']:.6g}",
                    "reference_lower_loss": result["mean_loss_diff"] > 0,
                    "significant": result["p_value"] < 0.05,
                }
                pairwise_rows.append(row)
                key = (dataset, reference_model, loss_name)
                item = summary.setdefault(key, {"comparisons": 0, "lower_loss": 0, "sig_lower_loss": 0, "diffs": []})
                item["comparisons"] += 1
                item["lower_loss"] += int(result["mean_loss_diff"] > 0)
                item["sig_lower_loss"] += int(result["mean_loss_diff"] > 0 and result["p_value"] < 0.05)
                item["diffs"].append(result["mean_loss_diff"])

    write_csv(output_dir / "diebold_mariano_results.csv", pairwise_rows, [
        "dataset", "reference_model", "baseline_model", "loss", "n", "mean_loss_diff", "dm_stat", "p_value", "reference_lower_loss", "significant"
    ])
    summary_rows = []
    for (dataset, reference_model, loss_name), item in sorted(summary.items()):
        diffs = sorted(item["diffs"])
        median_diff = diffs[len(diffs) // 2] if diffs else 0.0
        summary_rows.append({
            "dataset": dataset,
            "reference_model": reference_model,
            "loss": loss_name,
            "comparisons": item["comparisons"],
            "lower_loss": item["lower_loss"],
            "sig_lower_loss": item["sig_lower_loss"],
            "median_loss_diff": f"{median_diff:.6f}",
        })
    write_csv(output_dir / "diebold_mariano_summary.csv", summary_rows, [
        "dataset", "reference_model", "loss", "comparisons", "lower_loss", "sig_lower_loss", "median_loss_diff"
    ])


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Run PT and DM statistical tests from prediction CSV")
    parser.add_argument("--predictions", default="outputs/metrics/predictions.csv")
    parser.add_argument("--output-dir", default="outputs/metrics/stat_tests")
    parser.add_argument("--tests", nargs="+", choices=["pt", "dm"], default=["pt", "dm"])
    parser.add_argument("--reference-models", nargs="*", default=[])
    args = parser.parse_args(argv)

    predictions_path = resolve_path(args.predictions)
    output_dir = resolve_path(args.output_dir)
    rows = load_prediction_rows(predictions_path)
    if "pt" in args.tests:
        run_pt(rows, output_dir)
    if "dm" in args.tests:
        run_dm(rows, output_dir, args.reference_models)
    print(f"Statistical test outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
