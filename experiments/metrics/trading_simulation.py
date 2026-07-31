import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_KEYS = ("preset", "model", "ccn_dependency_variant", "prompt_format", "seed")
TABLE_KEYS = ("dataset", "preset", "model")
TRADING_DAYS = 252
METRIC_FIELDS = ("cum_return", "ann_return", "ann_vol", "sharpe", "mdd", "turnover")
CSV_FIELDS = ["dataset", "preset", "model", "seeds"] + list(METRIC_FIELDS)


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


def strategy_series(predictions, close, cost_per_change):
    series = []
    position_prev = 0
    for index, predicted_up in predictions:
        if index < 1 or index >= len(close) or close[index - 1] == 0:
            continue
        realized = close[index] / close[index - 1] - 1.0
        position = 1 if predicted_up else 0
        turn = abs(position - position_prev)
        strat = position * realized - cost_per_change * turn
        series.append((index, strat, turn))
        position_prev = position
    return series


def compute_metrics(returns, turns):
    if not returns:
        return None
    arr = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + arr)
    final = float(equity[-1])
    n = arr.size
    cum_return = final - 1.0
    ann_return = final ** (TRADING_DAYS / n) - 1.0 if final > 0 else -1.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ann_vol = std * math.sqrt(TRADING_DAYS)
    sharpe = float(np.mean(arr)) / std * math.sqrt(TRADING_DAYS) if std > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    mdd = float(np.max(drawdown)) if drawdown.size else 0.0
    turnover = mean(turns) if turns else 0.0
    return {
        "cum_return": cum_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "turnover": turnover,
    }


def portfolio_metrics(stock_series):
    index_returns = defaultdict(list)
    index_turns = defaultdict(list)
    for series in stock_series:
        for index, strat, turn in series:
            index_returns[index].append(strat)
            index_turns[index].append(turn)
    if not index_returns:
        return None
    ordered = sorted(index_returns.keys())
    returns = [mean(index_returns[index]) for index in ordered]
    turns = [mean(index_turns[index]) for index in ordered]
    return compute_metrics(returns, turns)


def evaluate(runs, include_splits, cost_per_change):
    seed_groups = defaultdict(lambda: defaultdict(list))
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
        series = strategy_series(predictions, close, cost_per_change)
        if not series:
            continue
        seed_key = (row.get("dataset", ""),) + tuple(str(row.get(field, "")) for field in SEED_KEYS)
        seed_groups[seed_key]["series"].append(series)

    table_groups = defaultdict(list)
    for seed_key, payload in seed_groups.items():
        metrics = portfolio_metrics(payload["series"])
        if metrics is None:
            continue
        dataset = seed_key[0]
        preset = seed_key[1]
        model = seed_key[2]
        table_groups[(dataset, preset, model)].append(metrics)

    summary = []
    for (dataset, preset, model), seed_metrics in sorted(table_groups.items()):
        row = {"dataset": dataset, "preset": preset, "model": model, "seeds": len(seed_metrics)}
        for field in METRIC_FIELDS:
            row[field] = mean(metric[field] for metric in seed_metrics)
        summary.append(row)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {field: row.get(field) for field in CSV_FIELDS}
            for field in METRIC_FIELDS:
                record[field] = f"{row[field]:.4f}"
            writer.writerow(record)


def write_tex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Trading-simulation results with 10 bps round-trip transaction costs.}",
        "\\begin{tabular}{lllrrrrrr}",
        "\\toprule",
        "Dataset & Preset & Model & Cum.\\ Ret. & Ann.\\ Ret. & Ann.\\ Vol. & SR & MDD & Turn. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['preset']} & {row['model']} & "
            f"{row['cum_return']:.4f} & {row['ann_return']:.4f} & {row['ann_vol']:.4f} & "
            f"{row['sharpe']:.4f} & {row['mdd']:.4f} & {row['turnover']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Long-cash trading simulation from prediction batches")
    parser.add_argument("--summary", default="outputs/metrics/batch_experiments/summary.json")
    parser.add_argument("--output-dir", default="outputs/metrics/trading_simulation")
    parser.add_argument("--include-splits", nargs="*", default=["Test"])
    parser.add_argument("--round-trip-bps", type=float, default=10.0)
    args = parser.parse_args(argv)
    cost_per_change = args.round_trip_bps / 2.0 / 10000.0
    runs = load_runs(args.summary)
    summary = evaluate(runs, set(args.include_splits) if args.include_splits else None, cost_per_change)
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "trading_simulation.csv", summary)
    write_tex(output_dir / "trading_simulation.tex", summary)
    print(f"Trading simulation written to: {output_dir}")
    for row in summary:
        print(
            f"{row['dataset']} {row['preset']} {row['model']}: "
            f"CumRet {row['cum_return']:.4f}, Sharpe {row['sharpe']:.4f}, "
            f"MDD {row['mdd']:.4f}, Turn {row['turnover']:.4f}"
        )


if __name__ == "__main__":
    main()
