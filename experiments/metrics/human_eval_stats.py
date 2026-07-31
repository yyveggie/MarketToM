import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = ("clarity", "actionability", "trust")
ORDERS = ("first", "second")
CSV_FIELDS = ["dimension", "first_mean", "first_sd", "second_mean", "second_sd", "n_pairs", "p_value"]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_data(input_path):
    return json.loads(resolve_path(input_path).read_text(encoding="utf-8"))


def participant_means(ratings):
    accumulator = defaultdict(list)
    for record in ratings:
        participant = record.get("participant")
        order = record.get("order")
        if participant is None or order not in ORDERS:
            continue
        for dimension in DIMENSIONS:
            if dimension in record and record[dimension] is not None:
                accumulator[(participant, order, dimension)].append(float(record[dimension]))
    return {key: mean(values) for key, values in accumulator.items() if values}


def dimension_stats(means):
    participants = sorted({key[0] for key in means})
    rows = []
    for dimension in DIMENSIONS:
        paired = [
            (means[(participant, "first", dimension)], means[(participant, "second", dimension)])
            for participant in participants
            if (participant, "first", dimension) in means and (participant, "second", dimension) in means
        ]
        if not paired:
            continue
        first = [pair[0] for pair in paired]
        second = [pair[1] for pair in paired]
        if any(a != b for a, b in paired):
            _, p_value = wilcoxon(first, second)
        else:
            p_value = 1.0
        rows.append({
            "dimension": dimension,
            "first_mean": mean(first),
            "first_sd": stdev(first) if len(first) > 1 else 0.0,
            "second_mean": mean(second),
            "second_sd": stdev(second) if len(second) > 1 else 0.0,
            "n_pairs": len(paired),
            "p_value": float(p_value),
        })
    return rows


def forced_choice_stats(forced_choice):
    total = sum(1 for record in forced_choice if record.get("preferred") in ORDERS)
    second = sum(1 for record in forced_choice if record.get("preferred") == "second")
    if total == 0:
        return {"second": 0, "total": 0, "ratio": 0.0, "p_value": 1.0}
    result = binomtest(second, total, 0.5)
    return {"second": second, "total": total, "ratio": second / total, "p_value": float(result.pvalue)}


def write_csv(path, rows, forced):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dimension": row["dimension"],
                "first_mean": f"{row['first_mean']:.4f}",
                "first_sd": f"{row['first_sd']:.4f}",
                "second_mean": f"{row['second_mean']:.4f}",
                "second_sd": f"{row['second_sd']:.4f}",
                "n_pairs": row["n_pairs"],
                "p_value": f"{row['p_value']:.6g}",
            })
        writer.writerow({
            "dimension": "forced_choice_second_preferred",
            "first_mean": "",
            "first_sd": "",
            "second_mean": f"{forced['ratio']:.4f}",
            "second_sd": f"{forced['second']}/{forced['total']}",
            "n_pairs": forced["total"],
            "p_value": f"{forced['p_value']:.6g}",
        })


def write_tex(path, rows, forced):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Paired human evaluation of perceived usefulness.}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Dimension & first-order (Mean $\\pm$ SD) & second-order (Mean $\\pm$ SD) & $p$-value \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dimension'].capitalize()} & "
            f"{row['first_mean']:.2f} $\\pm$ {row['first_sd']:.2f} & "
            f"{row['second_mean']:.2f} $\\pm$ {row['second_sd']:.2f} & "
            f"{row['p_value']:.3g} \\\\"
        )
    lines.append("\\midrule")
    percent = forced["ratio"] * 100
    lines.append(
        f"\\multicolumn{{3}}{{l}}{{Forced choice: second-order preferred in {percent:.1f}\\% "
        f"of instances ({forced['second']}/{forced['total']})}} & {forced['p_value']:.3g} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired human-evaluation statistics")
    parser.add_argument("--input", default="data/human_eval/ratings.json")
    parser.add_argument("--output-dir", default="outputs/metrics/human_eval")
    args = parser.parse_args(argv)
    data = load_data(args.input)
    means = participant_means(data.get("ratings", []))
    rows = dimension_stats(means)
    forced = forced_choice_stats(data.get("forced_choice", []))
    output_dir = resolve_path(args.output_dir)
    write_csv(output_dir / "human_eval.csv", rows, forced)
    write_tex(output_dir / "human_eval.tex", rows, forced)
    print(f"Human-evaluation statistics written to: {output_dir}")
    for row in rows:
        print(
            f"{row['dimension']}: first {row['first_mean']:.2f}±{row['first_sd']:.2f} "
            f"second {row['second_mean']:.2f}±{row['second_sd']:.2f} p={row['p_value']:.3g}"
        )
    print(f"forced_choice: second {forced['second']}/{forced['total']} ({forced['ratio']*100:.1f}%) p={forced['p_value']:.3g}")


if __name__ == "__main__":
    main()
