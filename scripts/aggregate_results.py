#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment summary.json files into CSV tables."
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/root/CS5782-LoRA-Final-Project/results",
            "/root/autodl-tmp/lora_sweeps",
            "/root/autodl-tmp/exp_merge_mnli",
        ],
        help="Root directories to recursively scan for summary.json",
    )
    parser.add_argument(
        "--detailed-output",
        default="/root/CS5782-LoRA-Final-Project/results/aggregated_results_detailed.csv",
        help="Output CSV path for per-run details",
    )
    parser.add_argument(
        "--leaderboard-output",
        default="/root/CS5782-LoRA-Final-Project/results/aggregated_leaderboard.csv",
        help="Output CSV path for aggregated leaderboard",
    )
    parser.add_argument(
        "--sort-by",
        choices=["mean_acc", "mean_f1", "median_acc"],
        default="mean_acc",
        help="Leaderboard sorting metric",
    )
    return parser.parse_args()


def to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_stat(values: list[float], mode: str) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan")
    if mode == "mean":
        return statistics.mean(clean)
    if mode == "median":
        return statistics.median(clean)
    if mode == "pstdev":
        return statistics.pstdev(clean) if len(clean) > 1 else 0.0
    raise ValueError(f"Unsupported stat mode: {mode}")


def collect_rows(roots: list[str]) -> list[dict]:
    rows = []
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            print(f"[skip] root not found: {root_path}")
            continue

        for summary_path in sorted(root_path.rglob("summary.json")):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[warn] invalid json: {summary_path}")
                continue

            eval_metrics = summary.get("eval_metrics", {})
            row = {
                "source_root": str(root_path),
                "summary_path": str(summary_path),
                "run_name": summary_path.parent.name,
                "task": summary.get("task"),
                "method": summary.get("method"),
                "lora_rank": summary.get("lora_rank"),
                "lora_alpha": summary.get("lora_alpha"),
                "merge_adapter": summary.get("merge_adapter"),
                "merged_lora_modules": summary.get("merged_lora_modules"),
                "merged_model_path": summary.get("merged_model_path"),
                "trainable_parameters": summary.get("trainable_parameters"),
                "total_parameters": summary.get("total_parameters"),
                "runtime_seconds": summary.get("runtime_seconds"),
                "eval_accuracy": eval_metrics.get("eval_accuracy"),
                "eval_f1": eval_metrics.get("eval_f1"),
                "eval_precision": eval_metrics.get("eval_precision"),
                "eval_recall": eval_metrics.get("eval_recall"),
                "eval_loss": eval_metrics.get("eval_loss"),
                "eval_runtime": eval_metrics.get("eval_runtime"),
                "epoch": eval_metrics.get("epoch"),
            }
            rows.append(row)
    return rows


def write_detailed_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_root",
        "summary_path",
        "run_name",
        "task",
        "method",
        "lora_rank",
        "lora_alpha",
        "merge_adapter",
        "merged_lora_modules",
        "merged_model_path",
        "trainable_parameters",
        "total_parameters",
        "runtime_seconds",
        "eval_accuracy",
        "eval_f1",
        "eval_precision",
        "eval_recall",
        "eval_loss",
        "eval_runtime",
        "epoch",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_leaderboard(rows: list[dict], sort_by: str) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (
            row.get("task"),
            row.get("method"),
            row.get("lora_rank"),
            row.get("lora_alpha"),
            row.get("merge_adapter"),
        )
        grouped.setdefault(key, []).append(row)

    board = []
    for key, items in grouped.items():
        acc_values = [to_float(x.get("eval_accuracy")) for x in items]
        f1_values = [to_float(x.get("eval_f1")) for x in items]
        acc_clean = [v for v in acc_values if not math.isnan(v)]
        f1_clean = [v for v in f1_values if not math.isnan(v)]

        board.append(
            {
                "task": key[0],
                "method": key[1],
                "lora_rank": key[2],
                "lora_alpha": key[3],
                "merge_adapter": key[4],
                "success_count": len(items),
                "mean_acc": safe_stat(acc_values, "mean"),
                "median_acc": safe_stat(acc_values, "median"),
                "std_acc": safe_stat(acc_values, "pstdev"),
                "best_acc": max(acc_clean) if acc_clean else float("nan"),
                "mean_f1": safe_stat(f1_values, "mean"),
                "best_f1": max(f1_clean) if f1_clean else float("nan"),
                "runs": ";".join(x.get("run_name", "") for x in items),
            }
        )

    board.sort(
        key=lambda x: (
            -1 if math.isnan(to_float(x.get(sort_by))) else to_float(x.get(sort_by))
        ),
        reverse=True,
    )
    return board


def write_leaderboard_csv(board: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "method",
        "lora_rank",
        "lora_alpha",
        "merge_adapter",
        "success_count",
        "mean_acc",
        "median_acc",
        "std_acc",
        "best_acc",
        "mean_f1",
        "best_f1",
        "runs",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(board)


def main() -> None:
    args = parse_args()
    detailed_out = Path(args.detailed_output).expanduser().resolve()
    leaderboard_out = Path(args.leaderboard_output).expanduser().resolve()

    rows = collect_rows(args.roots)
    if not rows:
        print("No summary.json found. Nothing written.")
        return

    write_detailed_csv(rows, detailed_out)
    board = build_leaderboard(rows, args.sort_by)
    write_leaderboard_csv(board, leaderboard_out)

    print(f"Collected runs: {len(rows)}")
    print(f"Wrote detailed CSV: {detailed_out}")
    print(f"Wrote leaderboard CSV: {leaderboard_out}")


if __name__ == "__main__":
    main()
