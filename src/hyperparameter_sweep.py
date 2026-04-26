from pathlib import Path
from dataclasses import dataclass, asdict
from statistics import median, mean, pstdev
import argparse
import json
import sys
import pandas as pd
import itertools
import subprocess

# -----------------------------
# Experiment definition
# -----------------------------
@dataclass(frozen=True)
class TrialConfig:
    task: str
    model: str
    rank: int
    alpha: int
    lr: float
    bs: int
    epochs: int
    max_len: int

@dataclass
class TrialResult:
    config: TrialConfig
    seed_scores: list
    success_count: int
    expected_count: int

    def to_row(self):
        scores = self.seed_scores
        return {
            **asdict(self.config),
            "success_count": self.success_count,
            "expected_count": self.expected_count,
            "median_acc": median(scores) if scores else None,
            "mean_acc": mean(scores) if scores else None,
            "std_acc": pstdev(scores) if len(scores) > 1 else 0.0 if len(scores) == 1 else None,
            "scores": json.dumps(scores),
        }

# -----------------------------
# Utilities
# -----------------------------
def build_output_dir(root: Path, cfg: TrialConfig, seed: int) -> Path:
    tag = f"{cfg.task}-rk{cfg.rank}-a{cfg.alpha}-lr{cfg.lr}-bs{cfg.bs}-sd{seed}"
    return root / tag

def launch_training(cfg: TrialConfig, seed: int, out_dir: Path) -> int:
    train_script = Path(__file__).resolve().parents[1] / "scripts" / "train_glue.py"
    cmd = [
        sys.executable, str(train_script),
        "--task", cfg.task,
        "--model_name", cfg.model,
        "--method", "lora",
        "--lora_rank", str(cfg.rank),
        "--lora_alpha", str(cfg.alpha),
        "--learning_rate", str(cfg.lr),
        "--batch_size", str(cfg.bs),
        "--epochs", str(cfg.epochs),
        "--max_length", str(cfg.max_len),
        "--seed", str(seed),
        "--output_dir", str(out_dir),
    ]
    print(f"[RUN] seed={seed} -> {out_dir.name}")
    return subprocess.run(cmd, check=False).returncode

def read_best_accuracy(result_csv: Path):
    if not result_csv.exists():
        return None
    df = pd.read_csv(result_csv)
    candidates = [c for c in ["Accuracy", "eval_accuracy", "accuracy"] if c in df.columns]
    if not candidates:
        return None
    return float(df[candidates[0]].max())

def evaluate_config(cfg: TrialConfig, seeds, save_root: Path) -> TrialResult:
    scores = []
    for sd in seeds:
        out_dir = build_output_dir(save_root, cfg, sd)
        out_dir.mkdir(parents=True, exist_ok=True)

        code = launch_training(cfg, sd, out_dir)
        if code != 0:
            print(f"  ! failed seed {sd} (exit={code})")
            continue

        best = read_best_accuracy(out_dir / "eval_history.csv")
        if best is None:
            print(f"  ! no usable accuracy for seed {sd}")
            continue

        print(f"  + seed {sd}: best_acc={best:.4f}")
        scores.append(best)

    return TrialResult(
        config=cfg,
        seed_scores=scores,
        success_count=len(scores),
        expected_count=len(seeds),
    )

def parse_int_list(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="LoRA hyperparameter sweep runner")
    parser.add_argument("--task", type=str, default="mnli")
    parser.add_argument("--model", type=str, default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=128, dest="max_len")
    parser.add_argument("--seeds", type=parse_int_list, default=42,
                        help="Comma-separated seeds, e.g. 42,43,44")

    parser.add_argument("--rank-list", type=parse_int_list, default=[4, 8, 16],
                        help="Comma-separated LoRA ranks")
    parser.add_argument("--alpha-list", type=parse_int_list, default=[8, 16, 32],
                        help="Comma-separated LoRA alphas")
    parser.add_argument("--lr-list", type=parse_float_list, default=1e-4,
                        help="Comma-separated learning rates")
    parser.add_argument("--bs-list", type=parse_int_list, default=32,
                        help="Comma-separated batch sizes")
    parser.add_argument("--results-root", type=str, default="../results")
    return parser


def main():
    args = build_parser().parse_args()

    search_space = {
        "rank": args.rank_list,
        "alpha": args.alpha_list,
        "lr": args.lr_list,
        "bs": args.bs_list,
    }
    common = {
        "task": args.task,
        "model": args.model,
        "epochs": args.epochs,
        "max_len": args.max_len,
    }
    seeds = args.seeds
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Run all configs
    # -----------------------------
    all_results = []
    grid = itertools.product(
        search_space["rank"],
        search_space["alpha"],
        search_space["lr"],
        search_space["bs"],
    )

    for i, (rk, a, lr, bs) in enumerate(grid, start=1):
        cfg = TrialConfig(
            task=common["task"],
            model=common["model"],
            rank=rk,
            alpha=a,
            lr=lr,
            bs=bs,
            epochs=common["epochs"],
            max_len=common["max_len"],
        )
        print(f"\n=== [{i}] config: rank={rk}, alpha={a}, lr={lr}, bs={bs} ===")
        all_results.append(evaluate_config(cfg, seeds, results_root))

    # -----------------------------
    # Save leaderboard
    # -----------------------------
    board = pd.DataFrame([r.to_row() for r in all_results])
    board = board.sort_values(["median_acc", "mean_acc"], ascending=False, na_position="last")

    leaderboard_csv = results_root / f"{common['task']}_sweep_leaderboard.csv"
    board.to_csv(leaderboard_csv, index=False)

    print("\nTop-10 configs by median_acc:")
    print(board.head(10).to_string(index=False))
    print(f"\nSaved: {leaderboard_csv}")


if __name__ == "__main__":
    main()