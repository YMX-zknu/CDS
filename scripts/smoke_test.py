import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Synthetic CPU execution check; no dataset download.")
    parser.add_argument("--out_dir", type=Path, default=ROOT / "results" / "smoke")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(command, *options):
        argv = [sys.executable, str(ROOT / "run.py"), command, *map(str, options)]
        print("Running:", " ".join(argv), flush=True)
        subprocess.run(argv, check=True, cwd=ROOT, env=env)

    common = [
        "--dry_run", "--epochs", "1", "--batch_size", "2", "--device", "cpu",
        "--num_workers", "0", "--dry_run_train_samples", "4", "--dry_run_test_samples", "4",
        "--seed", "2026", "--T", "10", "--analysis_batches", "1",
        "--isolated_perturbation_ratios", "0,0.1", "--isolated_perturbation_repeats", "1",
        "--out_dir", output / "training",
    ]
    run("train-dvs", *common, "--channels", "2", "--blocks", "1")
    run("train-shd", *common, "--hidden_dim", "8", "--hidden_layers", "1")
    run("train-stmnist", *common, "--channels", "2", "--blocks", "1")
    run(
        "ablate", "--datasets", "shd", "--variants", "constant_release,memory_free",
        *common[:-2], "--hidden_dim", "8", "--out_dir", output / "ablation",
    )
    for task in ("dvs", "shd", "stmnist"):
        run(
            f"analyze-{task}", "--results_26_root", output / "training",
            "--out_root", output / "analysis", "--seed", "2026", "--device", "cpu",
            "--num_workers", "0", "--perturbation_ratios", "0,0.1",
            "--perturbation_repeats", "1",
        )
    for dataset in ("dvsgesture", "shd", "stmnist"):
        for model in ("static", "cds"):
            assert (output / "training" / dataset / "seed_2026" / "checkpoints" / f"{model}.pt").is_file()
        assert (output / "analysis" / dataset / "seed_2026" / "logs" / "efficiency_summary.csv").is_file()
    print(f"Synthetic training, ablation and checkpoint-analysis checks completed: {output}")


if __name__ == "__main__":
    main()
