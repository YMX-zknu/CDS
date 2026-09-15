from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


PROJECT_ROOT = Path.cwd().resolve()
EXPECTED_ARCHITECTURE = "presynaptic_cds_v4_dataset_specific"
EXPECTED_ORDER = "CDS->Conv/Linear->BN(if used)->LIF"
EXPECTED_PERTURBATION_PROTOCOL = "amplitude_matched_maximum_guard_v4"


def resolve_project_path(value: str) -> Path:

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run(command):
    print("\n[run]", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=PROJECT_ROOT)


def verify_training_artifacts(out_dir: str, dataset: str, seed: int):
    log_dir = Path(out_dir) / dataset / f"seed_{seed}" / "logs"
    for filename in (
        "metrics_summary.csv",
        "isolated_perturbation_performance.csv",
        "transmission_selectivity.csv",
    ):
        path = log_dir / filename
        if not path.exists():
            raise RuntimeError(f"Missing Section 2.6 result file: {path}")
        table = pd.read_csv(path)
        if "isolated_perturbation_protocol" not in table.columns:
            raise RuntimeError(
                f"Result file lacks isolated-perturbation protocol metadata: {path}"
            )
        observed_protocols = set(
            table["isolated_perturbation_protocol"].dropna().astype(str)
        )
        if observed_protocols != {EXPECTED_PERTURBATION_PROTOCOL}:
            raise RuntimeError(
                f"Incompatible isolated-perturbation protocol in {path}: "
                f"{sorted(observed_protocols)}"
            )
    for model in ("static", "cds"):
        path = log_dir / f"architecture_verification_{model}.json"
        if not path.exists():
            raise RuntimeError(f"Missing architecture verification report: {path}")
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("verified") is not True:
            raise RuntimeError(f"Architecture verification failed or is absent in {path}")
        if report.get("architecture_version") != EXPECTED_ARCHITECTURE:
            raise RuntimeError(f"Incompatible architecture report in {path}: {report}")
        if report.get("transmission_order") != EXPECTED_ORDER:
            raise RuntimeError(f"Unexpected transmission order in {path}: {report}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--dvsgesture_root", default="data/DVS-Gesture")
    p.add_argument("--shd_root", default="data/SHD")
    p.add_argument("--stmnist_root", default="data/ST-MNIST")
    p.add_argument("--out_dir", default="results/section_2_6")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--with_ablation", action="store_true")
    p.add_argument("--ablation_variants", default="static,full,constant_release,memory_free")
    return p.parse_args()


def main():
    a = parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    out_dir = resolve_project_path(a.out_dir)
    perturbation_ratios = "0.0,0.025,0.05,0.1"
    perturbation_repeats = 3
    isolation_guard_steps = 4
    selectivity_ratio = 0.1
    trainers = [
        ("section_2_6_dvsgesture_train.py", resolve_project_path(a.dvsgesture_root)),
        ("section_2_6_shd_train.py", resolve_project_path(a.shd_root)),
        ("section_2_6_stmnist_train.py", resolve_project_path(a.stmnist_root)),
    ]
    for seed in seeds:
        for script, root in trainers:
            run([sys.executable, HERE / script, "--data_root", root,
                 "--out_dir", out_dir, "--seed", seed,
                 "--device", a.device, "--num_workers", a.num_workers,
                 "--isolated_perturbation_ratios", perturbation_ratios,
                 "--isolated_perturbation_repeats", perturbation_repeats,
                 "--isolation_guard_steps", isolation_guard_steps,
                 "--selectivity_ratio", selectivity_ratio])
            dataset = {"section_2_6_dvsgesture_train.py": "dvsgesture",
                       "section_2_6_shd_train.py": "shd",
                       "section_2_6_stmnist_train.py": "stmnist"}[script]
            verify_training_artifacts(out_dir, dataset, seed)

    if a.with_ablation:
        for seed in seeds:
            run([
                sys.executable, HERE / "section_2_6_cds_ablation.py",
                "--out_dir", out_dir / "ablation",
                "--seed", seed, "--device", a.device,
                "--num_workers", a.num_workers, "--variants", a.ablation_variants,
                "--isolated_perturbation_ratios", perturbation_ratios,
                "--isolated_perturbation_repeats", perturbation_repeats,
                "--isolation_guard_steps", isolation_guard_steps,
                "--selectivity_ratio", selectivity_ratio,
                "--dvsgesture_data_root", trainers[0][1],
                "--shd_data_root", trainers[1][1],
                "--stmnist_data_root", trainers[2][1],
            ])

    run([
        sys.executable, HERE / "section_2_6_multimodal_plot.py",
        "--root", out_dir, "--seeds", a.seeds,
        "--fig_dir", out_dir / "fig6",
        "--ablation_path", out_dir / "ablation" / "ablation_task_table.csv",
    ])


if __name__ == "__main__":
    main()
