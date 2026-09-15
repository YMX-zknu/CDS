from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path.cwd().resolve()


def resolve_project_path(value: str) -> Path:

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def execute(command, section_2_6_code_dir=None):
    rendered = list(map(str, command))
    print("\n[run]", " ".join(rendered), flush=True)
    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if section_2_6_code_dir is not None:
        env["SECTION_2_6_CODE_DIR"] = str(section_2_6_code_dir)
    if "--seed" in rendered:
        env["PYTHONHASHSEED"] = rendered[rendered.index("--seed") + 1]
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True, env=env)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--dvsgesture_root", default="data/DVS-Gesture")
    p.add_argument("--shd_root", default="data/SHD")
    p.add_argument("--stmnist_root", default="data/ST-MNIST")
    p.add_argument(
        "--section_2_6_code_dir",
        default=str(HERE.parent / "section_2_6"),
        help=(
            "Directory containing section_2_6_dvsgesture_train.py, "
            "section_2_6_shd_train.py and section_2_6_stmnist_train.py."
        ),
    )
    p.add_argument("--results_26_root", default="results/section_2_6")
    p.add_argument("--out_root", default="results/section_2_7")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--perturbation_ratios", default="0,0.025,0.05,0.10")
    p.add_argument(
        "--selectivity_ratio", type=float, default=0.10,
        help="Perturbation ratio displayed in Fig. 7f.",
    )
    p.add_argument("--perturbation_repeats", type=int, default=3)
    p.add_argument("--isolation_guard_steps", type=int, default=4)
    p.add_argument("--minimum_realized_fraction", type=float, default=0.95)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--include_constant_release", action="store_true")
    p.add_argument("--ablation_root", default=None)
    return p.parse_args()


def main():
    a = parse_args(); seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    analysed_ratios = [
        float(value) for value in a.perturbation_ratios.split(",")
        if value.strip()
    ]
    if not any(abs(value - a.selectivity_ratio) <= 1e-12 for value in analysed_ratios):
        raise ValueError(
            f"--selectivity_ratio={a.selectivity_ratio:g} must be included in "
            f"--perturbation_ratios={a.perturbation_ratios!r}."
        )
    section_2_6_code_dir = resolve_project_path(a.section_2_6_code_dir)
    required_trainers = [
        "section_2_6_dvsgesture_train.py",
        "section_2_6_shd_train.py",
        "section_2_6_stmnist_train.py",
    ]
    missing_trainers = [
        name for name in required_trainers
        if not (section_2_6_code_dir / name).is_file()
    ]
    if missing_trainers:
        raise FileNotFoundError(
            "The Section 2.6 code directory is invalid: "
            f"{section_2_6_code_dir}. Missing: {missing_trainers}. "
            "Set --section_2_6_code_dir to the directory containing the "
            "three latest Section 2.6 training scripts."
        )
    results_26_root = resolve_project_path(a.results_26_root)
    out_root = resolve_project_path(a.out_root)
    ablation_root = (
        resolve_project_path(a.ablation_root) if a.ablation_root else None
    )
    tasks = [
        ("section_2_7_dvsgesture_analysis.py", resolve_project_path(a.dvsgesture_root)),
        ("section_2_7_shd_analysis.py", resolve_project_path(a.shd_root)),
        ("section_2_7_stmnist_analysis.py", resolve_project_path(a.stmnist_root)),
    ]
    for seed in seeds:
        for script, data_root in tasks:
            command = [
                sys.executable, HERE / script, "--data_root", data_root,
                "--results_26_root", results_26_root, "--out_root", out_root,
                "--seed", seed, "--device", a.device, "--num_workers", a.num_workers,
                "--perturbation_ratios", a.perturbation_ratios,
                "--perturbation_repeats", a.perturbation_repeats,
                "--isolation_guard_steps", a.isolation_guard_steps,
                "--minimum_realized_fraction", a.minimum_realized_fraction,
                "--max_batches", a.max_batches,
            ]
            if a.include_constant_release:
                if not a.ablation_root:
                    raise ValueError("--include_constant_release requires --ablation_root")
                command += ["--include_constant_release", "--ablation_root", ablation_root]
            execute(command, section_2_6_code_dir=section_2_6_code_dir)
    execute([
        sys.executable, HERE / "section_2_7_multimodal_plot.py",
        "--root", out_root, "--out_dir", out_root / "fig7",
        "--seeds", a.seeds, "--selectivity_ratio", a.selectivity_ratio,
    ])


if __name__ == "__main__":
    main()
