import argparse
import os
import sys
from pathlib import Path


def _configure_section_2_6_import_path() -> Path:

    here = Path(__file__).resolve().parent
    candidates = []
    configured = os.environ.get("SECTION_2_6_CODE_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([
        here,
        here.parent / "section_2_6",
        Path.cwd() / "experiments" / "section_2_6",
    ])
    required = [
        "section_2_6_dvsgesture_train.py",
        "section_2_6_shd_train.py",
        "section_2_6_stmnist_train.py",
    ]
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if all((candidate / name).is_file() for name in required):
            sys.path.insert(0, str(candidate))
            return candidate
    raise ModuleNotFoundError(
        "Cannot locate the latest Section 2.6 training scripts. Checked: "
        f"{[str(path) for path in checked]}. Run through "
        "run_section_2_7_multiseed.py or set SECTION_2_6_CODE_DIR to the "
        "directory containing the three section_2_6_*_train.py files."
    )


SECTION_2_6_CODE_DIR = _configure_section_2_6_import_path()

from section_2_6_dvsgesture_train import DVSGestureConfig, build_dvsgesture_datasets, build_dvsgesture_model
from section_2_7_common import DEFAULT_ENERGY_COSTS, parse_float_list, run_dataset_analysis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/DVS-Gesture")
    p.add_argument("--results_26_root", default="results/section_2_6")
    p.add_argument("--out_root", default="results/section_2_7")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--perturbation_ratios", default="0,0.025,0.05,0.10")
    p.add_argument("--perturbation_repeats", type=int, default=3)
    p.add_argument("--isolation_guard_steps", type=int, default=4)
    p.add_argument("--minimum_realized_fraction", type=float, default=0.95)
    p.add_argument("--max_batches", type=int, default=0, help="0 analyses the complete official test set.")
    p.add_argument("--ablation_root", default=None)
    p.add_argument("--include_constant_release", action="store_true")
    for key, value in DEFAULT_ENERGY_COSTS.items():
        p.add_argument(f"--{key}", type=float, default=value)
    return p.parse_args()


def main():
    a = parse_args()
    costs = {k: getattr(a, k) for k in DEFAULT_ENERGY_COSTS}
    run_dataset_analysis(
        "dvsgesture", "visual", a.seed, a.data_root, a.results_26_root, a.out_root,
        DVSGestureConfig, build_dvsgesture_datasets, build_dvsgesture_model,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        device_text=a.device,
        perturbation_ratios=parse_float_list(a.perturbation_ratios),
        max_batches=a.max_batches,
        perturbation_repeats=a.perturbation_repeats,
        isolation_guard_steps=a.isolation_guard_steps,
        minimum_realized_fraction=a.minimum_realized_fraction,
        ablation_root=a.ablation_root,
        include_constant_release=a.include_constant_release,
        energy_costs=costs,
    )


if __name__ == "__main__":
    main()
