from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from torch.utils.data import DataLoader

from section_2_6_dvsgesture_train import (
    ARCHITECTURE_VERSION,
    ISOLATED_PERTURBATION_PROTOCOL_VERSION,
    TRANSMISSION_ORDER,
    DVSGestureConfig,
    build_dvsgesture_datasets,
    build_dvsgesture_model,
    parse_args as parse_dvsgesture_args,
    make_data_loader_generator,
    run_training_pipeline,
    set_seed,
)
from section_2_6_shd_train import (
    SHDConfig,
    build_shd_datasets,
    build_shd_model,
    parse_args as parse_shd_args,
)
from section_2_6_stmnist_train import (
    STMNISTConfig,
    build_stmnist_datasets,
    build_stmnist_model,
    parse_args as parse_stmnist_args,
)


DATASETS = {
    "dvsgesture": (
        DVSGestureConfig,
        build_dvsgesture_datasets,
        build_dvsgesture_model,
        parse_dvsgesture_args,
    ),
    "shd": (SHDConfig, build_shd_datasets, build_shd_model, parse_shd_args),
    "stmnist": (
        STMNISTConfig,
        build_stmnist_datasets,
        build_stmnist_model,
        parse_stmnist_args,
    ),
}

VARIANTS = ["static", "full", "constant_release", "memory_free"]

MECHANISM_LABELS = {
    "static": "static synaptic transmission",
    "full": "presynaptic residual calcium memory + bounded release probability",
    "constant_release": "removes time-varying release probability",
    "memory_free": "removes residual calcium carry-over",
}


def filtered_kwargs(cls, base: Dict):

    names = {f.name for f in fields(cls)}
    return {k: v for k, v in base.items() if k in names and v is not None}


def _set_if_not_none(base: Dict, key: str, value):
    if value is not None:
        base[key] = value


def _dataset_data_root(dataset: str, args) -> Optional[str]:

    specific = getattr(args, f"{dataset}_data_root", None)
    if specific:
        return specific
    if args.data_root:
        return args.data_root
    return None


def normalize_dataset_splits(build_output):

    if not isinstance(build_output, (tuple, list)):
        raise TypeError(
            f"Dataset builder must return a tuple/list, got {type(build_output).__name__}."
        )
    if len(build_output) == 2:
        return build_output[0], build_output[1]
    raise ValueError(
        f"Dataset builder returned {len(build_output)} values; expected exactly 2 (train, test)."
    )


def make_cfg(dataset: str, args, variant: str):


    cls, _, _, dataset_parse_args = DATASETS[dataset]
    base: Dict = vars(dataset_parse_args([])).copy()


    base["dataset_name"] = dataset
    base["out_dir"] = str(Path(args.out_dir) / "ablation_runs" / variant)
    base["model"] = "static" if variant == "static" else "cds"
    base["cds_variant"] = "full" if variant in {"static", "full"} else variant


    for key in [
        "epochs",
        "batch_size",
        "num_workers",
        "device",
        "lr",
        "eta_min",
        "seed",
        "analysis_batches",
        "isolated_perturbation_ratios",
        "isolated_perturbation_repeats",
        "isolation_guard_steps",
        "selectivity_ratio",
        "minimum_realized_fraction",
        "T",
        "channels",
        "blocks",
        "hidden_dim",
        "tau_m",
        "tau_ca_init",
        "dropout_p",
        "dry_run_train_samples",
        "dry_run_test_samples",
        "dry_run_hw",
    ]:
        _set_if_not_none(base, key, getattr(args, key, None))


    if args.TET:
        base["TET"] = True
    _set_if_not_none(base, "lamb", args.lamb)
    _set_if_not_none(base, "means", args.means)


    if args.amp:
        base["amp"] = True
    if args.dry_run:
        base["dry_run"] = True


    data_root = _dataset_data_root(dataset, args)
    if data_root is not None:
        base["data_root"] = data_root


    if dataset == "dvsgesture":
        _set_if_not_none(base, "delta_t", args.dvsgesture_delta_t)
    elif dataset == "shd":
        _set_if_not_none(base, "split_by", args.shd_split_by)
        _set_if_not_none(base, "normalize", args.shd_normalize)
    elif dataset == "stmnist":
        _set_if_not_none(base, "in_channels", args.stmnist_in_channels)
        _set_if_not_none(base, "sensor_h", args.stmnist_sensor_h)
        _set_if_not_none(base, "sensor_w", args.stmnist_sensor_w)
        _set_if_not_none(base, "train_ratio", args.stmnist_train_ratio)
        _set_if_not_none(base, "split_seed", args.stmnist_split_seed)
        _set_if_not_none(base, "time_window_ms", args.stmnist_time_window_ms)
        _set_if_not_none(base, "normalize", args.stmnist_normalize)
        if args.stmnist_force_reprocess:
            base["force_reprocess"] = True

    return cls(**filtered_kwargs(cls, base))


def run_one(dataset: str, variant: str, args):
    _, build_data, build_model, _ = DATASETS[dataset]
    cfg = make_cfg(dataset, args, variant)

    worker_init = set_seed(cfg.seed)
    train_set, test_set = normalize_dataset_splits(build_data(cfg))
    if min(len(train_set), len(test_set)) == 0:
        raise RuntimeError(f"{dataset} split is empty: train={len(train_set)}, test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=(dataset == "dvsgesture"),
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed + 1),
    )
    summary = run_training_pipeline(
        cfg, train_loader, test_loader, build_model, cfg.num_classes
    )
    summary["ablation_variant"] = variant
    summary["mechanism_tested"] = MECHANISM_LABELS[variant]
    summary["architecture_version"] = ARCHITECTURE_VERSION
    summary["transmission_order"] = TRANSMISSION_ORDER
    summary["isolated_perturbation_protocol"] = (
        ISOLATED_PERTURBATION_PROTOCOL_VERSION
    )
    summary["architecture_verified"] = True
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Run CDS ablations for Section 2.6.")


    p.add_argument("--out_dir", type=str, default="results/section_2_6/ablation")
    p.add_argument("--datasets", type=str, default="dvsgesture,shd,stmnist")
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))


    p.add_argument("--data_root", type=str, default=None,
                   help="Optional common root passed directly to every selected dataset. Prefer dataset-specific roots for real runs.")
    p.add_argument("--dvsgesture_data_root", type=str, default=None,
                   help="Override DVS-Gesture data_root; unset uses section_2_6_dvsgesture_train.py parse_args() default.")
    p.add_argument("--shd_data_root", type=str, default=None,
                   help="Override SHD data_root; unset uses section_2_6_shd_train.py parse_args() default.")
    p.add_argument("--stmnist_data_root", type=str, default=None,
                   help="Override ST-MNIST data_root; unset uses section_2_6_stmnist_train.py parse_args() default.")


    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eta_min", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--analysis_batches", type=int, default=None)
    p.add_argument("--isolated_perturbation_ratios", type=str, default=None)
    p.add_argument("--isolated_perturbation_repeats", type=int, default=None)
    p.add_argument("--isolation_guard_steps", type=int, default=None)
    p.add_argument("--selectivity_ratio", type=float, default=None)
    p.add_argument("--minimum_realized_fraction", type=float, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--blocks", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--tau_m", type=float, default=None)
    p.add_argument("--tau_ca_init", type=float, default=None)
    p.add_argument("--dropout_p", type=float, default=None)


    p.add_argument("--TET", action="store_true",
                   help="Use TET-style one-hot MSE loss for all ablation runs.")
    p.add_argument("--lamb", type=float, default=None,
                   help="TET regularization weight. Keep unset to use the dataset script default.")
    p.add_argument("--means", type=float, default=None,
                   help="TET target mean. Keep unset to use the dataset script default.")


    p.add_argument("--dvsgesture_delta_t", type=int, default=None)
    p.add_argument("--shd_split_by", choices=["time", "number"], default=None)
    p.add_argument("--shd_normalize", choices=["max", "none"], default=None)
    p.add_argument("--stmnist_in_channels", type=int, choices=[1, 2], default=None)
    p.add_argument("--stmnist_sensor_h", type=int, default=None)
    p.add_argument("--stmnist_sensor_w", type=int, default=None)
    p.add_argument("--stmnist_train_ratio", type=float, default=None)
    p.add_argument("--stmnist_split_seed", type=int, default=None)
    p.add_argument("--stmnist_time_window_ms", type=float, default=None)
    p.add_argument("--stmnist_normalize", choices=["none", "max", "log"], default=None)
    p.add_argument("--stmnist_force_reprocess", action="store_true")


    p.add_argument("--amp", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dry_run_train_samples", type=int, default=None)
    p.add_argument("--dry_run_test_samples", type=int, default=None)
    p.add_argument("--dry_run_hw", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for d in datasets:
        if d not in DATASETS:
            raise ValueError(f"Unknown dataset {d}. Choose from {list(DATASETS)}")
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"Unknown variant {v}. Choose from {VARIANTS}")

    rows: List[pd.DataFrame] = []
    for dataset in datasets:
        for variant in variants:
            print(f"\n[Section 2.6 ablation] dataset={dataset}, variant={variant}")
            rows.append(run_one(dataset, variant, args))

    if not rows:
        raise RuntimeError("No ablation runs were executed.")

    all_summary = pd.concat(rows, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_path = out_dir / "ablation_summary.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        all_summary = pd.concat([existing, all_summary], ignore_index=True)
        keys = [c for c in ["dataset", "seed", "ablation_variant"] if c in all_summary.columns]
        if keys:
            all_summary = all_summary.drop_duplicates(keys, keep="last")
    all_summary.to_csv(out_dir / "ablation_summary.csv", index=False)

    cols = [
        "dataset",
        "modality",
        "seed",
        "ablation_variant",
        "mechanism_tested",
        "model",
        "best_test_acc",
        "best_test_epoch",
        "test_acc",
        "test_balanced_acc",
        "test_macro_f1",
        "test_loss",
        "isolated_perturbation_auc",
        "mean_supported_release",
        "mean_isolated_release",
        "transmission_selectivity",
        "params",
        "TET",
        "tet_lamb",
        "tet_means",
        "architecture_version",
        "transmission_order",
        "isolated_perturbation_protocol",
        "architecture_verified",
    ]
    cols = [c for c in cols if c in all_summary.columns]
    all_summary[cols].to_csv(out_dir / "ablation_task_table.csv", index=False)


    value_cols = [c for c in [
        "test_acc", "test_balanced_acc", "test_macro_f1",
        "isolated_perturbation_auc", "mean_supported_release",
        "mean_isolated_release", "transmission_selectivity", "params"
    ] if c in all_summary.columns]
    if value_cols:
        agg = all_summary.groupby(["dataset", "modality", "ablation_variant", "mechanism_tested"], as_index=False)[value_cols].agg(["mean", "std"])
        agg.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in agg.columns]
        agg.to_csv(out_dir / "ablation_task_table_mean_sd.csv", index=False)

    print(f"\nSaved ablation tables to {out_dir}")


if __name__ == "__main__":
    main()
