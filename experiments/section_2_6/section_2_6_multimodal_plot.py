from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["dvsgesture", "shd", "stmnist"]
DATASET_LABELS = {
    "dvsgesture": "DVS Gesture\nvisual",
    "shd": "SHD\nauditory",
    "stmnist": "ST-MNIST\ntactile",
}
DATASET_SHORT = {
    "dvsgesture": "DVS Gesture",
    "shd": "SHD",
    "stmnist": "ST-MNIST",
}
MODEL_ORDER = ["static", "cds"]
MODEL_LABELS = {"static": "Static synapse", "cds": "CDS"}
MODEL_COLORS = {"static": "#377eb8", "cds": "#ff7f00"}
ABLATION_ORDER = ["static", "full", "constant_release", "memory_free"]
EXPECTED_ARCHITECTURE = "presynaptic_cds_v4_dataset_specific"
EXPECTED_ORDER = "CDS->Conv/Linear->BN(if used)->LIF"
EXPECTED_PROTOCOL = "amplitude_matched_maximum_guard_v4"


def _seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.match(r"seed_(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def parse_seeds(text: str) -> Optional[List[int]]:
    text = str(text).strip().lower()
    if text in {"", "all", "*"}:
        return None
    return [int(value) for value in re.split(r"[, ]+", text) if value]


def _read_protocol_csv(path: Path, dataset: str, seed: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    column = "isolated_perturbation_protocol"
    if column not in table.columns:
        raise RuntimeError(
            f"{path} predates the amplitude-matched perturbation protocol; rerun Section 2.6."
        )
    observed = set(table[column].dropna().astype(str))
    if observed != {EXPECTED_PROTOCOL}:
        raise RuntimeError(f"Incompatible perturbation protocol in {path}: {sorted(observed)}")
    if "dataset" not in table.columns:
        table["dataset"] = dataset
    if "seed" not in table.columns:
        table["seed"] = int(seed)
    return table


def collect_logs(
    root: Path,
    datasets: Sequence[str],
    seeds: Optional[Sequence[int]],
):
    metric_tables, performance_tables, selectivity_tables = [], [], []
    for dataset in datasets:
        dataset_dir = root / dataset
        seed_dirs = (
            sorted(dataset_dir.glob("seed_*"))
            if seeds is None
            else [dataset_dir / f"seed_{int(seed)}" for seed in seeds]
        )
        for seed_dir in seed_dirs:
            seed = _seed_from_path(seed_dir)
            if seed is None:
                continue
            logs = seed_dir / "logs"
            paths = {
                "metrics": logs / "metrics_summary.csv",
                "performance": logs / "isolated_perturbation_performance.csv",
                "selectivity": logs / "transmission_selectivity.csv",
            }
            if paths["metrics"].exists():
                table = _read_protocol_csv(paths["metrics"], dataset, seed)
                required = {
                    "architecture_version",
                    "transmission_order",
                    "architecture_verified",
                }
                missing = required.difference(table.columns)
                if missing:
                    raise RuntimeError(f"{paths['metrics']} lacks {sorted(missing)}")
                if not table["architecture_version"].eq(EXPECTED_ARCHITECTURE).all():
                    raise RuntimeError(f"Legacy architecture results in {paths['metrics']}")
                if not table["transmission_order"].eq(EXPECTED_ORDER).all():
                    raise RuntimeError(f"Unexpected CDS placement in {paths['metrics']}")
                metric_tables.append(table)
            if paths["performance"].exists():
                performance_tables.append(
                    _read_protocol_csv(paths["performance"], dataset, seed)
                )
            if paths["selectivity"].exists():
                selectivity_tables.append(
                    _read_protocol_csv(paths["selectivity"], dataset, seed)
                )
    concatenate = lambda tables: (
        pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    )
    return (
        concatenate(metric_tables),
        concatenate(performance_tables),
        concatenate(selectivity_tables),
    )


def available_datasets(metrics: pd.DataFrame, requested: Sequence[str]) -> List[str]:
    present = set(metrics["dataset"].astype(str)) if not metrics.empty else set()
    return [dataset for dataset in requested if dataset in present]


def grouped_seed_bar(
    ax,
    table: pd.DataFrame,
    value: str,
    ylabel: str,
    datasets: Sequence[str],
    ylim=None,
):
    width = 0.36
    x = np.arange(len(datasets))
    for model_index, model in enumerate(MODEL_ORDER):
        means, errors = [], []
        for dataset in datasets:
            values = table.loc[
                (table.dataset == dataset) & (table.model == model), value
            ].astype(float)
            means.append(float(values.mean()) if len(values) else np.nan)
            errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        positions = x + (model_index - 0.5) * width
        ax.bar(
            positions,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        for dataset_index, dataset in enumerate(datasets):
            values = table.loc[
                (table.dataset == dataset) & (table.model == model), value
            ].astype(float).to_numpy()
            if len(values):
                jitter = np.linspace(-0.035, 0.035, len(values))
                ax.scatter(
                    np.full(len(values), positions[dataset_index]) + jitter,
                    values,
                    s=13,
                    color="black",
                    alpha=0.65,
                    zorder=3,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[dataset] for dataset in datasets])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8)


def aggregate_performance(performance: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        performance.groupby(
            ["dataset", "model", "seed", "perturbation_ratio"], as_index=False
        )["balanced_acc"]
        .mean()
    )
    return (
        per_seed.groupby(["dataset", "model", "perturbation_ratio"])[
            "balanced_acc"
        ]
        .agg(mean="mean", sd="std")
        .reset_index()
    )


def plot_dataset_curve(ax, performance: pd.DataFrame, dataset: str):
    aggregate = aggregate_performance(performance)
    for model in MODEL_ORDER:
        subset = aggregate[
            (aggregate.dataset == dataset) & (aggregate.model == model)
        ].sort_values("perturbation_ratio")
        if subset.empty:
            continue
        x = subset.perturbation_ratio.to_numpy(float)
        y = subset["mean"].to_numpy(float)
        error = subset["sd"].fillna(0).to_numpy(float)
        ax.plot(
            x,
            y,
            marker="o",
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
        )
        ax.fill_between(x, y - error, y + error, color=MODEL_COLORS[model], alpha=0.15)
    ax.set_title(DATASET_SHORT[dataset], fontsize=9)
    ax.set_xlabel("Added isolated-input ratio")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=False, fontsize=7)


def paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for dataset in DATASET_ORDER:
        subset = metrics[metrics.dataset == dataset]
        for metric in ["test_acc", "isolated_perturbation_auc"]:
            if metric not in subset.columns:
                continue
            wide = subset.pivot_table(
                index="seed", columns="model", values=metric, aggfunc="first"
            ).dropna()
            if not {"static", "cds"}.issubset(wide.columns) or wide.empty:
                continue
            differences = (wide.cds - wide.static).to_numpy(float)
            bootstrap = np.array(
                [
                    rng.choice(differences, size=len(differences), replace=True).mean()
                    for _ in range(10000)
                ]
            )
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_paired_seeds": len(differences),
                "static_mean": float(wide.static.mean()),
                "cds_mean": float(wide.cds.mean()),
                "paired_difference_mean": float(differences.mean()),
                "paired_difference_sd": (
                    float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
                ),
                "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
            })
    return pd.DataFrame(rows)


def plot_auc_effects(ax, effects: pd.DataFrame, datasets: Sequence[str]):
    subset = effects[effects.metric == "isolated_perturbation_auc"].set_index("dataset")
    y = np.arange(len(datasets))
    means = np.array([subset.loc[d, "paired_difference_mean"] for d in datasets])
    low = np.array([subset.loc[d, "bootstrap_ci95_low"] for d in datasets])
    high = np.array([subset.loc[d, "bootstrap_ci95_high"] for d in datasets])
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - low, high - means]),
        fmt="o",
        color="#e66101",
        capsize=3,
    )
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_SHORT[d] for d in datasets])
    ax.set_xlabel(r"Paired $\Delta\mathcal{A}_{\mathrm{iso}}$ (CDS - static)")
    ax.invert_yaxis()


def plot_release_selectivity(ax, selectivity: pd.DataFrame, datasets: Sequence[str]):
    cds = selectivity[selectivity.model == "cds"].copy()
    per_seed = (
        cds.groupby(["dataset", "seed"], as_index=False)[
            [
                "mean_supported_release",
                "mean_isolated_release",
                "transmission_selectivity",
            ]
        ]
        .mean()
    )
    x = np.arange(len(datasets))
    width = 0.34
    columns = [
        ("mean_isolated_release", "Isolated additions", "#80b1d3"),
        ("mean_supported_release", "Supported inputs", "#fb8072"),
    ]
    for index, (column, label, color) in enumerate(columns):
        means, errors = [], []
        for dataset in datasets:
            values = per_seed.loc[per_seed.dataset == dataset, column].astype(float)
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=color,
            label=label,
        )
    for dataset_index, dataset in enumerate(datasets):
        value = per_seed.loc[
            per_seed.dataset == dataset, "transmission_selectivity"
        ].mean()
        if np.isfinite(value):
            ax.text(
                x[dataset_index],
                0.98,
                rf"$S_{{\mathrm{{tx}}}}={value:.2f}$",
                ha="center",
                va="top",
                fontsize=7,
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("First-layer release probability")
    ax.legend(frameon=False, fontsize=7)


def make_task_table(metrics: pd.DataFrame, datasets: Sequence[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for model in MODEL_ORDER:
            subset = metrics[(metrics.dataset == dataset) & (metrics.model == model)]
            row = {
                "dataset": dataset,
                "modality": subset.modality.iloc[0] if len(subset) else "",
                "model": model,
                "n_seeds": int(subset.seed.nunique()) if len(subset) else 0,
            }
            for column in [
                "test_acc",
                "test_balanced_acc",
                "test_macro_f1",
                "isolated_perturbation_auc",
                "mean_supported_release",
                "mean_isolated_release",
                "transmission_selectivity",
            ]:
                values = subset[column].astype(float) if column in subset else pd.Series(dtype=float)
                row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{column}_sd"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            rows.append(row)
    table = pd.DataFrame(rows)
    effects = paired_effects(metrics)
    delta = effects[effects.metric == "isolated_perturbation_auc"][
        ["dataset", "paired_difference_mean", "bootstrap_ci95_low", "bootstrap_ci95_high"]
    ].rename(columns={
        "paired_difference_mean": "delta_auc_cds_minus_static",
        "bootstrap_ci95_low": "delta_auc_ci95_low",
        "bootstrap_ci95_high": "delta_auc_ci95_high",
    })
    return table.merge(delta, on="dataset", how="left")


def read_ablation(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    protocol_column = "isolated_perturbation_protocol"
    if protocol_column not in table.columns:
        raise RuntimeError(f"{path} lacks new perturbation-protocol metadata.")
    observed = set(table[protocol_column].dropna().astype(str))
    if observed != {EXPECTED_PROTOCOL}:
        raise RuntimeError(f"Incompatible ablation protocol in {path}: {sorted(observed)}")
    return table


def make_ablation_table(ablation: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "ablation_variant",
        "seed",
        "test_acc",
        "isolated_perturbation_auc",
        "transmission_selectivity",
    }
    missing = required.difference(ablation.columns)
    if missing:
        raise RuntimeError(f"Ablation results lack Table 2 columns: {sorted(missing)}")
    table = (
        ablation.groupby(["dataset", "ablation_variant"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            clean_accuracy_mean=("test_acc", "mean"),
            clean_accuracy_sd=("test_acc", "std"),
            isolated_perturbation_auc_mean=("isolated_perturbation_auc", "mean"),
            isolated_perturbation_auc_sd=("isolated_perturbation_auc", "std"),
            transmission_selectivity_mean=("transmission_selectivity", "mean"),
            transmission_selectivity_sd=("transmission_selectivity", "std"),
        )
    )
    dataset_rank = {value: index for index, value in enumerate(DATASET_ORDER)}
    variant_rank = {value: index for index, value in enumerate(ABLATION_ORDER)}
    table["_dataset_rank"] = table.dataset.map(dataset_rank)
    table["_variant_rank"] = table.ablation_variant.map(variant_rank)
    table = table.sort_values(["_dataset_rank", "_variant_rank"]).drop(
        columns=["_dataset_rank", "_variant_rank"]
    )
    table["clean_accuracy_percent_mean"] = 100.0 * table.clean_accuracy_mean
    table["clean_accuracy_percent_sd"] = 100.0 * table.clean_accuracy_sd
    table["isolated_perturbation_protocol"] = EXPECTED_PROTOCOL
    return table.reset_index(drop=True)


def plot_all(
    root: Path,
    fig_dir: Path,
    seeds: Optional[Sequence[int]],
    datasets: Sequence[str],
    ablation_path: Optional[Path],
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics, performance, selectivity = collect_logs(root, datasets, seeds)
    if metrics.empty or performance.empty or selectivity.empty:
        raise FileNotFoundError(
            "Section 2.6 outputs are incomplete; run all three training scripts first."
        )
    datasets_present = available_datasets(metrics, datasets)
    if not datasets_present:
        raise FileNotFoundError("No requested datasets were found.")

    metrics.to_csv(fig_dir / "merged_metrics_summary.csv", index=False)
    performance.to_csv(
        fig_dir / "merged_isolated_perturbation_performance.csv", index=False
    )
    selectivity.to_csv(fig_dir / "merged_transmission_selectivity.csv", index=False)
    effects = paired_effects(metrics)
    effects.to_csv(fig_dir / "table_paired_seed_effects.csv", index=False)
    make_task_table(metrics, datasets_present).to_csv(
        fig_dir / "table_sensory_task_performance.csv", index=False
    )
    ablation = read_ablation(ablation_path)
    if not ablation.empty:
        make_ablation_table(ablation).to_csv(
            fig_dir / "table_cds_ablation_summary.csv", index=False
        )

    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    panel_labels = ["a", "b", "c", "d", "e", "f"]
    for axis, label in zip(axes.flat, panel_labels):
        axis.text(-0.17, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=12)

    grouped_seed_bar(
        axes[0, 0], metrics, "test_acc", "Best test accuracy", datasets_present, (0, 1.02)
    )
    plot_dataset_curve(axes[0, 1], performance, "dvsgesture")
    plot_dataset_curve(axes[0, 2], performance, "shd")
    plot_dataset_curve(axes[1, 0], performance, "stmnist")
    plot_auc_effects(axes[1, 1], effects, datasets_present)
    plot_release_selectivity(axes[1, 2], selectivity, datasets_present)
    figure.tight_layout()
    figure.savefig(fig_dir / "fig6_temporal_perturbation_robustness.png", dpi=600)
    figure.savefig(fig_dir / "fig6_temporal_perturbation_robustness.pdf")
    print(f"Saved Fig. 6 and manuscript source tables to {fig_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Section 2.6 results.")
    parser.add_argument(
        "--root",
        default="results/section_2_6",
    )
    parser.add_argument(
        "--fig_dir",
        default="results/section_2_6/fig6",
    )
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--datasets", default="dvsgesture,shd,stmnist")
    parser.add_argument(
        "--ablation_path",
        default="results/section_2_6/ablation/ablation_task_table.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_all(
        Path(args.root),
        Path(args.fig_dir),
        parse_seeds(args.seeds),
        [value.strip().lower() for value in args.datasets.split(",") if value.strip()],
        Path(args.ablation_path) if args.ablation_path else None,
    )


if __name__ == "__main__":
    main()
