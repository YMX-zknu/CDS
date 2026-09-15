from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


DATASETS = ["dvsgesture", "shd", "stmnist"]
LABELS = {"dvsgesture": "DVS Gesture", "shd": "SHD", "stmnist": "ST-MNIST"}
SHORT = {"dvsgesture": "DVS", "shd": "SHD", "stmnist": "ST"}
NUM_CLASSES = {"dvsgesture": 11, "shd": 20, "stmnist": 10}
MODELS = ["static", "constant_release", "cds"]
MODEL_LABELS = {"static": "Static", "constant_release": "Constant release", "cds": "Full CDS"}
COLORS = {"static": "#777777", "constant_release": "#D89C36", "cds": "#2B6CB0"}
PJ_PER_UJ = 1.0e6
EXPECTED_ARCHITECTURE = "presynaptic_cds_v4_dataset_specific"
EXPECTED_ORDER = "CDS->Conv/Linear->BN(if used)->LIF"
EXPECTED_PERTURBATION_PROTOCOL = "amplitude_matched_maximum_guard_v4"
EXPECTED_ANALYSIS_VERSION = "transmission_weighted_selectivity_v2"


def set_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.dpi": 600,
    })


def parse_seeds(text: str) -> Optional[List[int]]:
    if str(text).strip().lower() == "all": return None
    return [int(x) for x in str(text).split(",") if x.strip()]


def seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        hit = re.fullmatch(r"seed_(\d+)", part)
        if hit: return int(hit.group(1))
    return None


def collect(root: Path, seeds: Optional[Sequence[int]]):
    buckets = {"summary": [], "layer": [], "cds": [], "energy": []}
    names = {
        "summary": "efficiency_summary.csv", "layer": "layerwise_operations.csv",
        "cds": "cds_state_dynamics.csv", "energy": "energy_overhead_decomposition.csv",
    }
    for dataset in DATASETS:
        dirs = sorted((root / dataset).glob("seed_*")) if seeds is None else [root / dataset / f"seed_{s}" for s in seeds]
        for directory in dirs:
            seed = seed_from_path(directory)
            if seed is None: continue
            for key, filename in names.items():
                path = directory / "logs" / filename
                if path.exists():
                    frame = pd.read_csv(path)
                    required = {
                        "section_2_7_analysis_version",
                        "architecture_version",
                        "transmission_order",
                        "isolated_perturbation_protocol",
                    }
                    missing = required.difference(frame.columns)
                    if missing:
                        raise RuntimeError(
                            f"{path} lacks architecture metadata {sorted(missing)}; rerun Section 2.7."
                        )
                    if not frame["architecture_version"].eq(EXPECTED_ARCHITECTURE).all():
                        raise RuntimeError(f"Legacy architecture results detected in {path}.")
                    if not frame["section_2_7_analysis_version"].eq(
                        EXPECTED_ANALYSIS_VERSION
                    ).all():
                        raise RuntimeError(
                            f"Legacy Section 2.7 results detected in {path}; "
                            "rerun the three analysis scripts before plotting."
                        )
                    if not frame["transmission_order"].eq(EXPECTED_ORDER).all():
                        raise RuntimeError(f"Unexpected CDS placement detected in {path}.")
                    if not frame["isolated_perturbation_protocol"].eq(
                        EXPECTED_PERTURBATION_PROTOCOL
                    ).all():
                        raise RuntimeError(
                            f"Incompatible isolated-perturbation results detected in {path}."
                        )
                    if "seed" not in frame: frame["seed"] = seed
                    if "dataset" not in frame: frame["dataset"] = dataset
                    buckets[key].append(frame)
    return tuple(pd.concat(buckets[k], ignore_index=True) if buckets[k] else pd.DataFrame()
                 for k in ["summary", "layer", "cds", "energy"])


def seed_average(frame: pd.DataFrame, group_cols: Sequence[str], value_cols: Sequence[str]):
    cols = [c for c in value_cols if c in frame.columns]
    if frame.empty or not cols: return pd.DataFrame()
    first = frame.groupby(list(group_cols) + ["seed"], as_index=False)[cols].mean()
    return first


def clean_seed_tables(summary: pd.DataFrame, energy: pd.DataFrame):
    clean = summary[summary.perturbation_ratio == 0].copy()
    clean = seed_average(clean, ["dataset", "model"], [
        "acc", "balanced_acc", "output_spikes_per_sample", "synops_per_sample",
        "transmission_weighted_propagation_per_sample",
        "mac_ops_per_sample", "analog_mac_ops_per_sample", "graded_mac_ops_per_sample", "ac_ops_per_sample",
        "normalized_synops_vs_clean_static",
        "normalized_weighted_propagation_vs_clean_static",
        "propagation_efficiency",
        "chance_corrected_balanced_accuracy", "transmission_efficiency",
    ])
    en = energy[energy.perturbation_ratio == 0].copy()
    en = seed_average(en, ["dataset", "model"], [
        "synaptic_energy_pj", "neuron_energy_pj", "cds_arithmetic_memory_exact_pj",
        "cds_arithmetic_memory_lut_pj", "total_energy_exact_pj", "total_energy_lut_pj",
        "inference_energy_exact_uj_per_sample", "inference_energy_lut_uj_per_sample",
        "normalized_energy_exact_vs_clean_static", "normalized_energy_lut_vs_clean_static",
        "cds_state_updates_per_sample", "cds_state_accesses_per_sample",
        "cds_persistent_state_elements_per_sample", "cds_persistent_state_memory_fp32_bytes",
        "total_parameter_count", "cds_parameter_count",
        "mac_ops_per_sample", "analog_mac_ops_per_sample", "graded_mac_ops_per_sample", "ac_ops_per_sample",
    ])


    if "chance_corrected_balanced_accuracy" not in clean.columns:
        clean["chance_corrected_balanced_accuracy"] = np.nan
    if "transmission_efficiency" not in clean.columns:
        clean["transmission_efficiency"] = np.nan
    for dataset in DATASETS:
        mask = clean.dataset == dataset
        chance = 1.0 / float(NUM_CLASSES[dataset])
        quality = (clean.loc[mask, "balanced_acc"] - chance) / (1.0 - chance)
        clean.loc[mask, "chance_corrected_balanced_accuracy"] = quality
        clean.loc[mask, "transmission_efficiency"] = (
            quality
            / clean.loc[
                mask, "normalized_weighted_propagation_vs_clean_static"
            ].clip(lower=1e-12)
        )
    return clean, en


def inference_energy_seed_table(clean_energy: pd.DataFrame) -> pd.DataFrame:


    if clean_energy.empty:
        return pd.DataFrame()
    frame = clean_energy.copy()
    conversions = {
        "inference_energy_exact_uj_per_sample": "total_energy_exact_pj",
        "inference_energy_lut_uj_per_sample": "total_energy_lut_pj",
    }
    for target, source in conversions.items():
        if target not in frame.columns:
            if source not in frame.columns:
                raise KeyError(f"Energy results contain neither {target} nor {source}.")
            frame[target] = frame[source] / PJ_PER_UJ

    rows = []
    scenarios = [
        ("static", "Static", "inference_energy_lut_uj_per_sample"),
        ("cds", "CDS--LUT", "inference_energy_lut_uj_per_sample"),
        ("cds", "CDS--exact exp", "inference_energy_exact_uj_per_sample"),
    ]
    for dataset in DATASETS:
        for model, implementation, column in scenarios:
            subset = frame[(frame.dataset == dataset) & (frame.model == model)]
            for _, row in subset.iterrows():
                rows.append({
                    "dataset": dataset,
                    "seed": int(row.seed),
                    "model": model,
                    "implementation": implementation,
                    "inference_energy_uj_per_sample": float(row[column]),
                    "process_node_nm": 45,
                    "energy_type": "hardware-oriented operation-count estimate",
                })
    return pd.DataFrame(rows)


def inference_energy_summary(energy_seed: pd.DataFrame) -> pd.DataFrame:

    if energy_seed.empty:
        return pd.DataFrame()
    return energy_seed.groupby(
        ["dataset", "model", "implementation", "process_node_nm", "energy_type"],
        as_index=False,
    ).agg(
        n_seeds=("seed", "nunique"),
        inference_energy_uj_mean=("inference_energy_uj_per_sample", "mean"),
        inference_energy_uj_sd=("inference_energy_uj_per_sample", "std"),
    )


def perturbation_seed_table(summary: pd.DataFrame):
    perturbed = seed_average(summary, ["dataset", "model", "perturbation_ratio"], [
        "balanced_acc", "perturbation_balanced_acc_retention", "synops_per_sample",
        "transmission_weighted_propagation_per_sample",
        "normalized_weighted_propagation_vs_clean_static",
        "mean_supported_release", "mean_isolated_release",
        "transmission_selectivity", "added_event_bins_per_sample",
        "realized_perturbation_ratio", "realized_target_fraction",
    ])
    return perturbed


def paired_weighted_propagation(perturbed: pd.DataFrame):

    rows = []
    for (dataset, seed, ratio), sub in perturbed.groupby(
        ["dataset", "seed", "perturbation_ratio"]
    ):
        wide = sub.set_index("model")
        if {"static", "cds"}.issubset(wide.index):
            static_load = float(
                wide.loc["static", "transmission_weighted_propagation_per_sample"]
            )
            cds_load = float(
                wide.loc["cds", "transmission_weighted_propagation_per_sample"]
            )
            ratio_value = cds_load / static_load if static_load > 0 else np.nan
            rows.append({
                "dataset": dataset, "seed": seed,
                "perturbation_ratio": ratio,
                "static_weighted_propagation_per_sample": static_load,
                "cds_weighted_propagation_per_sample": cds_load,
                "cds_to_static_weighted_propagation": ratio_value,
                "weighted_propagation_suppression": 1.0 - ratio_value,
            })
    return pd.DataFrame(rows)


def manuscript_table(clean: pd.DataFrame, perturbed: pd.DataFrame,
                     weighted: pd.DataFrame, selectivity_ratio: float):
    rows = []
    for dataset in DATASETS:
        c = clean[clean.dataset == dataset]
        w = weighted[
            (weighted.dataset == dataset)
            & np.isclose(weighted.perturbation_ratio, selectivity_ratio)
        ]
        p = perturbed[
            (perturbed.dataset == dataset)
            & np.isclose(perturbed.perturbation_ratio, selectivity_ratio)
        ]
        for model in [m for m in MODELS if m in set(c.model)]:
            cm = c[c.model == model]
            pm = p[p.model == model]
            rows.append({
                "dataset": dataset, "model": model, "n_seeds": int(cm.seed.nunique()),
                "balanced_acc_mean": cm.balanced_acc.mean(), "balanced_acc_sd": cm.balanced_acc.std(),
                "normalized_synops_mean": cm.normalized_synops_vs_clean_static.mean(),
                "normalized_synops_sd": cm.normalized_synops_vs_clean_static.std(),
                "weighted_propagation_per_sample_mean": (
                    cm.transmission_weighted_propagation_per_sample.mean()
                ),
                "mac_ops_per_sample_mean": cm.mac_ops_per_sample.mean(),
                "analog_mac_ops_per_sample_mean": cm.analog_mac_ops_per_sample.mean(),
                "graded_mac_ops_per_sample_mean": cm.graded_mac_ops_per_sample.mean(),
                "ac_ops_per_sample_mean": cm.ac_ops_per_sample.mean(),
                "propagation_efficiency_mean": cm.propagation_efficiency.mean(),
                "transmission_efficiency_mean": cm.transmission_efficiency.mean(),
                "transmission_efficiency_sd": cm.transmission_efficiency.std(),
                "weighted_propagation_suppression_mean": (
                    w.weighted_propagation_suppression.mean()
                    if model == "cds" and len(w) else np.nan
                ),
                "transmission_selectivity_mean": (
                    pm.transmission_selectivity.mean() if len(pm) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def supplementary_overhead_table(energy: pd.DataFrame):

    columns = [
        "dataset", "model", "seed", "cds_state_updates_per_sample",
        "cds_persistent_state_memory_fp32_bytes", "cds_mul_ops_per_sample",
        "cds_add_ops_per_sample", "cds_nonlinear_ops_per_sample",
        "cds_state_accesses_per_sample", "cds_parameter_count",
        "normalized_energy_exact_vs_clean_static",
        "normalized_energy_lut_vs_clean_static",
        "inference_energy_exact_uj_per_sample",
        "inference_energy_lut_uj_per_sample",
    ]
    available = [column for column in columns if column in energy.columns]
    return energy[available].copy()


def paired_effects(clean: pd.DataFrame):
    rng, rows = np.random.default_rng(2027), []
    metrics = [
        (clean, "balanced_acc"), (clean, "normalized_synops_vs_clean_static"),
        (clean, "normalized_weighted_propagation_vs_clean_static"),
        (clean, "propagation_efficiency"),
        (clean, "transmission_efficiency"),
    ]
    for dataset in DATASETS:
        for frame, metric in metrics:
            sub = frame[frame.dataset == dataset]
            if metric not in sub: continue
            wide = sub.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean").dropna()
            if not {"static", "cds"}.issubset(wide.columns): continue
            diff = (wide.cds - wide.static).to_numpy()
            boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(10000)])
            rows.append({
                "dataset": dataset, "metric": metric, "n_paired_seeds": len(diff),
                "cds_minus_static_mean": diff.mean(),
                "cds_minus_static_sd": diff.std(ddof=1) if len(diff) > 1 else 0.0,
                "bootstrap_ci95_low": np.quantile(boot, .025),
                "bootstrap_ci95_high": np.quantile(boot, .975),
            })
    return pd.DataFrame(rows)


def origin_export(out: Path, clean: pd.DataFrame, perturbed: pd.DataFrame,
                  weighted: pd.DataFrame,
                  selectivity_ratio: float):
    origin = out / "origin_subplot_csv"; origin.mkdir(parents=True, exist_ok=True)
    clean.to_csv(origin / "Fig7b_clean_propagation.csv", index=False)
    for dataset in DATASETS:
        weighted[weighted.dataset == dataset].to_csv(
            origin / f"Fig7c_{dataset}_weighted_perturbation_propagation.csv",
            index=False,
        )
    clean.to_csv(origin / "Fig7d_accuracy_synops_pareto.csv", index=False)
    clean[[
        "dataset", "model", "seed", "balanced_acc",
        "chance_corrected_balanced_accuracy",
        "normalized_weighted_propagation_vs_clean_static",
        "transmission_efficiency",
    ]].to_csv(
        origin / "Fig7e_accuracy_normalized_transmission_efficiency.csv",
        index=False,
    )
    perturbed[
        np.isclose(perturbed.perturbation_ratio, selectivity_ratio)
    ].to_csv(
        origin / "Fig7f_isolated_perturbation_selectivity.csv", index=False
    )


def design_panel(ax):
    ax.axis("off"); ax.text(-.05, 1.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=11)
    boxes = [(0.02,.68,"Visual\nDVS"),(0.02,.38,"Auditory\nSHD"),(0.02,.08,"Tactile\nST-MNIST"),
             (.38,.38,"Static / CDS\nCDS before weights"),(.72,.55,"Spikes + SynOps\nisolated perturbations"),
             (.72,.20,"Weighted propagation +\ntransmission selectivity")]
    for x,y,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),.24,.16,boxstyle="round,pad=.02",fill=False,lw=.8))
        ax.text(x+.12,y+.08,text,ha="center",va="center",fontsize=7)
    for y in [.76,.46,.16]: ax.annotate("",(.38,.46),(.26,y),arrowprops={"arrowstyle":"->","lw":.8})
    ax.annotate("",(.72,.63),(.62,.49),arrowprops={"arrowstyle":"->","lw":.8})
    ax.annotate("",(.72,.28),(.62,.43),arrowprops={"arrowstyle":"->","lw":.8})


def panel_clean(ax, clean):
    ax.text(-.18, 1.03, "b", transform=ax.transAxes, fontweight="bold", fontsize=11)
    x = np.arange(len(DATASETS)); width=.28
    for j, model in enumerate(["static","cds"]):
        vals=[]; errs=[]
        for d in DATASETS:
            s=clean[(clean.dataset==d)&(clean.model==model)]
            vals.append(s.normalized_synops_vs_clean_static.mean()); errs.append(s.normalized_synops_vs_clean_static.std())
        ax.bar(x+(j-.5)*width,vals,width,yerr=errs,capsize=2,color=COLORS[model],label=MODEL_LABELS[model])
    ax.axhline(1,color="black",lw=.6,ls="--"); ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_ylabel("Normalized SynOps"); ax.set_title("Clean propagation load"); ax.legend(frameon=False,fontsize=7)


def panel_perturbation(ax, weighted, dataset, show_ylabel=False):
    s = weighted[weighted.dataset == dataset]
    agg = s.groupby("perturbation_ratio").cds_to_static_weighted_propagation.agg(
        ["mean", "std"]
    ).reset_index()
    ax.axhline(1.0, color=COLORS["static"], lw=1.0, ls="--",
               label="Static reference")
    ax.plot(agg.perturbation_ratio, agg["mean"], marker="o", ms=3,
            color=COLORS["cds"], label="Full CDS / Static")
    ax.fill_between(
        agg.perturbation_ratio,
        agg["mean"] - agg["std"].fillna(0),
        agg["mean"] + agg["std"].fillna(0),
        color=COLORS["cds"], alpha=.15
    )
    ax.set_title(LABELS[dataset],fontsize=8)
    ax.set_xlabel("Added isolated-bin ratio")
    if show_ylabel:
        ax.set_ylabel("CDS / Static weighted propagation")


def panel_pareto(ax, clean):
    ax.text(-.18, 1.03, "d", transform=ax.transAxes, fontweight="bold", fontsize=11)
    for dataset in DATASETS:
        points=[]
        for model in ["static","cds"]:
            s=clean[(clean.dataset==dataset)&(clean.model==model)]
            if s.empty: continue
            px,py=s.normalized_synops_vs_clean_static.mean(),s.balanced_acc.mean(); points.append((px,py))
            ax.scatter(px,py,color=COLORS[model],s=28,edgecolor="white",zorder=3)
            ax.text(px,py,SHORT[dataset],fontsize=6,ha="left",va="bottom")
        if len(points)==2: ax.annotate("",points[1],points[0],arrowprops={"arrowstyle":"->","lw":.8,"color":"#444444"})
    ax.set_xlabel("Normalized SynOps"); ax.set_ylabel("Balanced accuracy"); ax.set_title("Accuracy–propagation trade-off")


def panel_transmission_efficiency(ax, clean):
    ax.text(-.16, 1.03, "e", transform=ax.transAxes, fontweight="bold", fontsize=11)
    if clean.empty or "transmission_efficiency" not in clean:
        ax.text(.5, .5, "Transmission-efficiency data unavailable", ha="center")
        return
    x = np.arange(len(DATASETS)); width = .28
    for j, model in enumerate(["static", "cds"]):
        means, errors = [], []
        for dataset in DATASETS:
            values = clean[
                (clean.dataset == dataset) & (clean.model == model)
            ].transmission_efficiency.dropna()
            means.append(values.mean() if len(values) else np.nan)
            errors.append(values.std() if len(values) > 1 else 0.0)
        ax.bar(
            x + (j - .5) * width, means, width, yerr=errors, capsize=2,
            color=COLORS[model], label=MODEL_LABELS[model],
        )
    ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_ylabel(r"Transmission efficiency $\eta_{\mathrm{tx}}$")
    ax.set_title("Accuracy-normalized\ntransmission efficiency", fontsize=8)
    ax.legend(frameon=False, fontsize=7)


def panel_selectivity(ax, perturbed, selectivity_ratio):
    ax.text(-.10, 1.03, "f", transform=ax.transAxes, fontweight="bold", fontsize=11)
    selected = perturbed[np.isclose(
        perturbed.perturbation_ratio, selectivity_ratio
    )]
    x = np.arange(len(DATASETS)); width = .28
    for j, model in enumerate(["static", "cds"]):
        values, errors = [], []
        for dataset in DATASETS:
            s = selected[
                (selected.dataset == dataset) & (selected.model == model)
            ].transmission_selectivity.dropna()
            values.append(s.mean() if len(s) else np.nan)
            errors.append(s.std() if len(s) > 1 else 0.0)
        ax.bar(x + (j - .5) * width, values, width, yerr=errors,
               capsize=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax.axhline(0, color="black", lw=.6)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[d] for d in DATASETS])
    ax.set_ylabel(r"Isolated-event selectivity $\mathcal{S}_{iso}$")
    ax.set_title(
        f"Selective attenuation of isolated perturbations (r={selectivity_ratio:g})"
    )
    ax.legend(frameon=False, fontsize=7)


def plot_all(root: Path, out: Path, seeds: Optional[Sequence[int]],
             selectivity_ratio: float):
    out.mkdir(parents=True,exist_ok=True); set_style()
    summary, layer, cds, energy = collect(root,seeds)
    if summary.empty or energy.empty: raise FileNotFoundError(f"No Section 2.7 results found under {root}")
    clean,en=clean_seed_tables(summary,energy)
    energy_seed=inference_energy_seed_table(en)
    energy_table=inference_energy_summary(energy_seed)
    perturbed=perturbation_seed_table(summary)
    weighted=paired_weighted_propagation(perturbed)
    available_ratios = np.asarray(sorted(
        perturbed.loc[perturbed.perturbation_ratio > 0, "perturbation_ratio"].unique()
    ))
    if available_ratios.size == 0 or not np.any(
        np.isclose(available_ratios, selectivity_ratio)
    ):
        raise ValueError(
            f"selectivity_ratio={selectivity_ratio:g} is unavailable; "
            f"positive analysed ratios are {available_ratios.tolist()}."
        )
    table=manuscript_table(clean,perturbed,weighted,selectivity_ratio)
    effects=paired_effects(clean)
    overhead=supplementary_overhead_table(en)
    summary.to_csv(out/"merged_efficiency_summary.csv",index=False); layer.to_csv(out/"merged_layerwise_operations.csv",index=False)
    cds.to_csv(out/"merged_cds_state_dynamics.csv",index=False); energy.to_csv(out/"merged_energy_decomposition.csv",index=False)
    table.to_csv(out/"table2_propagation_efficiency_summary.csv",index=False)
    weighted.to_csv(out/"table_weighted_perturbation_propagation.csv",index=False)
    overhead.to_csv(out/"supplementary_table_cds_implementation_overhead.csv",index=False)
    energy_seed.to_csv(out/"supplementary_inference_energy_45nm_seed_level.csv",index=False)
    energy_table.to_csv(out/"supplementary_table_inference_energy_45nm_summary.csv",index=False)
    effects.to_csv(out/"table_paired_seed_effects.csv",index=False)
    if not cds.empty and "release_mean" in cds:
        state_time = cds[cds.perturbation_ratio == 0].groupby(
            ["dataset","model","seed","layer","t"], as_index=False
        )[["release_mean","calcium_mean","transmission_ratio"]].mean()
        state_summary = state_time.groupby(["dataset","model","seed","layer"], as_index=False).agg(
            release_mean=("release_mean","mean"), release_time_sd=("release_mean","std"),
            calcium_mean=("calcium_mean","mean"), transmission_ratio_mean=("transmission_ratio","mean"),
        )
        state_summary["release_time_cv"] = state_summary.release_time_sd / state_summary.release_mean.abs().clip(lower=1e-12)
        state_summary.to_csv(out/"supplementary_cds_dynamic_state_summary.csv",index=False)
    if not layer.empty:
        layer.groupby(["dataset","model","seed","layer"],as_index=False)[["synops","transmission_weighted_propagation","output_spikes","neuron_updates"]].sum().to_csv(out/"supplementary_layerwise_operations.csv",index=False)
    origin_export(out,clean,perturbed,weighted,selectivity_ratio)

    fig=plt.figure(figsize=(12,9)); gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.05])
    axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axd=fig.add_subplot(gs[0,2])
    caxes=[fig.add_subplot(gs[1,i]) for i in range(3)]; axe=fig.add_subplot(gs[2,0]); axf=fig.add_subplot(gs[2,1:])
    design_panel(axa); panel_clean(axb,clean); panel_pareto(axd,clean)
    for i,(ax,d) in enumerate(zip(caxes,DATASETS)):
        panel_perturbation(ax,weighted,d,show_ylabel=i==0)
    caxes[0].text(-.18,1.03,"c",transform=caxes[0].transAxes,fontweight="bold",fontsize=11)
    if len(caxes[-1].lines): caxes[-1].legend(frameon=False,fontsize=7)
    panel_transmission_efficiency(axe,clean)
    panel_selectivity(axf,perturbed,selectivity_ratio)
    fig.suptitle("CDS attenuates redundant propagation across sensory event streams",fontsize=12,y=.995)
    fig.tight_layout(); fig.savefig(out/"fig7_cross_modal_propagation_efficiency.png",dpi=600,bbox_inches="tight")
    fig.savefig(out/"fig7_cross_modal_propagation_efficiency.pdf",bbox_inches="tight"); plt.close(fig)
    print(f"Saved Fig. 7, source CSVs and tables to {out}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="results/section_2_7")
    p.add_argument("--out_dir",default="results/section_2_7/fig7")
    p.add_argument("--seeds",default="all")
    p.add_argument("--selectivity_ratio",type=float,default=0.1)
    a=p.parse_args()
    plot_all(Path(a.root),Path(a.out_dir),parse_seeds(a.seeds),a.selectivity_ratio)


if __name__=="__main__": main()
