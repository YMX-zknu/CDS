import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from spikingjelly.activation_based import base
    BaseMemoryModule = base.MemoryModule
except Exception:
    class BaseMemoryModule(nn.Module):
        def __init__(self):
            super().__init__()
            self._memories = {}
        def register_memory(self, name, value):
            self._memories[name] = value
            setattr(self, name, value)
        def reset(self):
            for name, value in self._memories.items():
                setattr(self, name, value.clone().detach() if torch.is_tensor(value) else value)

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def sigmoid_np(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))

def softplus_np(x: float) -> float:
    return float(np.log1p(np.exp(float(x))))

def inv_softplus_np(y: float) -> float:
    y = float(y)
    if y <= 0:
        raise ValueError("softplus inverse requires y > 0")
    return y if y > 20 else float(np.log(np.expm1(y)))

class CalciumDynamicSynapse(BaseMemoryModule):
    def __init__(self, tau_ca=20.0, ca_influx=0.5, u0=0.1, u_max=1.0, k_ca=1.0,
                 w_static=1.0, learnable=False, tau_is_effective=False):
        super().__init__()
        if tau_is_effective:
            tau_ca = inv_softplus_np(tau_ca)
        def make_param(x):
            return nn.Parameter(torch.tensor(float(x), dtype=torch.float32), requires_grad=learnable)
        self.tau_ca = make_param(tau_ca)
        self.ca_influx = make_param(ca_influx)
        self.u0 = make_param(u0)
        self.u_max = make_param(u_max)
        self.k_ca = make_param(k_ca)
        self.w_static = make_param(w_static)
        self.eps = 1e-6
        self.register_memory("ca", torch.tensor(0.0))
        self.register_memory("u", torch.tensor(0.0))
    def forward(self, spike: torch.Tensor):
        tau_ca = torch.nn.functional.softplus(self.tau_ca)
        ca_influx = torch.nn.functional.softplus(self.ca_influx)
        k_ca = torch.nn.functional.softplus(self.k_ca)
        u0 = torch.sigmoid(self.u0)
        u_max = torch.sigmoid(self.u_max)
        self.ca = self.ca * torch.exp(-1.0 / (tau_ca + self.eps)) + spike * ca_influx
        self.u = u0 + (u_max - u0) * (1.0 - torch.exp(-k_ca * self.ca))
        return self.w_static * self.u * spike
    def reset(self):
        super().reset()
        self.ca = torch.tensor(0.0)
        self.u = torch.tensor(0.0)

@dataclass
class CDSParams:
    tau_ca: float = 20.0
    ca_influx: float = 0.5
    u0: float = 0.1
    u_max: float = 1.0
    k_ca: float = 1.0
    w_static: float = 1.0
    tau_is_effective: bool = False

def effective_u0(params: CDSParams) -> float:
    return sigmoid_np(params.u0)

def effective_umax(params: CDSParams) -> float:
    return sigmoid_np(params.u_max)

def make_spike_train(T: int, times: List[int]) -> np.ndarray:
    s = np.zeros(T, dtype=float)
    for t in times:
        if 0 <= int(t) < T:
            s[int(t)] = 1.0
    return s

def canonical_spike_patterns(T: int = 180, seed: int = SEED) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    structured = list(range(70, 82))
    isolated = [15, 45, 125, 155]
    low_frequency_regular = list(range(20, 170, 25))
    burst_silence_burst = list(range(30, 40)) + list(range(110, 120))
    jittered_burst = [68, 70, 73, 77, 78, 82, 85]
    fragmented_burst = [70, 74, 78, 82, 86, 90]
    poisson_pool = np.setdiff1d(np.arange(10, T - 10), np.array(structured + isolated))
    poisson_background = sorted(rng.choice(poisson_pool, size=18, replace=False).tolist())
    patterns = {
        "isolated_noise": make_spike_train(T, isolated),
        "low_frequency_regular": make_spike_train(T, low_frequency_regular),
        "structured_burst": make_spike_train(T, structured),
        "burst_silence_burst": make_spike_train(T, burst_silence_burst),
        "poisson_background": make_spike_train(T, poisson_background),
        "burst_plus_noise": make_spike_train(T, sorted(set(structured + isolated))),
        "jittered_burst": make_spike_train(T, jittered_burst),
        "fragmented_burst": make_spike_train(T, fragmented_burst),
        "burst_plus_poisson": make_spike_train(T, sorted(set(structured + poisson_background))),
    }
    labels = {}
    for name, s in patterns.items():
        lab = np.zeros(T, dtype=int)
        lab[structured] = 1
        lab[isolated] = 2
        lab[jittered_burst] = 3
        lab[fragmented_burst] = 3
        lab[poisson_background] = 4
        labels[name] = lab
    return patterns, labels

def run_cds(spikes: np.ndarray, params: CDSParams) -> pd.DataFrame:
    syn = CalciumDynamicSynapse(**asdict(params), learnable=False)
    syn.reset()
    rows = []
    for t, sp in enumerate(spikes):
        with torch.no_grad():
            I = syn(torch.tensor(float(sp), dtype=torch.float32))
        ca = float(syn.ca.detach().cpu())
        u = float(syn.u.detach().cpu())
        I = float(I.detach().cpu())
        rows.append({"t": t, "spike": float(sp), "Ca": ca, "u": u,
                     "W": float(params.w_static), "I": I,
                     "E": I / (params.w_static + 1e-12) if sp > 0 else np.nan})
    return pd.DataFrame(rows)

def run_static(spikes: np.ndarray, w_static: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame([{"t": t, "spike": float(sp), "Ca": np.nan, "u": np.nan,
                          "W": w_static, "I": float(w_static * sp),
                          "E": float(sp) if sp > 0 else np.nan}
                         for t, sp in enumerate(spikes)])

def spike_density_before(spikes: np.ndarray, t: int, window: int = 12) -> float:
    return float(spikes[max(0, int(t)-window):int(t)].sum()) / float(window)

def per_spike_table(trace: pd.DataFrame, spikes: np.ndarray, scenario: str, model: str, labels=None) -> pd.DataFrame:
    df = trace[trace["spike"] > 0.5].copy()
    if len(df) == 0:
        return pd.DataFrame()
    first_I = float(df["I"].iloc[0])
    first_E = float(df["E"].iloc[0])
    rows = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        t = int(row["t"])
        rows.append({"scenario": scenario, "model": model, "spike_index": idx, "t": t,
                     "label": int(labels[t]) if labels is not None else -1,
                     "D_k": spike_density_before(spikes, t),
                     "local_density": spike_density_before(spikes, t),
                     "I": float(row["I"]), "E": float(row["E"]),
                     "relative_current": float(row["I"] / (first_I + 1e-12)),
                     "R_k": float(row["E"] / (first_E + 1e-12)),
                     "relative_efficacy": float(row["E"] / (first_E + 1e-12)),
                     "Ca": float(row["Ca"]) if not pd.isna(row["Ca"]) else np.nan,
                     "u": float(row["u"]) if not pd.isna(row["u"]) else np.nan})
    return pd.DataFrame(rows)

def normalized_cds_efficacy(E, params):
    u0 = effective_u0(params); umax = effective_umax(params)
    return np.clip((np.asarray(E, dtype=float) - u0) / (umax - u0 + 1e-12), 0, 1)

def safe_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def burst_noise_discrimination(e_burst, e_noise):
    return float((np.nanmean(e_burst) - np.nanmean(e_noise)) / (np.nanmean(e_burst) + np.nanmean(e_noise) + 1e-12))

def set_pub_style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
                         "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_spikes(ax, spikes, title="", ylabel=None, color="black", lw=1.6, xlabel=True):
    times = np.where(np.asarray(spikes) > 0.5)[0]
    ax.eventplot([times], lineoffsets=0.5, linelengths=0.75, linewidths=lw, colors=color)
    ax.set_ylim(0, 1); ax.set_yticks([])
    if xlabel:
        ax.set_xlabel("Time step")
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_title(title)
    despine(ax)

def save_origin_csv(df, data_dir, name):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(data_dir) / f"{name}.csv", index=False)


def summarize(df: pd.DataFrame) -> dict:
    sp = df[df.spike > 0.5]
    if len(sp) == 0:
        return {}
    return {
        "first_E": float(sp.E.iloc[0]),
        "last_E": float(sp.E.iloc[-1]),
        "R_last": float(sp.E.iloc[-1] / (sp.E.iloc[0] + 1e-12)),
        "facilitation_ratio_F": float(sp.E.iloc[-1] / (sp.E.iloc[0] + 1e-12)),
        "max_Ca": float(df.Ca.max()),
        "max_u": float(df.u.max()),
        "mean_E": float(sp.E.mean()),
    }

def plot_fig2a_mechanism_library(patterns, fig_dir, data_dir):
    keys = ["isolated_noise", "low_frequency_regular", "structured_burst",
            "burst_silence_burst", "poisson_background",
            "jittered_burst", "fragmented_burst", "burst_plus_noise"]
    rows = []
    for key in keys:
        for t in np.where(patterns[key] > 0.5)[0]:
            rows.append({"pattern": key, "t": int(t), "spike": 1})
    save_origin_csv(pd.DataFrame(rows), data_dir, "Fig2a_canonical_pattern_library")

    fig = plt.figure(figsize=(12.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.30)
    ax_mech = fig.add_subplot(gs[0,0])
    ax_mech.axis("off")
    nodes = [(0.10,0.58,r"$s[t]$","spike"),
             (0.34,0.58,r"$Ca[t]$","history state"),
             (0.58,0.58,r"$u[t]$","release"),
             (0.82,0.58,r"$E_k$","efficacy")]
    for x,y,sym,desc in nodes:
        ax_mech.text(x,y,sym+"\n"+desc, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.25"),
                     transform=ax_mech.transAxes)
    for x1,x2 in [(0.19,0.27),(0.43,0.51),(0.67,0.75)]:
        ax_mech.annotate("", xy=(x2,0.58), xytext=(x1,0.58), xycoords="axes fraction",
                         arrowprops=dict(arrowstyle="->", lw=1.4))
    ax_mech.text(0.5,0.24, r"$E_k=I[t_k]/W,\quad I[t]=Wu[t]s[t]$",
                 ha="center", va="center", fontsize=11, transform=ax_mech.transAxes)
    ax_mech.set_title("CDS computational chain")

    ax_lib = fig.add_subplot(gs[0,1])
    offsets = np.arange(len(keys))[::-1]
    for off, key in zip(offsets, keys):
        times = np.where(patterns[key] > 0.5)[0]
        ax_lib.eventplot([times], lineoffsets=off, linelengths=0.65, linewidths=1.4)
    ax_lib.set_yticks(offsets)
    ax_lib.set_yticklabels([k.replace("_"," ") for k in keys])
    ax_lib.set_xlabel("Time step")
    ax_lib.set_title("Canonical event-pattern library used across Fig. 2--Fig. 5")
    ax_lib.set_ylim(-0.8, len(keys)-0.2)
    despine(ax_lib)
    fig.savefig(fig_dir / "Fig2a_mechanism_and_pattern_library.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig2b_history_dependent_efficacy(traces, fig_dir, data_dir):
    keys = ["isolated_noise", "structured_burst", "burst_silence_burst"]
    rows = []
    for key in keys:
        df = traces[key]
        ev = per_spike_table(df, df.spike.values, key, "CDS")
        rows.append(ev)
    save_origin_csv(pd.concat(rows, ignore_index=True), data_dir, "Fig2b_per_spike_efficacy")

    fig, axes = plt.subplots(len(keys), 3, figsize=(13.5, 6.8), sharex=False)
    for r, key in enumerate(keys):
        df = traces[key]
        sp = df[df.spike > 0.5]
        plot_spikes(axes[r,0], df.spike.values, title="Input context" if r == 0 else "",
                    ylabel=key.replace("_","\n"), xlabel=False)
        axes[r,1].plot(df.t, df.Ca, label=r"$Ca[t]$", lw=1.5)
        axes[r,1].plot(df.t, df.u, label=r"$u[t]$", lw=1.5)
        axes[r,1].set_title("Synaptic state" if r == 0 else "")
        axes[r,1].set_ylabel("State value")
        axes[r,1].legend(frameon=False)
        despine(axes[r,1])
        axes[r,2].plot(range(1, len(sp)+1), sp.E, marker="o", lw=1.6)
        axes[r,2].set_title("Per-spike efficacy" if r == 0 else "")
        axes[r,2].set_xlabel("Spike index k")
        axes[r,2].set_ylabel(r"$E_k=I[t_k]/W$")
        despine(axes[r,2])
    for ax in axes[-1,:2]:
        ax.set_xlabel("Time step")
    fig.suptitle("Residual calcium assigns spike-specific efficacy without changing spike timing", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig2b_history_dependent_efficacy.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig2c_parameter_control(pattern, base_params, fig_dir, data_dir):
    tau_values = [2, 5, 10, 20, 50]
    k_values = [-2, -0.5, 1, 3, 6]
    ca_values = [0.1, 0.3, 0.5, 0.8, 1.0]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for vals, pname, ax in [(tau_values, "tau_ca", axes[0]), (k_values, "k_ca", axes[1]), (ca_values, "ca_influx", axes[2])]:
        for v in vals:
            p = CDSParams(**asdict(base_params)); setattr(p, pname, float(v))
            df = run_cds(pattern, p)
            ev = per_spike_table(df, pattern, f"{pname}_{v}", "CDS")
            ev[pname] = v
            rows.append(ev)
            ax.plot(ev.spike_index, ev.E, marker="o", lw=1.2, label=str(v))
        ax.set_xlabel("Spike index k")
        ax.set_ylabel(r"$E_k$")
        ax.set_title(pname)
        ax.legend(frameon=False, ncol=2, fontsize=7)
        despine(ax)
    scan = pd.concat(rows, ignore_index=True)
    save_origin_csv(scan, data_dir, "Fig2c_parameter_control_per_spike_efficacy")
    fig.suptitle("Calcium timescale, influx and sensitivity control context-dependent efficacy", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig2c_parameter_control_of_efficacy.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig2d_heatmap(pattern, base_params, fig_dir, data_dir):
    tau_values = [2, 5, 10, 20, 50, 80]
    k_values = [-2, -1, 0, 1, 2, 3, 5, 7]
    rows = []
    for tau in tau_values:
        for k in k_values:
            p = CDSParams(**asdict(base_params)); p.tau_ca = tau; p.k_ca = k
            df = run_cds(pattern, p)
            m = summarize(df); m.update({"tau_ca": tau, "k_ca": k}); rows.append(m)
    heat = pd.DataFrame(rows)
    save_origin_csv(heat, data_dir, "Fig2d_tauCa_kCa_facilitation_heatmap")
    pivot = heat.pivot(index="k_ca", columns="tau_ca", values="facilitation_ratio_F")
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel(r"$\tau_{Ca}$"); ax.set_ylabel(r"$k_{Ca}$")
    ax.set_title(r"Facilitation ratio $F$ across parameter space")
    cb = fig.colorbar(im, ax=ax); cb.set_label(r"$F=E_K/E_1$")
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig2d_facilitation_parameter_space.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig2e_summary(traces, fig_dir, data_dir):
    keys = ["isolated_noise", "low_frequency_regular", "structured_burst",
            "burst_silence_burst", "poisson_background"]
    rows = []
    for key in keys:
        m = summarize(traces[key]); m["pattern"] = key; rows.append(m)
    summ = pd.DataFrame(rows)
    save_origin_csv(summ, data_dir, "Fig2e_pattern_summary_metrics")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    x = np.arange(len(keys))
    labels = [k.replace("_","\n") for k in keys]
    axes[0].bar(x, summ["facilitation_ratio_F"])
    axes[0].axhline(1, ls=":", color="0.5")
    axes[0].set_ylabel(r"$F=E_K/E_1$")
    axes[0].set_title("Facilitation")
    axes[1].bar(x, summ["max_Ca"])
    axes[1].set_ylabel(r"max $Ca[t]$")
    axes[1].set_title("Calcium accumulation")
    axes[2].bar(x, summ["mean_E"])
    axes[2].set_ylabel(r"mean $E_k$")
    axes[2].set_title("Mean efficacy")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        despine(ax)
    fig.suptitle("Temporal context determines efficacy across canonical patterns", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig2e_summary_across_patterns.png", bbox_inches="tight")
    plt.close(fig)


def _safe_filename(text: str) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() else "_")
    name = "".join(out)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _reset_origin_subplot_dir(root_dir: Path) -> Path:

    out_dir = Path(root_dir) / "origin_subplot_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_file in out_dir.glob("*.csv"):
        old_file.unlink()
    return out_dir


def _save_subplot_csv(df: pd.DataFrame, out_dir: Path, filename: str):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / f"{filename}.csv", index=False)


def export_fig2a_subplot_csv(patterns: Dict[str, np.ndarray], out_dir: Path):

    mechanism_nodes = pd.DataFrame([
        {"node": "s[t]", "description": "spike", "x": 0.10, "y": 0.58, "plot_type": "text_box"},
        {"node": "Ca[t]", "description": "history state", "x": 0.34, "y": 0.58, "plot_type": "text_box"},
        {"node": "u[t]", "description": "release", "x": 0.58, "y": 0.58, "plot_type": "text_box"},
        {"node": "E_k", "description": "efficacy", "x": 0.82, "y": 0.58, "plot_type": "text_box"},
    ])
    mechanism_arrows = pd.DataFrame([
        {"arrow": "s_to_Ca", "x_start": 0.19, "y_start": 0.58, "x_end": 0.27, "y_end": 0.58},
        {"arrow": "Ca_to_u", "x_start": 0.43, "y_start": 0.58, "x_end": 0.51, "y_end": 0.58},
        {"arrow": "u_to_E", "x_start": 0.67, "y_start": 0.58, "x_end": 0.75, "y_end": 0.58},
    ])
    mechanism = pd.concat([mechanism_nodes, mechanism_arrows], axis=0, ignore_index=True)
    _save_subplot_csv(mechanism, out_dir, "Fig2a_01_CDS_computational_chain")

    keys = ["isolated_noise", "low_frequency_regular", "structured_burst",
            "burst_silence_burst", "poisson_background",
            "jittered_burst", "fragmented_burst", "burst_plus_noise"]
    rows = []
    for offset, key in zip(np.arange(len(keys))[::-1], keys):
        for t in np.where(patterns[key] > 0.5)[0]:
            rows.append({
                "pattern": key,
                "pattern_label": key.replace("_", " "),
                "t": int(t),
                "event_y_offset": int(offset),
                "event_linelength": 0.65,
                "spike": 1.0,
            })
    _save_subplot_csv(pd.DataFrame(rows), out_dir, "Fig2a_02_canonical_event_pattern_library")


def export_fig2b_subplot_csv(traces: Dict[str, pd.DataFrame], out_dir: Path):

    keys = ["isolated_noise", "structured_burst", "burst_silence_burst"]
    for key in keys:
        df = traces[key].copy()
        sp = df[df.spike > 0.5].copy().reset_index(drop=True)
        sp["spike_index"] = np.arange(1, len(sp) + 1)

        _save_subplot_csv(
            pd.DataFrame({
                "t": sp["t"].astype(int),
                "event_y_offset": 0.5,
                "event_linelength": 0.75,
                "spike": 1.0,
            }),
            out_dir,
            f"Fig2b_{_safe_filename(key)}_01_input_context",
        )
        _save_subplot_csv(
            pd.DataFrame({
                "t": df["t"].astype(int),
                "Ca_t": df["Ca"].astype(float),
                "u_t": df["u"].astype(float),
                "spike": df["spike"].astype(float),
                "Ca_at_spike": np.where(df["spike"] > 0.5, df["Ca"], np.nan),
                "u_at_spike": np.where(df["spike"] > 0.5, df["u"], np.nan),
            }),
            out_dir,
            f"Fig2b_{_safe_filename(key)}_02_synaptic_state",
        )
        _save_subplot_csv(
            pd.DataFrame({
                "spike_index_k": sp["spike_index"].astype(int),
                "t": sp["t"].astype(int),
                "E_k": sp["E"].astype(float),
            }),
            out_dir,
            f"Fig2b_{_safe_filename(key)}_03_per_spike_efficacy",
        )


def export_fig2c_subplot_csv(pattern: np.ndarray, base_params: CDSParams, out_dir: Path):

    scans = [
        ("tau_ca", [2, 5, 10, 20, 50]),
        ("k_ca", [-2, -0.5, 1, 3, 6]),
        ("ca_influx", [0.1, 0.3, 0.5, 0.8, 1.0]),
    ]
    for subplot_idx, (pname, values) in enumerate(scans, start=1):
        rows = []
        for v in values:
            p = CDSParams(**asdict(base_params))
            setattr(p, pname, float(v))
            df = run_cds(pattern, p)
            ev = per_spike_table(df, pattern, f"{pname}_{v}", "CDS")
            for _, row in ev.iterrows():
                rows.append({
                    "spike_index_k": int(row["spike_index"]),
                    "t": int(row["t"]),
                    pname: float(v),
                    "E_k": float(row["E"]),
                    "R_k": float(row["R_k"]),
                    "Ca_at_spike": float(row["Ca"]),
                    "u_at_spike": float(row["u"]),
                })
        _save_subplot_csv(
            pd.DataFrame(rows),
            out_dir,
            f"Fig2c_{subplot_idx:02d}_{_safe_filename(pname)}_control_of_Ek",
        )


def export_fig2d_subplot_csv(pattern: np.ndarray, base_params: CDSParams, out_dir: Path):

    tau_values = [2, 5, 10, 20, 50, 80]
    k_values = [-2, -1, 0, 1, 2, 3, 5, 7]
    rows = []
    for tau in tau_values:
        for k in k_values:
            p = CDSParams(**asdict(base_params))
            p.tau_ca = tau
            p.k_ca = k
            df = run_cds(pattern, p)
            m = summarize(df)
            rows.append({"tau_ca": tau, "k_ca": k, **m})
    heat = pd.DataFrame(rows)
    pivot = heat.pivot(index="k_ca", columns="tau_ca", values="facilitation_ratio_F")
    matrix = pivot.reset_index()

    for col in matrix.columns:
        matrix = matrix.rename(columns={col: f"tau_ca_{col}" if isinstance(col, (int, float)) else col})
    _save_subplot_csv(matrix, out_dir, "Fig2d_01_facilitation_parameter_space_heatmap_matrix")
    _save_subplot_csv(heat, out_dir, "Fig2d_01_facilitation_parameter_space_heatmap_longform")


def export_fig2e_subplot_csv(traces: Dict[str, pd.DataFrame], out_dir: Path):

    keys = ["isolated_noise", "low_frequency_regular", "structured_burst",
            "burst_silence_burst", "poisson_background"]
    rows = []
    for i, key in enumerate(keys):
        m = summarize(traces[key])
        rows.append({"pattern_index": i, "pattern": key, "pattern_label": key.replace("_", "\n"), **m})
    summ = pd.DataFrame(rows)
    _save_subplot_csv(
        summ[["pattern_index", "pattern", "pattern_label", "facilitation_ratio_F"]],
        out_dir,
        "Fig2e_01_facilitation_barplot",
    )
    _save_subplot_csv(
        summ[["pattern_index", "pattern", "pattern_label", "max_Ca"]],
        out_dir,
        "Fig2e_02_calcium_accumulation_barplot",
    )
    _save_subplot_csv(
        summ[["pattern_index", "pattern", "pattern_label", "mean_E"]],
        out_dir,
        "Fig2e_03_mean_efficacy_barplot",
    )


def export_all_origin_subplot_csv(patterns: Dict[str, np.ndarray], traces: Dict[str, pd.DataFrame], params: CDSParams, data_dir: Path):

    out_dir = _reset_origin_subplot_dir(Path(data_dir))
    export_fig2a_subplot_csv(patterns, out_dir)
    export_fig2b_subplot_csv(traces, out_dir)
    export_fig2c_subplot_csv(patterns["structured_burst"], params, out_dir)
    export_fig2d_subplot_csv(patterns["structured_burst"], params, out_dir)
    export_fig2e_subplot_csv(traces, out_dir)
    return out_dir


def main():
    set_pub_style()
    root = Path("results/section_2_2")
    fig_dir, data_dir = root / "figures", root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    patterns, labels = canonical_spike_patterns()
    params = CDSParams()
    traces = {k: run_cds(v, params) for k, v in patterns.items()}
    for k, df in traces.items():
        save_origin_csv(df, data_dir, f"trace_{k}")
    metrics = pd.DataFrame([dict(pattern=k, **summarize(df)) for k, df in traces.items()])
    save_origin_csv(metrics, data_dir, "summary_metrics_all_patterns")
    origin_subplot_dir = export_all_origin_subplot_csv(patterns, traces, params, data_dir)
    plot_fig2a_mechanism_library(patterns, fig_dir, data_dir)
    plot_fig2b_history_dependent_efficacy(traces, fig_dir, data_dir)
    plot_fig2c_parameter_control(patterns["structured_burst"], params, fig_dir, data_dir)
    plot_fig2d_heatmap(patterns["structured_burst"], params, fig_dir, data_dir)
    plot_fig2e_summary(traces, fig_dir, data_dir)
    print(f"Saved Section 2.2 results to {root.resolve()}")
    print(f"Origin-ready subplot CSV files saved to: {origin_subplot_dir.resolve()}")

if __name__ == "__main__":
    main()
