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

    pattern_times = {
        "isolated_noise": isolated,
        "low_frequency_regular": low_frequency_regular,
        "structured_burst": structured,
        "burst_silence_burst": burst_silence_burst,
        "poisson_background": poisson_background,
        "burst_plus_noise": sorted(set(structured + isolated)),
        "jittered_burst": jittered_burst,
        "fragmented_burst": fragmented_burst,
        "burst_plus_poisson": sorted(set(structured + poisson_background)),
    }
    patterns = {name: make_spike_train(T, times) for name, times in pattern_times.items()}

    def make_label_array(assignments):
        lab = np.zeros(T, dtype=int)
        for times, label_id in assignments:
            for t in times:
                if 0 <= int(t) < T:
                    lab[int(t)] = int(label_id)
        return lab

    labels = {
        "isolated_noise": make_label_array([(isolated, 2)]),
        "low_frequency_regular": make_label_array([(low_frequency_regular, 1)]),
        "structured_burst": make_label_array([(structured, 1)]),
        "burst_silence_burst": make_label_array([(burst_silence_burst, 1)]),
        "poisson_background": make_label_array([(poisson_background, 4)]),
        "burst_plus_noise": make_label_array([(structured, 1), (isolated, 2)]),
        "jittered_burst": make_label_array([(jittered_burst, 3)]),
        "fragmented_burst": make_label_array([(fragmented_burst, 3)]),
        "burst_plus_poisson": make_label_array([(structured, 1), (poisson_background, 4)]),
    }
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


def build_tables(patterns, labels, params):
    trace_rows, event_rows = [], []
    scenarios = ["burst_plus_noise", "structured_burst", "isolated_noise", "jittered_burst"]
    for scenario in scenarios:
        s = patterns[scenario]; lab = labels[scenario]
        cds = run_cds(s, params); cds["scenario"] = scenario; cds["model"] = "CDS"; cds["label"] = lab
        st = run_static(s, params.w_static); st["scenario"] = scenario; st["model"] = "Static"; st["label"] = lab
        trace_rows += [cds, st]
        for df, model in [(cds, "CDS"), (st, "Static")]:
            ps = per_spike_table(df, s, scenario, model, lab)
            if len(ps) and model == "CDS":
                ps["E_norm"] = normalized_cds_efficacy(ps["E"].values, params)
            elif len(ps):
                ps["E_norm"] = np.nan
            event_rows.append(ps)
    return pd.concat(trace_rows, ignore_index=True), pd.concat(event_rows, ignore_index=True)

def plot_fig3a_overview(fig_dir, data_dir):
    save_origin_csv(pd.DataFrame([
        {"model":"Static","principle":"same connection, same efficacy","current_equation":"I[t_k]=W"},
        {"model":"CDS","principle":"recent temporal context, dynamic efficacy","current_equation":"I[t_k]=W*u[t_k]"},
    ]), data_dir, "Fig3a_conceptual_comparison")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(0.20, 0.72, "Static synapse", ha="center", weight="bold", fontsize=12, transform=ax.transAxes)
    ax.text(0.20, 0.43, r"same structural connection"+"\n"+r"$\Rightarrow$ equal efficacy",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.2"), transform=ax.transAxes)
    ax.text(0.72, 0.72, "CDS", ha="center", weight="bold", fontsize=12, transform=ax.transAxes)
    ax.text(0.72, 0.43, r"local temporal context"+"\n"+r"$\Rightarrow$ dynamic efficacy",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.2"), transform=ax.transAxes)
    ax.annotate("", xy=(0.53,0.43), xytext=(0.37,0.43), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.46, 0.58, "from equal treatment\nto context-dependent transmission", ha="center", fontsize=10, transform=ax.transAxes)
    fig.savefig(fig_dir / "Fig3a_equal_treatment_to_context_dependent_transmission.png", bbox_inches="tight")
    plt.close(fig)

def event_type_name(label):
    return {1:"temporally supported", 2:"isolated noise", 3:"perturbed structure", 4:"background"}.get(int(label), "other")

def plot_fig3b_eventwise_context(events, patterns, labels, params, fig_dir, data_dir):
    ev = events[(events.scenario=="burst_plus_noise") & (events.model=="CDS")].copy()
    ev["event_type"] = ev["label"].map(event_type_name)
    save_origin_csv(ev, data_dir, "Fig3b_eventwise_context_vs_efficacy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    markers = {1:"o", 2:"x", 4:"^", 3:"s"}
    for label_id, name in [(2,"isolated noise"), (1,"temporally supported"), (4,"background")]:
        m = ev[ev.label==label_id]
        if len(m):
            axes[0].scatter(m.D_k, m.E_norm, marker=markers.get(label_id,"o"), label=name, s=42)
    corr = safe_corr(ev.D_k, ev.E_norm)
    axes[0].set_xlabel(r"Local event density $D_k$")
    axes[0].set_ylabel(r"Normalized efficacy $E^{norm}_k$")
    axes[0].set_title(r"Event efficacy follows local context"+"\n"+rf"corr = {corr:.2f}")
    axes[0].legend(frameon=False)
    despine(axes[0])


    s = patterns["burst_plus_noise"]; lab = labels["burst_plus_noise"]
    offsets = {2:0.25, 1:0.60, 4:0.95}
    for lid, name in [(2,"isolated noise"), (1,"temporally supported"), (4,"background")]:
        times = np.where((s>0.5) & (lab==lid))[0]
        if len(times):
            axes[1].eventplot([times], lineoffsets=offsets[lid], linelengths=0.25, linewidths=1.8, label=name)
    axes[1].set_ylim(0,1.15); axes[1].set_yticks([])
    axes[1].set_xlabel("Time step")
    axes[1].set_title("Mixed stream used for event-wise analysis")
    axes[1].legend(frameon=False)
    despine(axes[1])
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig3b_eventwise_context_efficacy_relationship.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig3c_distribution(events, params, fig_dir, data_dir):


    ev = events[(events.scenario=="burst_plus_noise") & (events.model=="CDS")].copy()
    noise = ev[ev.label==2].E_norm.values
    burst = ev[ev.label==1].E_norm.values
    disc = burst_noise_discrimination(burst, noise)
    corr = safe_corr(ev.D_k, ev.E_norm)

    dist_rows = pd.DataFrame({
        "event_group": (["isolated_noise"]*len(noise)) + (["temporally_supported"]*len(burst)),
        "E_norm": np.concatenate([noise, burst]) if len(noise)+len(burst)>0 else []
    })
    summary_rows = pd.DataFrame([
        {"metric":"history_density_correlation", "value":corr},
        {"metric":"burst_noise_discrimination", "value":disc},
        {"metric":"mean_noise_E_norm", "value":np.nanmean(noise)},
        {"metric":"mean_supported_E_norm", "value":np.nanmean(burst)},
    ])
    save_origin_csv(dist_rows, data_dir, "Fig3c_efficacy_distribution")
    save_origin_csv(summary_rows, data_dir, "Fig3c_selectivity_summary_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    box_data = [noise, burst]
    axes[0].boxplot(box_data, labels=["isolated\nnoise", "temporally\nsupported"], showfliers=False, widths=0.55)
    for i, vals in enumerate(box_data, start=1):
        if len(vals):
            jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals) > 1 else np.array([0.0])
            axes[0].scatter(np.ones_like(vals)*i + jitter, vals, s=22, alpha=0.85)
    axes[0].set_ylabel(r"$E^{norm}_k$")
    axes[0].set_title("Event-wise efficacy distribution")
    despine(axes[0])

    metric_names = [r"corr($D_k,E^{norm}_k$)", "discrimination"]
    metric_vals = [corr, disc]
    axes[1].bar(np.arange(2), metric_vals, width=0.55)
    axes[1].axhline(0, ls=":", color="0.5")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(metric_names, rotation=12, ha="right")
    axes[1].set_ylabel("Selectivity metric")
    axes[1].set_title("Context-efficacy coupling and separation")
    despine(axes[1])

    fig.tight_layout()
    fig.savefig(fig_dir / "Fig3c_distribution_level_event_selectivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_fig3d_channel_specific(patterns, labels, params, fig_dir, data_dir):

    channels = {"clustered channel": "structured_burst", "sparse channel": "isolated_noise"}
    rows = []
    for ch, key in channels.items():
        s = patterns[key]; lab = labels[key]
        df = run_cds(s, params)
        ev = per_spike_table(df, s, key, ch, lab)
        ev["channel"] = ch
        ev["E_norm"] = normalized_cds_efficacy(ev.E.values, params)
        rows.append(ev)
    ch_ev = pd.concat(rows, ignore_index=True)
    save_origin_csv(ch_ev, data_dir, "Fig3d_connection_specific_event_efficacy")
    summary = ch_ev.groupby("channel").agg(mean_E=("E","mean"),
                                           mean_E_norm=("E_norm","mean"),
                                           final_R=("R_k","last")).reset_index()
    gain = float(summary.loc[summary.channel=="clustered channel","mean_E"].iloc[0] /
                 (summary.loc[summary.channel=="sparse channel","mean_E"].iloc[0] + 1e-12))
    summary["channel_gain_ratio_clustered_over_sparse"] = gain
    save_origin_csv(summary, data_dir, "Fig3d_connection_specific_gain_summary")
    fig, axes = plt.subplots(1, 2, figsize=(10.5,4.0))
    for ch in channels:
        m = ch_ev[ch_ev.channel==ch]
        axes[0].plot(m.spike_index, m.E, marker="o", label=ch)
    axes[0].set_xlabel("Spike index k within each input stream")
    axes[0].set_ylabel(r"$E_k$")
    axes[0].set_title("Connection-specific efficacy trajectories")
    axes[0].legend(frameon=False)
    despine(axes[0])
    axes[1].bar(np.arange(len(summary)), summary.mean_E_norm)
    axes[1].set_xticks(np.arange(len(summary)))
    axes[1].set_xticklabels(summary.channel, rotation=20, ha="right")
    axes[1].set_ylabel(r"mean $E^{norm}_k$")
    axes[1].set_title(r"Channel gain ratio = %.2f" % gain)
    despine(axes[1])
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig3d_connection_specific_context_dependence.png", bbox_inches="tight")
    plt.close(fig)

def plot_fig3e_parameter_robustness(patterns, labels, params, fig_dir, data_dir):
    s = patterns["burst_plus_noise"]; lab = labels["burst_plus_noise"]
    scans = {"tau_ca":[2,5,10,20,50],
             "k_ca":[-2,-0.5,1,3,6],
             "ca_influx":[0.1,0.3,0.5,0.8,1.0]}
    rows = []
    for pname, vals in scans.items():
        for v in vals:
            p = CDSParams(**asdict(params)); setattr(p, pname, float(v))
            df = run_cds(s, p)
            ps = per_spike_table(df, s, "burst_plus_noise", "CDS", lab)
            ps["E_norm"] = normalized_cds_efficacy(ps.E.values, p)
            burst = ps[ps.label==1].E_norm.values
            noise = ps[ps.label==2].E_norm.values
            rows.append({"parameter":pname, "value":v,
                         "discrimination":burst_noise_discrimination(burst, noise),
                         "history_corr":safe_corr(ps.D_k, ps.E_norm)})
    scan = pd.DataFrame(rows)
    save_origin_csv(scan, data_dir, "Fig3e_parameter_robustness_of_event_selectivity")
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    for r, pname in enumerate(["tau_ca","k_ca","ca_influx"]):
        sdf = scan[scan.parameter==pname].sort_values("value")
        axes[r,0].plot(sdf.value, sdf.discrimination, marker="o")
        axes[r,0].set_ylabel("Discrimination")
        axes[r,0].set_xlabel(pname)
        despine(axes[r,0])
        axes[r,1].plot(sdf.value, sdf.history_corr, marker="o")
        axes[r,1].set_ylabel(r"corr($D_k,E^{norm}_k$)")
        axes[r,1].set_xlabel(pname)
        despine(axes[r,1])
    fig.suptitle("Event selectivity remains linked to local temporal context across CDS parameters", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "Fig3e_parameter_robustness_of_event_selectivity.png", bbox_inches="tight")
    plt.close(fig)


def _safe_filename(text: str) -> str:

    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() else "_")
    name = "".join(out)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def _prepare_origin_subplot_dir(data_dir: Path) -> Path:

    out_dir = Path(data_dir) / "origin_subplot_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    return out_dir


def _export_csv(df: pd.DataFrame, out_dir: Path, name: str):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / f"{_safe_filename(name)}.csv", index=False)


def export_fig3a_origin_subplot_csv(out_dir: Path):

    df = pd.DataFrame([
        {"x_box": 0.20, "y_box": 0.43, "model": "Static", "principle": "same structural connection -> equal efficacy", "equation": "I[t_k]=W"},
        {"x_box": 0.72, "y_box": 0.43, "model": "CDS", "principle": "local temporal context -> dynamic efficacy", "equation": "I[t_k]=W*u[t_k]"},
        {"x_arrow_start": 0.37, "x_arrow_end": 0.53, "y_arrow": 0.43, "annotation": "from equal treatment to context-dependent transmission"},
    ])
    _export_csv(df, out_dir, "Fig3a_01_equal_treatment_concept")


def export_fig3b_origin_subplot_csv(events, patterns, labels, params, out_dir: Path):

    ev = events[(events.scenario == "burst_plus_noise") & (events.model == "CDS")].copy()
    ev["event_type"] = ev["label"].map(event_type_name)
    _export_csv(
        ev[["t", "spike_index", "label", "event_type", "D_k", "local_density", "E", "E_norm", "Ca", "u"]],
        out_dir,
        "Fig3b_01_event_efficacy_vs_local_density",
    )

    s = patterns["burst_plus_noise"]
    lab = labels["burst_plus_noise"]
    rows = []
    offsets = {2: 0.25, 1: 0.60, 4: 0.95}
    for lid, name in [(2, "isolated_noise"), (1, "temporally_supported"), (4, "background")]:
        times = np.where((s > 0.5) & (lab == lid))[0]
        for t in times:
            rows.append({"t": int(t), "event_y_offset": offsets[lid], "label": lid, "event_type": name, "spike": 1.0})
    _export_csv(pd.DataFrame(rows), out_dir, "Fig3b_02_mixed_stream_raster")


def export_fig3c_origin_subplot_csv(events, params, out_dir: Path):

    ev = events[(events.scenario == "burst_plus_noise") & (events.model == "CDS")].copy()
    noise = ev[ev.label == 2].E_norm.values
    burst = ev[ev.label == 1].E_norm.values
    disc = burst_noise_discrimination(burst, noise)
    corr = safe_corr(ev.D_k, ev.E_norm)

    dist_rows = []
    for v in noise:
        dist_rows.append({"event_group": "isolated_noise", "x_category": 1, "E_norm": float(v)})
    for v in burst:
        dist_rows.append({"event_group": "temporally_supported", "x_category": 2, "E_norm": float(v)})
    _export_csv(pd.DataFrame(dist_rows), out_dir, "Fig3c_01_eventwise_efficacy_distribution")

    summary_rows = pd.DataFrame([
        {"metric": "history_density_correlation", "x_category": 1, "value": corr},
        {"metric": "burst_noise_discrimination", "x_category": 2, "value": disc},
    ])
    _export_csv(summary_rows, out_dir, "Fig3c_02_selectivity_summary_metrics")


def export_fig3d_origin_subplot_csv(patterns, labels, params, out_dir: Path):

    channels = {"clustered channel": "structured_burst", "sparse channel": "isolated_noise"}
    rows = []
    for ch, key in channels.items():
        s = patterns[key]
        lab = labels[key]
        df = run_cds(s, params)
        ev = per_spike_table(df, s, key, ch, lab)
        ev["channel"] = ch
        ev["E_norm"] = normalized_cds_efficacy(ev.E.values, params)
        rows.append(ev)
    ch_ev = pd.concat(rows, ignore_index=True)
    _export_csv(
        ch_ev[["channel", "scenario", "spike_index", "t", "label", "D_k", "E", "E_norm", "R_k", "Ca", "u"]],
        out_dir,
        "Fig3d_01_connection_specific_efficacy_trajectories",
    )

    summary = ch_ev.groupby("channel").agg(
        mean_E=("E", "mean"),
        mean_E_norm=("E_norm", "mean"),
        final_R=("R_k", "last"),
    ).reset_index()
    gain = float(summary.loc[summary.channel == "clustered channel", "mean_E"].iloc[0] /
                 (summary.loc[summary.channel == "sparse channel", "mean_E"].iloc[0] + 1e-12))
    summary["channel_gain_ratio_clustered_over_sparse"] = gain
    summary["x_category"] = np.arange(1, len(summary) + 1)
    _export_csv(summary, out_dir, "Fig3d_02_connection_specific_gain_summary")


def export_fig3e_origin_subplot_csv(patterns, labels, params, out_dir: Path):

    s = patterns["burst_plus_noise"]
    lab = labels["burst_plus_noise"]
    scans = {
        "tau_ca": [2, 5, 10, 20, 50],
        "k_ca": [-2, -0.5, 1, 3, 6],
        "ca_influx": [0.1, 0.3, 0.5, 0.8, 1.0],
    }
    rows = []
    for pname, vals in scans.items():
        for v in vals:
            p = CDSParams(**asdict(params))
            setattr(p, pname, float(v))
            df = run_cds(s, p)
            ps = per_spike_table(df, s, "burst_plus_noise", "CDS", lab)
            ps["E_norm"] = normalized_cds_efficacy(ps.E.values, p)
            burst = ps[ps.label == 1].E_norm.values
            noise = ps[ps.label == 2].E_norm.values
            rows.append({
                "parameter": pname,
                "value": v,
                "discrimination": burst_noise_discrimination(burst, noise),
                "history_corr": safe_corr(ps.D_k, ps.E_norm),
            })
    scan = pd.DataFrame(rows)
    for pname in ["tau_ca", "k_ca", "ca_influx"]:
        sdf = scan[scan.parameter == pname].sort_values("value")
        _export_csv(sdf[["value", "discrimination"]], out_dir, f"Fig3e_{pname}_01_discrimination")
        _export_csv(sdf[["value", "history_corr"]], out_dir, f"Fig3e_{pname}_02_history_correlation")


def export_all_origin_subplot_csv(patterns, labels, params, traces, events, data_dir: Path):

    out_dir = _prepare_origin_subplot_dir(data_dir)
    export_fig3a_origin_subplot_csv(out_dir)
    export_fig3b_origin_subplot_csv(events, patterns, labels, params, out_dir)
    export_fig3c_origin_subplot_csv(events, params, out_dir)
    export_fig3d_origin_subplot_csv(patterns, labels, params, out_dir)
    export_fig3e_origin_subplot_csv(patterns, labels, params, out_dir)
    return out_dir

def main():
    set_pub_style()
    root = Path("results/section_2_3")
    fig_dir, data_dir = root / "figures", root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)
    params = CDSParams()
    patterns, labels = canonical_spike_patterns()
    traces, events = build_tables(patterns, labels, params)
    save_origin_csv(traces, data_dir, "traces_all_section_2_2")
    save_origin_csv(events, data_dir, "event_table_all_section_2_2")
    origin_subplot_dir = export_all_origin_subplot_csv(patterns, labels, params, traces, events, data_dir)
    plot_fig3a_overview(fig_dir, data_dir)
    plot_fig3b_eventwise_context(events, patterns, labels, params, fig_dir, data_dir)
    plot_fig3c_distribution(events, params, fig_dir, data_dir)
    plot_fig3d_channel_specific(patterns, labels, params, fig_dir, data_dir)
    plot_fig3e_parameter_robustness(patterns, labels, params, fig_dir, data_dir)
    print(f"Saved Section 2.3 results to {root.resolve()}")
    print(f"Origin subplot CSV files saved to: {origin_subplot_dir.resolve()}")

if __name__ == "__main__":
    main()
