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


@dataclass
class LIFParams:
    tau_m: float = 2.0
    v_threshold: float = 1.6
    v_reset: float = 0.0

def resting_static_weight(params: CDSParams) -> float:
    return params.w_static * effective_u0(params)

class CDSLIF:
    def __init__(self, cds_params: CDSParams, lif_params: LIFParams):
        self.syn = CalciumDynamicSynapse(**asdict(cds_params), learnable=False)
        self.lif = lif_params
        self.reset()
    def reset(self):
        self.syn.reset()
        self.U = torch.tensor(0.0)
        self.prev_spike = torch.tensor(0.0)
    def step(self, sp):
        with torch.no_grad():
            I = self.syn(torch.tensor(float(sp), dtype=torch.float32))
            alpha = float(np.exp(-1.0 / self.lif.tau_m))
            self.U = self.U * alpha * (1.0 - self.prev_spike) + I
            U_before = float(self.U.detach().cpu())
            out = 1.0 if U_before >= self.lif.v_threshold else 0.0
            if out:
                self.U = torch.tensor(self.lif.v_reset)
            self.prev_spike = torch.tensor(out)
        return {"I": float(I.detach().cpu()), "Ca": float(self.syn.ca.detach().cpu()), "u": float(self.syn.u.detach().cpu()),
                "U": U_before, "output_spike": out}

class StaticLIF:
    def __init__(self, w_static, lif_params: LIFParams):
        self.w = float(w_static); self.lif = lif_params; self.reset()
    def reset(self):
        self.U = torch.tensor(0.0); self.prev_spike = torch.tensor(0.0)
    def step(self, sp):
        with torch.no_grad():
            I = torch.tensor(self.w * float(sp))
            alpha = float(np.exp(-1.0 / self.lif.tau_m))
            self.U = self.U * alpha * (1.0 - self.prev_spike) + I
            U_before = float(self.U.detach().cpu())
            out = 1.0 if U_before >= self.lif.v_threshold else 0.0
            if out:
                self.U = torch.tensor(self.lif.v_reset)
            self.prev_spike = torch.tensor(out)
        return {"I": float(I.detach().cpu()), "Ca": np.nan, "u": np.nan, "U": U_before, "output_spike": out}

def run_unit(spikes, unit, model, pattern, params=None):
    rows=[]
    for t, sp in enumerate(spikes):
        rec=unit.step(sp)
        E = rec["I"] / ((params.w_static if params else 1.0) + 1e-12) if sp > 0 else np.nan
        rec.update({"t":t,"spike":float(sp),"model":model,"pattern":pattern,"E":E})
        rows.append(rec)
    return pd.DataFrame(rows)

def metrics(df):
    return {
        "output_spike_count": float(df.output_spike.sum()),
        "mean_U": float(df.U.mean()),
        "max_U": float(df.U.max()),
        "mean_I_at_spikes": float(df.loc[df.spike>0.5,"I"].mean()) if (df.spike>0.5).any() else np.nan,
        "mean_E_at_spikes": float(df.loc[df.spike>0.5,"E"].mean()) if (df.spike>0.5).any() else np.nan,
        "latency_first_spike": float(df.loc[df.output_spike>0.5,"t"].iloc[0]) if (df.output_spike>0.5).any() else np.nan,
    }

def plot_fig4a_schematic(fig_dir, data_dir):
    save_origin_csv(pd.DataFrame([
        {"stage":1, "variable":"s[t]", "role":"presynaptic spike"},
        {"stage":2, "variable":"E_k / I[t]", "role":"context-dependent synaptic transmission"},
        {"stage":3, "variable":"U[t]", "role":"somatic membrane state"},
        {"stage":4, "variable":"o[t]", "role":"postsynaptic spike output"},
    ]), data_dir, "Fig4a_computational_pathway")
    fig, ax = plt.subplots(figsize=(9.5, 3))
    ax.axis("off")
    labels = [(0.10,r"$s[t]$","input"),
              (0.34,"CDS",r"$E_k,I[t]$"),
              (0.59,"LIF soma",r"$U[t]$"),
              (0.84,r"$o[t]$","output")]
    for x, top, bottom in labels:
        ax.text(x,0.58,top+"\n"+bottom,ha="center",va="center",
                bbox=dict(boxstyle="round,pad=0.5",fc="white",ec="0.2"),
                transform=ax.transAxes)
    for x1,x2 in [(0.18,0.26),(0.43,0.51),(0.68,0.76)]:
        ax.annotate("",xy=(x2,0.58),xytext=(x1,0.58),xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->",lw=1.5))
    ax.text(0.34,0.20,"synaptic context state",ha="center",fontsize=10,transform=ax.transAxes)
    ax.text(0.59,0.20,"somatic integration state",ha="center",fontsize=10,transform=ax.transAxes)
    fig.savefig(fig_dir/"Fig4a_synapse_to_soma_pathway.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig4b_representative(patterns, params, lif_params, fig_dir, data_dir):
    s = patterns["burst_silence_burst"]
    cds = run_unit(s, CDSLIF(params, lif_params), "CDS-LIF", "burst_silence_burst", params)
    static = run_unit(s, StaticLIF(resting_static_weight(params), lif_params), "Static-LIF", "burst_silence_burst", params)
    all_df = pd.concat([cds,static], ignore_index=True)
    save_origin_csv(all_df, data_dir, "Fig4b_representative_synapse_to_soma_traces")
    fig, axes = plt.subplots(4,1,figsize=(9.5,7.2),sharex=True)
    plot_spikes(axes[0], s, title="Shared burst-silence-burst input", ylabel="input")
    for model, df, ls in [("CDS-LIF", cds, "-"), ("Static-LIF", static, "--")]:
        sp = df[df.spike > 0.5]
        axes[1].vlines(sp.t, 0, sp.I, lw=1.7, label=model, linestyles=ls)
    axes[1].set_ylabel(r"$I[t_k]$")
    axes[1].set_title("Transmission at spike times")
    axes[1].legend(frameon=False); despine(axes[1])
    axes[2].plot(cds.t, cds.U, label="CDS-LIF")
    axes[2].plot(static.t, static.U, "--", label="Static-LIF")
    axes[2].set_ylabel(r"$U[t]$")
    axes[2].set_title("Membrane integration")
    axes[2].legend(frameon=False); despine(axes[2])
    axes[3].eventplot([cds.loc[cds.output_spike>0.5,"t"].values], lineoffsets=0.70, linelengths=0.25, label="CDS-LIF")
    axes[3].eventplot([static.loc[static.output_spike>0.5,"t"].values], lineoffsets=0.30, linelengths=0.25, label="Static-LIF")
    axes[3].set_ylim(0,1); axes[3].set_yticks([]); axes[3].set_xlabel("Time step")
    axes[3].set_title("Output spikes"); axes[3].legend(frameon=False); despine(axes[3])
    fig.suptitle("Context-dependent synaptic transmission reshapes somatic integration", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig4b_representative_synapse_to_soma_transformation.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig4c_tau_scan(patterns, params, lif_params, fig_dir, data_dir):
    s = patterns["burst_silence_burst"]
    tau_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows=[]
    for tau_eff in tau_values:
        p=CDSParams(**asdict(params)); p.tau_ca=tau_eff; p.tau_is_effective=True
        df=run_unit(s, CDSLIF(p,lif_params), "CDS-LIF", "burst_silence_burst", p)
        m=metrics(df); m.update({"tau_Ca_effective":tau_eff}); rows.append(m)
    scan=pd.DataFrame(rows)
    save_origin_csv(scan, data_dir, "Fig4c_tauCa_somatic_response_scan")
    fig, axes=plt.subplots(1,3,figsize=(13,3.8))
    axes[0].plot(scan.tau_Ca_effective, scan.max_U, marker="o")
    axes[0].set_xscale("log"); axes[0].set_xlabel(r"Effective $\tau_{Ca}$"); axes[0].set_ylabel(r"max $U[t]$"); axes[0].set_title("Peak membrane state"); despine(axes[0])
    axes[1].plot(scan.tau_Ca_effective, scan.mean_I_at_spikes, marker="o")
    axes[1].set_xscale("log"); axes[1].set_xlabel(r"Effective $\tau_{Ca}$"); axes[1].set_ylabel("Mean current at spikes"); axes[1].set_title("Synaptic drive"); despine(axes[1])
    axes[2].plot(scan.tau_Ca_effective, scan.output_spike_count, marker="o")
    axes[2].set_xscale("log"); axes[2].set_xlabel(r"Effective $\tau_{Ca}$"); axes[2].set_ylabel("Output spike count"); axes[2].set_title("Somatic response"); despine(axes[2])
    fig.suptitle("Synaptic timescale controls the conversion from temporal history to somatic firing", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig4c_tauCa_controls_somatic_response.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig4d_state_space(patterns, params, lif_params, fig_dir, data_dir):
    keys=["isolated_noise","structured_burst","burst_silence_burst"]
    fig, axes=plt.subplots(1,3,figsize=(13,3.8))
    rows=[]
    for ax,key in zip(axes,keys):
        df=run_unit(patterns[key], CDSLIF(params,lif_params), "CDS-LIF", key, params)
        rows.append(df)
        im=ax.scatter(df.Ca, df.U, c=df.t, s=16)
        ax.set_xlabel(r"$Ca[t]$"); ax.set_ylabel(r"$U[t]$")
        ax.set_title(key.replace("_"," "))
        despine(ax)
    concat = pd.concat(rows, ignore_index=True)
    save_origin_csv(concat, data_dir, "Fig4d_coupled_state_space_trajectories")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Time step")
    fig.suptitle("Synaptic context state and somatic state form input-dependent trajectories", y=1.02)
    fig.savefig(fig_dir/"Fig4d_coupled_state_space_trajectories.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig4e_summary(patterns, params, lif_params, fig_dir, data_dir):
    keys=["isolated_noise","structured_burst","burst_silence_burst"]
    rows=[]
    for key in keys:
        for model, unit in [("Static-LIF", StaticLIF(resting_static_weight(params), lif_params)),
                            ("CDS-LIF", CDSLIF(params, lif_params))]:
            df = run_unit(patterns[key], unit, model, key, params)
            m = metrics(df); m.update({"pattern":key, "model":model}); rows.append(m)
    summ = pd.DataFrame(rows)
    save_origin_csv(summ, data_dir, "Fig4e_static_vs_cds_lif_summary")
    fig, axes=plt.subplots(1,3,figsize=(13,3.8))
    x = np.arange(len(keys)); width=0.36
    for i, metric_name in enumerate(["max_U","output_spike_count","mean_I_at_spikes"]):
        ax=axes[i]
        for j, model in enumerate(["Static-LIF","CDS-LIF"]):
            vals=[summ[(summ.pattern==k)&(summ.model==model)][metric_name].iloc[0] for k in keys]
            ax.bar(x+(j-0.5)*width, vals, width=width, label=model)
        ax.set_xticks(x); ax.set_xticklabels([k.replace("_","\n") for k in keys])
        ax.set_ylabel(metric_name.replace("_"," "))
        ax.set_title(metric_name.replace("_"," "))
        despine(ax)
    axes[0].legend(frameon=False)
    fig.suptitle("Context-dependent transmission changes somatic integration across input regimes", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig4e_summary_static_vs_cds_lif.png",bbox_inches="tight")
    plt.close(fig)


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_")


def _prepare_origin_subplot_dir(root: Path) -> Path:
    out_dir = root / "origin_subplot_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    return out_dir


def _save_subplot_csv(df: pd.DataFrame, out_dir: Path, filename: str):
    df.to_csv(out_dir / f"{filename}.csv", index=False)


def export_fig4a_origin_subplot_csv(out_dir: Path):
    df = pd.DataFrame([
        {"stage": 1, "x_position": 0.10, "variable": "s[t]", "role": "input", "connection_to_next": "s[t] -> CDS"},
        {"stage": 2, "x_position": 0.34, "variable": "E_k / I[t]", "role": "context-dependent synaptic transmission", "connection_to_next": "CDS -> LIF soma"},
        {"stage": 3, "x_position": 0.59, "variable": "U[t]", "role": "somatic membrane state", "connection_to_next": "U[t] -> output spike"},
        {"stage": 4, "x_position": 0.84, "variable": "o[t]", "role": "postsynaptic spike output", "connection_to_next": ""},
    ])
    _save_subplot_csv(df, out_dir, "Fig4a_01_synapse_to_soma_pathway")


def export_fig4b_origin_subplot_csv(patterns, params, lif_params, out_dir: Path):
    s = patterns["burst_silence_burst"]
    cds = run_unit(s, CDSLIF(params, lif_params), "CDS-LIF", "burst_silence_burst", params)
    static = run_unit(s, StaticLIF(resting_static_weight(params), lif_params), "Static-LIF", "burst_silence_burst", params)

    _save_subplot_csv(
        pd.DataFrame({"t": np.arange(len(s)), "input_spike": s}),
        out_dir,
        "Fig4b_01_shared_input_spikes",
    )

    current_rows = []
    for model, df in [("CDS-LIF", cds), ("Static-LIF", static)]:
        sp = df[df.spike > 0.5]
        for _, row in sp.iterrows():
            current_rows.append({"model": model, "t": int(row["t"]), "I_at_spike": float(row["I"]), "E_at_spike": float(row["E"])})
    _save_subplot_csv(pd.DataFrame(current_rows), out_dir, "Fig4b_02_transmission_at_spike_times")

    u_df = pd.DataFrame({
        "t": cds["t"].astype(int),
        "CDS_LIF_U": cds["U"].astype(float),
        "Static_LIF_U": static["U"].astype(float),
        "threshold": float(lif_params.v_threshold),
    })
    _save_subplot_csv(u_df, out_dir, "Fig4b_03_membrane_integration")

    output_rows = []
    for model, df, offset in [("CDS-LIF", cds, 0.70), ("Static-LIF", static, 0.30)]:
        out_times = df.loc[df.output_spike > 0.5, "t"].values
        for t in out_times:
            output_rows.append({"model": model, "t": int(t), "event_lineoffset": offset, "output_spike": 1.0})
    _save_subplot_csv(pd.DataFrame(output_rows, columns=["model", "t", "event_lineoffset", "output_spike"]), out_dir, "Fig4b_04_output_spikes")


def export_fig4c_origin_subplot_csv(patterns, params, lif_params, out_dir: Path):
    s = patterns["burst_silence_burst"]
    tau_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    rows = []
    for tau_eff in tau_values:
        p = CDSParams(**asdict(params)); p.tau_ca = tau_eff; p.tau_is_effective = True
        df = run_unit(s, CDSLIF(p, lif_params), "CDS-LIF", "burst_silence_burst", p)
        m = metrics(df); m.update({"tau_Ca_effective": tau_eff}); rows.append(m)
    scan = pd.DataFrame(rows)
    _save_subplot_csv(scan[["tau_Ca_effective", "max_U"]], out_dir, "Fig4c_01_peak_membrane_state")
    _save_subplot_csv(scan[["tau_Ca_effective", "mean_I_at_spikes", "mean_E_at_spikes"]], out_dir, "Fig4c_02_synaptic_drive")
    _save_subplot_csv(scan[["tau_Ca_effective", "output_spike_count"]], out_dir, "Fig4c_03_somatic_response")


def export_fig4d_origin_subplot_csv(patterns, params, lif_params, out_dir: Path):
    for idx, key in enumerate(["isolated_noise", "structured_burst", "burst_silence_burst"], start=1):
        df = run_unit(patterns[key], CDSLIF(params, lif_params), "CDS-LIF", key, params)
        _save_subplot_csv(
            df[["t", "Ca", "U", "I", "u", "spike", "output_spike"]].copy(),
            out_dir,
            f"Fig4d_{idx:02d}_{_safe_filename(key)}_state_space",
        )


def export_fig4e_origin_subplot_csv(patterns, params, lif_params, out_dir: Path):
    keys = ["isolated_noise", "structured_burst", "burst_silence_burst"]
    rows = []
    for key in keys:
        for model, unit in [("Static-LIF", StaticLIF(resting_static_weight(params), lif_params)),
                            ("CDS-LIF", CDSLIF(params, lif_params))]:
            df = run_unit(patterns[key], unit, model, key, params)
            m = metrics(df); m.update({"pattern": key, "model": model}); rows.append(m)
    summ = pd.DataFrame(rows)
    for idx, metric_name in enumerate(["max_U", "output_spike_count", "mean_I_at_spikes"], start=1):
        _save_subplot_csv(
            summ[["pattern", "model", metric_name]].copy(),
            out_dir,
            f"Fig4e_{idx:02d}_{metric_name}",
        )


def export_all_origin_subplot_csv(patterns, params, lif_params, data_dir: Path):
    out_dir = _prepare_origin_subplot_dir(data_dir)
    export_fig4a_origin_subplot_csv(out_dir)
    export_fig4b_origin_subplot_csv(patterns, params, lif_params, out_dir)
    export_fig4c_origin_subplot_csv(patterns, params, lif_params, out_dir)
    export_fig4d_origin_subplot_csv(patterns, params, lif_params, out_dir)
    export_fig4e_origin_subplot_csv(patterns, params, lif_params, out_dir)
    return out_dir

def main():
    set_pub_style()
    root=Path("results/section_2_4")
    fig_dir,data_dir=root/"figures",root/"data"
    fig_dir.mkdir(parents=True,exist_ok=True); data_dir.mkdir(parents=True,exist_ok=True)
    patterns,labels=canonical_spike_patterns()
    params=CDSParams(tau_ca=2.0,tau_is_effective=True)
    lif=LIFParams()
    plot_fig4a_schematic(fig_dir, data_dir)
    plot_fig4b_representative(patterns,params,lif,fig_dir,data_dir)
    plot_fig4c_tau_scan(patterns,params,lif,fig_dir,data_dir)
    plot_fig4d_state_space(patterns,params,lif,fig_dir,data_dir)
    plot_fig4e_summary(patterns,params,lif,fig_dir,data_dir)
    origin_subplot_dir = export_all_origin_subplot_csv(patterns, params, lif, data_dir)
    print(f"Saved Section 2.4 results to {root.resolve()}")
    print(f"Origin subplot CSV files saved to {origin_subplot_dir.resolve()}")

if __name__=="__main__":
    main()
