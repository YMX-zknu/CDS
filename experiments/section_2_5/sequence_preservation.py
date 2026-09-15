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
    v_threshold: float = 2.2
    v_reset: float = 0.0

def resting_static_weight(params: CDSParams) -> float:
    return params.w_static * effective_u0(params)

class CDSLIF:
    def __init__(self, cds_params: CDSParams, lif_params: LIFParams):
        self.syn = CalciumDynamicSynapse(**asdict(cds_params), learnable=False)
        self.lif = lif_params
        self.reset()
    def reset(self):
        self.syn.reset(); self.U=torch.tensor(0.0); self.prev_spike=torch.tensor(0.0)
    def step(self, sp):
        with torch.no_grad():
            I=self.syn(torch.tensor(float(sp),dtype=torch.float32))
            alpha=float(np.exp(-1.0/self.lif.tau_m))
            self.U=self.U*alpha*(1.0-self.prev_spike)+I
            U=float(self.U.detach().cpu())
            out=1.0 if U>=self.lif.v_threshold else 0.0
            if out: self.U=torch.tensor(self.lif.v_reset)
            self.prev_spike=torch.tensor(out)
        return {"I":float(I.detach().cpu()),"Ca":float(self.syn.ca.detach().cpu()),"u":float(self.syn.u.detach().cpu()),"U":U,"output_spike":out}

class StaticLIF:
    def __init__(self,w,lif_params):
        self.w=float(w); self.lif=lif_params; self.reset()
    def reset(self):
        self.U=torch.tensor(0.0); self.prev_spike=torch.tensor(0.0)
    def step(self, sp):
        with torch.no_grad():
            I=torch.tensor(self.w*float(sp))
            alpha=float(np.exp(-1.0/self.lif.tau_m))
            self.U=self.U*alpha*(1.0-self.prev_spike)+I
            U=float(self.U.detach().cpu())
            out=1.0 if U>=self.lif.v_threshold else 0.0
            if out: self.U=torch.tensor(self.lif.v_reset)
            self.prev_spike=torch.tensor(out)
        return {"I":float(I.detach().cpu()),"Ca":np.nan,"u":np.nan,"U":U,"output_spike":out}

def run_unit(spikes, labels, unit, model, pattern, params=None):
    rows=[]
    for t,sp in enumerate(spikes):
        rec=unit.step(sp)
        raw=rec["I"]/((params.w_static if params else 1.0)+1e-12) if sp>0 else np.nan
        rec.update({"t":t,"spike":float(sp),"label":int(labels[t]),"model":model,"pattern":pattern,"E":raw})
        rows.append(rec)
    return pd.DataFrame(rows)

def pattern_metrics(df):
    signal=df[(df.spike>0.5)&(df.label==1)]
    noise=df[(df.spike>0.5)&(df.label.isin([2,4]))]
    signal_E=float(signal.E.mean()) if len(signal) else np.nan
    noise_E=float(noise.E.mean()) if len(noise) else np.nan
    signal_I=float(signal.I.mean()) if len(signal) else np.nan
    noise_I=float(noise.I.mean()) if len(noise) else np.nan
    contrast=(signal_E-noise_E)/(signal_E+noise_E+1e-12) if np.isfinite(signal_E) and np.isfinite(noise_E) else np.nan
    noise_signal_ratio=(noise_E/(signal_E+1e-12)) if np.isfinite(signal_E) and np.isfinite(noise_E) else np.nan
    return {"output_spike_count":float(df.output_spike.sum()),
            "mean_signal_efficacy":signal_E,"mean_noise_efficacy":noise_E,
            "mean_signal_current":signal_I,"mean_noise_current":noise_I,
            "temporal_contrast_TC":float(contrast) if np.isfinite(contrast) else np.nan,
            "noise_to_signal_E_ratio":float(noise_signal_ratio) if np.isfinite(noise_signal_ratio) else np.nan,
            "max_U":float(df.U.max()),
            "mean_U":float(df.U.mean())}


def build_robustness_patterns(patterns, labels):


    robust_patterns = dict(patterns)
    robust_labels = {k: v.copy() for k, v in labels.items()}

    def combine_signal_noise(new_key, signal_key, noise_key, noise_label=2):
        s = np.clip(patterns[signal_key] + patterns[noise_key], 0, 1)
        lab = np.zeros_like(labels[signal_key])
        signal_times = np.where(patterns[signal_key] > 0.5)[0]
        noise_times = np.where(patterns[noise_key] > 0.5)[0]
        lab[signal_times] = 1
        lab[noise_times] = noise_label
        robust_patterns[new_key] = s
        robust_labels[new_key] = lab


    robust_labels["burst_plus_noise"] = np.zeros_like(labels["burst_plus_noise"])
    robust_labels["burst_plus_noise"][np.where(patterns["structured_burst"] > 0.5)[0]] = 1
    robust_labels["burst_plus_noise"][np.where(patterns["isolated_noise"] > 0.5)[0]] = 2

    robust_labels["burst_plus_poisson"] = np.zeros_like(labels["burst_plus_poisson"])
    robust_labels["burst_plus_poisson"][np.where(patterns["structured_burst"] > 0.5)[0]] = 1
    robust_labels["burst_plus_poisson"][np.where(patterns["poisson_background"] > 0.5)[0]] = 4


    combine_signal_noise("jittered_burst_plus_noise", "jittered_burst", "isolated_noise", noise_label=2)
    combine_signal_noise("fragmented_burst_plus_noise", "fragmented_burst", "isolated_noise", noise_label=2)

    return robust_patterns, robust_labels


def build_outputs(patterns, labels, params, lif):
    rows=[]; metrics=[]
    use_keys=["burst_plus_noise","jittered_burst_plus_noise",
              "fragmented_burst_plus_noise","burst_plus_poisson"]
    for key in use_keys:
        s=patterns[key]; lab=labels[key]
        for model, unit in [("Static-LIF", StaticLIF(resting_static_weight(params),lif)),
                            ("CDS-LIF", CDSLIF(params,lif))]:
            df=run_unit(s,lab,unit,model,key,params)
            rows.append(df)
            m=pattern_metrics(df); m.update({"pattern":key,"model":model}); metrics.append(m)
    return pd.concat(rows,ignore_index=True), pd.DataFrame(metrics)

def plot_fig5a_perturbation_design(patterns, fig_dir, data_dir):
    keys=["structured_burst","burst_plus_noise","jittered_burst_plus_noise","fragmented_burst_plus_noise","burst_plus_poisson"]
    design = pd.DataFrame([
        {"pattern":"structured_burst","interpretation":"clean temporally supported sequence"},
        {"pattern":"burst_plus_noise","interpretation":"structured sequence with isolated temporal noise"},
        {"pattern":"jittered_burst","interpretation":"temporal jitter in structured sequence"},
        {"pattern":"fragmented_burst","interpretation":"loss of compact temporal support"},
        {"pattern":"burst_plus_poisson","interpretation":"structured sequence with irregular background"},
    ])
    save_origin_csv(design, data_dir, "Fig5a_perturbation_design")
    rows=[]
    for key in keys:
        for t in np.where(patterns[key]>0.5)[0]:
            rows.append({"pattern":key,"t":int(t),"spike":1})
    save_origin_csv(pd.DataFrame(rows), data_dir, "Fig5a_perturbation_raster_data")
    fig, axes=plt.subplots(len(keys),1,figsize=(9,5.8),sharex=True)
    for ax,key in zip(axes,keys):
        plot_spikes(ax,patterns[key],title="",ylabel=key.replace("_","\n"),xlabel=False)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Perturbation design for testing structured-event preservation", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig5a_perturbation_design.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig5b_mixed_stream_decomposition(traces, patterns, labels, fig_dir, data_dir):
    key="burst_plus_noise"
    cds=traces[(traces.pattern==key)&(traces.model=="CDS-LIF")]
    st=traces[(traces.pattern==key)&(traces.model=="Static-LIF")]
    combined=pd.concat([cds,st],ignore_index=True)
    save_origin_csv(combined, data_dir, "Fig5b_mixed_stream_decomposition_traces")
    fig, axes=plt.subplots(2,2,figsize=(12,7))
    s=patterns[key]; lab=labels[key]
    for lid,name,offset in [(2,"isolated noise",0.25),(1,"structured event",0.65),(4,"background",1.00)]:
        times=np.where((s>0.5)&(lab==lid))[0]
        if len(times):
            axes[0,0].eventplot([times],lineoffsets=offset,linelengths=0.28,linewidths=1.8,label=name)
    axes[0,0].set_ylim(0,1.15); axes[0,0].set_yticks([]); axes[0,0].set_xlabel("Time step")
    axes[0,0].set_title("Mixed stream with event labels"); axes[0,0].legend(frameon=False); despine(axes[0,0])
    ev=cds[cds.spike>0.5]
    for lid,name,marker in [(2,"isolated noise","x"),(1,"structured event","o"),(4,"background","^")]:
        m=ev[ev.label==lid]
        if len(m):
            axes[0,1].scatter(m.t,m.E,marker=marker,label=name,s=42)
    axes[0,1].set_xlabel("Spike time"); axes[0,1].set_ylabel(r"$E_k$")
    axes[0,1].set_title("Event-wise transmission in the same mixed stream")
    axes[0,1].legend(frameon=False); despine(axes[0,1])
    axes[1,0].plot(cds.t,cds.U,label="CDS-LIF")
    axes[1,0].plot(st.t,st.U,"--",label="Static-LIF")
    axes[1,0].set_xlabel("Time step"); axes[1,0].set_ylabel(r"$U[t]$")
    axes[1,0].set_title("Postsynaptic integration")
    axes[1,0].legend(frameon=False); despine(axes[1,0])
    rows=[]
    for model,df in [("Static-LIF",st),("CDS-LIF",cds)]:
        m=pattern_metrics(df)
        rows.append({"model":model,"signal":m["mean_signal_efficacy"],"noise":m["mean_noise_efficacy"],"TC":m["temporal_contrast_TC"]})
    bar=pd.DataFrame(rows)
    save_origin_csv(bar, data_dir, "Fig5b_signal_noise_efficacy_decomposition")
    x=np.arange(len(bar)); width=0.35
    axes[1,1].bar(x-width/2,bar.signal,width=width,label="structured events")
    axes[1,1].bar(x+width/2,bar.noise,width=width,label="noise/background")
    axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(bar.model)
    axes[1,1].set_ylabel(r"Mean $E_k$")
    axes[1,1].set_title("Signal-noise contribution")
    axes[1,1].legend(frameon=False); despine(axes[1,1])
    fig.suptitle("CDS separates structured and isolated events within a single mixed stream", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig5b_mixed_stream_decomposition.png",bbox_inches="tight")
    plt.close(fig)

def plot_fig5c_temporal_contrast(metrics, fig_dir, data_dir):
    keys=["burst_plus_noise","jittered_burst_plus_noise","fragmented_burst_plus_noise","burst_plus_poisson"]
    df=metrics[metrics.pattern.isin(keys)].copy()
    save_origin_csv(df, data_dir, "Fig5c_temporal_contrast_across_perturbations")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(keys)); width=0.36
    for j,model in enumerate(["Static-LIF","CDS-LIF"]):
        sdf=df[df.model==model].set_index("pattern").loc[keys]
        axes[0].bar(x+(j-0.5)*width,sdf.temporal_contrast_TC,width=width,label=model)
        axes[1].bar(x+(j-0.5)*width,sdf.noise_to_signal_E_ratio,width=width,label=model)
    axes[0].set_xticks(x); axes[0].set_xticklabels([k.replace("_plus_","+").replace("_","\n") for k in keys])
    axes[0].set_ylabel(r"Temporal contrast $TC$")
    axes[0].set_title("Signal-noise separation")
    axes[1].set_xticks(x); axes[1].set_xticklabels([k.replace("_plus_","+").replace("_","\n") for k in keys])
    axes[1].set_ylabel(r"Noise/signal efficacy ratio")
    axes[1].set_title("Noise transmission relative to signal")
    for ax in axes:
        ax.legend(frameon=False); despine(ax)
    fig.suptitle("CDS preserves structured transmission across noise and structural perturbations", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig5c_temporal_contrast_across_perturbations.png",bbox_inches="tight")
    plt.close(fig)


def plot_fig5d_corruption_sweep(patterns, labels, params, lif, fig_dir, data_dir):
    base_times=np.where(patterns["structured_burst"]>0.5)[0].tolist()
    candidate_noise=[15,45,125,155,25,55,135,165,95,145]
    rows=[]
    for n in [0,2,4,6,8,10]:
        times=sorted(set(base_times+candidate_noise[:n]))
        s=make_spike_train(180,times)
        lab=np.zeros(180,dtype=int); lab[base_times]=1; lab[candidate_noise[:n]]=2
        for model,unit in [("Static-LIF",StaticLIF(resting_static_weight(params),lif)),("CDS-LIF",CDSLIF(params,lif))]:
            df=run_unit(s,lab,unit,model,f"noise_{n}",params)
            m=pattern_metrics(df); m.update({"noise_count":n,"model":model}); rows.append(m)
    sweep=pd.DataFrame(rows)
    save_origin_csv(sweep, data_dir, "Fig5d_corruption_sweep")
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for model in ["Static-LIF","CDS-LIF"]:
        sdf=sweep[sweep.model==model]
        axes[0].plot(sdf.noise_count,sdf.temporal_contrast_TC,marker="o",label=model)
        axes[1].plot(sdf.noise_count,sdf.noise_to_signal_E_ratio,marker="o",label=model)
    axes[0].set_xlabel("Number of added isolated noise events"); axes[0].set_ylabel(r"Temporal contrast $TC$"); axes[0].set_title("Filtering under increasing noise"); axes[0].legend(frameon=False); despine(axes[0])
    axes[1].set_xlabel("Number of added isolated noise events"); axes[1].set_ylabel(r"Noise/signal efficacy ratio"); axes[1].set_title("Relative noise transmission"); axes[1].legend(frameon=False); despine(axes[1])
    fig.tight_layout(); fig.savefig(fig_dir/"Fig5d_corruption_sweep_noise_robustness.png",bbox_inches="tight"); plt.close(fig)


def plot_fig5e_summary(metrics, fig_dir, data_dir):
    keys=["burst_plus_noise","jittered_burst_plus_noise","fragmented_burst_plus_noise","burst_plus_poisson"]
    df=metrics[metrics.pattern.isin(keys)].copy()
    summary=df.groupby("model").agg(mean_signal_E=("mean_signal_efficacy","mean"),
                                    mean_noise_E=("mean_noise_efficacy","mean"),
                                    mean_TC=("temporal_contrast_TC","mean"),
                                    mean_noise_signal_ratio=("noise_to_signal_E_ratio","mean")).reset_index()
    save_origin_csv(summary, data_dir, "Fig5e_robustness_summary_dashboard")
    fig,axes=plt.subplots(1,4,figsize=(14,3.6))
    metrics_to_plot=[("mean_signal_E",r"Signal $E_k$"),
                     ("mean_noise_E",r"Noise $E_k$"),
                     ("mean_TC",r"$TC$"),
                     ("mean_noise_signal_ratio",r"Noise/signal $E_k$")]
    x=np.arange(len(summary))
    for ax,(col,title) in zip(axes,metrics_to_plot):
        ax.bar(x,summary[col])
        ax.set_xticks(x); ax.set_xticklabels(summary.model,rotation=20,ha="right")
        ax.set_title(title); despine(ax)
    fig.suptitle("Summary of structured-event preservation and noise attenuation", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir/"Fig5e_robustness_summary_dashboard.png",bbox_inches="tight")
    plt.close(fig)


def _safe_filename(name: str) -> str:

    s = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _prepare_origin_subplot_dir(root: Path) -> Path:

    out_dir = Path(root) / "origin_subplot_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()
    return out_dir


def _write_csv(df: pd.DataFrame, out_dir: Path, filename: str):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / filename, index=False)


def _eventplot_origin_data(spikes: np.ndarray, pattern: str, label_values: np.ndarray = None) -> pd.DataFrame:

    times = np.where(np.asarray(spikes) > 0.5)[0]
    rows = []
    for t in times:
        lab = int(label_values[t]) if label_values is not None else -1
        rows.append({
            "pattern": pattern,
            "t": int(t),
            "spike": 1.0,
            "label": lab,
            "event_type": {1: "structured_event", 2: "isolated_noise", 3: "perturbed_structure", 4: "background"}.get(lab, "unlabeled"),
            "y_offset": 0.5,
            "y_start": 0.125,
            "y_end": 0.875,
        })
    return pd.DataFrame(rows)


def export_fig5a_subplot_csv(patterns, labels, out_dir: Path):

    keys = ["structured_burst", "burst_plus_noise", "jittered_burst_plus_noise", "fragmented_burst_plus_noise", "burst_plus_poisson"]
    for i, key in enumerate(keys, start=1):
        df = _eventplot_origin_data(patterns[key], key, labels.get(key))
        _write_csv(df, out_dir, f"Fig5a_{i:02d}_{_safe_filename(key)}_raster.csv")


def export_fig5b_subplot_csv(traces, patterns, labels, params, out_dir: Path):

    key = "burst_plus_noise"
    s = patterns[key]
    lab = labels[key]
    cds = traces[(traces.pattern == key) & (traces.model == "CDS-LIF")].copy()
    st = traces[(traces.pattern == key) & (traces.model == "Static-LIF")].copy()


    raster_rows = []
    offsets = {2: 0.25, 1: 0.65, 4: 1.00}
    names = {2: "isolated_noise", 1: "structured_event", 4: "background"}
    for lid in [2, 1, 4]:
        for t in np.where((s > 0.5) & (lab == lid))[0]:
            offset = offsets[lid]
            raster_rows.append({
                "t": int(t), "label": int(lid), "event_type": names[lid],
                "y_offset": float(offset), "y_start": float(offset - 0.14), "y_end": float(offset + 0.14),
                "spike": 1.0,
            })
    _write_csv(pd.DataFrame(raster_rows), out_dir, "Fig5b_01_mixed_stream_event_labels.csv")


    ev = cds[cds.spike > 0.5].copy()
    ev["event_type"] = ev["label"].map({1: "structured_event", 2: "isolated_noise", 4: "background"}).fillna("other")
    _write_csv(ev[["t", "label", "event_type", "E", "I", "Ca", "u", "U", "output_spike"]], out_dir, "Fig5b_02_eventwise_transmission.csv")


    u_rows = []
    for model, df in [("CDS-LIF", cds), ("Static-LIF", st)]:
        tmp = df[["t", "U"]].copy()
        tmp["model"] = model
        u_rows.append(tmp)
    _write_csv(pd.concat(u_rows, ignore_index=True), out_dir, "Fig5b_03_postsynaptic_integration_U.csv")


    rows = []
    for model, df in [("Static-LIF", st), ("CDS-LIF", cds)]:
        m = pattern_metrics(df)
        rows.extend([
            {"model": model, "event_group": "structured_events", "mean_E": m["mean_signal_efficacy"], "TC": m["temporal_contrast_TC"]},
            {"model": model, "event_group": "noise_background", "mean_E": m["mean_noise_efficacy"], "TC": m["temporal_contrast_TC"]},
        ])
    _write_csv(pd.DataFrame(rows), out_dir, "Fig5b_04_signal_noise_efficacy_decomposition.csv")


def export_fig5c_subplot_csv(metrics_df: pd.DataFrame, out_dir: Path):

    keys = ["burst_plus_noise", "jittered_burst_plus_noise", "fragmented_burst_plus_noise", "burst_plus_poisson"]
    df = metrics_df[metrics_df.pattern.isin(keys)].copy()
    _write_csv(df[["pattern", "model", "temporal_contrast_TC"]], out_dir, "Fig5c_01_temporal_contrast_TC.csv")
    _write_csv(df[["pattern", "model", "noise_to_signal_E_ratio"]], out_dir, "Fig5c_02_noise_to_signal_efficacy_ratio.csv")


def compute_corruption_sweep_dataframe(patterns, labels, params, lif) -> pd.DataFrame:

    base_times = np.where(patterns["structured_burst"] > 0.5)[0].tolist()
    candidate_noise = [15, 45, 125, 155, 25, 55, 135, 165, 95, 145]
    rows = []
    for n in [0, 2, 4, 6, 8, 10]:
        times = sorted(set(base_times + candidate_noise[:n]))
        s = make_spike_train(180, times)
        lab = np.zeros(180, dtype=int)
        lab[base_times] = 1
        lab[candidate_noise[:n]] = 2
        for model, unit in [("Static-LIF", StaticLIF(resting_static_weight(params), lif)), ("CDS-LIF", CDSLIF(params, lif))]:
            df = run_unit(s, lab, unit, model, f"noise_{n}", params)
            m = pattern_metrics(df)
            m.update({"noise_count": n, "model": model})
            rows.append(m)
    return pd.DataFrame(rows)


def export_fig5d_subplot_csv(patterns, labels, params, lif, out_dir: Path):

    sweep = compute_corruption_sweep_dataframe(patterns, labels, params, lif)
    _write_csv(sweep[["noise_count", "model", "temporal_contrast_TC"]], out_dir, "Fig5d_01_temporal_contrast_under_added_noise.csv")
    _write_csv(sweep[["noise_count", "model", "noise_to_signal_E_ratio"]], out_dir, "Fig5d_02_noise_signal_ratio_under_added_noise.csv")


def export_fig5e_subplot_csv(metrics_df: pd.DataFrame, out_dir: Path):

    keys = ["burst_plus_noise", "jittered_burst_plus_noise", "fragmented_burst_plus_noise", "burst_plus_poisson"]
    df = metrics_df[metrics_df.pattern.isin(keys)].copy()
    summary = df.groupby("model").agg(
        mean_signal_E=("mean_signal_efficacy", "mean"),
        mean_noise_E=("mean_noise_efficacy", "mean"),
        mean_TC=("temporal_contrast_TC", "mean"),
        mean_noise_signal_ratio=("noise_to_signal_E_ratio", "mean"),
    ).reset_index()
    for i, col in enumerate(["mean_signal_E", "mean_noise_E", "mean_TC", "mean_noise_signal_ratio"], start=1):
        _write_csv(summary[["model", col]], out_dir, f"Fig5e_{i:02d}_{col}.csv")


def export_all_origin_subplot_csv(patterns, labels, traces, metrics, params, lif, root: Path):

    out_dir = _prepare_origin_subplot_dir(root)
    export_fig5a_subplot_csv(patterns, labels, out_dir)
    export_fig5b_subplot_csv(traces, patterns, labels, params, out_dir)
    export_fig5c_subplot_csv(metrics, out_dir)
    export_fig5d_subplot_csv(patterns, labels, params, lif, out_dir)
    export_fig5e_subplot_csv(metrics, out_dir)
    return out_dir


def main():
    set_pub_style()
    root=Path("results/section_2_5")
    fig_dir,data_dir=root/"figures",root/"data"
    fig_dir.mkdir(parents=True,exist_ok=True); data_dir.mkdir(parents=True,exist_ok=True)
    patterns,labels=canonical_spike_patterns()
    patterns,labels=build_robustness_patterns(patterns, labels)
    params=CDSParams(tau_ca=5.0, ca_influx=0.3, u0=-3.0, u_max=3.0, k_ca=1.0, tau_is_effective=True)
    lif=LIFParams()
    traces,metrics=build_outputs(patterns,labels,params,lif)
    save_origin_csv(traces, data_dir, "traces_all_section_2_4")
    save_origin_csv(metrics, data_dir, "metrics_all_section_2_4")
    export_all_origin_subplot_csv(patterns, labels, traces, metrics, params, lif, root)
    plot_fig5a_perturbation_design(patterns,fig_dir,data_dir)
    plot_fig5b_mixed_stream_decomposition(traces,patterns,labels,fig_dir,data_dir)
    plot_fig5c_temporal_contrast(metrics,fig_dir,data_dir)
    plot_fig5d_corruption_sweep(patterns,labels,params,lif,fig_dir,data_dir)
    plot_fig5e_summary(metrics,fig_dir,data_dir)
    print(f"Saved Section 2.5 results to {root.resolve()}")

if __name__=="__main__":
    main()
