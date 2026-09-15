from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
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
np.random.seed(SEED)
torch.manual_seed(SEED)


class CalciumDynamicSynapse(BaseMemoryModule):


    def __init__(
        self,
        tau_ca: float = 20.0,
        ca_influx: float = 0.5,
        u0: float = 0.1,
        u_max: float = 1.0,
        k_ca: float = 1.0,
        w_static: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        def make_param(x: float):
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
        self.register_memory("step_count", 0)

    def forward(self, spike: torch.Tensor):
        self.step_count += 1

        tau_ca = torch.nn.functional.softplus(self.tau_ca)
        self.ca = self.ca * torch.exp(-1.0 / (tau_ca + self.eps))

        ca_influx = torch.nn.functional.softplus(self.ca_influx)
        self.ca = self.ca + spike * ca_influx

        k_ca = torch.nn.functional.softplus(self.k_ca)
        u0 = torch.sigmoid(self.u0)
        u_max = torch.sigmoid(self.u_max)

        self.u = u0 + (u_max - u0) * (1.0 - torch.exp(-k_ca * self.ca))

        w_eff = self.w_static * self.u
        synaptic_current = w_eff * spike
        return synaptic_current

    def reset(self):
        super().reset()
        self.ca = torch.tensor(0.0)
        self.u = torch.tensor(0.0)
        self.step_count = 0


@dataclass
class CDSParams:

    tau_ca: float = 18.0
    ca_influx: float = 0.65
    u0: float = 0.1
    u_max: float = 1.0
    k_ca: float = 1.2
    w_static: float = 1.0


def generate_context_probe_spike_train(T: int = 150) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:


    t = np.arange(T)
    s = np.zeros(T, dtype=float)


    s[[16, 29, 41]] = 1.0


    burst1 = np.array([58, 60, 62, 64, 66, 68, 70])
    s[burst1] = 1.0


    burst2 = np.array([96, 98, 100, 102, 104])
    s[burst2] = 1.0


    s[[132]] = 1.0

    probes = {
        "isolated_probe": 16,
        "post_sparse_probe": 41,
        "early_burst_probe": 58,
        "late_burst_probe": 70,
        "second_burst_probe": 96,
        "late_isolated_probe": 132,
    }
    return t, s, probes


def run_cds(spikes: np.ndarray, params: CDSParams) -> Dict[str, np.ndarray]:
    syn = CalciumDynamicSynapse(**asdict(params), learnable=False)
    syn.reset()

    records = {"spike": [], "Ca": [], "u": [], "I": [], "W_eff": []}
    for sp in spikes:
        spike_tensor = torch.tensor(float(sp), dtype=torch.float32)
        with torch.no_grad():
            current = syn(spike_tensor)
        records["spike"].append(float(sp))
        records["Ca"].append(float(syn.ca.detach().cpu()))
        records["u"].append(float(syn.u.detach().cpu()))
        records["I"].append(float(current.detach().cpu()))
        records["W_eff"].append(float((syn.w_static * syn.u).detach().cpu()))

    return {k: np.asarray(v, dtype=float) for k, v in records.items()}


def set_pub_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def stem_binary(ax, t, spikes, color="black", label=None, height=1.0, lw=1.3, alpha=1.0):
    for tt, sp in zip(t, spikes):
        if sp > 0.5:
            ax.vlines(tt, 0, height, color=color, lw=lw, alpha=alpha)
    if label is not None:
        ax.plot([], [], color=color, lw=lw, label=label)


def stem_values(ax, t, values, mask=None, color="#d62728", label=None, lw=1.4, alpha=1.0):
    if mask is None:
        mask = values > 0
    for tt, vv, m in zip(t, values, mask):
        if m:
            ax.vlines(tt, 0, vv, color=color, lw=lw, alpha=alpha)
    if label is not None:
        ax.plot([], [], color=color, lw=lw, label=label)


def add_context_shading(ax, label_y=1.04):
    regions = [
        (10, 46, "sparse isolated context", "#1f77b4"),
        (54, 73, "correlated burst", "#2ca02c"),
        (92, 106, "second burst", "#2ca02c"),
        (126, 138, "late isolated", "#1f77b4"),
    ]
    for x0, x1, label, color in regions:
        ax.axvspan(x0, x1, color=color, alpha=0.07)
        ax.text((x0 + x1) / 2, label_y, label, color=color, ha="center", va="bottom",
                fontsize=8, transform=ax.get_xaxis_transform())


def save_panel_csvs(out_dir: Path, t: np.ndarray, spikes: np.ndarray, cds: Dict[str, np.ndarray],
                    spike_times: np.ndarray, per_spike_I: np.ndarray, per_spike_u: np.ndarray):

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        csv_dir / "Fig1b_panelA_binary_spike_train.csv",
        np.column_stack([t, spikes]),
        delimiter=",",
        header="time,s",
        comments="",
    )

    np.savetxt(
        csv_dir / "Fig1b_panelB_same_positions_dynamic_current.csv",
        np.column_stack([t, spikes, cds["u"], cds["I"]]),
        delimiter=",",
        header="time,s,u,I",
        comments="",
    )

    np.savetxt(
        csv_dir / "Fig1b_panelC_Ca_u_dynamics.csv",
        np.column_stack([t, cds["Ca"], cds["u"]]),
        delimiter=",",
        header="time,Ca,u",
        comments="",
    )

    np.savetxt(
        csv_dir / "Fig1b_panelD_per_spike_efficacy.csv",
        np.column_stack([np.arange(1, len(spike_times) + 1), spike_times, per_spike_u, per_spike_I]),
        delimiter=",",
        header="spike_index,spike_time,u_at_spike,I_over_W",
        comments="",
    )

    np.savetxt(
        csv_dir / "Fig1b_all_time_series.csv",
        np.column_stack([t, spikes, cds["Ca"], cds["u"], cds["W_eff"], cds["I"]]),
        delimiter=",",
        header="time,s,Ca,u,W_eff,I",
        comments="",
    )


def make_fig1b(out_dir: str = "results/section_2_1") -> Path:
    set_pub_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CDSParams()
    t, spikes, probes = generate_context_probe_spike_train()
    cds = run_cds(spikes, params)


    non_spike_mask = spikes < 0.5
    if not np.allclose(cds["I"][non_spike_mask], 0.0):
        raise RuntimeError("CDS current should be zero whenever input spike s[t] is zero.")

    spike_mask = spikes > 0.5
    spike_times = t[spike_mask]
    per_spike_I = cds["I"][spike_mask] / params.w_static
    per_spike_u = cds["u"][spike_mask]

    probe_times = np.array(list(probes.values()))
    probe_I = np.array([cds["I"][p] / params.w_static for p in probe_times])
    probe_Ca = np.array([cds["Ca"][p] for p in probe_times])
    probe_u = np.array([cds["u"][p] for p in probe_times])

    fig = plt.figure(figsize=(13.6, 9.0))
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[0.85, 2.1, 2.1], hspace=0.60, wspace=0.30)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    ax0.text(0.5, 0.78, "History-dependent adaptive transmission without changing spike timing",
             ha="center", va="center", fontsize=16, weight="bold")
    ax0.text(0.5, 0.42,
             '"chatter ... car-horn burst ... silent gap ... car-horn burst ... late isolated sound"',
             ha="center", va="center", fontsize=13)
    ax0.text(0.5, 0.12,
             r"CDS preserves the binary spike pattern: $s[t]\in\{0,1\}$. "
             r"It changes only the transmitted strength $I[t]=W\,u[t]\,s[t]$ at spike times.",
             ha="center", va="center", fontsize=11.5, color="#b22222")


    ax1 = fig.add_subplot(gs[1, 0])
    stem_binary(ax1, t, spikes, color="black", label=r"input $s[t]$", height=1.0, lw=1.2)
    for p in probe_times:
        ax1.scatter([p], [1.05], s=30, zorder=4)
    add_context_shading(ax1)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(0, 1.35)
    ax1.set_yticks([0, 1])
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(r"Binary spike $s[t]$")
    ax1.set_title("Binary spike train")
    ax1.legend(frameon=False, loc="upper right")
    despine(ax1)


    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(t, cds["Ca"], color="#9467bd", lw=1.7, label=r"Residual calcium $Ca[t]$")
    ax2.scatter(probe_times, probe_Ca, color="#9467bd", edgecolor="black", s=34, zorder=5)
    ax2b = ax2.twinx()
    ax2b.plot(t, cds["u"], color="#d62728", lw=1.7, label=r"Release probability $u[t]$")
    ax2b.scatter(probe_times, probe_u, color="#d62728", edgecolor="black", s=34, zorder=5)
    add_context_shading(ax2)
    ax2.set_xlim(0, t[-1])
    ax2.set_xlabel("Time step")
    ax2.set_ylabel(r"Residual calcium $Ca[t]$")
    ax2b.set_ylabel(r"Release probability $u[t]$")
    ax2.set_title(r"Recent spike history is stored in $Ca[t]$ and $u[t]$")
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, frameon=False, loc="upper right")
    despine(ax2)
    ax2b.spines["top"].set_visible(False)


    ax3 = fig.add_subplot(gs[1, 1])
    stem_binary(ax3, t, spikes, color="0.72", label=r"input positions $s[t]=1$", height=0.15, lw=0.9)
    stem_values(ax3, t, cds["I"], mask=spike_mask, color="#d62728",
                label=r"dynamic strength $I[t]/W=u[t]$ at spikes", lw=1.8)
    ax3.scatter(probe_times, probe_I, color="#d62728", edgecolor="black", s=38, zorder=5)
    add_context_shading(ax3)
    ax3.set_xlim(0, t[-1])
    ax3.set_ylim(0, max(1.05, cds["I"].max() * 1.15))
    ax3.set_xlabel("Time step")
    ax3.set_ylabel(r"Spike position / transmitted strength")
    ax3.set_title("Same spike positions, different transmission strengths")
    ax3.legend(frameon=False, loc="upper right")
    despine(ax3)


    ax4 = fig.add_subplot(gs[2, 1])
    spike_indices = np.arange(1, len(per_spike_I) + 1)
    ax4.plot(spike_indices, per_spike_I, marker="o", color="#d62728", lw=1.6)
    for p in probe_times:
        si = np.where(spike_times == p)[0][0] + 1
        ax4.scatter([si], [cds["I"][p] / params.w_static], s=46, edgecolor="black", color="#d62728", zorder=5)
    ax4.set_xlabel("Spike index")
    ax4.set_ylabel(r"Per-spike efficacy $I[t_k]/W$")
    ax4.set_title("Transmission efficacy changes across identical spikes")
    despine(ax4)

    fig.suptitle("Fig. 1b | CDS modulates strength while preserving the input spike pattern",
                 fontsize=17, weight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    png = out_dir / "Fig1b_history_dependent_adaptive_transmission.png"
    svg = out_dir / "Fig1b_history_dependent_adaptive_transmission.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    save_panel_csvs(out_dir, t, spikes, cds, spike_times, per_spike_I, per_spike_u)
    return png


if __name__ == "__main__":
    print(f"Saved: {make_fig1b()}")
