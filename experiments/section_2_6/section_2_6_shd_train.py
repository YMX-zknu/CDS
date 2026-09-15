from __future__ import annotations

import os


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Dict

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from section_2_6_dvsgesture_train import (
    ARCHITECTURE_VERSION,
    ISOLATED_PERTURBATION_PROTOCOL_VERSION,
    TRANSMISSION_ORDER,

    make_data_loader_generator,
    run_training_pipeline,
    set_seed,
    functional,
    neuron,
    surrogate,
)


class CDSLayer(nn.Module):


    def __init__(
        self,
        channels: int,
        param_dims: int = 4,
        tau_ca_init: float = 2.0,
        ca_influx_init: float = 0.5,
        k_ca_init: float = 1.0,
        u0_init: float = 0.1,
        u_max_init: float = 1.0,
        learnable: bool = True,
        variant: str = "full",
    ):
        super().__init__()
        if variant not in {"full", "constant_release", "memory_free"}:
            raise ValueError(f"Unknown CDS variant: {variant}")
        self.channels = int(channels)
        self.param_dims = int(param_dims)
        self.variant = variant
        self.eps = 1e-6
        shape = (1, self.channels) + (() if self.param_dims == 2 else (1,) * (self.param_dims - 2))

        def parameter(value: float):
            return nn.Parameter(
                torch.full(shape, float(value)), requires_grad=learnable
            )

        def inverse_softplus(value: float) -> float:
            value = max(float(value), self.eps)
            return value + math.log(-math.expm1(-value))

        def logit(value: float) -> float:
            value = min(max(float(value), self.eps), 1.0 - self.eps)
            return math.log(value) - math.log1p(-value)


        u0_target = min(max(float(u0_init), self.eps), 1.0 - 2.0 * self.eps)
        umax_target = min(
            max(float(u_max_init), u0_target + self.eps), 1.0 - self.eps
        )
        relative_upper = (umax_target - u0_target) / (1.0 - u0_target)
        self.tau_ca = parameter(inverse_softplus(tau_ca_init))
        self.ca_influx = parameter(inverse_softplus(ca_influx_init))
        self.k_ca = parameter(inverse_softplus(k_ca_init))
        self.u0 = parameter(logit(u0_target))
        self.u_max = parameter(logit(relative_upper))
        self.calcium = None
        self.release_prob = None
        self.last_stats: Dict[str, float] = {}

    def _reset_state(self, x):
        if self.calcium is None or tuple(self.calcium.shape) != tuple(x.shape):
            self.calcium = torch.zeros_like(x)
        if self.release_prob is None or tuple(self.release_prob.shape) != tuple(x.shape):
            self.release_prob = torch.zeros_like(x)

    def forward(self, x):
        self._reset_state(x)
        tau_ca = F.softplus(self.tau_ca) + self.eps
        ca_influx = F.softplus(self.ca_influx)
        k_ca = F.softplus(self.k_ca)
        u0 = torch.sigmoid(self.u0)

        u_max = u0 + (1.0 - u0) * torch.sigmoid(self.u_max)


        drive = x

        if self.variant == "constant_release":
            self.calcium = torch.zeros_like(x)
            self.release_prob = torch.clamp(u0.expand_as(x), 0.0, 1.0)
        else:
            if self.variant == "memory_free":
                self.calcium = drive * ca_influx
            else:
                self.calcium = self.calcium * torch.exp(-1.0 / tau_ca) + drive * ca_influx

            self.release_prob = u0 + (u_max - u0) * (1.0 - torch.exp(-k_ca * self.calcium))
            self.release_prob = torch.clamp(self.release_prob, 0.0, 1.0)

        out = x * self.release_prob
        with torch.no_grad():
            x_abs = x.detach().abs().mean()
            out_abs = out.detach().abs().mean()
            self.last_stats = {
                "calcium_mean": float(self.calcium.detach().mean().cpu()),
                "calcium_max": float(self.calcium.detach().max().cpu()),
                "release_mean": float(self.release_prob.detach().mean().cpu()),
                "release_std": float(self.release_prob.detach().std().cpu()),
                "transmission_ratio": float((out_abs / (x_abs + 1e-12)).cpu()),
                "tau_ca_eff_mean": float(tau_ca.detach().mean().cpu()),
                "k_ca_eff_mean": float(k_ca.detach().mean().cpu()),
                "ca_influx_eff_mean": float(ca_influx.detach().mean().cpu()),
            }
        return out

    def reset(self):
        self.calcium = None
        self.release_prob = None
        self.last_stats = {}


@dataclass
class BaseConfig:


    data_root: str = "data/SHD"
    out_dir: str = "results/section_2_6"
    dataset_name: str = "shd"
    modality: str = "auditory"
    T: int = 100
    batch_size: int = 128
    epochs: int = 200
    num_workers: int = 4
    device: str = "cuda:0"
    lr: float = 1e-3
    eta_min: float = 1e-5
    seed: int = 2026
    model: str = "both"
    hidden_dim: int = 256
    hidden_layers: int = 2
    tau_m: float = 2.0
    tau_ca_init: float = 2.0
    dropout_p: float = 0.2
    cds_variant: str = "full"
    amp: bool = False
    TET: bool = False
    lamb: float = 0.0
    means: float = 1.0
    loss_mode: str = "rate_mse"
    true_spike_count: float = 60.0
    false_spike_count: float = 10.0
    analysis_batches: int = 32
    isolated_perturbation_ratios: str = "0.0,0.025,0.05,0.1"
    isolated_perturbation_repeats: int = 3
    isolation_guard_steps: int = 4
    selectivity_ratio: float = 0.1
    minimum_realized_fraction: float = 0.95
    dry_run: bool = False
    dry_run_train_samples: int = 24
    dry_run_test_samples: int = 12
    dry_run_hw: int = 128

    def perturbation_ratios(self) -> List[float]:
        values = [
            float(v)
            for v in str(self.isolated_perturbation_ratios).split(",")
            if str(v).strip()
        ]
        if not values:
            raise ValueError("isolated_perturbation_ratios must contain at least one value.")
        if any(v < 0 for v in values):
            raise ValueError("isolated_perturbation_ratios must be non-negative.")
        return sorted(set(values))


class TemporalMLPSNN(nn.Module):


    def __init__(
        self,
        input_dim: int = 700,
        num_classes: int = 20,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        model_type: str = "cds",
        tau_m: float = 2.0,
        tau_ca_init: float = 2.0,
        dropout_p: float = 0.2,
        cds_variant: str = "full",
    ):
        super().__init__()
        assert model_type in {"static", "cds"}
        self.model_type = model_type
        self.architecture_version = ARCHITECTURE_VERSION
        self.transmission_order = TRANSMISSION_ORDER
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.linears = nn.ModuleList()
        self.cds_layers = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        dim_in = int(input_dim)
        for _ in range(int(hidden_layers)):
            if model_type == "cds":
                self.cds_layers.append(
                    CDSLayer(dim_in, param_dims=2, tau_ca_init=tau_ca_init, variant=cds_variant)
                )
            self.linears.append(nn.Linear(dim_in, hidden_dim, bias=False))
            self.lifs.append(
                neuron.LIFNode(tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True)
            )
            self.dropouts.append(nn.Dropout(dropout_p))
            dim_in = hidden_dim
        self.readout = nn.Linear(dim_in, num_classes)
        self.lif_out = neuron.LIFNode(
            tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True
        )
        self.feature_dim = dim_in

    def reset(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset"):
                try:
                    module.reset()
                except TypeError:
                    pass

    def _single_step(self, x, t: int, return_dynamics: bool = False):
        spike_row = {"t": t}
        cds_rows = []
        for i, linear in enumerate(self.linears):
            if self.model_type == "cds":
                x = self.cds_layers[i](x)
                if return_dynamics:
                    row = {
                        "t": t,
                        "layer": f"fc{i + 1}",
                        "cds_location": "presynaptic_before_linear",
                    }
                    row.update(self.cds_layers[i].last_stats)
                    cds_rows.append(row)
            x = linear(x)
            x = self.lifs[i](x)
            if return_dynamics:
                spike_row[f"fc{i + 1}_spike_mean"] = float(x.detach().mean().cpu())
            x = self.dropouts[i](x)
        feature = x
        x = self.readout(x)
        x = self.lif_out(x)
        if return_dynamics:
            spike_row["out_spike_mean"] = float(x.detach().mean().cpu())
        return x, feature, spike_row, cds_rows

    def forward(
        self,
        x_seq,
        return_dynamics: bool = False,
        return_input_release: bool = False,
    ):
        out_seq, feature_seq = [], []
        spike_stats, cds_stats = [], []
        input_release_seq = []
        for t in range(int(x_seq.shape[0])):
            out, feature, spike_row, cds_rows = self._single_step(
                x_seq[t], t, return_dynamics
            )
            out_seq.append(out)
            feature_seq.append(feature)
            if return_input_release and self.model_type == "cds":
                input_release_seq.append(
                    self.cds_layers[0].release_prob.detach().clone()
                )
            if return_dynamics:
                spike_stats.append(spike_row)
                cds_stats.extend(cds_rows)
        out_seq = torch.stack(out_seq, dim=0)
        feature_seq = torch.stack(feature_seq, dim=0)
        if return_dynamics or return_input_release:
            dynamics = {
                "features": feature_seq.mean(dim=0).detach(),
                "feature_seq": feature_seq.detach(),
                "logits_seq": out_seq.detach(),
                "spike_stats": spike_stats,
                "cds_stats": cds_stats,
            }
            if return_input_release:
                dynamics["input_release_seq"] = (
                    torch.stack(input_release_seq, dim=0)
                    if input_release_seq
                    else None
                )
            return out_seq, dynamics
        return out_seq


@dataclass
class SHDConfig(BaseConfig):
    dataset_name: str = "shd"
    modality: str = "auditory"
    T: int = 100
    batch_size: int = 128
    epochs: int = 200
    hidden_dim: int = 256
    hidden_layers: int = 2
    dropout_p: float = 0.2
    num_classes: int = 20
    input_dim: int = 700
    split_by: str = "time"
    normalize: str = "max"


class SHDEventsToFrames:


    def __init__(self, T: int = 100, W: int = 700, split_by: Literal["time", "number"] = "time", normalize: str = "max"):
        self.T = int(T)
        self.W = int(W)
        self.split_by = str(split_by)
        self.normalize = str(normalize).lower()
        if self.split_by not in {"time", "number"}:
            raise ValueError("split_by must be 'time' or 'number'")

    def _integrate_by_time(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        frames = np.zeros((self.T, self.W), dtype=np.float32)
        if t.size == 0:
            return frames


        order = np.argsort(t)
        t = t[order]
        x = x[order]

        t0, t1 = float(t[0]), float(t[-1])
        if not np.isfinite(t0) or not np.isfinite(t1):
            return frames
        if t1 <= t0:

            valid = (x >= 0) & (x < self.W)
            np.add.at(frames, (np.zeros(valid.sum(), dtype=np.int64), x[valid]), 1.0)
            return frames


        edges = np.linspace(t0, t1, self.T + 1, dtype=np.float64)


        bin_idx = np.searchsorted(edges, t, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.T - 1).astype(np.int64)
        x = x.astype(np.int64, copy=False)
        valid = (x >= 0) & (x < self.W)
        np.add.at(frames, (bin_idx[valid], x[valid]), 1.0)
        return frames

    def _integrate_by_number(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        frames = np.zeros((self.T, self.W), dtype=np.float32)
        if t.size == 0:
            return frames
        order = np.argsort(t)
        x = x[order].astype(np.int64, copy=False)
        chunks = np.array_split(np.arange(x.size), self.T)
        for i, idx in enumerate(chunks):
            if idx.size == 0:
                continue
            xi = x[idx]
            valid = (xi >= 0) & (xi < self.W)
            if valid.any():
                counts = np.bincount(xi[valid], minlength=self.W).astype(np.float32)
                frames[i] = counts[: self.W]
        return frames

    def __call__(self, events):

        t = np.asarray(events["t"], dtype=np.float64)
        x = np.asarray(events["x"], dtype=np.int64)
        if self.split_by == "time":
            frames = self._integrate_by_time(t, x)
        else:
            frames = self._integrate_by_number(t, x)

        if self.normalize == "max":
            m = float(frames.max())
            if m > 0:
                frames = frames / m
        elif self.normalize not in {"none", "false", "0"}:
            raise ValueError("normalize must be 'max' or 'none'")
        return torch.from_numpy(frames).float()


def build_shd_datasets(cfg: SHDConfig):
    if cfg.dry_run:
        x_train = torch.bernoulli(
            torch.full((cfg.dry_run_train_samples, cfg.T, cfg.input_dim), 0.01)
        )
        y_train = torch.randint(0, cfg.num_classes, (cfg.dry_run_train_samples,))
        x_test = torch.bernoulli(
            torch.full((cfg.dry_run_test_samples, cfg.T, cfg.input_dim), 0.01)
        )
        y_test = torch.randint(0, cfg.num_classes, (cfg.dry_run_test_samples,))
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    try:
        from spikingjelly.datasets.shd import SpikingHeidelbergDigits
    except Exception:
        from spikingjelly.activation_based.datasets.shd import SpikingHeidelbergDigits

    transform = SHDEventsToFrames(T=cfg.T, W=cfg.input_dim, split_by=cfg.split_by, normalize=cfg.normalize)

    official_train = SpikingHeidelbergDigits(
        root=cfg.data_root,
        train=True,
        data_type="event",
        transform=transform,
    )
    test_set = SpikingHeidelbergDigits(
        root=cfg.data_root,
        train=False,
        data_type="event",
        transform=transform,
    )

    n_train, n_test = len(official_train), len(test_set)
    if n_train == 0 or n_test == 0:
        raise RuntimeError(
            f"Loaded SHD event dataset is empty (train={n_train}, test={n_test}). "
            f"Check that {Path(cfg.data_root) / 'extract'} contains shd_train.h5 and shd_test.h5."
        )
    print(f"[SHD] Loaded event dataset: train={n_train}, test={n_test}; on-the-fly frames T={cfg.T}, W={cfg.input_dim}, split_by={cfg.split_by}")
    return official_train, test_set


def build_shd_model(name: str, cfg: SHDConfig, device: torch.device):
    model = TemporalMLPSNN(
        input_dim=cfg.input_dim,
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        hidden_layers=cfg.hidden_layers,
        model_type=name,
        tau_m=cfg.tau_m,
        tau_ca_init=cfg.tau_ca_init,
        dropout_p=cfg.dropout_p,
        cds_variant=cfg.cds_variant,
    )
    try:
        functional.set_step_mode(model, "s")
    except Exception:
        pass
    if (model.architecture_version != ARCHITECTURE_VERSION or
            model.transmission_order != TRANSMISSION_ORDER):
        raise RuntimeError("SHD loaded an incompatible network definition.")
    return model.to(device)


def parse_args(argv=None):

    p = argparse.ArgumentParser(description="Train static/CDS SNNs on SHD for Section 2.6.")
    p.add_argument("--data_root", type=str, default="data/SHD")
    p.add_argument("--out_dir", type=str, default="results/section_2_6")
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--model", choices=["static", "cds", "both"], default="both")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--tau_m", type=float, default=2.0)
    p.add_argument("--tau_ca_init", type=float, default=2.0)
    p.add_argument("--dropout_p", type=float, default=0.2)
    p.add_argument("--cds_variant", choices=["full", "constant_release", "memory_free"], default="full")
    p.add_argument("--analysis_batches", type=int, default=32)
    p.add_argument("--isolated_perturbation_ratios", type=str,
                   default="0.0,0.025,0.05,0.1")
    p.add_argument("--isolated_perturbation_repeats", type=int, default=3)
    p.add_argument("--isolation_guard_steps", type=int, default=4)
    p.add_argument("--selectivity_ratio", type=float, default=0.1)
    p.add_argument("--minimum_realized_fraction", type=float, default=0.95)
    p.add_argument("--split_by", choices=["time", "number"], default="time",
                   help="Safe on-the-fly event-to-frame split method. 'time' allows empty bins; 'number' balances event counts.")
    p.add_argument("--normalize", choices=["max", "none"], default="max")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--TET", action="store_true", help="Use TET-style one-hot MSE over all time steps.")
    p.add_argument("--lamb", type=float, default=0.0, help="TET regularization weight. Keep 0.0 for standard time-step averaged TET loss.")
    p.add_argument("--means", type=float, default=1.0, help="TET regularization target mean used when --lamb > 0.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dry_run_train_samples", type=int, default=24)
    p.add_argument("--dry_run_test_samples", type=int, default=12)
    return p.parse_args(argv)


def config_from_args(args=None) -> SHDConfig:

    if args is None:
        args = parse_args([])
    return SHDConfig(**vars(args))


def main():
    cfg = config_from_args(parse_args())
    print(
        f"[shd] isolated-perturbation protocol="
        f"{ISOLATED_PERTURBATION_PROTOCOL_VERSION}"
    )
    worker_init = set_seed(cfg.seed)
    train_set, test_set = build_shd_datasets(cfg)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed),
        pin_memory=("cuda" in str(cfg.device).lower()),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed + 1),
        pin_memory=("cuda" in str(cfg.device).lower()),
    )
    run_training_pipeline(cfg, train_loader, test_loader, build_shd_model, cfg.num_classes)


if __name__ == "__main__":
    main()
