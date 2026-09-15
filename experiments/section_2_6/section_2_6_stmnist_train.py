from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import random
import torch.nn.functional as F

from section_2_6_dvsgesture_train import (
    ARCHITECTURE_VERSION,
    ISOLATED_PERTURBATION_PROTOCOL_VERSION,
    TRANSMISSION_ORDER,

    load_torch_file,
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
        u0_init: float = 0.3,
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


    data_root: str = "data/ST-MNIST"
    out_dir: str = "results/section_2_6"
    dataset_name: str = "stmnist"
    modality: str = "tactile"
    T: int = 30
    batch_size: int = 16
    epochs: int = 100
    num_workers: int = 4
    device: str = "cuda:0"
    lr: float = 1e-3
    eta_min: float = 1e-5
    seed: int = 2026
    model: str = "both"
    channels: int = 64
    blocks: int = 2
    hidden_dim: int = 256
    tau_m: float = 2.0
    tau_ca_init: float = 2.0
    dropout_p: float = 0.5
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


class STMNISTSNN(nn.Module):


    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        input_hw: Tuple[int, int] = (10, 10),
        channels: int = 64,
        blocks: int = 2,
        fc_dim: int = 512,
        votes: int = 10,
        model_type: str = "cds",
        tau_m: float = 2.0,
        tau_ca_init: float = 2.0,
        dropout_p: float = 0.5,
        cds_variant: str = "full",
    ):
        super().__init__()
        assert model_type in {"static", "cds"}
        self.model_type = model_type
        self.architecture_version = ARCHITECTURE_VERSION
        self.transmission_order = TRANSMISSION_ORDER
        self.num_classes = int(num_classes)
        self.votes = int(votes)
        self.blocks = int(blocks)
        self.channels = int(channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.cds_layers = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = int(in_channels)
        for _ in range(self.blocks):
            if model_type == "cds":
                self.cds_layers.append(
                    CDSLayer(c_in, param_dims=4, tau_ca_init=tau_ca_init, variant=cds_variant)
                )
            self.convs.append(nn.Conv2d(c_in, channels, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(channels))
            self.lifs.append(
                neuron.LIFNode(tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True)
            )
            self.pools.append(nn.MaxPool2d(2, 2))
            c_in = channels

        pooled_h = int(input_hw[0]) // (2 ** self.blocks)
        pooled_w = int(input_hw[1]) // (2 ** self.blocks)
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(
                f"input_hw={input_hw} is too small for {self.blocks} pooling stages."
            )
        self.adaptive_pool = nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(channels * pooled_h * pooled_w, fc_dim)
        self.lif_fc1 = neuron.LIFNode(
            tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True
        )
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, num_classes * votes)
        self.lif_fc2 = neuron.LIFNode(
            tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True
        )
        self.feature_dim = fc_dim

    def reset(self):
        for module in self.modules():
            if module is not self and hasattr(module, "reset"):
                try:
                    module.reset()
                except TypeError:
                    pass

    def voting(self, x):
        return x.view(x.shape[0], self.num_classes, self.votes).mean(dim=2)

    def _single_step(self, x, t: int, return_dynamics: bool = False):
        spike_row = {"t": t}
        cds_rows = []
        for i in range(self.blocks):
            if self.model_type == "cds":
                x = self.cds_layers[i](x)
                if return_dynamics:
                    row = {
                        "t": t,
                        "layer": f"block{i + 1}",
                        "cds_location": "presynaptic_before_conv",
                    }
                    row.update(self.cds_layers[i].last_stats)
                    cds_rows.append(row)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.lifs[i](x)
            if return_dynamics:
                spike_row[f"block{i + 1}_spike_mean"] = float(x.detach().mean().cpu())
            x = self.pools[i](x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.lif_fc1(x)
        if return_dynamics:
            spike_row["fc1_spike_mean"] = float(x.detach().mean().cpu())
        feature = x
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.lif_fc2(x)
        if return_dynamics:
            spike_row["fc2_spike_mean"] = float(x.detach().mean().cpu())
        return self.voting(x), feature, spike_row, cds_rows

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
class STMNISTConfig(BaseConfig):
    dataset_name: str = "stmnist"
    modality: str = "tactile"
    T: int = 30
    batch_size: int = 16
    epochs: int = 100
    channels: int = 64
    blocks: int = 2
    dropout_p: float = 0.5
    num_classes: int = 10
    in_channels: int = 2
    sensor_h: int = 10
    sensor_w: int = 10
    train_ratio: float = 0.8
    split_seed: int = 2026
    time_window_ms: float = 0.0
    normalize: str = "max"
    force_reprocess: bool = False


def find_data_submission(data_root: str) -> Path:

    root = Path(data_root).expanduser().resolve()
    candidates = [
        root,
        root / "data_submission",
        root / "STMNIST" / "raw" / "data_submission",
        root / "STMNIST dataset NUS Tee Research Group" / "data_submission",
        root / "STMNIST_dataset_NUS_Tee_Research_Group" / "data_submission",
    ]
    for c in candidates:
        if c.name == "data_submission" and c.is_dir():
            return c
        if (c / "data_submission").is_dir():
            return c / "data_submission"


    matches = []
    if root.exists():
        for p in root.rglob("data_submission"):
            if p.is_dir():
                matches.append(p)
    if matches:

        matches = sorted(matches, key=lambda p: (len(str(p)), str(p)))
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a ST-MNIST 'data_submission' directory under {root}. "
        "Please set --data_root to the extracted ST-MNIST folder or its parent directory."
    )


def numeric_sort_key(path: Path):
    try:
        return (0, int(path.name))
    except Exception:
        return (1, path.name)


def list_stmnist_mat_files(data_submission: Path) -> List[Tuple[Path, int]]:

    items: List[Tuple[Path, int]] = []
    label_dirs = [p for p in data_submission.iterdir() if p.is_dir()]
    for d in sorted(label_dirs, key=numeric_sort_key):
        try:
            label = int(d.name)
        except ValueError:

            continue
        for f in sorted(d.glob("*.mat")):
            if f.name.lower() == "lut.mat":
                continue
            items.append((f, label))
    if not items:
        raise RuntimeError(f"No .mat samples were found in {data_submission}/<label>/*.mat")
    return items


def make_label_mapping(raw_labels: Sequence[int], num_classes: int) -> Dict[int, int]:
    unique = sorted(set(int(x) for x in raw_labels))
    if len(unique) != int(num_classes):
        print(
            f"[Warning] Found {len(unique)} unique labels {unique}, but cfg.num_classes={num_classes}. "
            "Labels will still be remapped to contiguous indices."
        )
    return {raw: i for i, raw in enumerate(unique)}


def stratified_split_indices(labels: Sequence[int], train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:

    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels, dtype=np.int64)
    train_idx, test_idx = [], []
    for cls in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            n_train = 1
        else:
            n_train = int(round(len(idx) * float(train_ratio)))
            n_train = min(max(n_train, 1), len(idx) - 1)
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def reference_matrix_10x10() -> np.ndarray:


    return np.arange(100, dtype=np.int64).reshape(10, 10).T.reshape(-1)


def load_spiketrain_mat(path: Path) -> np.ndarray:
    try:
        import scipy.io
    except Exception as e:
        raise ImportError("Reading ST-MNIST .mat files requires scipy. Please install scipy.") from e
    mat = scipy.io.loadmat(str(path))
    if "spiketrain" not in mat:
        raise KeyError(f"{path} does not contain variable 'spiketrain'. Available keys: {list(mat.keys())}")
    arr = np.asarray(mat["spiketrain"])
    if arr.ndim != 2 or arr.shape[0] < 101:
        raise ValueError(f"Unexpected spiketrain shape in {path}: {arr.shape}; expected [101, num_events].")
    return arr


def spiketrain_to_frames(
    spiketrain: np.ndarray,
    T: int,
    in_channels: int = 2,
    sensor_h: int = 10,
    sensor_w: int = 10,
    time_window_ms: float = 2000.0,
    normalize: str = "max",
) -> torch.Tensor:


    if int(sensor_h) != 10 or int(sensor_w) != 10:
        raise ValueError("ST-MNIST uses a 10 x 10 tactile sensor. Keep sensor_h=sensor_w=10.")
    if int(in_channels) not in (1, 2):
        raise ValueError("ST-MNIST local reader supports in_channels=1 or 2.")

    values = np.asarray(spiketrain[:-1, :], dtype=np.float32)
    times = np.asarray(spiketrain[-1, :], dtype=np.float64)
    if times.size == 0:
        return torch.zeros((int(T), int(in_channels), 10, 10), dtype=torch.float32)


    t_ms = times * 1000.0 if np.nanmax(times) <= 10.0 else times

    ref = reference_matrix_10x10()
    frames_flat = np.zeros((int(T), int(in_channels), 100), dtype=np.float32)

    if int(in_channels) == 1:

        pos_count = (values == 1.0).sum(axis=0)
        neg_count = (values == -1.0).sum(axis=0)
        valid = (pos_count == 1) & (neg_count == 0) & np.isfinite(t_ms)
        if np.any(valid):
            row_idx = np.argmax(values[:, valid] == 1.0, axis=0)
            taxel_idx = ref[row_idx]
            duration = float(time_window_ms) if float(time_window_ms) > 0 else max(float(np.nanmax(t_ms[valid])), 1e-9)
            bins = np.floor(t_ms[valid] / duration * int(T)).astype(np.int64)
            bins = np.clip(bins, 0, int(T) - 1)
            np.add.at(frames_flat[:, 0, :], (bins, taxel_idx), 1.0)
    else:

        abs_count = (np.abs(values) > 0.0).sum(axis=0)
        valid = (abs_count == 1) & np.isfinite(t_ms)
        if np.any(valid):
            sub = values[:, valid]
            row_idx = np.argmax(np.abs(sub) > 0.0, axis=0)
            signs = sub[row_idx, np.arange(sub.shape[1])]
            channels = (signs < 0).astype(np.int64)
            taxel_idx = ref[row_idx]
            duration = float(time_window_ms) if float(time_window_ms) > 0 else max(float(np.nanmax(t_ms[valid])), 1e-9)
            bins = np.floor(t_ms[valid] / duration * int(T)).astype(np.int64)
            bins = np.clip(bins, 0, int(T) - 1)
            np.add.at(frames_flat, (bins, channels, taxel_idx), 1.0)

    frames = frames_flat.reshape(int(T), int(in_channels), 10, 10)
    if normalize == "max":
        m = float(frames.max())
        if m > 0:
            frames = frames / m
    elif normalize == "log":
        frames = np.log1p(frames)
        m = float(frames.max())
        if m > 0:
            frames = frames / m
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize mode: {normalize}")
    return torch.from_numpy(frames.astype(np.float32))


class LocalSTMNISTDataset(Dataset):
    def __init__(self, frames: torch.Tensor, labels: torch.Tensor):
        self.frames = frames.float()
        self.labels = labels.long()
        if len(self.frames) != len(self.labels):
            raise ValueError(f"frames/labels length mismatch: {len(self.frames)} vs {len(self.labels)}")

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.frames[index], self.labels[index]


def cache_paths(cfg: STMNISTConfig):
    processed = Path(cfg.data_root).expanduser().resolve() / "STMNIST_local_processed"
    tag = (
        f"T{cfg.T}_C{cfg.in_channels}_H{cfg.sensor_h}_W{cfg.sensor_w}_"
        f"tw{int(cfg.time_window_ms)}_norm{cfg.normalize}_split{cfg.split_seed}_"
        f"train{int(round(cfg.train_ratio * 100))}"
    )
    return (processed / f"train_{tag}.pt", processed / f"test_{tag}.pt",
            processed / f"meta_{tag}.json")


def preprocess_stmnist(cfg: STMNISTConfig) -> Tuple[LocalSTMNISTDataset, LocalSTMNISTDataset]:
    train_cache, test_cache, meta_cache = cache_paths(cfg)
    if train_cache.exists() and test_cache.exists() and (not cfg.force_reprocess):
        train = load_torch_file(train_cache, map_location="cpu")
        test = load_torch_file(test_cache, map_location="cpu")
        return (LocalSTMNISTDataset(train["frames"], train["labels"]),
                LocalSTMNISTDataset(test["frames"], test["labels"]))

    data_submission = find_data_submission(cfg.data_root)
    print(f"[ST-MNIST] Using raw data directory: {data_submission}")
    items = list_stmnist_mat_files(data_submission)
    raw_labels = [label for _, label in items]
    label_map = make_label_mapping(raw_labels, cfg.num_classes)
    mapped_labels = [label_map[int(y)] for y in raw_labels]
    train_idx, test_idx = stratified_split_indices(
        mapped_labels, cfg.train_ratio, cfg.split_seed)

    def build(indices: Sequence[int], split: str):
        frames_list, labels_list = [], []
        for n, idx in enumerate(indices):
            f, raw_label = items[int(idx)]
            if n % 500 == 0:
                print(f"[ST-MNIST] Processing {split} sample {n}/{len(indices)}: {f.name}")
            st = load_spiketrain_mat(f)
            frames = spiketrain_to_frames(
                st,
                T=cfg.T,
                in_channels=cfg.in_channels,
                sensor_h=cfg.sensor_h,
                sensor_w=cfg.sensor_w,
                time_window_ms=cfg.time_window_ms,
                normalize=cfg.normalize,
            )
            frames_list.append(frames)
            labels_list.append(label_map[int(raw_label)])
        if frames_list:
            x = torch.stack(frames_list, dim=0)
            y = torch.tensor(labels_list, dtype=torch.long)
        else:
            x = torch.empty((0, cfg.T, cfg.in_channels, cfg.sensor_h, cfg.sensor_w), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.long)
        return {"frames": x, "labels": y}

    train = build(train_idx, "train")
    test = build(test_idx, "test")
    train_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train, train_cache)
    torch.save(test, test_cache)
    meta = {
        "data_submission": str(data_submission),
        "num_total": len(items),
        "num_train": int(train["labels"].numel()),
        "num_test": int(test["labels"].numel()),
        "raw_label_to_contiguous": {str(k): int(v) for k, v in label_map.items()},
        "T": cfg.T,
        "in_channels": cfg.in_channels,
        "time_window_ms": cfg.time_window_ms,
        "normalize": cfg.normalize,
        "train_ratio": cfg.train_ratio,
        "split_seed": cfg.split_seed,
    }
    with open(meta_cache, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[ST-MNIST] Saved processed cache to {train_cache.parent}")
    return (LocalSTMNISTDataset(train["frames"], train["labels"]),
            LocalSTMNISTDataset(test["frames"], test["labels"]))


def build_stmnist_datasets(cfg: STMNISTConfig):
    if cfg.dry_run:
        x_train = torch.bernoulli(
            torch.full(
                (cfg.dry_run_train_samples, cfg.T, cfg.in_channels, cfg.sensor_h, cfg.sensor_w),
                0.03,
            )
        )
        y_train = torch.randint(0, cfg.num_classes, (cfg.dry_run_train_samples,))
        x_test = torch.bernoulli(
            torch.full(
                (cfg.dry_run_test_samples, cfg.T, cfg.in_channels, cfg.sensor_h, cfg.sensor_w),
                0.03,
            )
        )
        y_test = torch.randint(0, cfg.num_classes, (cfg.dry_run_test_samples,))
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
    return preprocess_stmnist(cfg)


def build_stmnist_model(name: str, cfg: STMNISTConfig, device: torch.device):
    model = STMNISTSNN(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        input_hw=(cfg.sensor_h, cfg.sensor_w),
        channels=cfg.channels,
        blocks=cfg.blocks,
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
        raise RuntimeError("ST-MNIST loaded an incompatible network definition.")
    return model.to(device)


def parse_args(argv=None):

    p = argparse.ArgumentParser(description="Train static/CDS SNNs on locally loaded ST-MNIST for Section 2.6.")
    p.add_argument("--data_root", type=str, default="data/ST-MNIST", help="Extracted ST-MNIST folder or its parent directory.")
    p.add_argument("--out_dir", type=str, default="results/section_2_6")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--model", choices=["static", "cds", "both"], default="both")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--tau_m", type=float, default=2.0)
    p.add_argument("--tau_ca_init", type=float, default=2.0)
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--cds_variant", choices=["full", "constant_release", "memory_free"], default="full")
    p.add_argument("--in_channels", type=int, choices=[1, 2], default=2,
                   help="2 preserves ON/OFF events as separate channels; 1 retains ON events only.")
    p.add_argument("--sensor_h", type=int, default=10)
    p.add_argument("--sensor_w", type=int, default=10)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--time_window_ms", type=float, default=0.0, help="<=0 covers each sample's complete duration.")
    p.add_argument("--normalize", choices=["none", "max", "log"], default="max")
    p.add_argument("--force_reprocess", action="store_true", help="Rebuild local ST-MNIST frame cache from .mat files.")
    p.add_argument("--analysis_batches", type=int, default=32)
    p.add_argument("--isolated_perturbation_ratios", type=str,
                   default="0.0,0.025,0.05,0.1")
    p.add_argument("--isolated_perturbation_repeats", type=int, default=3)
    p.add_argument("--isolation_guard_steps", type=int, default=4)
    p.add_argument("--selectivity_ratio", type=float, default=0.1)
    p.add_argument("--minimum_realized_fraction", type=float, default=0.95)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--TET", action="store_true", help="Use TET-style one-hot MSE over all time steps.")
    p.add_argument("--lamb", type=float, default=0.0, help="TET regularization weight. Keep 0.0 for standard time-step averaged TET loss.")
    p.add_argument("--means", type=float, default=1.0, help="TET regularization target mean used when --lamb > 0.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dry_run_train_samples", type=int, default=24)
    p.add_argument("--dry_run_test_samples", type=int, default=12)
    return p.parse_args(argv)


def config_from_args(args=None) -> STMNISTConfig:

    if args is None:
        args = parse_args([])
    return STMNISTConfig(**vars(args))


def main():
    cfg = config_from_args(parse_args())
    print(
        f"[stmnist] isolated-perturbation protocol="
        f"{ISOLATED_PERTURBATION_PROTOCOL_VERSION}"
    )
    worker_init = set_seed(cfg.seed)
    train_set, test_set = build_stmnist_datasets(cfg)
    if min(len(train_set), len(test_set)) == 0:
        raise RuntimeError(f"Empty ST-MNIST split: train={len(train_set)}, test={len(test_set)}")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed),
        pin_memory=("cuda" in cfg.device and torch.cuda.is_available()),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0 if cfg.dry_run else cfg.num_workers,
        worker_init_fn=worker_init,
        generator=make_data_loader_generator(cfg.seed + 1),
        pin_memory=("cuda" in cfg.device and torch.cuda.is_available()),
    )
    run_training_pipeline(cfg, train_loader, test_loader, build_stmnist_model, cfg.num_classes)


if __name__ == "__main__":
    main()
