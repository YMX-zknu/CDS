from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
    HAS_SPIKINGJELLY = True
except Exception:
    HAS_SPIKINGJELLY = False

    class _SpikeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0).to(x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            grad = 1.0 / (1.0 + np.pi * np.pi * x * x)
            return grad_output * grad

    class _FallbackLIFNode(nn.Module):
        def __init__(self, tau: float = 2.0, threshold: float = 1.0, **kwargs):
            super().__init__()
            self.tau = float(tau)
            self.threshold = float(threshold)
            self.v = None

        def forward(self, x):
            if self.v is None or self.v.shape != x.shape:
                self.v = torch.zeros_like(x)
            self.v = self.v + (x - self.v) / self.tau
            s = _SpikeFn.apply(self.v - self.threshold)
            self.v = self.v * (1.0 - s.detach())
            return s

        def reset(self):
            self.v = None

    class _NeuronNamespace:
        LIFNode = _FallbackLIFNode

    class _SurrogateNamespace:
        class ATan:
            def __init__(self, *args, **kwargs):
                pass

    class _FunctionalNamespace:
        @staticmethod
        def reset_net(module):
            for m in module.modules():
                if hasattr(m, "reset") and m is not module:
                    try:
                        m.reset()
                    except TypeError:
                        pass

        @staticmethod
        def set_step_mode(module, mode):
            return None

    neuron = _NeuronNamespace()
    surrogate = _SurrogateNamespace()
    functional = _FunctionalNamespace()


def set_seed(seed: int = 2026):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    def worker_init_fn(worker_id: int):
        worker_seed = int(seed) + int(worker_id)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def make_data_loader_generator(seed: int) -> torch.Generator:

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def reset_data_loader_generator(loader: DataLoader, seed: int) -> None:

    if loader.generator is not None:
        loader.generator.manual_seed(int(seed))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_torch_file(path: Path, map_location="cpu"):


    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def reset_net(model: nn.Module) -> None:


    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "reset"):
            try:
                m.reset()
            except TypeError:
                pass


class CutOrPadFrames:


    def __init__(self, fixed_frames_num: int):
        self.T = int(fixed_frames_num)

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        frames_num = int(x.shape[0])
        if frames_num >= self.T:
            return x[: self.T]
        padding_shape = (self.T - frames_num,) + tuple(x.shape[1:])
        return torch.cat([x, torch.zeros(padding_shape, dtype=x.dtype)], dim=0)


def to_time_first(x: torch.Tensor, device: torch.device, expected_channels: Optional[int] = None) -> torch.Tensor:

    x = torch.as_tensor(x).float()
    if x.ndim == 5:


        if x.shape[2] in [1, 2, 700] or (expected_channels is not None and x.shape[2] == expected_channels):
            x = x.transpose(0, 1).contiguous()
        elif x.shape[-1] in [1, 2] or (expected_channels is not None and x.shape[-1] == expected_channels):
            x = x.permute(1, 0, 4, 2, 3).contiguous()
        else:

            x = x.transpose(0, 1).contiguous()
    elif x.ndim == 3:

        x = x.transpose(0, 1).contiguous()
    else:
        raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
    return x.to(device)


def labels_to_device(y, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(y, dtype=torch.long, device=device)


ARCHITECTURE_VERSION = "presynaptic_cds_v4_dataset_specific"


TRANSMISSION_ORDER = "CDS->Conv/Linear->BN(if used)->LIF"


ISOLATED_PERTURBATION_PROTOCOL_VERSION = "amplitude_matched_maximum_guard_v4"


ISOLATED_NOISE_PROTOCOL_VERSION = ISOLATED_PERTURBATION_PROTOCOL_VERSION


class CDSLayer(nn.Module):


    def __init__(
        self,
        channels: int,
        param_dims: int = 4,
        tau_ca_init: float = 2.0,
        ca_influx_init: float = 0.5,
        k_ca_init: float = 1.0,
        u0_init: float = 0.5,
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


        u_max = torch.sigmoid(self.u_max)


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


class DVSGestureSNN(nn.Module):


    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        input_hw: Tuple[int, int] = (128, 128),
        channels: int = 128,
        blocks: int = 5,
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
            self.lifs.append(neuron.LIFNode(tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True))
            self.pools.append(nn.MaxPool2d(2, 2))
            c_in = channels


        self.adaptive_pool = nn.Identity()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(channels * 4 * 4, fc_dim)
        self.lif_fc1 = neuron.LIFNode(tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.dropout2 = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(fc_dim, num_classes * votes)
        self.lif_fc2 = neuron.LIFNode(tau=tau_m, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.feature_dim = fc_dim

    def reset(self):
        for m in self.modules():
            if m is not self and hasattr(m, "reset"):
                try:
                    m.reset()
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
                        "layer": f"block{i+1}",
                        "cds_location": "presynaptic_before_conv",
                    }
                    row.update(self.cds_layers[i].last_stats)
                    cds_rows.append(row)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.lifs[i](x)
            if return_dynamics:
                spike_row[f"block{i+1}_spike_mean"] = float(x.detach().mean().cpu())
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
        out_seq, feat_seq = [], []
        spike_stats, cds_stats = [], []
        input_release_seq = []
        for t in range(int(x_seq.shape[0])):
            out_t, feat_t, spike_row, cds_rows = self._single_step(x_seq[t], t, return_dynamics)
            out_seq.append(out_t)
            feat_seq.append(feat_t)
            if return_input_release and self.model_type == "cds":
                input_release_seq.append(self.cds_layers[0].release_prob.detach().clone())
            if return_dynamics:
                spike_stats.append(spike_row)
                cds_stats.extend(cds_rows)
        out_seq = torch.stack(out_seq, dim=0)
        feat_seq = torch.stack(feat_seq, dim=0)
        if return_dynamics or return_input_release:
            dynamics = {
                "features": feat_seq.mean(dim=0).detach(),
                "feature_seq": feat_seq.detach(),
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


def verify_presynaptic_forward_order(model: nn.Module, x_seq: torch.Tensor) -> Dict[str, object]:


    if x_seq.ndim < 3 or x_seq.shape[0] < 1 or x_seq.shape[1] < 1:
        raise ValueError(f"Expected non-empty time-first input [T,N,...], got {tuple(x_seq.shape)}")

    stages: List[Tuple[str, nn.Module]] = []
    model_type = str(getattr(model, "model_type", ""))
    if model_type not in {"static", "cds"}:
        raise TypeError(f"Unsupported model_type={model_type!r} for architecture verification.")

    if hasattr(model, "convs") and hasattr(model, "lifs"):
        if model_type == "cds" and len(model.cds_layers) != len(model.convs):
            raise RuntimeError("Every CDS convolutional block must own one presynaptic CDS layer.")
        bns = getattr(model, "bns", None)
        if bns is not None and len(bns) not in {0, len(model.convs)}:
            raise RuntimeError("Convolutional models must define either zero or one BN module per block.")
        for i, (conv, lif) in enumerate(zip(model.convs, model.lifs)):
            if model_type == "cds":
                cds = model.cds_layers[i]
                if int(cds.channels) != int(conv.in_channels):
                    raise RuntimeError(
                        f"block{i+1}: CDS channels={cds.channels} must match Conv input channels={conv.in_channels}."
                    )
                stages.append((f"block{i+1}.cds", cds))
            stages.append((f"block{i+1}.conv", conv))
            if bns is not None and len(bns) == len(model.convs):
                stages.append((f"block{i+1}.bn", bns[i]))
            stages.append((f"block{i+1}.lif", lif))
        architecture = "conv"
    elif hasattr(model, "linears") and hasattr(model, "lifs"):
        if model_type == "cds" and len(model.cds_layers) != len(model.linears):
            raise RuntimeError("Every CDS MLP hidden block must own one presynaptic CDS layer.")
        for i, (linear, lif) in enumerate(zip(model.linears, model.lifs)):
            if model_type == "cds":
                cds = model.cds_layers[i]
                if int(cds.channels) != int(linear.in_features):
                    raise RuntimeError(
                        f"fc{i+1}: CDS units={cds.channels} must match Linear input units={linear.in_features}."
                    )
                stages.append((f"fc{i+1}.cds", cds))
            stages.extend([(f"fc{i+1}.linear", linear), (f"fc{i+1}.lif", lif)])
        architecture = "mlp"
    else:
        raise TypeError(f"Unsupported model class for architecture verification: {type(model).__name__}")

    expected = [name for name, _ in stages]
    observed: List[str] = []
    handles = []

    def record(name: str):
        def hook(_module, _inputs):
            observed.append(name)
        return hook

    for name, module in stages:
        handles.append(module.register_forward_pre_hook(record(name)))

    was_training = model.training
    try:
        model.eval()
        reset_net(model)
        with torch.no_grad():
            model(x_seq[:1, :1])
    finally:
        for handle in handles:
            handle.remove()
        reset_net(model)
        model.train(was_training)

    if observed != expected:
        raise RuntimeError(
            "Forward-order verification failed. "
            f"Expected {expected}, observed {observed}."
        )
    if model_type == "cds":
        for name in expected:
            if name.endswith(".conv"):
                prefix = name.rsplit(".", 1)[0]
                if expected.index(f"{prefix}.cds") > expected.index(name):
                    raise RuntimeError(f"{prefix}: CDS was called after Conv.")
            if name.endswith(".linear"):
                prefix = name.rsplit(".", 1)[0]
                if expected.index(f"{prefix}.cds") > expected.index(name):
                    raise RuntimeError(f"{prefix}: CDS was called after Linear.")

    return {
        "verified": True,
        "architecture_version": ARCHITECTURE_VERSION,
        "model_type": model_type,
        "architecture": architecture,
        "transmission_order": TRANSMISSION_ORDER,
        "observed_hidden_stage_order": observed,
    }


@dataclass
class BaseConfig:


    data_root: str = "data/DVS-Gesture"
    out_dir: str = "results/section_2_6"
    dataset_name: str = "dvsgesture"
    modality: str = "visual"
    T: int = 20
    batch_size: int = 16
    epochs: int = 100
    num_workers: int = 4
    device: str = "cuda:0"
    lr: float = 1e-4
    eta_min: float = 1e-5
    seed: int = 2026
    model: str = "both"
    channels: int = 128
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
            raise ValueError(
                "isolated_perturbation_ratios must contain at least one value."
            )
        if any(v < 0 for v in values):
            raise ValueError("isolated_perturbation_ratios must be non-negative.")
        return sorted(set(values))


def model_names(model_arg: str) -> List[str]:
    if model_arg == "both":
        return ["static", "cds"]
    if model_arg in {"static", "cds"}:
        return [model_arg]
    raise ValueError("--model must be static, cds or both")


def classification_from_confusion(confusion: np.ndarray) -> Dict[str, float]:

    cm = np.asarray(confusion, dtype=np.float64)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(np.diag(cm), support, out=np.zeros_like(support), where=support > 0)
    precision = np.divide(np.diag(cm), predicted, out=np.zeros_like(predicted), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    valid = support > 0
    return {
        "acc": float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else np.nan,
        "balanced_acc": float(recall[valid].mean()) if valid.any() else np.nan,
        "macro_f1": float(f1[valid].mean()) if valid.any() else np.nan,
    }


def tet_mse_loss_onehot(outputs: torch.Tensor, labels_onehot: torch.Tensor, means: float = 1.0, lamb: float = 0.0) -> torch.Tensor:


    loss = 0.0
    for t in range(outputs.shape[0]):
        loss = loss + F.mse_loss(outputs[t], labels_onehot)
    loss = loss / outputs.shape[0]
    if float(lamb) > 0:
        target = torch.ones_like(outputs) * float(means)
        loss = (1.0 - float(lamb)) * loss + float(lamb) * F.mse_loss(outputs, target)
    return loss


def compute_mse_onehot_loss(
    out_seq: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    cfg: Optional["BaseConfig"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_fr = out_seq.mean(dim=0)
    if cfg is not None and str(getattr(cfg, "loss_mode", "rate_mse")) == "spike_count":
        spike_counts = out_seq.sum(dim=0)
        target_counts = torch.full_like(
            spike_counts, float(getattr(cfg, "false_spike_count", 10.0))
        )
        target_counts.scatter_(
            1, labels.unsqueeze(1), float(getattr(cfg, "true_spike_count", 60.0))
        )
        return F.mse_loss(spike_counts, target_counts), spike_counts
    target = F.one_hot(labels, num_classes).float()
    if cfg is not None and bool(getattr(cfg, "TET", False)):
        loss = tet_mse_loss_onehot(
            out_seq,
            target,
            means=float(getattr(cfg, "means", 1.0)),
            lamb=float(getattr(cfg, "lamb", 0.0)),
        )
    else:
        loss = F.mse_loss(out_fr, target)
    return loss, out_fr


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device, cfg: BaseConfig, num_classes: int):
    model.train()
    total_loss, total_correct, total_num = 0.0, 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    for frame, label in loader:
        optimizer.zero_grad(set_to_none=True)
        x_seq = to_time_first(frame, device)
        y = labels_to_device(label, device)
        with torch.cuda.amp.autocast(enabled=bool(cfg.amp and device.type == "cuda")):
            out_seq = model(x_seq)
            loss, out_fr = compute_mse_onehot_loss(out_seq, y, num_classes, cfg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * y.numel()
        total_correct += int((out_fr.argmax(dim=1) == y).sum().detach().cpu())
        total_num += int(y.numel())
        reset_net(model)
    return {"loss": total_loss / max(total_num, 1), "acc": total_correct / max(total_num, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: BaseConfig, num_classes: int, return_extra: bool = False, analysis_batches: Optional[set] = None):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    spike_rows, cds_rows, emb_rows, conf_rows = [], [], [], []
    analysis_batches = analysis_batches or set()
    for batch_idx, (frame, label) in enumerate(loader):
        x_seq = to_time_first(frame, device)
        y = labels_to_device(label, device)
        record = bool(return_extra and batch_idx in analysis_batches)
        if record:
            out_seq, dyn = model(x_seq, return_dynamics=True)
        else:
            out_seq, dyn = model(x_seq), None
        loss, out_fr = compute_mse_onehot_loss(out_seq, y, num_classes, cfg)
        pred = out_fr.argmax(dim=1)
        total_loss += float(loss.detach().cpu()) * y.numel()
        total_correct += int((pred == y).sum().detach().cpu())
        total_num += int(y.numel())
        for gt, pr in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[int(gt), int(pr)] += 1
        if dyn is not None:
            mname = getattr(model, "model_type", "model")
            for row in dyn["spike_stats"]:
                r = dict(row); r.update({"batch": batch_idx, "model": mname})
                spike_rows.append(r)
            for row in dyn["cds_stats"]:
                r = dict(row); r.update({"batch": batch_idx, "model": mname})
                cds_rows.append(r)
            prob = dyn["logits_seq"].mean(0).softmax(dim=-1).cpu()
            conf_rows.append({
                "batch": batch_idx,
                "model": mname,
                "mean_max_confidence": float(prob.max(1).values.mean()),
                "mean_true_class_probability": float(prob[torch.arange(prob.shape[0]), y.cpu()].mean()),
            })
            feats = dyn["features"].cpu().numpy()
            for i in range(feats.shape[0]):
                row = {"batch": batch_idx, "sample": i, "model": mname, "label": int(y.cpu()[i]), "pred": int(pred.cpu()[i])}
                for d in range(min(32, feats.shape[1])):
                    row[f"f{d}"] = float(feats[i, d])
                emb_rows.append(row)
        reset_net(model)
    out = {"loss": total_loss / max(total_num, 1), "confusion": confusion, **classification_from_confusion(confusion)}
    if return_extra:
        out.update({
            "spike_stats": pd.DataFrame(spike_rows),
            "cds_stats": pd.DataFrame(cds_rows),
            "embeddings": pd.DataFrame(emb_rows),
            "confidence": pd.DataFrame(conf_rows),
        })
    return out


@dataclass
class IsolatedPerturbationBatch:
    perturbed: torch.Tensor
    added_mask: torch.Tensor
    supported_original_mask: torch.Tensor
    n_original: int
    n_target: int
    n_added: int
    n_available: int


def temporal_support_mask(x_seq: torch.Tensor, guard_steps: int) -> torch.Tensor:


    guard_steps = int(guard_steps)
    if guard_steps < 1:
        raise ValueError("isolation_guard_steps must be at least 1.")
    T, batch_size = int(x_seq.shape[0]), int(x_seq.shape[1])
    occupied = x_seq.reshape(T, batch_size, -1) != 0
    previous_support = torch.zeros_like(occupied)
    for lag in range(1, min(guard_steps, T - 1) + 1):
        previous_support[lag:] |= occupied[:-lag]
    return (occupied & previous_support).reshape_as(x_seq)


def _same_polarity_group(feature_index: int, x_seq: torch.Tensor) -> int:

    if x_seq.ndim < 4:
        return 0
    spatial_size = int(np.prod(x_seq.shape[3:]))
    return int(feature_index) // spatial_size


def _maximum_isolated_candidate_mask(
    eligible_sample: torch.Tensor,
    guard_steps: int,
) -> torch.Tensor:


    if eligible_sample.device.type != "cpu":
        eligible_sample = eligible_sample.cpu()
    eligible_sample = eligible_sample.bool()
    T, feature_count = map(int, eligible_sample.shape)
    selected = torch.zeros_like(eligible_sample)
    last_selected = torch.full(
        (feature_count,), -(int(guard_steps) + 1), dtype=torch.long
    )
    for time_index in range(T):
        selectable = eligible_sample[time_index] & (
            time_index - last_selected > int(guard_steps)
        )
        selected[time_index] = selectable
        last_selected[selectable] = time_index
    return selected


def _time_balanced_candidate_subsample(
    maximum_mask: torch.Tensor,
    n_target: int,
    generator: torch.Generator,
) -> List[Tuple[int, int]]:


    maximum_mask = maximum_mask.cpu().bool()
    T = int(maximum_mask.shape[0])
    capacity = maximum_mask.sum(dim=1).to(torch.long)
    n_select = min(max(int(n_target), 0), int(capacity.sum().item()))
    if n_select == 0:
        return []

    quota = torch.zeros(T, dtype=torch.long)
    residual = capacity.clone()
    remaining = n_select
    while remaining > 0:
        active = torch.nonzero(residual > 0, as_tuple=False).flatten()
        if active.numel() == 0:
            break
        active = active[torch.randperm(active.numel(), generator=generator)]
        if remaining < int(active.numel()):
            chosen_times = active[:remaining]
            quota[chosen_times] += 1
            residual[chosen_times] -= 1
            remaining = 0
            break
        base = max(1, remaining // int(active.numel()))
        take = torch.minimum(
            residual[active], torch.full_like(active, base, dtype=torch.long)
        )
        quota[active] += take
        residual[active] -= take
        remaining -= int(take.sum().item())

    selected: List[Tuple[int, int]] = []
    for time_index in range(T):
        count = int(quota[time_index].item())
        if count == 0:
            continue
        features = torch.nonzero(
            maximum_mask[time_index], as_tuple=False
        ).flatten()
        order = torch.randperm(features.numel(), generator=generator)[:count]
        selected.extend(
            (time_index, int(feature_index))
            for feature_index in features[order].tolist()
        )
    if len(selected) != n_select:
        raise RuntimeError(
            f"Internal isolated-candidate allocation error: selected "
            f"{len(selected)} of {n_select}."
        )
    return selected


def add_amplitude_matched_isolated_perturbations(
    x_seq: torch.Tensor,
    ratio: float,
    seed: int,
    guard_steps: int = 4,
    minimum_realized_fraction: float = 0.95,
) -> IsolatedPerturbationBatch:


    ratio = float(ratio)
    guard_steps = int(guard_steps)
    minimum_realized_fraction = float(minimum_realized_fraction)
    if ratio < 0:
        raise ValueError("Isolated-perturbation ratio must be non-negative.")
    if guard_steps < 1:
        raise ValueError("guard_steps must be at least 1.")
    if not 0 < minimum_realized_fraction <= 1:
        raise ValueError("minimum_realized_fraction must lie in (0, 1].")

    x = x_seq.clone()
    T, batch_size = int(x.shape[0]), int(x.shape[1])
    flat = x.reshape(T, batch_size, -1)
    original_flat = x_seq.reshape(T, batch_size, -1)
    occupied = original_flat != 0
    n_original_total = int(occupied.sum().item())
    supported = temporal_support_mask(x_seq, guard_steps)
    added_flat = torch.zeros_like(occupied)
    if ratio == 0:
        return IsolatedPerturbationBatch(
            x, added_flat.reshape_as(x_seq), supported, n_original_total, 0, 0, 0
        )

    original_neighbourhood = occupied.clone()
    for lag in range(1, min(guard_steps, T - 1) + 1):
        original_neighbourhood[lag:] |= occupied[:-lag]
        original_neighbourhood[:-lag] |= occupied[lag:]
    active_feature = occupied.any(dim=0).unsqueeze(0).expand_as(occupied)
    eligible = (~original_neighbourhood) & active_feature

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    feature_count = int(flat.shape[2])
    n_target_total = 0
    n_added_total = 0
    n_available_total = 0

    for batch_index in range(batch_size):
        n_original = int(occupied[:, batch_index].sum().item())
        n_target = int(round(n_original * ratio))
        n_target_total += n_target
        if n_target <= 0:
            continue

        maximum_mask = _maximum_isolated_candidate_mask(
            eligible[:, batch_index], guard_steps
        )
        n_available = int(maximum_mask.sum().item())
        n_available_total += n_available
        selected = _time_balanced_candidate_subsample(
            maximum_mask, n_target, generator
        )

        if not selected:
            continue


        group_to_selected: Dict[int, List[Tuple[int, int]]] = {}
        for time_index, feature_index in selected:
            group = _same_polarity_group(feature_index, x_seq)
            group_to_selected.setdefault(group, []).append(
                (time_index, feature_index)
            )
        spatial_size = int(np.prod(x_seq.shape[3:])) if x_seq.ndim >= 4 else feature_count
        for group, group_selected in group_to_selected.items():
            if x_seq.ndim >= 4:
                start, end = group * spatial_size, (group + 1) * spatial_size
            else:
                start, end = 0, feature_count
            pool = original_flat[:, batch_index, start:end]
            pool = pool[pool != 0]
            if pool.numel() == 0:
                raise RuntimeError(
                    "Amplitude matching found no nonzero source values for an active group."
                )
            draw = torch.randint(
                0, int(pool.numel()), (len(group_selected),), generator=generator
            ).to(pool.device)
            amplitudes = pool[draw]
            for index, (time_index, feature_index) in enumerate(group_selected):
                flat[time_index, batch_index, feature_index] = amplitudes[index]
                added_flat[time_index, batch_index, feature_index] = True
        n_added_total += len(selected)

    perturbed = flat.reshape_as(x)
    added_mask = added_flat.reshape_as(x_seq)
    validate_amplitude_matched_isolated_perturbations(
        original=x_seq,
        perturbed=perturbed,
        added_mask=added_mask,
        guard_steps=guard_steps,
        expected_added=n_added_total,
    )
    return IsolatedPerturbationBatch(
        perturbed,
        added_mask,
        supported,
        n_original_total,
        n_target_total,
        n_added_total,
        n_available_total,
    )


def validate_amplitude_matched_isolated_perturbations(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    added_mask: torch.Tensor,
    guard_steps: int,
    expected_added: int,
) -> None:

    if tuple(original.shape) != tuple(perturbed.shape):
        raise RuntimeError("Original and perturbed tensors must have identical shapes.")
    if not torch.equal(perturbed[original != 0], original[original != 0]):
        raise RuntimeError("The intervention changed an original occupied input bin.")

    T, batch_size = int(original.shape[0]), int(original.shape[1])
    original_flat = original.reshape(T, batch_size, -1)
    added_flat = added_mask.reshape(T, batch_size, -1)
    observed = int(added_flat.sum().item())
    if observed != int(expected_added):
        raise RuntimeError(
            f"Expected {expected_added} additions but validated {observed}."
        )
    if torch.any(added_flat & (original_flat != 0)):
        raise RuntimeError("An added perturbation overlaps original activity.")
    for lag in range(0, min(int(guard_steps), T - 1) + 1):
        if lag == 0:
            conflict = added_flat & (original_flat != 0)
        else:
            conflict = (
                (added_flat[lag:] & (original_flat[:-lag] != 0))
                | (added_flat[:-lag] & (original_flat[lag:] != 0))
                | (added_flat[lag:] & added_flat[:-lag])
            )
        if torch.any(conflict):
            raise RuntimeError(
                "A perturbation violates the symmetric feature-wise isolation guard."
            )


def add_isolated_event_bin_noise(
    x_seq: torch.Tensor,
    ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, int, int]:

    batch = add_amplitude_matched_isolated_perturbations(
        x_seq, ratio, seed, guard_steps=4, minimum_realized_fraction=0.95
    )
    return batch.perturbed, batch.n_added, batch.n_original


@torch.no_grad()
def evaluate_isolated_perturbations(
    models: Dict[str, nn.Module],
    loader: DataLoader,
    device: torch.device,
    cfg: BaseConfig,
    num_classes: int,
    ratios: Sequence[float],
    repeats: int,
) -> pd.DataFrame:

    rows = []
    chance = 1.0 / float(num_classes)
    for ratio in ratios:
        n_repeats = 1 if float(ratio) == 0 else int(repeats)
        if n_repeats < 1:
            raise ValueError("isolated_perturbation_repeats must be at least 1.")
        for repeat in range(n_repeats):
            perturbation_seed = int(
                cfg.seed * 100003 + round(float(ratio) * 10000) * 101 + repeat
            )
            for name, model in models.items():
                model.eval()
                confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
                n_added = n_original = n_target = n_available = 0
                supported_release_sum = isolated_release_sum = 0.0
                supported_count = isolated_count = 0
                for batch_index, (frame, label) in enumerate(loader):
                    x_seq = to_time_first(frame, device)
                    perturbation = add_amplitude_matched_isolated_perturbations(
                        x_seq,
                        float(ratio),
                        seed=perturbation_seed + 1000003 * batch_index,
                        guard_steps=cfg.isolation_guard_steps,
                        minimum_realized_fraction=cfg.minimum_realized_fraction,
                    )
                    y = labels_to_device(label, device)
                    if name == "cds":
                        out_seq, dynamics = model(
                            perturbation.perturbed, return_input_release=True
                        )
                        release = dynamics["input_release_seq"]
                        supported_values = release[
                            perturbation.supported_original_mask
                        ]
                        isolated_values = release[perturbation.added_mask]
                        supported_release_sum += float(supported_values.sum().cpu())
                        isolated_release_sum += float(isolated_values.sum().cpu())
                        supported_count += int(supported_values.numel())
                        isolated_count += int(isolated_values.numel())
                    else:
                        out_seq = model(perturbation.perturbed)
                        supported_count += int(
                            perturbation.supported_original_mask.sum().item()
                        )
                        isolated_count += int(perturbation.added_mask.sum().item())
                        supported_release_sum += float(
                            perturbation.supported_original_mask.sum().item()
                        )
                        isolated_release_sum += float(
                            perturbation.added_mask.sum().item()
                        )
                    out_fr = out_seq.mean(0)
                    pred = out_fr.argmax(1)
                    for ground_truth, prediction in zip(
                        y.cpu().numpy(), pred.cpu().numpy()
                    ):
                        confusion[int(ground_truth), int(prediction)] += 1
                    n_added += int(perturbation.n_added)
                    n_target += int(perturbation.n_target)
                    n_original += int(perturbation.n_original)
                    n_available += int(perturbation.n_available)
                    reset_net(model)

                realized_target_fraction = (
                    n_added / n_target if n_target > 0 else 1.0
                )
                if (
                    n_target > 0
                    and realized_target_fraction + 1e-12
                    < float(cfg.minimum_realized_fraction)
                ):
                    raise RuntimeError(
                        "The complete test split cannot realize the requested "
                        "isolated-perturbation load under the symmetric guard: "
                        f"dataset={cfg.dataset_name}, model={name}, "
                        f"ratio={float(ratio):g}, repeat={repeat}, "
                        f"added={n_added}, target={n_target}, "
                        f"maximum_available={n_available}, "
                        f"fraction={realized_target_fraction:.3f} < "
                        f"{float(cfg.minimum_realized_fraction):.3f}. "
                        "Reduce the maximum perturbation ratio or the isolation guard."
                    )

                metrics = classification_from_confusion(confusion)
                mean_supported = (
                    supported_release_sum / supported_count
                    if supported_count
                    else np.nan
                )
                mean_isolated = (
                    isolated_release_sum / isolated_count
                    if isolated_count
                    else np.nan
                )
                selectivity = (
                    1.0 - mean_isolated / mean_supported
                    if supported_count and isolated_count and mean_supported > 0
                    else np.nan
                )
                rows.append({
                    "dataset": cfg.dataset_name,
                    "modality": cfg.modality,
                    "model": name,
                    "isolated_perturbation_protocol": ISOLATED_PERTURBATION_PROTOCOL_VERSION,
                    "perturbation_ratio": float(ratio),
                    "perturbation_repeat": int(repeat),
                    **metrics,
                    "chance_level": chance,
                    "chance_normalized_balanced_accuracy": (
                        (metrics["balanced_acc"] - chance) / (1.0 - chance)
                    ),
                    "mean_supported_release": mean_supported,
                    "mean_isolated_release": mean_isolated,
                    "transmission_selectivity": selectivity,
                    "n_supported_original_bins": supported_count,
                    "n_original_occupied_bins": n_original,
                    "n_target_additions": n_target,
                    "n_added_bins": n_added,
                    "n_available_isolated_bins": n_available,
                    "available_to_target_ratio": (
                        n_available / n_target if n_target > 0 else np.nan
                    ),
                    "realized_perturbation_ratio": n_added / max(n_original, 1),
                    "realized_target_fraction": realized_target_fraction,
                    "isolation_guard_steps": int(cfg.isolation_guard_steps),
                    "perturbation_seed": perturbation_seed,
                })
    return pd.DataFrame(rows)


def isolated_perturbation_auc(results: pd.DataFrame) -> pd.DataFrame:

    rows = []
    for model, sub in results.groupby("model"):
        curve = (
            sub.groupby("perturbation_ratio", as_index=False)[
                "chance_normalized_balanced_accuracy"
            ]
            .mean()
            .sort_values("perturbation_ratio")
        )
        x = curve.perturbation_ratio.to_numpy(float)
        y = curve.chance_normalized_balanced_accuracy.to_numpy(float)
        auc = (
            float(np.trapz(y, x) / (x.max() - x.min()))
            if len(x) > 1 and x.max() > x.min()
            else np.nan
        )
        rows.append({"model": model, "isolated_perturbation_auc": auc})
    return pd.DataFrame(rows)


def sample_analysis_batches(loader: DataLoader, n: int, seed: int) -> set:
    if n <= 0:
        return set()
    total = len(loader)
    k = min(int(n), total)
    rng = np.random.default_rng(seed)
    return set(int(i) for i in rng.choice(np.arange(total), size=k, replace=False))


def run_training_pipeline(
    cfg: BaseConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    build_model_fn,
    num_classes: int,
):
    worker_dir = Path(cfg.out_dir) / cfg.dataset_name / f"seed_{cfg.seed}"
    logs_dir = ensure_dir(worker_dir / "logs")
    ckpt_dir = ensure_dir(worker_dir / "checkpoints")
    save_json(asdict(cfg), logs_dir / "config.json")

    device = torch.device(cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu")
    all_curves, summary_rows = [], []
    trained_models: Dict[str, nn.Module] = {}
    for name in model_names(cfg.model):


        set_seed(cfg.seed)
        reset_data_loader_generator(train_loader, cfg.seed)
        reset_data_loader_generator(test_loader, cfg.seed + 1)
        model = build_model_fn(name, cfg, device)
        try:
            verification_frame, _ = next(iter(train_loader))
        except StopIteration as exc:
            raise RuntimeError("Training loader is empty; architecture verification cannot run.") from exc
        verification_x = to_time_first(verification_frame, device)
        verification = verify_presynaptic_forward_order(model, verification_x)


        set_seed(cfg.seed)
        reset_data_loader_generator(train_loader, cfg.seed)
        reset_data_loader_generator(test_loader, cfg.seed + 1)
        verification.update({"dataset": cfg.dataset_name, "seed": int(cfg.seed)})
        save_json(verification, logs_dir / f"architecture_verification_{name}.json")
        print(
            f"[{cfg.dataset_name}][{name}] verified forward order: "
            f"{verification['observed_hidden_stage_order']}"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.eta_min)
        max_test_acc, max_test_acc_epoch = -1.0, -1
        best_path = ckpt_dir / f"{name}.pt"
        for epoch in range(int(cfg.epochs)):
            tr = train_one_epoch(model, train_loader, optimizer, device, cfg, num_classes)
            te = evaluate(model, test_loader, device, cfg, num_classes, return_extra=False)
            scheduler.step()
            is_best = float(te["acc"]) > max_test_acc
            if is_best:
                max_test_acc = float(te["acc"])
                max_test_acc_epoch = int(epoch + 1)
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "config": asdict(cfg),
                    "model": name,
                    "epoch": int(epoch + 1),
                    "best_test_acc": max_test_acc,
                    "best_test_epoch": max_test_acc_epoch,
                    "architecture_version": ARCHITECTURE_VERSION,
                    "transmission_order": TRANSMISSION_ORDER,
                    "isolated_perturbation_protocol": ISOLATED_PERTURBATION_PROTOCOL_VERSION,
                    "isolated_noise_protocol": ISOLATED_PERTURBATION_PROTOCOL_VERSION,
                    "architecture_verified": True,
                    "observed_hidden_stage_order": verification["observed_hidden_stage_order"],
                }, best_path)
            row = {
                "dataset": cfg.dataset_name,
                "modality": cfg.modality,
                "model": name,
                "seed": cfg.seed,
                "epoch": epoch + 1,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "test_loss": te["loss"],
                "test_acc": te["acc"],
                "max_test_acc_so_far": max_test_acc,
                "max_test_acc_epoch_so_far": max_test_acc_epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "TET": bool(getattr(cfg, "TET", False)),
                "tet_lamb": float(getattr(cfg, "lamb", 0.0)),
                "tet_means": float(getattr(cfg, "means", 1.0)),
                "architecture_version": ARCHITECTURE_VERSION,
                "transmission_order": TRANSMISSION_ORDER,
                "isolated_perturbation_protocol": ISOLATED_PERTURBATION_PROTOCOL_VERSION,
            }
            all_curves.append(row)
            print(f"[{cfg.dataset_name}][{name}] epoch {epoch+1}/{cfg.epochs} "
                  f"train={tr['acc']:.4f} test={te['acc']:.4f} "
                  f"max_test={max_test_acc:.4f} max_test_epoch={max_test_acc_epoch}"
                  f"{' [saved best]' if is_best else ''}")
        if best_path.exists():
            state = load_torch_file(best_path, map_location=device)
            model.load_state_dict(state["model_state"])
        else:
            raise RuntimeError(f"No best-test checkpoint was saved for {cfg.dataset_name}/{name}.")
        print(
            f"[{cfg.dataset_name}][{name}] best test accuracy={max_test_acc:.4f} "
            f"at epoch={max_test_acc_epoch}; checkpoint={best_path}"
        )
        trained_models[name] = model
        final = evaluate(model, test_loader, device, cfg, num_classes, return_extra=True, analysis_batches=sample_analysis_batches(test_loader, cfg.analysis_batches, cfg.seed))
        summary_rows.append({
            "dataset": cfg.dataset_name,
            "modality": cfg.modality,
            "model": name,
            "seed": cfg.seed,
            "best_test_acc": max_test_acc,
            "best_test_epoch": max_test_acc_epoch,
            "test_acc": final["acc"],
            "test_balanced_acc": final["balanced_acc"],
            "test_macro_f1": final["macro_f1"],
            "test_loss": final["loss"],
            "params": sum(p.numel() for p in model.parameters()),
            "TET": bool(getattr(cfg, "TET", False)),
            "tet_lamb": float(getattr(cfg, "lamb", 0.0)),
            "tet_means": float(getattr(cfg, "means", 1.0)),
            "architecture_version": ARCHITECTURE_VERSION,
            "transmission_order": TRANSMISSION_ORDER,
            "isolated_perturbation_protocol": ISOLATED_PERTURBATION_PROTOCOL_VERSION,
            "architecture_verified": True,
        })
        pd.DataFrame(final["confusion"]).to_csv(logs_dir / f"confusion_{name}.csv", index=False)
        final["spike_stats"].to_csv(logs_dir / f"spike_stats_{name}.csv", index=False)
        final["cds_stats"].to_csv(logs_dir / f"cds_stats_{name}.csv", index=False)
        final["embeddings"].to_csv(logs_dir / f"embeddings_{name}.csv", index=False)

    curves = pd.DataFrame(all_curves)
    summary = pd.DataFrame(summary_rows)
    isolated_perturbations = evaluate_isolated_perturbations(
        trained_models,
        test_loader,
        device,
        cfg,
        num_classes,
        cfg.perturbation_ratios(),
        cfg.isolated_perturbation_repeats,
    )
    robustness = isolated_perturbation_auc(isolated_perturbations)
    selectivity = (
        isolated_perturbations[
            np.isclose(
                isolated_perturbations["perturbation_ratio"],
                float(cfg.selectivity_ratio),
            )
        ]
        .groupby("model", as_index=False)[
            [
                "mean_supported_release",
                "mean_isolated_release",
                "transmission_selectivity",
            ]
        ]
        .mean()
    )
    if selectivity.empty:
        raise RuntimeError(
            "selectivity_ratio must be included in isolated_perturbation_ratios."
        )
    summary = summary.merge(robustness, on="model", how="left").merge(
        selectivity, on="model", how="left"
    )

    curves.to_csv(logs_dir / "training_curves.csv", index=False)
    summary.to_csv(logs_dir / "metrics_summary.csv", index=False)
    isolated_perturbations.to_csv(
        logs_dir / "isolated_perturbation_performance.csv", index=False
    )
    isolated_perturbations[
        np.isclose(
            isolated_perturbations["perturbation_ratio"],
            float(cfg.selectivity_ratio),
        )
    ].to_csv(logs_dir / "transmission_selectivity.csv", index=False)
    print(f"Saved standardized Section 2.6 outputs to {logs_dir}")
    return summary


@dataclass
class DVSGestureConfig(BaseConfig):
    dataset_name: str = "dvsgesture"
    modality: str = "visual"
    T: int = 20
    delta_t: int = 125
    batch_size: int = 16
    epochs: int = 100
    channels: int = 128
    dropout_p: float = 0.5
    num_classes: int = 11
    input_hw: Tuple[int, int] = (128, 128)
    blocks: int = 5
    tau_ca_init: float = 2.0


def build_dvsgesture_datasets(cfg: DVSGestureConfig):
    if cfg.dry_run:
        hw = int(getattr(cfg, "dry_run_hw", 128))
        x_train = torch.bernoulli(
            torch.full((cfg.dry_run_train_samples, cfg.T, 2, hw, hw), 0.02)
        )
        y_train = torch.randint(0, cfg.num_classes, (cfg.dry_run_train_samples,))
        x_test = torch.bernoulli(
            torch.full((cfg.dry_run_test_samples, cfg.T, 2, hw, hw), 0.02)
        )
        y_test = torch.randint(0, cfg.num_classes, (cfg.dry_run_test_samples,))
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
    try:
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    except Exception:
        from spikingjelly.activation_based.datasets.dvs128_gesture import DVS128Gesture
    official_train = DVS128Gesture(
        root=cfg.data_root,
        train=True,
        data_type="frame",
        split_by="time",
        duration=int(cfg.delta_t) * 1000,
        transform=CutOrPadFrames(cfg.T),
    )
    test_set = DVS128Gesture(
        root=cfg.data_root,
        train=False,
        data_type="frame",
        split_by="time",
        duration=int(cfg.delta_t) * 1000,
        transform=CutOrPadFrames(cfg.T),
    )
    return official_train, test_set


def build_dvsgesture_model(name: str, cfg: DVSGestureConfig, device: torch.device):
    model = DVSGestureSNN(
        in_channels=2,
        num_classes=cfg.num_classes,
        input_hw=cfg.input_hw,
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
        raise RuntimeError("DVS Gesture loaded an incompatible network definition.")
    return model.to(device)


def parse_args(argv=None):


    p = argparse.ArgumentParser(description="Train static/CDS SNNs on DVS Gesture for Section 2.6.")
    p.add_argument("--data_root", type=str, default='data/DVS-Gesture')
    p.add_argument("--out_dir", type=str, default="results/section_2_6")
    p.add_argument("--T", type=int, default=10,
                   help="Fixed number of DVS frames retained after duration-based integration; shorter sequences are zero-padded.")
    p.add_argument("--delta_t", type=int, default=125,
                   help="DVS frame-integration window in milliseconds (passed to SpikingJelly as duration=delta_t*1000 microseconds).")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--model", choices=["static", "cds", "both"], default="both")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--tau_m", type=float, default=2.0)
    p.add_argument("--tau_ca_init", type=float, default=2.0)
    p.add_argument("--dropout_p", type=float, default=0.5)
    p.add_argument("--cds_variant", choices=["full", "constant_release", "memory_free"], default="full")
    p.add_argument("--analysis_batches", type=int, default=32)
    p.add_argument(
        "--isolated_perturbation_ratios",
        type=str,
        default="0.0,0.025,0.05,0.1",
        help="Added amplitude-matched perturbations relative to occupied bins.",
    )
    p.add_argument(
        "--isolated_perturbation_repeats",
        type=int,
        default=3,
        help="Independent realizations for each nonzero perturbation ratio.",
    )
    p.add_argument(
        "--isolation_guard_steps",
        type=int,
        default=4,
        help="Symmetric feature-wise empty window around every added input.",
    )
    p.add_argument(
        "--selectivity_ratio",
        type=float,
        default=0.1,
        help="Perturbation ratio used for input-layer release-selectivity analysis.",
    )
    p.add_argument(
        "--minimum_realized_fraction",
        type=float,
        default=0.95,
        help="Minimum fraction of requested additions required in every batch.",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--TET", action="store_true", help="Use TET-style one-hot MSE over all time steps.")
    p.add_argument("--lamb", type=float, default=0.0, help="TET regularization weight. Keep 0.0 for standard time-step averaged TET loss.")
    p.add_argument("--means", type=float, default=1.0, help="TET regularization target mean used when --lamb > 0.")
    p.add_argument("--dry_run", action="store_true", help="Run on synthetic tensors to verify code without downloading datasets.")
    p.add_argument("--dry_run_train_samples", type=int, default=24)
    p.add_argument("--dry_run_test_samples", type=int, default=12)
    p.add_argument("--dry_run_hw", type=int, default=128)
    return p.parse_args(argv)


def config_from_args(args=None) -> DVSGestureConfig:

    if args is None:
        args = parse_args([])
    return DVSGestureConfig(**vars(args))


def main():
    cfg = config_from_args(parse_args())
    worker_init = set_seed(cfg.seed)
    train_set, test_set = build_dvsgesture_datasets(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0 if cfg.dry_run else cfg.num_workers, worker_init_fn=worker_init, generator=make_data_loader_generator(cfg.seed))
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0 if cfg.dry_run else cfg.num_workers, worker_init_fn=worker_init, generator=make_data_loader_generator(cfg.seed + 1))
    run_training_pipeline(cfg, train_loader, test_loader, build_dvsgesture_model, cfg.num_classes)


if __name__ == "__main__":
    main()
