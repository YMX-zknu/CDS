from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SECTION_2_6 = ROOT / "experiments" / "section_2_6"
if str(SECTION_2_6) not in sys.path:
    sys.path.insert(0, str(SECTION_2_6))

from section_2_6_dvsgesture_train import (  # noqa: E402
    CDSLayer,
    reset_net,
    set_seed,
)


REGIMES = ("aligned", "matched", "anti_aligned")
VARIANTS = ("static", "constant_release", "memory_free", "full")
EXPERIMENT_VERSION = "context_relevance_boundary_v1"


@dataclass(frozen=True)
class BoundarySpec:
    time_steps: int = 48
    cue_steps: int = 6
    cue_amplitude: float = 0.5
    candidate_start: int = 10
    channels_per_symbol: int = 2
    events_per_channel: int = 4
    isolated_gap: int = 8

    @property
    def input_dim(self) -> int:
        # Two cue channels and two candidate banks. Each bank has two symbols.
        return 2 + 2 * 2 * self.channels_per_symbol

    @property
    def events_per_candidate(self) -> int:
        return self.channels_per_symbol * self.events_per_channel


@dataclass
class BoundaryBatch:
    inputs: np.ndarray
    labels: np.ndarray
    distractor_labels: np.ndarray
    cue_bank: np.ndarray
    relevant_mask: np.ndarray
    distractor_mask: np.ndarray
    regime: str

    def tensors(self) -> Tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.inputs),
            torch.from_numpy(self.labels),
            torch.from_numpy(self.distractor_labels),
            torch.from_numpy(self.cue_bank),
            torch.from_numpy(self.relevant_mask),
            torch.from_numpy(self.distractor_mask),
        )


def parse_int_list(value: str) -> List[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one seed is required.")
    if len(parsed) > 5:
        raise ValueError("This experiment permits at most five random seeds.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("Seeds must be unique.")
    return parsed


def parse_float_list(value: str) -> List[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("tau_values must contain positive values.")
    return sorted(set(parsed))


def parse_choice_list(value: str, allowed: Iterable[str]) -> List[str]:
    allowed = tuple(allowed)
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(allowed))
    if not parsed or unknown:
        raise ValueError(f"Expected a non-empty subset of {allowed}; unknown={unknown}.")
    return parsed


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _schedule(pattern: str, rng: np.random.Generator, spec: BoundarySpec) -> np.ndarray:
    count = spec.events_per_channel
    if pattern == "burst":
        latest = spec.time_steps - count
        start = int(rng.integers(spec.candidate_start, latest + 1))
        return np.arange(start, start + count, dtype=np.int64)
    if pattern == "isolated":
        latest = spec.time_steps - 1 - (count - 1) * spec.isolated_gap
        if latest < spec.candidate_start:
            raise ValueError("BoundarySpec is too short for the requested isolated schedule.")
        start = int(rng.integers(spec.candidate_start, latest + 1))
        return start + np.arange(count, dtype=np.int64) * spec.isolated_gap
    raise ValueError(f"Unknown pattern: {pattern}")


def _candidate_channels(bank: int, symbol: int, spec: BoundarySpec) -> np.ndarray:
    if bank not in (0, 1) or symbol not in (0, 1):
        raise ValueError("bank and symbol must be binary.")
    bank_width = 2 * spec.channels_per_symbol
    start = 2 + bank * bank_width + symbol * spec.channels_per_symbol
    return np.arange(start, start + spec.channels_per_symbol, dtype=np.int64)


def generate_boundary_batch(
    n_samples: int,
    regime: str,
    seed: int,
    spec: BoundarySpec,
) -> BoundaryBatch:
    """Generate a sample-wise rate- and count-matched contextual selection task.

    Each example contains two candidate streams. A cue chooses the relevant stream, and
    the other stream carries the opposite label. Candidate roles switch randomly across
    examples, so fixed channel weights cannot identify relevance. Relevant and distractor
    candidates always contain exactly the same number and amplitude of events.
    """

    if regime not in REGIMES:
        raise ValueError(f"Unknown regime: {regime}")
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if n_samples % 4 != 0:
        raise ValueError(
            "n_samples must be divisible by four so cue bank and class are exactly balanced."
        )
    rng = np.random.default_rng(int(seed))
    x = np.zeros((n_samples, spec.time_steps, spec.input_dim), dtype=np.float32)
    relevant = np.zeros_like(x, dtype=bool)
    distractor = np.zeros_like(x, dtype=bool)
    # Each four-example block contains every cue-bank/class combination. Reusing the
    # schedules within the block makes target and distractor histories exactly balanced
    # across channel identities as well as within each matched example.
    combinations = np.asarray([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=np.int64)
    tiled = np.tile(combinations, (n_samples // 4, 1))
    cue_bank = tiled[:, 0].copy()
    labels = tiled[:, 1].copy()
    distractor_labels = 1 - labels

    for sample in range(n_samples):
        target_bank = int(cue_bank[sample])
        other_bank = 1 - target_bank
        target_symbol = int(labels[sample])
        other_symbol = int(distractor_labels[sample])

        # The cue remains locally available during evidence presentation. Its temporal
        # statistics are identical across classes and candidate-history regimes.
        x[sample, : spec.cue_steps, target_bank] = spec.cue_amplitude
        x[sample, spec.candidate_start :, target_bank] = spec.cue_amplitude

        if sample % 4 == 0:
            burst_times = _schedule("burst", rng, spec)
            isolated_times = _schedule("isolated", rng, spec)
            matched_pattern = "burst" if int(rng.integers(0, 2)) == 0 else "isolated"
            matched_times = _schedule(matched_pattern, rng, spec)
        if regime == "aligned":
            target_times, other_times = burst_times, isolated_times
        elif regime == "anti_aligned":
            target_times, other_times = isolated_times, burst_times
        else:
            # Exact equality, rather than equality only in expectation, makes the null
            # condition a direct invariance test for an input-local transmission rule.
            target_times, other_times = matched_times, matched_times

        target_channels = _candidate_channels(target_bank, target_symbol, spec)
        other_channels = _candidate_channels(other_bank, other_symbol, spec)
        x[sample][np.ix_(target_times, target_channels)] = 1.0
        x[sample][np.ix_(other_times, other_channels)] = 1.0
        relevant[sample][np.ix_(target_times, target_channels)] = True
        distractor[sample][np.ix_(other_times, other_channels)] = True

    order = rng.permutation(n_samples)
    batch = BoundaryBatch(
        inputs=x[order],
        labels=labels[order],
        distractor_labels=distractor_labels[order],
        cue_bank=cue_bank[order],
        relevant_mask=relevant[order],
        distractor_mask=distractor[order],
        regime=regime,
    )
    validate_boundary_batch(batch, spec)
    return batch


def prior_trace(inputs: np.ndarray, tau: float) -> np.ndarray:
    """Return the causal trace immediately before each current input is incorporated."""

    if tau <= 0:
        raise ValueError("tau must be positive.")
    decay = math.exp(-1.0 / float(tau))
    state = np.zeros((inputs.shape[0], inputs.shape[2]), dtype=np.float64)
    traces = np.zeros_like(inputs, dtype=np.float64)
    for time in range(inputs.shape[1]):
        state *= decay
        traces[:, time] = state
        state += inputs[:, time]
    return traces


def validate_boundary_batch(batch: BoundaryBatch, spec: BoundarySpec) -> None:
    if batch.inputs.shape != batch.relevant_mask.shape or batch.inputs.shape != batch.distractor_mask.shape:
        raise RuntimeError("Input and annotation shapes differ.")
    if np.any(batch.relevant_mask & batch.distractor_mask):
        raise RuntimeError("Relevant and distractor event masks overlap.")
    relevant_counts = batch.relevant_mask.sum(axis=(1, 2))
    distractor_counts = batch.distractor_mask.sum(axis=(1, 2))
    expected = np.full(len(batch.labels), spec.events_per_candidate)
    if not np.array_equal(relevant_counts, expected):
        raise RuntimeError("Relevant event counts are not fixed at the preregistered value.")
    if not np.array_equal(distractor_counts, expected):
        raise RuntimeError("Distractor event counts are not fixed at the preregistered value.")
    relevant_amplitude = (batch.inputs * batch.relevant_mask).sum(axis=(1, 2))
    distractor_amplitude = (batch.inputs * batch.distractor_mask).sum(axis=(1, 2))
    if not np.array_equal(relevant_amplitude, distractor_amplitude):
        raise RuntimeError("Relevant and distractor total input amplitudes differ.")
    if not np.array_equal(
        batch.relevant_mask.sum(axis=(0, 1)),
        batch.distractor_mask.sum(axis=(0, 1)),
    ):
        raise RuntimeError("Relevant and distractor roles are not balanced by channel identity.")
    if batch.regime == "matched":
        trace = prior_trace(batch.inputs, tau=2.0)
        for sample in range(len(batch.labels)):
            lhs = np.sort(trace[sample][batch.relevant_mask[sample]])
            rhs = np.sort(trace[sample][batch.distractor_mask[sample]])
            if not np.array_equal(lhs, rhs):
                raise RuntimeError("Matched condition does not have identical local histories.")


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC without adding a machine-learning dependency."""

    positive = np.asarray(positive, dtype=np.float64).ravel()
    negative = np.asarray(negative, dtype=np.float64).ravel()
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    position = 0
    while position < values.size:
        end = position + 1
        while end < values.size and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + 1 + end)
        position = end
    rank_sum = ranks[: positive.size].sum()
    u_statistic = rank_sum - positive.size * (positive.size + 1) / 2.0
    return float(u_statistic / (positive.size * negative.size))


def _fixed_release(
    inputs: np.ndarray,
    variant: str,
    tau: float,
    device: torch.device,
) -> np.ndarray:
    if variant == "static":
        return np.ones_like(inputs, dtype=np.float32)
    layer = CDSLayer(
        channels=inputs.shape[-1],
        param_dims=2,
        tau_ca_init=float(tau),
        ca_influx_init=0.5,
        k_ca_init=1.0,
        u0_init=0.1,
        u_max_init=1.0,
        learnable=False,
        variant=variant,
    ).to(device)
    x = torch.from_numpy(inputs).to(device)
    releases = []
    with torch.no_grad():
        for time in range(x.shape[1]):
            layer(x[:, time])
            releases.append(layer.release_prob.detach().cpu())
    layer.reset()
    return torch.stack(releases, dim=1).numpy()


def transmission_metrics(
    release: np.ndarray,
    batch: BoundaryBatch,
    tau: float,
) -> Dict[str, float]:
    target_release = release[batch.relevant_mask]
    distractor_release = release[batch.distractor_mask]
    target_mean = float(target_release.mean())
    distractor_mean = float(distractor_release.mean())
    trace = prior_trace(batch.inputs, tau)
    target_trace = float(trace[batch.relevant_mask].mean())
    distractor_trace = float(trace[batch.distractor_mask].mean())
    sample_contrasts = []
    for sample in range(batch.inputs.shape[0]):
        sample_contrasts.append(
            float(release[sample][batch.relevant_mask[sample]].mean())
            - float(release[sample][batch.distractor_mask[sample]].mean())
        )
    sample_contrasts = np.asarray(sample_contrasts, dtype=np.float64)
    return {
        "relevant_event_count": int(batch.relevant_mask.sum()),
        "distractor_event_count": int(batch.distractor_mask.sum()),
        "relevant_prior_trace": target_trace,
        "distractor_prior_trace": distractor_trace,
        "prior_trace_contrast": target_trace - distractor_trace,
        "relevant_release": target_mean,
        "distractor_release": distractor_mean,
        "release_contrast": target_mean - distractor_mean,
        "release_ratio": target_mean / max(distractor_mean, 1e-12),
        "relevance_auc": binary_auc(target_release, distractor_release),
        "sample_contrast_mean": float(sample_contrasts.mean()),
        "sample_contrast_std": float(sample_contrasts.std(ddof=1)) if len(sample_contrasts) > 1 else 0.0,
        "sample_contrast_sem": float(sample_contrasts.std(ddof=1) / math.sqrt(len(sample_contrasts))) if len(sample_contrasts) > 1 else 0.0,
    }


def run_transmission_audit(
    seeds: Sequence[int],
    regimes: Sequence[str],
    variants: Sequence[str],
    tau_values: Sequence[float],
    n_samples: int,
    spec: BoundarySpec,
    device: torch.device,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for seed in seeds:
        for regime in regimes:
            batch = generate_boundary_batch(n_samples, regime, seed, spec)
            for tau in tau_values:
                for variant in variants:
                    release = _fixed_release(batch.inputs, variant, tau, device)
                    rows.append({
                        "experiment_version": EXPERIMENT_VERSION,
                        "seed": int(seed),
                        "regime": regime,
                        "variant": variant,
                        "tau_ca": float(tau),
                        "n_samples": int(n_samples),
                        **transmission_metrics(release, batch, tau),
                    })
    return pd.DataFrame(rows)


class _BoundarySpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (distance,) = ctx.saved_tensors
        return grad_output / (1.0 + math.pi * math.pi * distance.square())


class _BoundaryLIF(nn.Module):
    """Small deterministic LIF cell used identically with and without SpikingJelly."""

    def __init__(self, tau: float, threshold: float = 0.2):
        super().__init__()
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.voltage = None

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        if self.voltage is None or self.voltage.shape != current.shape:
            self.voltage = torch.zeros_like(current)
        self.voltage = self.voltage + (current - self.voltage) / self.tau
        spike = _BoundarySpike.apply(self.voltage - self.threshold)
        self.voltage = self.voltage * (1.0 - spike.detach())
        return spike

    def reset(self) -> None:
        self.voltage = None


class ContextSelectionSNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        variant: str,
        tau_m: float,
        tau_ca: float,
        input_gain: float = 5.0,
        hidden_gain: float = 3.0,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.input_gain = float(input_gain)
        self.hidden_gain = float(hidden_gain)
        self.input_cds = None
        if variant != "static":
            self.input_cds = CDSLayer(
                input_dim,
                param_dims=2,
                tau_ca_init=tau_ca,
                ca_influx_init=0.5,
                k_ca_init=1.0,
                u0_init=0.1,
                u_max_init=1.0,
                learnable=True,
                variant=variant,
            )
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.lif1 = _BoundaryLIF(tau=tau_m, threshold=0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.lif2 = _BoundaryLIF(tau=tau_m, threshold=0.2)
        self.readout = nn.Linear(hidden_dim, 2, bias=True)

    def forward(self, x: torch.Tensor, return_release: bool = False):
        # Input convention is batch x time x features.
        logits, releases = [], []
        for time in range(x.shape[1]):
            current = x[:, time]
            if self.input_cds is not None:
                current = self.input_cds(current)
                if return_release:
                    releases.append(self.input_cds.release_prob)
            elif return_release:
                releases.append(torch.ones_like(current))
            hidden = self.lif1(self.fc1(current * self.input_gain))
            hidden = self.lif2(self.fc2(hidden) * self.hidden_gain)
            logits.append(self.readout(hidden))
        logit_seq = torch.stack(logits, dim=1)
        if return_release:
            return logit_seq, torch.stack(releases, dim=1)
        return logit_seq


def _make_loader(
    batch: BoundaryBatch,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*batch.tensors()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def remove_distractor_events(batch: BoundaryBatch) -> BoundaryBatch:
    inputs = batch.inputs.copy()
    inputs[batch.distractor_mask] = 0.0
    return BoundaryBatch(
        inputs=inputs,
        labels=batch.labels.copy(),
        distractor_labels=batch.distractor_labels.copy(),
        cue_bank=batch.cue_bank.copy(),
        relevant_mask=batch.relevant_mask.copy(),
        distractor_mask=batch.distractor_mask.copy(),
        regime="target_only",
    )


@torch.no_grad()
def evaluate_context_model(
    model: ContextSelectionSNN,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = correct = switched_correct = changed = 0
    relevant_release: List[np.ndarray] = []
    distractor_release: List[np.ndarray] = []
    for x, label, distractor_label, _cue_bank, relevant, distractor in loader:
        x = x.to(device)
        label = label.to(device)
        distractor_label = distractor_label.to(device)
        logits, release = model(x, return_release=True)
        prediction = logits.mean(dim=1).argmax(dim=1)
        total += int(label.numel())
        correct += int((prediction == label).sum().cpu())

        rel = relevant.to(device)
        dis = distractor.to(device)
        relevant_release.append(release[rel].detach().cpu().numpy())
        distractor_release.append(release[dis].detach().cpu().numpy())
        reset_net(model)

        swapped = x.clone()
        swapped[:, :, [0, 1]] = swapped[:, :, [1, 0]]
        swapped_prediction = model(swapped).mean(dim=1).argmax(dim=1)
        switched_correct += int((swapped_prediction == distractor_label).sum().cpu())
        changed += int((swapped_prediction != prediction).sum().cpu())
        reset_net(model)

    positive = np.concatenate(relevant_release)
    negative = np.concatenate(distractor_release)
    target_mean = float(positive.mean())
    distractor_mean = float(negative.mean())
    return {
        "accuracy": correct / max(total, 1),
        "cue_swap_accuracy": switched_correct / max(total, 1),
        "cue_swap_prediction_change": changed / max(total, 1),
        "relevant_release": target_mean,
        "distractor_release": distractor_mean,
        "release_contrast": target_mean - distractor_mean,
        "release_ratio": target_mean / max(distractor_mean, 1e-12),
        "relevance_auc": binary_auc(positive, negative),
    }


def fit_context_model(
    train_batch: BoundaryBatch,
    validation_batch: BoundaryBatch,
    variant: str,
    seed: int,
    device: torch.device,
    hidden_dim: int,
    tau_m: float,
    tau_ca: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> Tuple[ContextSelectionSNN, Dict[str, float]]:
    set_seed(seed)
    model = ContextSelectionSNN(
        input_dim=train_batch.inputs.shape[-1],
        hidden_dim=hidden_dim,
        variant=variant,
        tau_m=tau_m,
        tau_ca=tau_ca,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = _make_loader(train_batch, batch_size, True, seed + 31)
    validation_loader = _make_loader(validation_batch, batch_size, False, seed + 37)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, label, *_ in train_loader:
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).mean(dim=1)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            reset_net(model)
        validation = evaluate_context_model(model, validation_loader, device)
        if validation["accuracy"] > best_validation:
            best_validation = validation["accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    learned = {}
    if model.input_cds is not None:
        with torch.no_grad():
            learned = {
                "tau_ca_eff_mean": float(F.softplus(model.input_cds.tau_ca).mean().cpu()),
                "ca_influx_eff_mean": float(F.softplus(model.input_cds.ca_influx).mean().cpu()),
                "k_ca_eff_mean": float(F.softplus(model.input_cds.k_ca).mean().cpu()),
                "u0_eff_mean": float(torch.sigmoid(model.input_cds.u0).mean().cpu()),
            }
    fit_metrics = {
        "best_epoch": int(best_epoch),
        "validation_accuracy": float(best_validation),
        **learned,
    }
    return model, fit_metrics


def run_network_experiment(
    seeds: Sequence[int],
    regimes: Sequence[str],
    variants: Sequence[str],
    spec: BoundarySpec,
    device: torch.device,
    train_samples: int,
    validation_samples: int,
    test_samples: int,
    hidden_dim: int,
    tau_m: float,
    tau_ca: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for seed in seeds:
        base_seed = int(seed) * 1009
        # Training contains one cued candidate and no distractor. Target history is mixed
        # between bursts and isolated events, preventing either history type from becoming
        # a training shortcut. The three target-distractor relations are test-only shifts.
        train_batch = remove_distractor_events(
            generate_boundary_batch(train_samples, "matched", base_seed + 1, spec)
        )
        validation_batch = remove_distractor_events(
            generate_boundary_batch(validation_samples, "matched", base_seed + 2, spec)
        )
        test_batches = {
            regime: generate_boundary_batch(
                test_samples, regime, base_seed + 100_003 * (index + 1), spec
            )
            for index, regime in enumerate(regimes)
        }
        for variant in variants:
            model, fit_metrics = fit_context_model(
                train_batch,
                validation_batch,
                variant,
                seed,
                device,
                hidden_dim,
                tau_m,
                tau_ca,
                batch_size,
                epochs,
                learning_rate,
            )
            for regime, test_batch in test_batches.items():
                test_loader = _make_loader(test_batch, batch_size, False, seed + 41)
                metrics = evaluate_context_model(model, test_loader, device)
                rows.append({
                    "experiment_version": EXPERIMENT_VERSION,
                    "seed": int(seed),
                    "regime": regime,
                    "variant": variant,
                    "training_condition": "target_only_mixed_history",
                    "train_samples": int(train_samples),
                    "validation_samples": int(validation_samples),
                    "test_samples": int(test_samples),
                    "epochs": int(epochs),
                    **fit_metrics,
                    **metrics,
                })
                print(
                    f"[network] seed={seed} regime={regime:12s} variant={variant:16s} "
                    f"accuracy={metrics['accuracy']:.4f} auc={metrics['relevance_auc']:.4f}"
                )
    return pd.DataFrame(rows)


def summarize_runs(
    frame: pd.DataFrame,
    value_columns: Sequence[str],
    group_columns: Sequence[str] = ("regime", "variant"),
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [column for column in value_columns if column in frame.columns]
    grouped = frame.groupby(list(group_columns), as_index=False)[columns].agg(["mean", "std"])
    grouped.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped.reset_index(drop=True)


def add_paired_deltas(network: pd.DataFrame) -> pd.DataFrame:
    if network.empty or not {"static", "full"}.issubset(set(network["variant"])):
        return pd.DataFrame()
    pivot = network.pivot_table(index=["seed", "regime"], columns="variant", values="accuracy")
    pivot = pivot.dropna(subset=["static", "full"]).reset_index()
    pivot["delta_full_minus_static"] = pivot["full"] - pivot["static"]
    return pivot


def plot_audit(audit: pd.DataFrame, out_path: Path) -> None:
    full = audit[audit["variant"] == "full"]
    if full.empty:
        return
    summary = full.groupby(["regime", "tau_ca"], as_index=False)["release_contrast"].agg(["mean", "std"]).reset_index()
    colors = {"aligned": "#27864B", "matched": "#3A78C2", "anti_aligned": "#C94B40"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for regime in REGIMES:
        current = summary[summary["regime"] == regime]
        if current.empty:
            continue
        ax.errorbar(
            current["tau_ca"],
            current["mean"],
            yerr=current["std"].fillna(0.0),
            marker="o",
            linewidth=2,
            capsize=3,
            label=regime.replace("_", " "),
            color=colors[regime],
        )
    ax.axhline(0.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Calcium time constant")
    ax.set_ylabel("Relevant minus distractor release")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_network(network: pd.DataFrame, out_path: Path) -> None:
    if network.empty:
        return
    summary = network.groupby(["regime", "variant"], as_index=False)["accuracy"].agg(["mean", "std"]).reset_index()
    regimes = [item for item in REGIMES if item in set(summary["regime"])]
    variants = [item for item in VARIANTS if item in set(summary["variant"])]
    x = np.arange(len(regimes), dtype=float)
    width = 0.8 / max(len(variants), 1)
    colors = {
        "static": "#777777",
        "constant_release": "#5B8E7D",
        "memory_free": "#D6A84B",
        "full": "#8C4B9E",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, variant in enumerate(variants):
        means, stds = [], []
        for regime in regimes:
            row = summary[(summary["regime"] == regime) & (summary["variant"] == variant)]
            means.append(float(row["mean"].iloc[0]))
            stds.append(float(row["std"].fillna(0.0).iloc[0]))
        offset = (index - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=3, label=variant.replace("_", " "), color=colors[variant])
    ax.set_xticks(x, [item.replace("_", " ") for item in regimes])
    ax.set_ylim(0.45, 1.01)
    ax.set_ylabel("Test accuracy")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_decision_report(audit: pd.DataFrame, network: pd.DataFrame, tau: float) -> Dict:
    report: Dict = {"experiment_version": EXPERIMENT_VERSION, "primary_tau_ca": float(tau)}
    primary = audit[(audit["variant"] == "full") & np.isclose(audit["tau_ca"], tau)]
    audit_means = primary.groupby("regime")["release_contrast"].mean().to_dict()
    audit_supported = (
        audit_means.get("aligned", 0.0) > 0.0
        and abs(audit_means.get("matched", float("inf"))) < 1e-7
        and audit_means.get("anti_aligned", 0.0) < 0.0
    )
    report["transmission_audit"] = {
        "mean_release_contrast": {key: float(value) for key, value in audit_means.items()},
        "predicted_ordering_observed": bool(audit_supported),
    }
    deltas = add_paired_deltas(network)
    if not deltas.empty:
        means = deltas.groupby("regime")["delta_full_minus_static"].mean().to_dict()
        monotonic = (
            means.get("aligned", -float("inf"))
            > means.get("matched", -float("inf"))
            > means.get("anti_aligned", -float("inf"))
        )
        report["network_test"] = {
            "mean_accuracy_delta_full_minus_static": {key: float(value) for key, value in means.items()},
            "aligned_to_anti_aligned_ordering_observed": bool(monotonic),
            "paired_seed_rows": deltas.to_dict(orient="records"),
        }
        if audit_supported and monotonic:
            recommendation = "mechanism_boundary_and_network_consequence_supported"
        elif audit_supported:
            recommendation = "transmission_boundary_supported_but_network_consequence_not_established"
        else:
            recommendation = "core_transmission_prediction_not_supported"
    else:
        recommendation = (
            "transmission_boundary_supported_network_test_pending"
            if audit_supported
            else "core_transmission_prediction_not_supported"
        )
    report["decision"] = recommendation
    report["interpretation_rule"] = (
        "The network claim requires paired full-minus-static accuracy to decrease from "
        "aligned through matched to anti_aligned conditions. Transmission results alone "
        "support a synaptic bias, not a reliable-computation claim."
    )
    return report


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; using CPU instead of {requested}.")
        return torch.device("cpu")
    return torch.device(requested)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Controlled test of when input-local CDS history agrees with task relevance."
    )
    parser.add_argument("--mode", choices=["audit", "network", "all"], default="all")
    parser.add_argument("--out_dir", default="results/mechanism_boundary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--tau_values", default="1,2,4,8,16")
    parser.add_argument("--primary_tau", type=float, default=2.0)
    parser.add_argument("--audit_samples", type=int, default=1024)
    parser.add_argument("--train_samples", type=int, default=2048)
    parser.add_argument("--validation_samples", type=int, default=512)
    parser.add_argument("--test_samples", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--tau_m", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true", help="Small execution check; not scientific evidence.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = parse_int_list(args.seeds)
    regimes = parse_choice_list(args.regimes, REGIMES)
    variants = parse_choice_list(args.variants, VARIANTS)
    tau_values = parse_float_list(args.tau_values)
    if not any(math.isclose(args.primary_tau, value) for value in tau_values):
        raise ValueError("primary_tau must be included in tau_values.")
    if args.quick:
        seeds = seeds[:1]
        args.audit_samples = min(args.audit_samples, 32)
        args.train_samples = min(args.train_samples, 64)
        args.validation_samples = min(args.validation_samples, 32)
        args.test_samples = min(args.test_samples, 32)
        args.epochs = min(args.epochs, 2)
        args.hidden_dim = min(args.hidden_dim, 16)
    device = resolve_device(args.device)
    spec = BoundarySpec()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "arguments": vars(args),
            "executed_seeds": seeds,
            "regimes": regimes,
            "variants": variants,
            "tau_values": tau_values,
            "device": str(device),
            "boundary_spec": asdict(spec),
            "hypotheses": {
                "aligned": "CDS release favors task-relevant events.",
                "matched": "Input-local CDS cannot identify relevance when local histories are exactly matched.",
                "anti_aligned": "CDS release favors locally repeated distractor events.",
            },
        },
        out_dir / "experiment_config.json",
    )

    audit = pd.DataFrame()
    if args.mode in {"audit", "all"}:
        audit = run_transmission_audit(
            seeds,
            regimes,
            variants,
            tau_values,
            args.audit_samples,
            spec,
            device,
        )
        audit.to_csv(out_dir / "transmission_audit.csv", index=False)
        summarize_runs(
            audit,
            ["release_contrast", "release_ratio", "relevance_auc", "prior_trace_contrast"],
            group_columns=("regime", "variant", "tau_ca"),
        ).to_csv(out_dir / "transmission_summary.csv", index=False)
        plot_audit(audit, out_dir / "transmission_boundary.png")

    network = pd.DataFrame()
    if args.mode in {"network", "all"}:
        network = run_network_experiment(
            seeds,
            regimes,
            variants,
            spec,
            device,
            args.train_samples,
            args.validation_samples,
            args.test_samples,
            args.hidden_dim,
            args.tau_m,
            args.primary_tau,
            args.batch_size,
            args.epochs,
            args.learning_rate,
        )
        network.to_csv(out_dir / "network_runs.csv", index=False)
        summarize_runs(
            network,
            ["accuracy", "cue_swap_accuracy", "cue_swap_prediction_change", "release_contrast", "relevance_auc"],
        ).to_csv(out_dir / "network_summary.csv", index=False)
        add_paired_deltas(network).to_csv(out_dir / "paired_accuracy_deltas.csv", index=False)
        plot_network(network, out_dir / "network_boundary.png")

    if audit.empty and (out_dir / "transmission_audit.csv").is_file():
        audit = pd.read_csv(out_dir / "transmission_audit.csv")
    if network.empty and (out_dir / "network_runs.csv").is_file():
        network = pd.read_csv(out_dir / "network_runs.csv")
    report = make_decision_report(audit, network, args.primary_tau) if not audit.empty else {
        "decision": "transmission_audit_required_before_interpretation"
    }
    save_json(report, out_dir / "decision_report.json")
    print(f"Saved controlled mechanism experiment to {out_dir}")
    print(f"Decision status: {report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
