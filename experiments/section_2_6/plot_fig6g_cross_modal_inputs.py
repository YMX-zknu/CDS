from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm, to_rgb
import numpy as np


torch = None


EXPECTED_PROTOCOL = "amplitude_matched_maximum_guard_v4"


EVENT_ON = "#FF3B30"
EVENT_OFF = "#168BFF"
SHD_EVENT = "#168BFF"
PERTURBATION = "#35F06F"
BACKGROUND = "#030507"
FOREGROUND = "#F2F4F7"
GRID_COLOR = "#3B424C"


AXIS_FONT_SCALE = 4.0
AXIS_LINE_SCALE = 2.0


@dataclass
class DatasetSpec:
    key: str
    display_name: str
    module_name: str
    build_name: str
    parse_name: str
    config_name: str
    root: str
    expected_shape: str
    batch_size: int
    explicit_index: int


@dataclass
class PreparedSample:
    dataset: str
    display_name: str
    sample_index: int
    label: int
    original: object
    perturbed: object
    added_mask: object
    supported_mask: object
    n_original: int
    n_target: int
    n_added: int
    n_available: int
    perturbation_seed: int
    batch_index: int
    batch_local_index: int
    selection_rule: str

    @property
    def realized_ratio(self) -> float:
        return self.n_added / max(self.n_original, 1)

    @property
    def realized_fraction(self) -> float:
        return self.n_added / self.n_target if self.n_target else 1.0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize DVS-Gesture, SHD, and ST-MNIST inputs with the exact "
            "Section 2.6 amplitude-matched isolated perturbation."
        )
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the three Section 2.6 training scripts.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/section_2_6/input_visualizations"),
        help="Directory in which the twelve TIFF files are written.",
    )
    parser.add_argument(
        "--datasets",
        default="dvsgesture,shd,stmnist",
        help="Comma-separated subset of dvsgesture,shd,stmnist.",
    )
    parser.add_argument(
        "--dvsgesture_root",
        default="data/DVS-Gesture",
    )
    parser.add_argument("--shd_root", default="data/SHD")
    parser.add_argument("--stmnist_root", default="data/ST-MNIST")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Added isolated-bin ratio; the manuscript uses 0.10 in Fig. 6g.",
    )
    parser.add_argument(
        "--guard_steps",
        type=int,
        default=4,
        help="Symmetric same-feature isolation window in time steps.",
    )
    parser.add_argument(
        "--training_seed",
        type=int,
        default=2026,
        help="Training-run seed used by the experiment's perturbation-seed formula.",
    )
    parser.add_argument(
        "--perturbation_repeat",
        type=int,
        default=0,
        help="Existing perturbation realization to reproduce (0, 1, or 2).",
    )
    parser.add_argument(
        "--minimum_realized_fraction",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--dvsgesture_sample_index",
        type=int,
        default=-1,
        help="Test index; -1 selects an objective median-occupancy example.",
    )
    parser.add_argument(
        "--shd_sample_index",
        type=int,
        default=-1,
        help="Test index; -1 selects an objective median-occupancy example.",
    )
    parser.add_argument(
        "--stmnist_sample_index",
        type=int,
        default=-1,
        help="Test index; -1 selects an objective median-occupancy example.",
    )
    parser.add_argument(
        "--selection_limit",
        type=int,
        default=0,
        help=(
            "Maximum evenly spaced test samples used for median selection; "
            "0 uses the complete test split."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--force_reprocess_stmnist",
        action="store_true",
        help="Rebuild the maintained ST-MNIST frame cache.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use built-in deterministic arrays to test plotting and TIFF export.",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.ratio <= 1.0:
        parser.error("--ratio must lie in (0, 1].")
    if args.guard_steps < 1:
        parser.error("--guard_steps must be at least 1.")
    if args.perturbation_repeat < 0:
        parser.error("--perturbation_repeat must be non-negative.")
    if not 0.0 < args.minimum_realized_fraction <= 1.0:
        parser.error("--minimum_realized_fraction must lie in (0, 1].")
    if args.dpi < 150:
        parser.error("--dpi must be at least 150 for a publication figure.")
    return args


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Arial", "DejaVu Serif", "Times New Roman"],
            "font.size": 9,
            "axes.labelsize": 10 * AXIS_FONT_SCALE,
            "xtick.labelsize": 8 * AXIS_FONT_SCALE,
            "ytick.labelsize": 8 * AXIS_FONT_SCALE,
            "axes.linewidth": 0.8 * AXIS_LINE_SCALE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": FOREGROUND,
            "axes.labelcolor": FOREGROUND,
            "axes.edgecolor": FOREGROUND,
            "xtick.color": FOREGROUND,
            "ytick.color": FOREGROUND,
            "savefig.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
        }
    )


def import_section26_modules(source_dir: Path) -> Dict[str, object]:
    source_dir = source_dir.expanduser().resolve()
    required = {
        "dvsgesture": "section_2_6_dvsgesture_train.py",
        "shd": "section_2_6_shd_train.py",
        "stmnist": "section_2_6_stmnist_train.py",
    }
    missing = [name for name in required.values() if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing companion Section 2.6 file(s) in "
            f"{source_dir}: {', '.join(missing)}. Use --source_dir to point to "
            "the directory containing the supplied training scripts."
        )
    sys.path.insert(0, str(source_dir))
    modules = {
        key: importlib.import_module(Path(filename).stem)
        for key, filename in required.items()
    }
    protocol = getattr(
        modules["dvsgesture"], "ISOLATED_PERTURBATION_PROTOCOL_VERSION", None
    )
    if protocol != EXPECTED_PROTOCOL:
        raise RuntimeError(
            f"Expected perturbation protocol {EXPECTED_PROTOCOL!r}, found "
            f"{protocol!r}. Use the Section 2.6 code supplied with this script."
        )
    return modules


def selected_dataset_specs(args: argparse.Namespace) -> List[DatasetSpec]:
    requested = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    valid = {"dvsgesture", "shd", "stmnist"}
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}; choose from {sorted(valid)}")
    specifications = {
        "dvsgesture": DatasetSpec(
            "dvsgesture",
            "DVS-Gesture",
            "section_2_6_dvsgesture_train",
            "build_dvsgesture_datasets",
            "parse_args",
            "config_from_args",
            args.dvsgesture_root,
            "[T=10, C=2, H=128, W=128]",
            16,
            args.dvsgesture_sample_index,
        ),
        "shd": DatasetSpec(
            "shd",
            "SHD",
            "section_2_6_shd_train",
            "build_shd_datasets",
            "parse_args",
            "config_from_args",
            args.shd_root,
            "[T=100, F=700]",
            128,
            args.shd_sample_index,
        ),
        "stmnist": DatasetSpec(
            "stmnist",
            "ST-MNIST",
            "section_2_6_stmnist_train",
            "build_stmnist_datasets",
            "parse_args",
            "config_from_args",
            args.stmnist_root,
            "[T=30, C=2, H=10, W=10]",
            16,
            args.stmnist_sample_index,
        ),
    }
    return [specifications[key] for key in requested]


def build_test_dataset(
    spec: DatasetSpec,
    module: object,
    args: argparse.Namespace,
):
    namespace = getattr(module, spec.parse_name)([])
    namespace.data_root = spec.root
    namespace.dry_run = bool(args.dry_run)
    namespace.seed = int(args.training_seed)
    namespace.isolation_guard_steps = int(args.guard_steps)
    namespace.selectivity_ratio = float(args.ratio)
    namespace.minimum_realized_fraction = float(args.minimum_realized_fraction)
    if spec.key == "dvsgesture":
        namespace.T = 10
        namespace.delta_t = 125
    elif spec.key == "shd":
        namespace.T = 100
        namespace.split_by = "time"
        namespace.normalize = "max"
    elif spec.key == "stmnist":
        namespace.T = 30
        namespace.in_channels = 2
        namespace.sensor_h = 10
        namespace.sensor_w = 10
        namespace.normalize = "max"
        namespace.time_window_ms = 0.0
        namespace.force_reprocess = bool(args.force_reprocess_stmnist)


    torch.manual_seed(int(args.training_seed) + {"dvsgesture": 0, "shd": 1, "stmnist": 2}[spec.key])
    config = getattr(module, spec.config_name)(namespace)
    train_set, test_set = getattr(module, spec.build_name)(config)
    if len(test_set) == 0:
        raise RuntimeError(f"{spec.display_name} test split is empty.")
    return config, test_set


def unpack_dataset_item(item) -> Tuple[torch.Tensor, int]:
    if not isinstance(item, (tuple, list)) or len(item) < 2:
        raise TypeError("Each dataset item must contain at least (input, label).")
    x = torch.as_tensor(item[0]).detach().cpu().float()
    y = int(torch.as_tensor(item[1]).reshape(-1)[0].item())
    return x, y


def standardize_sample_shape(x: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset in {"dvsgesture", "stmnist"}:
        if x.ndim != 4:
            raise ValueError(f"{dataset} sample must be rank 4, found {tuple(x.shape)}")
        if x.shape[1] == 2:
            out = x
        elif x.shape[-1] == 2:
            out = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"Cannot identify ON/OFF channel axis in {dataset} shape {tuple(x.shape)}"
            )
    elif dataset == "shd":
        if x.ndim != 2:
            raise ValueError(f"SHD sample must be rank 2, found {tuple(x.shape)}")
        out = x if x.shape[-1] == 700 else x.transpose(0, 1).contiguous()
        if out.shape[-1] != 700:
            raise ValueError(f"Expected 700 SHD channels, found {tuple(x.shape)}")
    else:
        raise ValueError(dataset)
    if torch.any(~torch.isfinite(out)):
        raise ValueError(f"{dataset} sample contains non-finite values.")
    if torch.any(out < 0):
        raise ValueError(
            f"{dataset} processed tensor contains negative amplitudes; expected "
            "separate non-negative polarity channels."
        )
    return out


def candidate_indices(length: int, limit: int) -> List[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, int(limit), dtype=int).tolist()))


def occupancy_order(dataset, dataset_key: str, explicit: int, limit: int) -> Tuple[List[int], str]:
    if explicit >= 0:
        if explicit >= len(dataset):
            raise IndexError(
                f"Requested {dataset_key} sample index {explicit}, but test split "
                f"contains {len(dataset)} samples."
            )
        return [explicit], f"explicit test index {explicit}"

    indices = candidate_indices(len(dataset), limit)
    counts: List[int] = []
    for index in indices:
        x, _ = unpack_dataset_item(dataset[index])
        x = standardize_sample_shape(x, dataset_key)
        counts.append(int(torch.count_nonzero(x).item()))
    median = float(np.median(np.asarray(counts, dtype=np.float64)))
    ordered = [
        index
        for _, index in sorted(
            zip((abs(count - median) for count in counts), indices),
            key=lambda pair: (pair[0], pair[1]),
        )
    ]
    scope = "complete test split" if len(indices) == len(dataset) else f"{len(indices)} evenly spaced test samples"
    return ordered, f"occupied-bin count closest to the median over the {scope}"


def stack_evaluation_batch(
    dataset,
    dataset_key: str,
    batch_index: int,
    batch_size: int,
) -> Tuple["torch.Tensor", List[int]]:
    start = int(batch_index) * int(batch_size)
    stop = min(start + int(batch_size), len(dataset))
    samples: List[torch.Tensor] = []
    labels: List[int] = []
    for index in range(start, stop):
        x, label = unpack_dataset_item(dataset[index])
        samples.append(standardize_sample_shape(x, dataset_key))
        labels.append(label)
    if not samples:
        raise RuntimeError("Resolved an empty evaluation batch.")
    return torch.stack(samples, dim=0), labels


def experiment_perturbation_seed(training_seed: int, ratio: float, repeat: int, batch_index: int) -> int:
    base = int(training_seed) * 100003 + round(float(ratio) * 10000) * 101 + int(repeat)
    return int(base + 1000003 * int(batch_index))


def prepare_sample(
    spec: DatasetSpec,
    dataset,
    dvs_module: object,
    args: argparse.Namespace,
) -> PreparedSample:
    ordered_indices, selection_rule = occupancy_order(
        dataset,
        spec.key,
        spec.explicit_index,
        int(args.selection_limit),
    )
    perturb_fn = getattr(dvs_module, "add_amplitude_matched_isolated_perturbations")
    to_time_first = getattr(dvs_module, "to_time_first")
    failures: List[str] = []

    for sample_index in ordered_indices:
        batch_index = int(sample_index) // int(spec.batch_size)
        local_index = int(sample_index) % int(spec.batch_size)
        raw_batch, labels = stack_evaluation_batch(
            dataset,
            spec.key,
            batch_index,
            spec.batch_size,
        )
        expected_channels = 700 if spec.key == "shd" else 2
        x_seq = to_time_first(raw_batch, torch.device("cpu"), expected_channels)
        seed = experiment_perturbation_seed(
            args.training_seed,
            args.ratio,
            args.perturbation_repeat,
            batch_index,
        )
        batch = perturb_fn(
            x_seq,
            float(args.ratio),
            seed=seed,
            guard_steps=int(args.guard_steps),
            minimum_realized_fraction=float(args.minimum_realized_fraction),
        )
        original = x_seq[:, local_index].detach().cpu()
        perturbed = batch.perturbed[:, local_index].detach().cpu()
        added = batch.added_mask[:, local_index].detach().cpu().bool()
        supported = batch.supported_original_mask[:, local_index].detach().cpu().bool()
        n_original = int(torch.count_nonzero(original).item())
        n_target = int(round(n_original * float(args.ratio)))
        n_added = int(added.sum().item())
        fraction = n_added / n_target if n_target else 1.0
        if n_target > 0 and fraction + 1e-12 < float(args.minimum_realized_fraction):
            failures.append(
                f"index={sample_index}, added={n_added}, target={n_target}, fraction={fraction:.3f}"
            )
            if spec.explicit_index >= 0:
                break
            continue
        return PreparedSample(
            dataset=spec.key,
            display_name=spec.display_name,
            sample_index=int(sample_index),
            label=int(labels[local_index]),
            original=original,
            perturbed=perturbed,
            added_mask=added,
            supported_mask=supported,
            n_original=n_original,
            n_target=n_target,
            n_added=n_added,
            n_available=int(batch.n_available),
            perturbation_seed=seed,
            batch_index=batch_index,
            batch_local_index=local_index,
            selection_rule=selection_rule,
        )

    detail = failures[:5]
    raise RuntimeError(
        f"Could not find a {spec.display_name} sample meeting the required "
        f"realized fraction {args.minimum_realized_fraction:.3f}. Examples: {detail}"
    )


def scaled_marker_sizes(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    maximum = float(np.nanmax(values))
    if not np.isfinite(maximum) or maximum <= 0:
        return np.full(values.shape, lower, dtype=np.float64)
    normalized = np.sqrt(np.clip(values / maximum, 0.0, 1.0))
    return lower + (upper - lower) * normalized


def as_numpy(value: object) -> np.ndarray:

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def without_added_perturbations(sample: PreparedSample) -> PreparedSample:

    empty_added_mask = np.zeros_like(as_numpy(sample.added_mask), dtype=bool)
    return replace(
        sample,
        perturbed=sample.original,
        added_mask=empty_added_mask,
        n_target=0,
        n_added=0,
    )


def style_2d_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(FOREGROUND)
    ax.spines["bottom"].set_color(FOREGROUND)
    ax.spines["left"].set_linewidth(0.8 * AXIS_LINE_SCALE)
    ax.spines["bottom"].set_linewidth(0.8 * AXIS_LINE_SCALE)
    ax.tick_params(
        colors=FOREGROUND,
        width=0.8 * AXIS_LINE_SCALE,
        length=3 * AXIS_LINE_SCALE,
    )
    ax.xaxis.label.set_color(FOREGROUND)
    ax.yaxis.label.set_color(FOREGROUND)


def save_tiff(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="tiff",
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=BACKGROUND,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"TIFF export failed: {path}")


def plot_dvsgesture(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Unexpected DVS-Gesture shape {x.shape}")
    t_count, _, height, width = x.shape
    fig, ax = plt.subplots(figsize=(13.0, 3.6), facecolor=BACKGROUND)


    depth_x = 0.34
    depth_y = 0.14
    time_left = -0.5
    time_right = t_count - 0.5

    def project(time, x_pixel, y_pixel):
        time = np.asarray(time, dtype=np.float64)
        x_norm = np.asarray(x_pixel, dtype=np.float64) / max(width - 1, 1)
        y_norm = np.asarray(y_pixel, dtype=np.float64) / max(height - 1, 1)
        screen_x = time + depth_x * x_norm
        screen_y = 1.0 - y_norm + depth_y * x_norm
        return screen_x, screen_y


    for time in (time_left, time_right):
        corners_x, corners_y = project(
            [time, time, time, time, time],
            [0, width - 1, width - 1, 0, 0],
            [height - 1, height - 1, 0, 0, height - 1],
        )
        ax.plot(corners_x, corners_y, color=GRID_COLOR, lw=0.42, alpha=0.55, zorder=0)
    for x_pixel, y_pixel in (
        (0, height - 1),
        (width - 1, height - 1),
        (width - 1, 0),
        (0, 0),
    ):
        line_x, line_y = project(
            [time_left, time_right],
            [x_pixel, x_pixel],
            [y_pixel, y_pixel],
        )
        ax.plot(line_x, line_y, color=GRID_COLOR, lw=0.52, alpha=0.72, zorder=0)

    def add_continuous_event_cloud(mask, channel, color, alpha):

        time, yy, xx = np.nonzero(mask)
        if time.size == 0:
            return


        key = (
            np.asarray(xx, dtype=np.uint64) * np.uint64(73856093)
            ^ np.asarray(yy, dtype=np.uint64) * np.uint64(19349663)
            ^ np.asarray(time, dtype=np.uint64) * np.uint64(83492791)
            ^ np.uint64(channel + 1) * np.uint64(2654435761)
        )
        fraction = (key & np.uint64(0xFFFFFFFF)).astype(np.float64) / float(2**32)
        display_time = time.astype(np.float64) - 0.5 + fraction
        screen_x, screen_y = project(display_time, xx, yy)
        ax.scatter(
            screen_x,
            screen_y,
            s=5.0,
            c=color,
            marker="o",
            alpha=alpha,
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    for channel, color in [(0, EVENT_ON), (1, EVENT_OFF)]:
        add_continuous_event_cloud(x[:, channel] > 0, channel, color, 0.88)
    for channel in (0, 1):
        add_continuous_event_cloud(added[:, channel], channel, PERTURBATION, 1.0)

    ax.set_xlabel("Time step", labelpad=5)
    ax.set_ylabel("y pixel", labelpad=5)
    ax.set_xlim(time_left - 0.12, time_right + depth_x + 0.12)
    ax.set_ylim(-0.04, 1.0 + depth_y + 0.04)
    ax.set_xticks([0, 3, 6, t_count - 1])
    y_ticks = np.linspace(0.0, 1.0, 3)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(value) for value in (127, 64, 0)])

    axis_x, axis_y = project(
        [time_right, time_right],
        [0, width - 1],
        [height - 1, height - 1],
    )
    ax.plot(
        axis_x,
        axis_y,
        color=FOREGROUND,
        linewidth=0.8 * AXIS_LINE_SCALE,
        solid_capstyle="butt",
        zorder=6,
        clip_on=False,
    )
    ax.annotate(
        "x pixel",
        xy=(axis_x[-1], axis_y[-1]),
        xytext=(8, 1),
        textcoords="offset points",
        color=FOREGROUND,
        fontsize=8 * AXIS_FONT_SCALE,
        rotation=22,
        va="bottom",
        clip_on=False,
    )
    ax.grid(False)
    style_2d_axis(ax)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.34, top=0.95)
    save_tiff(fig, output, dpi)


def plot_shd(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    if x.ndim != 2 or x.shape[1] != 700:
        raise ValueError(f"Unexpected SHD shape {x.shape}")
    t_count, channel_count = x.shape
    nonzero = x[x > 0]
    vmax = float(np.quantile(nonzero, 0.995)) if nonzero.size else 1.0
    vmax = max(vmax, np.finfo(np.float32).eps)
    blue_map = LinearSegmentedColormap.from_list(
        "shd_black_blue", [BACKGROUND, "#082A55", SHD_EVENT]
    )

    fig, ax = plt.subplots(figsize=(13.0, 3.6))
    ax.imshow(
        x.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, t_count - 0.5, -0.5, channel_count - 0.5),
        cmap=blue_map,
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax),
        rasterized=True,
    )
    overlay = np.ma.masked_where(~added.T, added.T.astype(float))
    ax.imshow(
        overlay,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, t_count - 0.5, -0.5, channel_count - 0.5),
        cmap=ListedColormap([PERTURBATION]),
        vmin=0.0,
        vmax=1.0,
        zorder=4,
        rasterized=True,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cochlear\nchannel")
    ax.set_xlim(-0.5, t_count - 0.5)
    ax.set_ylim(-0.5, channel_count - 0.5)
    ax.set_xticks([0, 25, 50, 75, t_count - 1])
    ax.set_yticks([0, channel_count // 2, channel_count - 1])
    style_2d_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.34, top=0.95)
    save_tiff(fig, output, dpi)


def plot_stmnist(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    if x.ndim != 4 or x.shape[1:] != (2, 10, 10):
        raise ValueError(f"Unexpected ST-MNIST shape {x.shape}")
    t_count, _, height, width = x.shape

    fig, ax = plt.subplots(figsize=(13.0, 3.6))
    for channel, color in [(0, EVENT_ON), (1, EVENT_OFF)]:
        time, yy, xx = np.nonzero(x[:, channel] > 0)
        taxel = yy * width + xx
        ax.scatter(
            time,
            taxel,
            s=11.0,
            c=color,
            marker="o",
            alpha=0.9,
            linewidths=0,
            rasterized=True,
        )

    for channel in (0, 1):
        time, yy, xx = np.nonzero(added[:, channel])
        taxel = yy * width + xx
        ax.scatter(
            time,
            taxel,
            s=11.0,
            c=PERTURBATION,
            marker="o",
            alpha=1.0,
            linewidths=0,
            zorder=5,
            rasterized=True,
        )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Taxel\nindex")
    ax.set_xlim(-0.5, t_count - 0.5)
    ax.set_ylim(-1, height * width)
    ax.set_xticks([0, 10, 20, t_count - 1])
    ax.set_yticks([0, height * width // 2, height * width - 1])
    ax.grid(True, color=GRID_COLOR, linewidth=0.45, alpha=0.55)
    style_2d_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.34, top=0.95)
    save_tiff(fig, output, dpi)


def selected_frame_indices(t_count: int, number: int = 10) -> np.ndarray:

    if t_count < number:
        raise ValueError(f"Need at least {number} time steps, found {t_count}.")
    indices = np.rint(np.linspace(0, t_count - 1, number)).astype(int)
    if np.unique(indices).size != number:
        raise RuntimeError("Frame-index selection produced duplicate time steps.")
    return indices


def polarity_frame_rgb(frame: np.ndarray, added: np.ndarray) -> np.ndarray:

    if frame.shape[0] != 2 or frame.shape != added.shape:
        raise ValueError(f"Expected matching [2,H,W] arrays, found {frame.shape} and {added.shape}")
    height, width = frame.shape[-2:]
    rgb = np.broadcast_to(np.asarray(to_rgb(BACKGROUND)), (height, width, 3)).copy()
    for channel, color in ((0, EVENT_ON), (1, EVENT_OFF)):
        values = np.clip(frame[channel], 0.0, None)
        maximum = float(values.max())
        if maximum <= 0:
            continue
        strength = np.clip(values / maximum, 0.0, 1.0)
        contribution = strength[..., None] * np.asarray(to_rgb(color))
        rgb = np.maximum(rgb, contribution)
    rgb[np.any(added, axis=0)] = np.asarray(to_rgb(PERTURBATION))
    return np.clip(rgb, 0.0, 1.0)


def format_frame_strip(fig: plt.Figure, axes: Sequence[plt.Axes], times: Sequence[int], ylabel: str) -> None:
    for column, (ax, time) in enumerate(zip(axes, times)):
        ax.set_facecolor(BACKGROUND)
        ax.set_xlabel(
            str(int(time)),
            labelpad=2,
            fontsize=7 * AXIS_FONT_SCALE,
        )
        ax.tick_params(
            colors=FOREGROUND,
            length=2 * AXIS_LINE_SCALE,
            width=0.6 * AXIS_LINE_SCALE,
            labelsize=6 * AXIS_FONT_SCALE,
        )
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(0.55 * AXIS_LINE_SCALE)
        if column == 0:
            ax.set_ylabel(
                ylabel,
                labelpad=3,
                fontsize=8 * AXIS_FONT_SCALE,
            )
        else:
            ax.set_yticklabels([])
    fig.supxlabel(
        "Time step",
        y=-0.16,
        fontsize=9 * AXIS_FONT_SCALE,
        color=FOREGROUND,
    )


def plot_dvsgesture_frames(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    times = selected_frame_indices(x.shape[0])
    fig, axes = plt.subplots(1, 10, figsize=(12.0, 2.5), facecolor=BACKGROUND)
    for ax, time in zip(axes, times):
        ax.imshow(
            polarity_frame_rgb(x[time], added[time]),
            origin="upper",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([0, x.shape[-2] - 1])
    format_frame_strip(fig, axes, times, "y pixel")
    fig.subplots_adjust(left=0.08, right=0.998, bottom=0.34, top=0.96, wspace=0.10)
    save_tiff(fig, output, dpi)


def shd_frame_rgb(values: np.ndarray, additions: np.ndarray, width: int = 12) -> np.ndarray:
    maximum = max(float(values.max()), np.finfo(np.float32).eps)
    strength = np.clip(values / maximum, 0.0, 1.0)
    base = np.asarray(to_rgb(BACKGROUND))
    event = np.asarray(to_rgb(SHD_EVENT))
    rgb = base[None, :] * (1.0 - strength[:, None]) + event[None, :] * strength[:, None]
    rgb[additions.astype(bool)] = np.asarray(to_rgb(PERTURBATION))
    return np.repeat(rgb[:, None, :], width, axis=1)


def plot_shd_frames(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    times = selected_frame_indices(x.shape[0])
    fig, axes = plt.subplots(1, 10, figsize=(12.0, 3.4), facecolor=BACKGROUND)
    for ax, time in zip(axes, times):
        ax.imshow(
            shd_frame_rgb(x[time], added[time]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(-0.5, 0.5, -0.5, x.shape[1] - 0.5),
        )
        ax.set_xticks([])
        ax.set_yticks([0, 350, 699])
    format_frame_strip(fig, axes, times, "Cochlear\nchannel")
    fig.subplots_adjust(left=0.12, right=0.998, bottom=0.30, top=0.96, wspace=0.10)
    save_tiff(fig, output, dpi)


def plot_stmnist_frames(sample: PreparedSample, output: Path, dpi: int) -> None:
    x = as_numpy(sample.original)
    added = as_numpy(sample.added_mask).astype(bool, copy=False)
    times = selected_frame_indices(x.shape[0])
    fig, axes = plt.subplots(1, 10, figsize=(12.0, 2.5), facecolor=BACKGROUND)
    for ax, time in zip(axes, times):
        ax.imshow(
            polarity_frame_rgb(x[time], added[time]),
            origin="upper",
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([0, x.shape[-2] - 1])
    format_frame_strip(fig, axes, times, "Taxel\nrow")
    fig.subplots_adjust(left=0.08, right=0.998, bottom=0.34, top=0.96, wspace=0.10)
    save_tiff(fig, output, dpi)


def print_metadata(
    sample: PreparedSample,
    stream_output: Path,
    frame_output: Path,
    clean_stream_output: Path,
    clean_frame_output: Path,
    args: argparse.Namespace,
) -> None:
    stream_titles = {
        "dvsgesture": "Spatiotemporal DVS-Gesture event stream with isolated additions",
        "shd": "Cochlear-channel representation of SHD with isolated additions",
        "stmnist": "ST-MNIST tactile event stream with isolated additions",
    }
    frame_titles = {
        "dvsgesture": "Processed DVS-Gesture frames with isolated additions",
        "shd": "Processed SHD input frames with isolated additions",
        "stmnist": "Processed ST-MNIST frames with isolated additions",
    }
    legends = {
        "dvsgesture": (
            f"ON events: red {EVENT_ON}; OFF events: blue {EVENT_OFF}; "
            f"added isolated inputs: green {PERTURBATION}."
        ),
        "shd": (
            f"Original normalized event-bin amplitude: blue {SHD_EVENT}; "
            f"added isolated inputs: green {PERTURBATION}."
        ),
        "stmnist": (
            f"ON events: red {EVENT_ON}; OFF events: blue {EVENT_OFF}; "
            f"added isolated inputs: green {PERTURBATION}."
        ),
    }
    clean_legends = {
        "dvsgesture": f"ON events: red {EVENT_ON}; OFF events: blue {EVENT_OFF}.",
        "shd": f"Original normalized event-bin amplitude: blue {SHD_EVENT}.",
        "stmnist": f"ON events: red {EVENT_ON}; OFF events: blue {EVENT_OFF}.",
    }
    axis_text = {
        "dvsgesture": "Stream axes: time step; x pixel; y pixel.",
        "shd": "Axes: time step; cochlear channel.",
        "stmnist": "Axes: time step; taxel index (row-major 0-99).",
    }
    print("\n" + "=" * 78)
    print(f"DATASET: {sample.display_name}")
    print(f"PERTURBED STREAM OUTPUT: {stream_output.resolve()}")
    print(f"PERTURBED FRAME OUTPUT: {frame_output.resolve()}")
    print(f"CLEAN STREAM OUTPUT: {clean_stream_output.resolve()}")
    print(f"CLEAN FRAME OUTPUT: {clean_frame_output.resolve()}")
    print(f"SUGGESTED PERTURBED STREAM TITLE (not embedded): {stream_titles[sample.dataset]}")
    print(f"SUGGESTED PERTURBED FRAME TITLE (not embedded): {frame_titles[sample.dataset]}")
    print(
        "SUGGESTED CLEAN STREAM TITLE (not embedded): "
        + stream_titles[sample.dataset].replace(" with isolated additions", " without added perturbations")
    )
    print(
        "SUGGESTED CLEAN FRAME TITLE (not embedded): "
        + frame_titles[sample.dataset].replace(" with isolated additions", " without added perturbations")
    )
    print(f"PERTURBED LEGEND TEXT (not embedded): {legends[sample.dataset]}")
    print(f"CLEAN LEGEND TEXT (not embedded): {clean_legends[sample.dataset]}")
    print(axis_text[sample.dataset])
    if sample.dataset == "dvsgesture":
        print(
            "DVS STREAM RENDERING NOTE (not embedded): Markers are distributed "
            "deterministically within their original unit-width processed time bins "
            "to present a continuous event stream. Bin assignments are unchanged, "
            "and original asynchronous event timestamps are not inferred."
        )
    print(
        "FRAME INDICES (not embedded): "
        + ", ".join(str(int(value)) for value in selected_frame_indices(as_numpy(sample.original).shape[0]))
        + "."
    )
    print(
        "SAMPLE NOTE (not embedded): "
        f"test index={sample.sample_index}, class label={sample.label}, "
        f"selection rule={sample.selection_rule}."
    )
    print(
        "PERTURBATION NOTE (not embedded): "
        f"requested ratio={args.ratio:.4f}, guard=+/-{args.guard_steps} time steps, "
        f"training seed={args.training_seed}, repeat={args.perturbation_repeat}, "
        f"evaluation batch={sample.batch_index}, local index={sample.batch_local_index}, "
        f"exact perturbation seed={sample.perturbation_seed}."
    )
    print(
        "COUNTS (not embedded): "
        f"original occupied bins={sample.n_original}, target additions={sample.n_target}, "
        f"realized additions={sample.n_added}, realized ratio={sample.realized_ratio:.6f}, "
        f"realized target fraction={sample.realized_fraction:.6f}."
    )
    print(
        "CAPTION NOTE (not embedded): The stream view displays all processed time "
        "steps, whereas the frame view displays ten time steps spanning the complete "
        "input duration. Perturbed and clean views use the same processed sample, "
        "viewpoint, axes and selected frame indices. "
        "Original occupied bins are unchanged. Added bins are amplitude-matched "
        "to the sample and isolated from original activity and from one another "
        "within the same feature stream."
    )


def synthetic_original(dataset: str, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    if dataset == "dvsgesture":
        x = np.zeros((10, 2, 128, 128), dtype=np.float32)
        for t in range(10):
            cx = 30 + 6 * t
            cy = 68 + int(round(14 * math.sin(2 * math.pi * t / 10)))
            for channel, shift in ((0, 0), (1, -5)):
                count = 170
                xx = np.clip(np.rint(rng.normal(cx + shift, 8, count)).astype(int), 0, 127)
                yy = np.clip(np.rint(rng.normal(cy, 15, count)).astype(int), 0, 127)
                x[t, channel, yy, xx] = rng.uniform(0.35, 1.0, count)
        return x
    if dataset == "shd":
        x = np.zeros((100, 700), dtype=np.float32)
        for t in range(100):
            centres = (
                110 + 55 * math.sin(2 * math.pi * t / 37),
                340 + 80 * math.sin(2 * math.pi * (t + 13) / 53),
                560 + 35 * math.sin(2 * math.pi * (t + 5) / 29),
            )
            for centre in centres:
                channels = np.clip(
                    np.rint(rng.normal(centre, 7, 8)).astype(int), 0, 699
                )
                x[t, channels] = np.maximum(
                    x[t, channels], rng.uniform(0.25, 1.0, channels.size)
                )
        return x
    if dataset == "stmnist":
        x = np.zeros((30, 2, 10, 10), dtype=np.float32)

        for t in range(30):
            row = int(round(2 + 5 * t / 29))
            col = int(round(2 + 5 * t / 29))
            for offset in range(-2, 3):
                rr = int(np.clip(row + offset, 0, 9))
                cc = int(np.clip(col - offset, 0, 9))
                x[t, 0, rr, col] = rng.uniform(0.45, 1.0)
                x[t, 1, row, cc] = rng.uniform(0.35, 0.9)
        return x
    raise ValueError(dataset)


def numpy_isolated_perturbation(
    original: np.ndarray,
    ratio: float,
    guard_steps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:

    rng = np.random.default_rng(seed)
    shape = original.shape
    occupied = original.reshape(shape[0], -1) > 0
    eligible = ~occupied.copy()
    for delta in range(-guard_steps, guard_steps + 1):
        if delta == 0:
            continue
        if delta > 0:
            eligible[delta:] &= ~occupied[:-delta]
        else:
            eligible[:delta] &= ~occupied[-delta:]
    candidates = np.argwhere(eligible)
    target = int(round(np.count_nonzero(occupied) * ratio))
    rng.shuffle(candidates)
    chosen: List[Tuple[int, int]] = []
    added_by_feature: Dict[int, List[int]] = {}
    for time, feature in candidates:
        if all(abs(int(time) - previous) > guard_steps for previous in added_by_feature.get(int(feature), [])):
            chosen.append((int(time), int(feature)))
            added_by_feature.setdefault(int(feature), []).append(int(time))
            if len(chosen) >= target:
                break
    added_flat = np.zeros_like(occupied, dtype=bool)
    values = original[original > 0]
    if not values.size:
        values = np.asarray([1.0], dtype=np.float32)
    perturbed_flat = original.reshape(shape[0], -1).copy()
    for time, feature in chosen:
        added_flat[time, feature] = True
        perturbed_flat[time, feature] = float(rng.choice(values))
    supported = np.zeros_like(occupied, dtype=bool)
    for delta in range(1, guard_steps + 1):
        supported[delta:] |= occupied[delta:] & occupied[:-delta]
    return (
        perturbed_flat.reshape(shape),
        added_flat.reshape(shape),
        supported.reshape(shape),
        int(candidates.shape[0]),
    )


def prepare_synthetic_sample(
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> PreparedSample:
    original = synthetic_original(
        spec.key,
        int(args.training_seed) + {"dvsgesture": 0, "shd": 1, "stmnist": 2}[spec.key],
    )
    seed = experiment_perturbation_seed(
        args.training_seed,
        args.ratio,
        args.perturbation_repeat,
        0,
    )
    perturbed, added, supported, n_available = numpy_isolated_perturbation(
        original,
        float(args.ratio),
        int(args.guard_steps),
        seed,
    )
    n_original = int(np.count_nonzero(original))
    n_target = int(round(n_original * float(args.ratio)))
    return PreparedSample(
        dataset=spec.key,
        display_name=spec.display_name,
        sample_index=0,
        label=0,
        original=original,
        perturbed=perturbed,
        added_mask=added,
        supported_mask=supported,
        n_original=n_original,
        n_target=n_target,
        n_added=int(np.count_nonzero(added)),
        n_available=n_available,
        perturbation_seed=seed,
        batch_index=0,
        batch_local_index=0,
        selection_rule="deterministic synthetic sample for plotting validation",
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    global torch
    args = parse_args(argv)
    configure_matplotlib()
    specs = selected_dataset_specs(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stream_plotters = {
        "dvsgesture": plot_dvsgesture,
        "shd": plot_shd,
        "stmnist": plot_stmnist,
    }
    frame_plotters = {
        "dvsgesture": plot_dvsgesture_frames,
        "shd": plot_shd_frames,
        "stmnist": plot_stmnist_frames,
    }
    outputs = {
        "dvsgesture": {
            "stream": args.output_dir / "fig6g_dvsgesture_stream.tif",
            "frames": args.output_dir / "fig6g_dvsgesture_frames.tif",
            "stream_clean": args.output_dir / "fig6g_dvsgesture_stream_clean.tif",
            "frames_clean": args.output_dir / "fig6g_dvsgesture_frames_clean.tif",
        },
        "shd": {
            "stream": args.output_dir / "fig6g_shd_stream.tif",
            "frames": args.output_dir / "fig6g_shd_frames.tif",
            "stream_clean": args.output_dir / "fig6g_shd_stream_clean.tif",
            "frames_clean": args.output_dir / "fig6g_shd_frames_clean.tif",
        },
        "stmnist": {
            "stream": args.output_dir / "fig6g_stmnist_stream.tif",
            "frames": args.output_dir / "fig6g_stmnist_frames.tif",
            "stream_clean": args.output_dir / "fig6g_stmnist_stream_clean.tif",
            "frames_clean": args.output_dir / "fig6g_stmnist_frames_clean.tif",
        },
    }

    print("No network training or checkpoint loading will be performed.")
    if args.dry_run:
        modules = None
        print(
            "Dry run: using deterministic built-in arrays and a NumPy isolation "
            "analogue; no dataset or companion module is read."
        )
    else:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Real-data plotting requires PyTorch and the dependencies used by "
                "the supplied Section 2.6 scripts. Install them in the experiment "
                "environment, or use --dry_run to verify TIFF generation."
            ) from exc
        torch = torch_module
        modules = import_section26_modules(args.source_dir)
        print(f"Perturbation protocol: {EXPECTED_PROTOCOL}")

    for spec in specs:
        if args.dry_run:
            prepared = prepare_synthetic_sample(spec, args)
        else:
            module = modules[spec.key]
            _, test_set = build_test_dataset(spec, module, args)
            prepared = prepare_sample(spec, test_set, modules["dvsgesture"], args)
        clean_prepared = without_added_perturbations(prepared)
        stream_plotters[spec.key](prepared, outputs[spec.key]["stream"], int(args.dpi))
        frame_plotters[spec.key](prepared, outputs[spec.key]["frames"], int(args.dpi))
        stream_plotters[spec.key](
            clean_prepared,
            outputs[spec.key]["stream_clean"],
            int(args.dpi),
        )
        frame_plotters[spec.key](
            clean_prepared,
            outputs[spec.key]["frames_clean"],
            int(args.dpi),
        )
        print_metadata(
            prepared,
            outputs[spec.key]["stream"],
            outputs[spec.key]["frames"],
            outputs[spec.key]["stream_clean"],
            outputs[spec.key]["frames_clean"],
            args,
        )

    print("\nCompleted TIFF files:")
    for spec in specs:
        for view in ("stream", "frames", "stream_clean", "frames_clean"):
            path = outputs[spec.key][view]
            print(
                f"  {spec.display_name} {view}: {path} "
                f"({path.stat().st_size / 1024**2:.2f} MiB)"
            )


if __name__ == "__main__":
    main()
