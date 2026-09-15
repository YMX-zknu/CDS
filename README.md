# CDS · Calcium-modulated dynamic synapses

[![Python checks](https://github.com/YMX-zknu/CDS/actions/workflows/ci.yml/badge.svg)](https://github.com/YMX-zknu/CDS/actions/workflows/ci.yml)

Implementation of **Calcium-modulated dynamic synapses enable temporal context-dependent transmission in spiking neural networks**.

CDS uses a decaying calcium state to modulate synaptic release before neuronal integration. This repository contains the mechanism experiments, matched static/CDS sensory-task experiments, component ablations, input visualizations and checkpoint-based propagation analyses.

**Start here:** [Installation](#1-install-the-environment) → [Data](#3-data-preparation) → [Training](#5-train-the-sensory-task-models) → [Analysis](#8-analyze-network-wide-propagation).

## 1. Install the environment

Use Linux with Python **3.10 or 3.11**. An NVIDIA GPU is recommended for full training; CPU execution is suitable for mechanism experiments and small checks. The code uses PyTorch operations and does not require CuPy. GPU memory requirements depend on the dataset, batch size and sequence length.

```bash
git clone https://github.com/YMX-zknu/CDS.git
cd CDS
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** matching PyTorch/torchvision pair. For a CUDA 12.1-compatible driver:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only execution, use this command instead:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

The pinned CPU environment is validated by GitHub Actions using Python 3.11. Full dataset training requires the data preparation below. The scripts use the legacy SpikingJelly `activation_based` API, so use the pinned release rather than upgrading SpikingJelly independently. See [PyTorch's official installation commands](https://pytorch.org/get-started/previous-versions/) for other CUDA builds.

Run the commands below **from the repository root**. Relative data and output paths are interpreted from the working directory. For headless servers:

```bash
export MPLBACKEND=Agg
python run.py --help
```

## 2. Check the installation without downloading datasets

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
```

The smoke test runs one small synthetic training epoch for each task, loads the resulting checkpoints for propagation analysis, and exercises both component ablations on SHD. It writes to `results/smoke/`. These checks verify execution, **not scientific performance**.

To check all twelve TIFF exports independently:

```bash
python run.py visualize --dry_run --dpi 150 --output_dir results/visualization_smoke
```

The visualization dry run uses synthetic arrays and a NumPy perturbation analogue. Real-data visualizations use the actual task preprocessing and perturbation implementation.

## 3. Data Preparation

Datasets and pretrained checkpoints are not bundled with this source release. Download datasets from their original providers and retain their terms of use. Detailed layouts and preprocessing are described in [docs/datasets.md](docs/datasets.md).

| Dataset | Official download | Default root | Processed sample |
|---|---|---|---|
| DVS-Gesture · vision | [IBM dataset files](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794) | `data/DVS-Gesture` | 10 × 2 × 128 × 128 |
| SHD · audition | [Zenke Lab dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) · [Data files](https://zenkelab.org/datasets/) | `data/SHD` | 100 × 700 |
| ST-MNIST · touch | [NUS ScholarBank dataset](https://scholarbank.nus.edu.sg/handle/10635/168106) | `data/ST-MNIST` | 30 × 2 × 10 × 10 |

```bash
mkdir -p data/DVS-Gesture data/SHD data/ST-MNIST
```

1. **DVS-Gesture:** manually download `DvsGesture.tar.gz`, `gesture_mapping.csv`, `LICENSE.txt` and `README.txt` from the IBM folder. Place all four files in `data/DVS-Gesture/download/`. SpikingJelly then checks and extracts them and builds the frame cache.
2. **SHD:** the loader can download its archives automatically. For offline preparation, download the training and test HDF5 archives from Zenke Lab, extract them, and place `shd_train.h5` and `shd_test.h5` in `data/SHD/extract/`.
3. **ST-MNIST:** download and extract the NUS archive under `data/ST-MNIST/`, retaining `data_submission/<numeric-label>/*.mat`. The local reader builds the frame cache and fixed class-stratified split.

These are the dataset sources listed in the manuscript's Data availability statement. Dataset roots must be writable for extraction and caching. See [the detailed preparation instructions](docs/datasets.md) for file layouts and preprocessing.

## 4. Run the mechanism experiments

These experiments do not require sensory datasets or trained checkpoints.

```bash
python run.py synapse
python run.py calcium
python run.py selectivity
python run.py lif
python run.py sequences
```

| Command | Manuscript section | Analysis | Output directory |
|---|---|---|---|
| `synapse` | 2.1 | History-dependent transmission example | `results/section_2_1/` |
| `calcium` | 2.2 | Calcium dynamics and parameter scans | `results/section_2_2/` |
| `selectivity` | 2.3 | Context-dependent transmission | `results/section_2_3/` |
| `lif` | 2.4 | Coupled CDS–LIF responses | `results/section_2_4/` |
| `sequences` | 2.5 | Sequence preservation under perturbations | `results/section_2_5/` |

The scripts export figures and CSV source data, including tables suitable for Origin. They retain the numerical definitions of the supplied experiments.

## 5. Train the sensory-task models

Run paired static and CDS models across all three datasets and three explicitly specified training seeds:

```bash
python run.py train \
  --seeds 2026,2027,2028 \
  --dvsgesture_root data/DVS-Gesture \
  --shd_root data/SHD \
  --stmnist_root data/ST-MNIST \
  --out_dir results/section_2_6 \
  --device cuda:0 --num_workers 4
```

The launcher runs tasks and seeds sequentially. Each training script saves its best-test-accuracy checkpoint, evaluates clean recognition and isolated-input perturbations, and exports release selectivity. The launcher verifies protocol metadata and executed layer order before aggregating results.

| Task | Time steps | Epochs | Batch size | Initial learning rate | Backbone |
|---|---:|---:|---:|---:|---|
| DVS-Gesture | 10 | 200 | 16 | 0.0001 | Five 128-channel convolutional blocks |
| SHD | 100 | 200 | 128 | 0.001 | Two 256-unit hidden layers |
| ST-MNIST | 30 | 100 | 16 | 0.001 | Two 64-channel convolutional blocks |

These are the **command-line defaults**. Training uses Adam, cosine learning-rate decay and rate-based one-hot MSE by default. `--TET` enables the separate optional temporal loss. Consult each trainer's help for all options:

```bash
python run.py train-dvs --help
python run.py train-shd --help
python run.py train-stmnist --help
```

For a single task or custom hyperparameters, call its trainer directly:

```bash
python run.py train-shd \
  --data_root data/SHD --out_dir results/section_2_6 \
  --seed 2026 --model both --device cuda:0 --num_workers 4
```

Results are stored as `results/section_2_6/<dataset>/seed_<seed>/`, with `checkpoints/static.pt`, `checkpoints/cds.pt` and a `logs/` directory. Reusing the same dataset, seed and output root overwrites that run's checkpoints. Use a different output root for exploratory configurations.

Both multi-seed launchers default to **2026, 2027 and 2028**. Each seed corresponds to an independent training run. Perturbation realizations are averaged within each training seed before means and sample standard deviations are computed across the three seeds.

## 6. Run component ablations

Add `--with_ablation` to the training command to run all four variants after the paired experiments. The flag enables ablations; without it only the paired models are trained.

To run the ablation suite separately:

```bash
for seed in 2026 2027 2028; do
  python run.py ablate \
    --datasets dvsgesture,shd,stmnist \
    --variants static,full,constant_release,memory_free \
    --dvsgesture_data_root data/DVS-Gesture \
    --shd_data_root data/SHD \
    --stmnist_data_root data/ST-MNIST \
    --out_dir results/section_2_6/ablation \
    --seed "$seed" --device cuda:0 --num_workers 4
done
```

| Variant | Transmission |
|---|---|
| `static` | No CDS layer |
| `full` | Calcium carryover and time-varying release |
| `constant_release` | Learned constant release, without calcium dynamics |
| `memory_free` | Instantaneous calcium drive, without carryover |

Each variant is trained independently. The ablation runner inherits the corresponding task's CLI defaults and exports per-seed and aggregated tables under `results/section_2_6/ablation/`.

## 7. Plot recognition and sensory inputs

Regenerate recognition and perturbation plots from saved CSV files without retraining:

```bash
python run.py plot-tasks \
  --root results/section_2_6 --seeds 2026,2027,2028 \
  --fig_dir results/section_2_6/figures \
  --ablation_path results/section_2_6/ablation/ablation_task_table.csv
```

Generate real-input visualizations without training or loading checkpoints:

```bash
python run.py visualize \
  --dvsgesture_root data/DVS-Gesture \
  --shd_root data/SHD \
  --stmnist_root data/ST-MNIST \
  --ratio 0.10 --guard_steps 4 --training_seed 2026 \
  --output_dir results/section_2_6/input_visualizations --dpi 600
```

Each dataset produces four TIFFs: clean and perturbed streams, plus clean and perturbed frame strips. Streams show every processed time step; strips show ten frames. Red/blue distinguish polarity in DVS-Gesture and ST-MNIST, blue indicates SHD channel activity, and green indicates added inputs. Captions, legends and sample-selection metadata are printed to the terminal. Use `--dvsgesture_sample_index`, `--shd_sample_index` and `--stmnist_sample_index` to select fixed examples.

## 8. Analyze network-wide propagation

Complete Section 2.6 training first. This stage reads its checkpoints and performs **no additional training**:

```bash
python run.py analyze \
  --seeds 2026,2027,2028 \
  --dvsgesture_root data/DVS-Gesture \
  --shd_root data/SHD \
  --stmnist_root data/ST-MNIST \
  --results_26_root results/section_2_6 \
  --out_root results/section_2_7 \
  --device cuda:0 --num_workers 4
```

The analysis defaults to the complete test split (`--max_batches 0`), perturbation ratios `0,0.025,0.05,0.10`, three perturbation realizations per nonzero ratio and a four-step isolation guard. It inherits architecture and preprocessing settings from each checkpoint. Keep the training batch size when matching the exact perturbation realizations, whose seeds also depend on batch index.

To include the constant-release control, add:

```bash
--include_constant_release --ablation_root results/section_2_6/ablation
```

To replot existing analysis CSVs:

```bash
python run.py plot-propagation \
  --root results/section_2_7 --seeds 2026,2027,2028 \
  --out_dir results/section_2_7/figures --selectivity_ratio 0.10
```

Outputs include operation counts, transmission-weighted propagation, release selectivity, per-layer dynamics and aggregated tables. Implementation-cost estimates are exported separately and are **not hardware energy measurements**. See [docs/reproducibility.md](docs/reproducibility.md) for output meanings and figure mapping.

## Repository organization

| Location | Purpose |
|---|---|
| `run.py` | Common entry point for experiment commands |
| `experiments/section_2_1/`–`section_2_5/` | Independent mechanism analyses |
| `experiments/section_2_6/` | Task training, shared task utilities, ablations and visualization |
| `experiments/section_2_7/` | Checkpoint-based propagation analysis and plotting |
| `scripts/` | Environment and synthetic execution checks |
| `tests/` | Command routing, perturbation and model regression checks |
| `docs/` | Dataset preparation and reproducibility details |

## License and attribution

Code is distributed under the [MIT License](LICENSE). Dataset licenses remain with their respective providers. Please acknowledge this repository and the manuscript **Calcium-modulated dynamic synapses enable temporal context-dependent transmission in spiking neural networks** when using the implementation. Bibliographic publication details will be added when available.
