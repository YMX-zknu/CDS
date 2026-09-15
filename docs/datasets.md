# Dataset preparation

Paths below are relative to the repository root. Dataset files and preprocessing caches are excluded from version control.

## DVS-Gesture

Obtain the original DVS128 Gesture dataset from [IBM's dataset repository](https://github.com/IBM/DvsGesture). The loader is `spikingjelly.datasets.dvs128_gesture.DVS128Gesture` from the pinned SpikingJelly release.

Place the provider's `DvsGesture.tar.gz` archive in `data/DVS-Gesture/download/`. Follow any additional file/checksum instructions printed by SpikingJelly if required. The loader extracts the archive, converts events and creates its frame cache. Alternatively, point `--data_root` at an existing compatible SpikingJelly dataset root.

The repository uses the provider's training/test split. `--delta_t 125` integrates 125 ms windows and `--T 10` retains ten frames, padding shorter samples with zeros. The sample layout is `[T, 2, 128, 128]`. The data root must be writable for extraction and cache generation.

```bash
mkdir -p data/DVS-Gesture/download
python run.py train-dvs --data_root data/DVS-Gesture --device cuda:0
```

## Spiking Heidelberg Digits (SHD)

Use the [Zenke Lab dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) for the official data and usage information.

SpikingJelly loads SHD events and can download/extract the files if the provider is reachable. For offline use, put the extracted `shd_train.h5` and `shd_test.h5` files in `data/SHD/events_h5/`. The repository's `SHDEventsToFrames` transform integrates each sample into 100 equal-duration bins across that sample's event duration, preserving empty bins, then divides by the sample's maximum count. The resulting shape is `[100, 700]`.

```bash
mkdir -p data/SHD/events_h5
python run.py train-shd --data_root data/SHD --device cuda:0
```

`--split_by number` and `--normalize none` are supported alternatives. The default experiment uses `--split_by time --normalize max`.

## ST-MNIST

Download the original archive from [NUS ScholarBank](https://scholarbank.nus.edu.sg/handle/10635/168106), associated with [the ST-MNIST dataset paper](https://arxiv.org/abs/2005.04319). Extract the archive under `data/ST-MNIST/`.

The reader searches recursively for a directory named `data_submission` and reads numeric class folders containing `.mat` files. Each sample must contain the `spiketrain` variable with at least 101 rows, as in the original dataset. The reader handles label remapping; do not rename the numeric class folders.

Expected path pattern:

```text
data/ST-MNIST/data_submission/<numeric-label>/<sample>.mat
```

A parent directory wrapping `data_submission` is also supported. `LUT.mat` is excluded from sample loading. The code uses a fixed taxel mapping, two polarity channels, 30 bins across the complete sample duration, and maximum-count normalization. An 80/20 class-stratified split uses `--split_seed 2026`, independently of the training seed.

```bash
python run.py train-stmnist --data_root data/ST-MNIST --device cuda:0
```

Preprocessed tensors and split metadata are cached under the dataset root. Use `--force_reprocess` after replacing raw data or when intentionally rebuilding the cache. Do not use `--force_reprocess` on every run.

## Storage and runtime

The repository does not redistribute raw data. Allow additional disk space for extracted events, frame caches, checkpoints and exported figures. The first dataset load can take substantially longer than later loads. Exact download size, preparation time and training time are not measured by the source-only checks in this repository.
