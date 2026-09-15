# Reproducibility and outputs

## Scope

The source release contains the supplied Section 2.1–2.7 Python experiments. It does not contain trained checkpoints, recorded training results or the full manuscript figure artwork. Commands generate results from the selected datasets, seeds and configuration; no reported accuracy values are embedded as replacement experiment data.

The section organization is retained because it makes the numerical analyses traceable to the manuscript. The Section 2.6 DVS-Gesture module also exports shared training, perturbation and metric utilities imported by SHD, ST-MNIST and Section 2.7. Dataset-specific CDS implementations and parameterizations are retained rather than merging them into a new model implementation.

## Running and recording experiments

- Pass training seeds explicitly. Both multi-seed launchers and the README examples use `2026,2027,2028`; each is a separate training run.
- Main comparisons use paired static/CDS networks, with each trainer's default preprocessing, architecture and optimization settings.
- Each saved checkpoint is selected by the maximum test accuracy during training. This is the supplied checkpoint-selection procedure.
- Nonzero perturbation ratios use three realizations. Average within each training seed before computing the mean and sample standard deviation across seeds. Realizations do not count as additional independent training runs.
- `--analysis_batches` selects batches for auxiliary embeddings/dynamics exports in Section 2.6. Clean task metrics and isolated-input evaluations still traverse the complete test loader.
- Section 2.7 uses `--max_batches 0` for full evaluation. A nonzero value is a partial check and should not be mixed with full results.
- Record the repository commit, `python -m pip freeze`, GPU/driver details, commands and output configuration files alongside a run. Numerical reproducibility across devices and software versions is not guaranteed by seed selection alone.

## Section 2.6 output files

Paths are relative to `results/section_2_6/<dataset>/seed_<seed>/`.

| File | Contents |
|---|---|
| `checkpoints/static.pt`, `checkpoints/cds.pt` | Selected weights, configuration, epoch and protocol metadata |
| `logs/training_curves.csv` | Epoch-level training/test accuracy and loss |
| `logs/metrics_summary.csv` | Clean metrics, integrated perturbation score and release selectivity |
| `logs/isolated_perturbation_performance.csv` | Metrics and realized perturbation counts per ratio and realization |
| `logs/transmission_selectivity.csv` | Release measurements at the selected ratio |
| `logs/architecture_verification_*.json` | Observed presynaptic CDS/transform/neuron execution order |
| `logs/confusion_*.csv` | Confusion matrices |
| `logs/spike_stats_*.csv`, `logs/cds_stats_*.csv`, `logs/embeddings_*.csv` | Auxiliary sampled dynamics/features |

Ablation checkpoints are stored under `ablation/ablation_runs/<variant>/<dataset>/seed_<seed>/checkpoints/`. Consolidated tables are written directly under `ablation/`.

## Section 2.7 output files

Paths are relative to `results/section_2_7/<dataset>/seed_<seed>/logs/`.

| File | Contents |
|---|---|
| `efficiency_summary.csv` | Recognition, operation counts, propagation and normalized metrics |
| `layerwise_operations.csv` | Per-layer fan-out, operations and weighted propagation |
| `temporal_activity.csv` | Time-resolved activity |
| `cds_state_dynamics.csv` | Calcium and release measurements |
| `analysis_config.json` | Checkpoint paths, loaded settings, protocol and verification results |
| `energy_overhead_decomposition.csv`, `energy_cost_assumptions.csv` | Supplementary implementation-cost estimates and their assumptions |

Transmission-weighted propagation measures graded signal magnitude with structural fan-out. It is distinct from SynOps, physical energy consumption and total implementation cost. The optional 45-nm arithmetic estimates retained in the source are analytical cost scenarios, not hardware measurements.

## Input visualization

The twelve TIFFs comprise four views per dataset. The real-data path uses the task's test split, preprocessing and perturbation builder. Original occupied bins remain unchanged. By default, the program searches for a representative example near median occupancy that satisfies the requested isolated-addition load. It prints the selected index, class, perturbation seed, counts and frame indices.

DVS-Gesture's stream view distributes markers inside their processed time bins for continuous display. It does not reconstruct original asynchronous timestamps. The clean and perturbed images use matching viewpoints and frame selections. The `--dry_run` path is synthetic and must not be used as experimental evidence.

## Figure names

Some filenames and panel labels retain the supplied plotting scripts' historical numbering. In the manuscript with the split input/recognition figures, use this mapping:

| Script output | Content | Manuscript destination |
|---|---|---|
| Mechanism scripts | Transmission example and mechanism analyses | Figures 1–5 |
| `fig6g_*` TIFF files | Three datasets, clean/perturbed streams and frames | Figure 6 input panels |
| `fig6_temporal_perturbation_robustness.*` | Recognition, perturbation curves and selectivity | Figure 7 |
| `fig7_cross_modal_propagation_efficiency.*` | Propagation metrics | Figure 8 |

The supplied Section 2.7 overview also contains a workflow panel. It is a plotting overview, not a pixel-identical copy of the five-panel manuscript Figure 8. The source CSV exports allow the final panel layout to be assembled independently. Existing generated filenames are retained so external plotting workflows remain compatible.

## Refactoring boundaries

The release organizes scripts under `experiments/section_2_*`, adds the common `run.py` entry point, removes narrative comments/docstrings, standardizes data/result paths, fixes the inverted `--with_ablation` switch, and sets the multi-seed defaults to `2026,2027,2028`. Numerical model definitions, learned parameter keys, training losses, checkpoint selection, perturbation construction and metric calculations are retained from the supplied code. Architecture/protocol identifiers remain unchanged for checkpoint compatibility.
