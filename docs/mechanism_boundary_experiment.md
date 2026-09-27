# Mechanism boundary experiment

## Scientific question

The current CDS state is input local: the release assigned to an event depends on the
current input at that feature and its own recent history. The experiment asks whether
this local history is sufficient evidence that an event is task relevant.

Every synthetic example contains two candidate streams and a context cue that identifies
which candidate is relevant. The relevant candidate carries the class label; the other
candidate carries the opposite label. The cue selects each candidate bank equally often,
so a fixed channel preference cannot solve the task. Relevant and distractor candidates
have exactly the same event count, event amplitude, and total input mass in every sample.

Three preregistered conditions change only the relationship between local repetition and
task relevance:

| Condition | Relevant candidate | Distractor candidate | Mechanistic prediction |
|---|---|---|---|
| `aligned` | Burst | Temporally isolated events | CDS favors relevant events |
| `matched` | Same schedule as distractor | Same schedule as relevant | Input-local CDS has no relevance information |
| `anti_aligned` | Temporally isolated events | Burst | CDS favors distractor events |

The `matched` condition is exact within every sample. It is not a distributional match.
The sorted causal history values at relevant and distractor events are asserted to be
identical before any model is run. Each four-example block also contains every cue-bank
and class combination with shared schedules, balancing target and distractor roles across
channel identities.

## Two stages

### 1. Transmission audit

The audit uses fixed CDS parameters and requires no training. It measures:

- causal history immediately before each event;
- mean release for relevant and distractor events;
- relevant minus distractor release and their ratio;
- tie-aware AUROC for predicting event relevance from release;
- sample-level contrast mean, standard deviation, and standard error.

The audit sweeps the calcium time constant and includes `static`, `constant_release`, and
`memory_free` controls. Its central prediction is a positive, zero, and negative release
contrast for `aligned`, `matched`, and `anti_aligned`, respectively.

### 2. Controlled SNN test

A two-hidden-layer LIF network is trained once per seed and transmission variant using
target-only examples. Target history alternates between burst and isolated schedules, and
no distractor is present during training. The three target-distractor relationships are
therefore test-only distribution shifts that the network cannot adapt to separately. The
input transmission variants are `static`, `constant_release`, `memory_free`, and full CDS.
All variants receive the same samples, split seeds, optimization schedule, and matched
initial random seed. The experiment reports classification accuracy and two additional checks:

- release AUROC for distinguishing relevant from distractor candidate events;
- counterfactual cue-swap accuracy, obtained by switching the two cue channels while
  retaining both candidates.

The scientific prediction is directional: the paired accuracy difference between full
CDS and the static network should decrease from `aligned` to `matched` to
`anti_aligned`. The script does not convert a small number of seeds into an overstated
significance claim. It exports every paired seed result.

## Run

Run the full direction experiment with three seeds:

```bash
python run.py mechanism-boundary \
  --mode all \
  --seeds 2026,2027,2028 \
  --device cuda:0 \
  --out_dir results/mechanism_boundary
```

The code rejects more than five seeds. A five-seed confirmation can be run with:

```bash
python run.py mechanism-boundary \
  --mode all \
  --seeds 2026,2027,2028,2029,2030 \
  --device cuda:0 \
  --out_dir results/mechanism_boundary_5seeds
```

Use the quick mode only to verify execution:

```bash
python run.py mechanism-boundary --quick --device cpu \
  --out_dir results/mechanism_boundary_smoke
```

Run only the non-training audit when a GPU is unavailable:

```bash
python run.py mechanism-boundary --mode audit --device cpu \
  --out_dir results/mechanism_boundary_audit
```

## Outputs

| File | Meaning |
|---|---|
| `experiment_config.json` | Executed protocol, hypotheses, seeds, and generator constants |
| `transmission_audit.csv` | Per-seed, per-condition, per-time-constant transmission results |
| `transmission_summary.csv` | Descriptive aggregation of the audit |
| `network_runs.csv` | Every independently trained controlled-network run |
| `network_summary.csv` | Descriptive network aggregation |
| `paired_accuracy_deltas.csv` | Seed-matched full-CDS minus static accuracy |
| `decision_report.json` | Mechanical decision rule and observed directional pattern |
| `transmission_boundary.png` | Release contrast across calcium time constants |
| `network_boundary.png` | Accuracy across the three evidence conditions |

## Decision rule

The current direction is supported only if both levels agree:

1. the fixed transmission audit produces the predicted positive, zero, and negative
   release contrast; and
2. the paired full-CDS minus static network effect decreases from `aligned` through
   `matched` to `anti_aligned`.

If only the first result holds, the experiment establishes a transmission bias but does
not establish a consequence for reliable network computation. If the matched condition
shows a relevance AUROC above chance at the input CDS, first inspect the annotations or
implementation because the local histories are constructed to be exactly identical.
