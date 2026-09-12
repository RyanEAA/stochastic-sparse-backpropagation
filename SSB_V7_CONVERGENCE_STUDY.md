# SSB-V7 convergence study

## Audited registries

- Datasets: `mnist`, `fashion_mnist`, `kmnist`, `cifar10`, `cifar100`, `svhn`
- Architectures: `mlp`, `cnn`
- V7 selectors: `random`, `gradient_l2`, `weight_l2`, `taylor`
- Requested V7 strategies: `layerwise`, `layerwise_warmup`,
  `layerwise_correction`, `layerwise_hybrid`
- Other registered V7 presets: `pure`, `warmup`, `correction`,
  `warmup_correction`

The selector option is named `--v6-selection-methods` in `master_train.py`
because V7 inherits the V6 selector implementation. The single-run option in
`train.py` is `--v6-selection-method`.

## What V7 does

V7 owns a dense master and a physically smaller, ordinary dense child. Hidden
Conv2d output channels and Linear output neurons are selected; the complete
input boundary is retained, and the complete class-output boundary is retained
so every class logit exists. Child inputs are the selected outputs of the
previous layer. The child performs sparse-phase forward, backward, and Adam
updates. At a scoring event, child weights and Adam moments are synchronized to
master-shaped storage, importance is recomputed, and a new child is gathered.
V7 avoids V6's full master-shaped Adam scatter after every ordinary child step.

`random` samples random importance scores. `weight_l2` ranks structured units
without a dense backward. `gradient_l2` and `taylor` run a dense scoring
forward/backward but no dense optimizer update. Taylor uses
`sum(abs(weight * gradient))` per output unit.

Dense warmup trains the master for complete initial epochs. A dense correction
replaces one sparse batch with a full-master Adam update after the configured
number of sparse updates. Corrections reconstruct the child to gather corrected
weights, but after the audit patch only the score-refresh schedule changes the
mask. Thus scoring/mask refresh and child reconstruction are distinct events.

## Parameter-budget correction

The historical `effective_keep_ratio` is a hidden structured-unit ratio, not a
parameter ratio. On the current CNNs:

| Configuration | Dataset family | Hidden-unit ratio | Child/master parameter ratio |
| --- | --- | ---: | ---: |
| Uniform width 0.20 | MNIST family | 19.90% | 4.10% |
| Uniform width 0.20 | CIFAR-10/SVHN | 20.01% | 4.02% |
| Uniform width 0.20 | CIFAR-100 | 20.01% | 4.17% |
| Layer profile `[1.0, 0.5, 0.2]` | MNIST family | 27.30% | 10.47% |
| Layer profile `[1.0, 0.5, 0.2]` | CIFAR-10/SVHN | 28.38% | 10.88% |
| Layer profile `[1.0, 0.5, 0.2]` | CIFAR-100 | 28.38% | 10.97% |

Use `--v7-target-parameter-ratio 0.20`. V7 scales the relative layer profile
per architecture to the closest realizable structured child. Because channel
and neuron counts are integers, the recorded ratio may differ from 0.20 by a
small rounding amount. Metadata records the requested target, calibrated layer
ratios, child/master counts, and measured parameter ratio. Every selector uses
the same calibrated dimensions.

## Stage 1 validation

Run locally or as a CPU utility job inside the configured container:

```bash
python -m pytest -q
python smoke_test.py
```

The smoke test includes a synthetic V7 optimization step and checks that the
CIFAR-10 CNN child is within 0.001 of the requested 20% parameter budget.

Dry-run the complete screening plan without training:

```bash
python master_train.py \
  --models dense ssb-v7 \
  --datasets mnist fashion_mnist kmnist cifar10 cifar100 svhn \
  --architectures cnn \
  --v6-keep-ratio 0.20 \
  --v6-selection-methods random gradient_l2 weight_l2 taylor \
  --score-refresh-steps 25 100 \
  --v7-hybrid-configs layerwise layerwise_warmup layerwise_correction layerwise_hybrid \
  --v7-warmup-epochs 1 \
  --v7-correction-steps 25 \
  --v7-layer-keep-ratios 1.0 0.5 0.2 \
  --v7-target-parameter-ratio 0.20 \
  --v6-disable-early-bird \
  --runs 1 \
  --stop-at-convergence --max-epochs 50 --patience 7 --min-delta 0.0001 \
  --validation-fraction 0.10 --split-seed 2026 \
  --timing-detail basic \
  --device cuda \
  --protocol-version protocol-v7-screen-convergence-v2 \
  --experiment-tag v7-screen-convergence \
  --results-dir results/v7-screen-convergence-v2 \
  --dry-run
```

Expected plan: 198 runs = 6 dense +
`6 datasets * 4 selectors * 2 refresh windows * 4 V7 strategies`.

Before that grid, run this five-run, one-epoch GPU validation:

```bash
python master_train.py \
  --models dense ssb-v7 \
  --datasets cifar10 \
  --architectures cnn \
  --v6-keep-ratio 0.20 \
  --v6-selection-methods taylor \
  --score-refresh-steps 25 \
  --v7-hybrid-configs layerwise layerwise_warmup layerwise_correction layerwise_hybrid \
  --v7-warmup-epochs 1 \
  --v7-correction-steps 25 \
  --v7-layer-keep-ratios 1.0 0.5 0.2 \
  --v7-target-parameter-ratio 0.20 \
  --v6-disable-early-bird \
  --runs 1 --epochs 1 --subset 2048 \
  --validation-fraction 0.10 --split-seed 2026 \
  --timing-detail full --device cuda \
  --protocol-version protocol-v7-validation-v2 \
  --experiment-tag v7-validation \
  --results-dir results/v7-validation-v2
```

The four V7 directories differ by `hybrid_<strategy>`, and selector, budget,
refresh, protocol, and seed are also represented. Dense is generated once.

## Stage 2 convergence screening

Submit one dataset per job. Replace `<dataset>` in the dry-run command above,
remove `--dry-run`, and use a dataset-specific protocol, tag, and result root.
Each job has 33 runs: one dense plus 32 V7 configurations. Early-Bird is
disabled so the 25-vs-100 mask-refresh comparison remains interpretable.

Do not assume a 33-run convergence job fits eight hours. From the five-run
validation log, calculate a conservative estimate:

```text
estimated hours = 33 * chosen expected epochs * slowest observed epoch seconds / 3600
```

Use the observed convergence epochs from the first completed dataset to refine
the estimate. If the estimate exceeds seven hours, split that dataset by
selector into four jobs. Reuse the same result root: the already completed dense
directory will be skipped after the first selector job. Do not launch selector
jobs concurrently against the same root.

## Convergence definition

The screening rule is validation-loss improvement greater than `0.0001`, with
patience 7 and a 50-epoch safety cap. The fixed train/validation split uses
`split_seed=2026` for every run; experimental seed controls initialization and
training order. The official dataset test split is evaluated only once after
stopping. For final experiments, use at least `--max-epochs 100 --patience 10`
after inspecting screening curves. A safety cap is still required; reaching it
must be reported as `max_epochs`, not convergence.

## Minsky Job Manager fields

For the five-run GPU validation, start `minsky`, choose **Submit a GPU job** and
**Launch a new job**, then enter:

- Python script: `/home/ain21/stochastic-sparse-backpropagation/master_train.py`
- Working directory: `/home/ain21/stochastic-sparse-backpropagation`
- Log directory: `/home/ain21/logs`
- Job name: `ssbv7-validation-v2`
- Program arguments: everything after `python master_train.py` in the validation command
- Maximum hours: `1`
- Additional minutes: `0`
- System memory: `32` GB
- CPU cores: `4`
- Environment: registered `ssbv7`, or manually select the supplied PyTorch
  container and `/home/ain21/environments/ssbv7`

On the review screen verify one GPU, partition `batch` in the generated script,
the 1-hour limit, the exact project directory, central log directory, container,
virtual environment, and every program argument. Submit only after that review.

For a per-dataset convergence screening job, request 8 hours, 32 GB, and 4 CPU
cores initially. Job names may contain only letters, digits, hyphens, and
underscores.

## Summaries

After a result root finishes:

```bash
python summarize_results.py --results-dir results/<result-root>
python visualize_results.py \
  --summary results/<result-root>/summary.csv \
  --output-dir results/<result-root>/plots
```

Use a CPU utility job in Minsky Job Manager for these scripts. The summary now
separates final validation accuracy, maximum validation accuracy, accuracy at
minimum validation loss, final child accuracy, final test accuracy,
generalization gap, training time, component timings, peak memory, structured
unit ratio, and child/master parameter ratio.

## Remaining limitations to report

- The supplied ZIP had no raw `results/` directory, so prior numerical claims
  could not be independently recomputed.
- Dataset transforms are only `ToTensor`; there is no normalization or data
  augmentation. This is matched across methods but is not a modern accuracy
  recipe for CIFAR/SVHN.
- Basic mode still synchronizes GPU forward/backward timing every batch. Run a
  separate no-component-timing throughput protocol before making the strongest
  wall-clock claim.
- The wrapper registers both master and child parameters, so legacy
  `trainable_parameters` metadata is not the sparse budget. Use the explicit
  child/master fields added by this audit.
- Reaching `max_epochs` is not convergence. Inspect `stop_reason` and
  `converged_by_patience` for every run.
