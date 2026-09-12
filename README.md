# Stochastic Sparse Backpropagation (SSB)

Research harness for investigating whether neural-network training can reduce gradient computation by allowing only a stochastic subset of output neurons to participate in backpropagation while retaining useful model accuracy.

The project is intentionally organized so that **algorithm mechanics, architectures, training, orchestration, and analysis remain separate**. New architectures and SSB refinements can therefore be tested as new experiment families without rerunning unrelated historical experiments.

## SSB variants

| Model ID | Forward | Backward selection | Scaling |
|---|---|---|---|
| `ssb-v0` | Dense | Individual neurons | None; historical instrumentation |
| `ssb-v1` | Dense | Individual neurons | None |
| `ssb-v2` | Dense | Individual neurons | `1 / keep_ratio` |
| `ssb-v3` | Active neurons only during training | Same selected individual neurons | None |
| `ssb-v1-block` | Dense | Contiguous neuron blocks | None |
| `ssb-v2-block` | Dense | Contiguous neuron blocks | `1 / keep_ratio` |
| `ssb-v3-block` | Active blocks only during training | Same selected contiguous blocks | None |
| `ssb-v4` | Physically smaller random child | Same child | Child Adam resets on refresh |
| `ssb-v5` | Physically smaller random child | Same child | Master-owned persistent Adam state |
| `ssb-v5.1` | Full dense master | Physically smaller random surrogate child | Master-owned persistent Adam state |
| `ssb-v6` | Physically smaller ranked child | Same child; periodic selector refresh | Master-owned persistent Adam state |
| `ssb-v7` | Optimized V6 ranked child | Same child; periodic selector refresh | Adam state synchronized only at refresh |

### V7 optimized implementation

V7 preserves V6 selection and Early-Bird behavior while removing redundant
per-batch master optimizer-state scattering. It also skips topology hashing and
importance statistics on batches without a scoring event. Use
`--timing-detail full` only for short profiling runs because synchronized phase
timers add overhead. Normal experiments default to low-overhead timing and defer
training loss/accuracy conversion to the end of each epoch.

See `V7_EXPERIMENT.md` for diagnostic and multi-dataset commands.
For the audited true-parameter-budget, convergence, and Minsky workflow, see
`SSB_V7_CONVERGENCE_STUDY.md`.

### V6 Stage 2

V6 selects hidden Linear neurons and Conv2d output channels for a physical child.
Available selectors are gradient L2, weight L2, and first-order Taylor
`sum(abs(weight * gradient))`. Weight L2 avoids the dense scoring pass entirely;
the gradient and Taylor selectors perform a dense scoring backward without a
dense optimizer update. Optional Early-Bird detection stops topology refreshes
after a configured window of selected-unit mask distances remains stable.

```bash
python train.py \
  --dataset mnist \
  --model ssb-v6 \
  --architecture cnn \
  --keep-ratio 0.2 \
  --score-refresh-steps 25 \
  --v6-selection-method taylor \
  --v6-early-bird \
  --v6-stability-window 5 \
  --v6-stability-threshold 0.10 \
  --epochs 3 \
  --output-dir results/v6-smoke
```

The batch CSV records selector and dense-scoring time separately, mask distance,
the Early-Bird freeze step, child-rebuild time, active units, and score statistics.
Total epoch/training wall time includes all selector and rebuild overhead.

See `V6_SELECTOR_EXPERIMENT.md` for the fixed-20% selector comparison.

Dynamic V6 uses `--v6-selection-mode gradient_retention` and chooses a separate
child size per hidden layer. The smallest top-ranked set retaining the requested
fraction of squared gradient energy is kept. See `CONVERGENCE_EXPERIMENT.md` for
the controlled dense-versus-dynamic-V6 experiment, where only
`--score-refresh-steps` is swept.

The naming convention deliberately separates the **SSB rule/version** from its **sparsity structure**. For example, `ssb-v1` and `ssb-v1-block` use the same V1 gradient rule but different selection structures. The family now has neuron-level and block-structured counterparts for V1, V2, and V3, keeping algorithm rule and sparsity structure as separate dimensions.

### V3

V3 is a separate forward-and-backward sparsity experiment. During training it samples output neurons, computes only those active forward outputs, scatters them into the normal output shape, and reuses the same selection in backward. It does not perform a full dense forward followed by `output * mask`.

### Block SSB

Block variants partition output neurons into aligned contiguous blocks. Each block is independently selected with probability `keep_ratio`, so the requested neuron keep ratio is preserved in expectation. Block size is an experiment parameter rather than a new algorithm version:

```text
8, 16, 32, 64, 128, ...
```

`ssb-v1-block` isolates block structure using the V1 rule. `ssb-v2-block` tests the same structure with V2 inverse-probability gradient scaling. `ssb-v3-block` applies the V3 rule with one block mask shared by sparse forward and backward, so inactive blocks are not included in the training forward matrix multiplication.

## Repository architecture

```text
algorithms/ssb/<variant>/   SSB mechanics only
models/<dataset>/dense/     dense dataset model builders
models/<dataset>/sparse/    models accepting an SSB linear implementation
models/<dataset>/dropout/   dropout comparison models
models/<dataset>/pruning/   pruning comparison models
data/                       dataset loaders/acquisition
training/                   shared training/runtime helpers
train.py                    one experiment
master_train.py             experiment-grid orchestration only
summarize_results.py        post-experiment statistics
visualize_results.py        post-experiment plots
results/                    raw and summarized results
```

Training-loop logic does not belong in model files, SSB mechanics do not belong in dataset models, and model definitions do not belong in `master_train.py`.

## Architectures

### MLP

```bash
--architecture mlp
```

The flattened MLPs are primarily **systems benchmarks**, not claims of dataset SOTA architectures. Larger configurations allow testing whether SSB reaches a model-size/compute crossover where reduced matrix work outweighs sparse-selection overhead.

### CNN

```bash
--architecture cnn
```

The current CNN experiment uses a shared **dense convolutional feature extractor** followed by a classifier whose linear layers use the selected dense/dropout/pruning/SSB implementation.

This intentionally does **not** claim sparse convolution. Channel-sparse convolution should be introduced as a separate algorithmic experiment rather than silently changing the meaning of linear SSB.

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Smoke test

Before launching expensive grids, run:

```bash
python smoke_test.py
```

The smoke suite checks algorithm registration, V0/V1 compatibility, keep=1 equivalence where appropriate, V3/V3-block inactive-output behavior, block structure, matched initialization, and synthetic MLP/CNN optimization steps.

## Running one experiment

### V1

```bash
python train.py \
  --dataset cifar10 \
  --architecture mlp \
  --model ssb-v1 \
  --keep-ratio 0.5 \
  --seed 1 \
  --output-dir results/cifar10/ssb-v1/architecture_mlp/protocol-v2/keep_0_5/seed_01
```

### V3

```bash
python train.py \
  --dataset cifar10 \
  --architecture mlp \
  --model ssb-v3 \
  --keep-ratio 0.5 \
  --seed 1 \
  --output-dir results/cifar10/ssb-v3/architecture_mlp/protocol-v2/keep_0_5/seed_01
```

### V1 Block

```bash
python train.py \
  --dataset cifar10 \
  --architecture mlp \
  --model ssb-v1-block \
  --keep-ratio 0.5 \
  --block-size 32 \
  --seed 1 \
  --output-dir results/cifar10/ssb-v1-block/architecture_mlp/protocol-v2/keep_0_5/block_32/seed_01
```

### V2 Block

```bash
python train.py \
  --dataset cifar10 \
  --architecture mlp \
  --model ssb-v2-block \
  --keep-ratio 0.5 \
  --block-size 32 \
  --seed 1 \
  --output-dir results/cifar10/ssb-v2-block/architecture_mlp/protocol-v2/keep_0_5/block_32/seed_01
```

### V3 Block

```bash
python train.py \
  --dataset cifar10 \
  --architecture mlp \
  --model ssb-v3-block \
  --keep-ratio 0.5 \
  --block-size 32 \
  --seed 1 \
  --output-dir results/cifar10/ssb-v3-block/architecture_mlp/protocol-v2/keep_0_5/block_32/seed_01
```

## Master experiment grids

A small exploratory grid should be used before multi-seed experiments:

```bash
python master_train.py \
  --datasets cifar10 \
  --architectures mlp cnn \
  --models dense ssb-v1 ssb-v2 ssb-v3 ssb-v1-block ssb-v2-block ssb-v3-block \
  --keep-ratios 0.5 \
  --block-sizes 16 32 64 \
  --runs 1 \
  --epochs 1 \
  --subset 2048
```

The default master grid remains conservative and does not automatically add V3 or block experiments to large historical-style sweeps.

For publication-quality comparisons, increase the seed count only after the smoke/exploratory grid behaves correctly. The existing convention is 20 seeds.

## Result storage

New orchestrated experiments use:

```text
results/
  <dataset>/
    <model>/
      architecture_<architecture>/
        <protocol-version>/
          keep_<ratio>/
            [block_<size>/]
              seed_<seed>/
                metadata.json
                batches.csv
                epochs.csv
```

Dense experiments omit the keep/block levels where they are not meaningful.

This hierarchy allows new architectures and algorithm variants to coexist with existing results. A new architecture does **not** require rerunning unrelated historical experiments; only experiments required for a controlled comparison with that architecture need to be run.

## Experiment metadata

Every new run writes `metadata.json` and includes identifying fields in the raw CSVs. Metadata includes:

- dataset and architecture;
- model/SSB variant;
- keep ratio and block size;
- seed and protocol version;
- batch size, epochs, learning rate, and subset size;
- optimizer and criterion;
- initialization policy;
- parameter counts;
- requested and resolved device;
- worker count;
- Python, PyTorch, and platform versions;
- Git commit when available;
- UTC timestamp;
- memory-measurement semantics;
- deterministic run ID.

For block experiments, the model identifier encodes the SSB rule (`ssb-v1-block`, `ssb-v2-block`, or `ssb-v3-block`) while `block_size` records the structural granularity.

## Metrics

Raw batch measurements include:

- forward wall time;
- backward wall time;
- memory measurement;
- batch accuracy.

Epoch measurements include training/validation loss and accuracy plus total epoch time.

Experiment-wide means and standard deviations are **not calculated in the training hot path**. They are produced only after raw runs complete.

## Summarizing results

```bash
python summarize_results.py --dataset cifar10
```

Historical CSVs without new metadata are treated as `historical_mlp` / `historical`, preventing them from being silently averaged with the newer controlled protocol.

## Visualizing results

```bash
python visualize_results.py --summary results/cifar10/summary.csv
```

Plots separate dataset, architecture, protocol, model, and block size where relevant.

## Experimental protocol

New `protocol-v2` runs explicitly initialize comparable linear and convolutional layers with Xavier-uniform weights and zero biases using the experiment seed, then reset the stochastic training seed.

Within a controlled comparison, preserve:

- architecture;
- initialization policy;
- optimizer;
- learning rate;
- epochs;
- batch size;
- dataset ordering/seed policy;
- hardware/device conditions.

If one of these changes, treat it as a new protocol or explicitly record the change rather than silently combining the results.

## Baseline caveat

The current random static pruning control masks outputs after a dense matrix multiplication. It therefore **does not remove matrix compute** and should be interpreted as a learning/control baseline rather than a structural speed-pruning implementation.

A stronger magnitude/structured pruning baseline remains part of the baseline-comparison stage.

## Research roadmap

Current near-term sequence:

1. **Scaling:** larger MLP/CNN workloads and crossover analysis.
2. **Baselines:** dense, dropout, random static pruning, then stronger structured/magnitude pruning.
3. **V3:** forward + backward stochastic neuron sparsity.
4. **Block SSB:** compare `ssb-v1-block`, `ssb-v2-block`, and `ssb-v3-block` against their neuron-level counterparts while sweeping block size.
5. **Later refinements:** mask persistence, gradient-magnitude/EMA-aware selection with exploration/starvation prevention, adaptive/layer-specific keep ratios.
6. **Low-level optimization only if profiling justifies it:** Triton/CUDA/C++ or other fused/block-sparse kernels.

The central systems question is whether reduced SSB matrix work can eventually overcome the indexing, masking, memory-movement, and kernel-launch overhead relative to highly optimized dense PyTorch operations.
