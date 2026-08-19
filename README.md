# Stochastic Sparse Backpropagation Research Harness

A reorganized research harness for testing Stochastic Sparse Backpropagation (SSB) against dense, dropout, and pruning baselines across multiple datasets.

The current implemented SSB variants are V0, V1, and V2. V3 and block SSB are intentionally separate future experiments. See `ROADMAP.md`.

## Single experiment

```bash
python train.py --dataset mnist --model dense --seed 1 --output-dir results/mnist/dense/seed_01
python train.py --dataset mnist --model ssb-v1 --keep-ratio 0.7 --seed 1 --output-dir results/mnist/ssb-v1/keep_0_7/seed_01
python train.py --dataset cifar10 --model dropout --keep-ratio 0.7 --seed 1 --output-dir results/cifar10/dropout/keep_0_7/seed_01
```

## Master experiment

```bash
python master_train.py --datasets mnist fashion_mnist cifar10 --runs 20
```

For a smoke test:

```bash
python master_train.py --datasets mnist --models dense ssb-v1 dropout pruning --keep-ratios 0.7 --runs 1 --epochs 1 --subset 2000
```

## Post-processing

```bash
python summarize_results.py --dataset mnist
python visualize_results.py --summary results/mnist/summary.csv
```

Means and standard deviations are intentionally computed only in `summarize_results.py`, after raw experiments are complete.

## Important baseline note

The included dropout and random-static-pruning implementations are simple comparison controls. The pruning baseline currently masks dense outputs; it is **not** structural pruning and therefore should not be claimed as a speed-optimized pruning implementation. A stronger magnitude/structured pruning baseline is a Stage 2 task.
