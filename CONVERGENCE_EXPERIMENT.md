# Dense vs Dynamic SSB-V6 Convergence Experiment

## Experimental question

Does dynamic SSB-V6 reach comparable validation performance with less total
training cost than dense backpropagation when only the dense-scoring/topology
refresh interval changes?

V6 uses a fixed `gradient_retention=0.90`. For each hidden layer, it keeps the
smallest top-ranked set of neurons/channels whose squared gradient norms contain
at least 90% of that layer's measured gradient energy. The effective keep ratio
is therefore measured, not configured.

## Recommended first experiment: MNIST CNN

This produces 3 dense controls and 12 V6 runs (four refresh rates × three seeds):

```bash
python master_train.py \
  --models dense ssb-v6 \
  --datasets mnist \
  --architectures cnn \
  --score-refresh-steps 1 10 25 100 \
  --v6-gradient-retention 0.90 \
  --runs 3 \
  --stop-at-convergence \
  --max-epochs 100 \
  --patience 8 \
  --min-delta 1e-4 \
  --batch-size 128 \
  --lr 0.001 \
  --device mps \
  --protocol-version protocol-v6-dynamic-convergence-v1 \
  --experiment-tag dense-v6-dynamic-convergence \
  --results-dir results/dense-v6-dynamic-convergence
```

Use `--device cuda` on Colab or `--device cpu` on a CPU-only machine. Omit
`--subset` (or leave it at `0`) for the full training set.

## Summarize

```bash
python summarize_results.py \
  --results-dir results/dense-v6-dynamic-convergence
```

Compare validation accuracy/loss, epochs to convergence, total training time,
mean forward/backward time, memory, scoring overhead, and the observed effective
keep-ratio range. Dense scoring and child rebuilding are included in total epoch
and training wall time.

## Full six-dataset experiment

After the MNIST CNN run validates the protocol, replace the dataset and
architecture arguments with:

```text
--datasets mnist fashion_mnist kmnist cifar10 cifar100 svhn
--architectures mlp cnn
```

With three seeds and four refresh rates, this creates 36 dense runs and 144 V6
runs (180 total).
