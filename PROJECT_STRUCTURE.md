# Project Structure

```text
stochastic_sparse_backprop_research/
├── algorithms/
│   └── ssb/
│       ├── registry.py
│       ├── v0/                  # historical instrumented SSB
│       ├── v1/                  # same SSB math, instrumentation removed
│       ├── v2/                  # inverse-probability scaled SSB
│       ├── v3/                  # planned forward+backward sparsity; separate test
│       └── block/               # planned block SSB; separate test
├── data/
│   ├── dataset_utils.py         # supplied dataset/setup helpers
│   └── loaders.py               # training DataLoader construction
├── models/
│   ├── common/                  # generic model components only
│   ├── mnist/
│   │   ├── config.py
│   │   ├── dense/model.py
│   │   ├── sparse/model.py
│   │   ├── dropout/model.py
│   │   └── pruning/model.py
│   ├── fashion_mnist/           # same dataset-local layout
│   ├── kmnist/
│   ├── cifar10/
│   ├── cifar100/
│   └── svhn/
├── training/
│   └── runtime.py               # timing/device/seed helpers
├── results/                     # generated; not committed
├── train.py                     # ONE experiment; --dataset + --model
├── master_train.py              # orchestrates experiment grid
├── summarize_results.py         # mean/std only after runs finish
├── visualize_results.py
├── ROADMAP.md
├── PROJECT_STRUCTURE.md
└── README.md
```

## Rules
1. Dataset architecture belongs under `models/<dataset>/`.
2. SSB algorithm mechanics belong under `algorithms/ssb/<variant>/`, never inside dataset models.
3. `train.py` trains one configuration only.
4. `master_train.py` schedules configurations only; it does not implement models or calculate statistics.
5. Summary statistics and plots are post-processing steps.
6. V3 and block SSB remain separate experiments.
7. Preserve V0/V1/V2 semantics; do not silently rewrite earlier variants.
