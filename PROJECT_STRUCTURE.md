# Project Structure

```text
stochastic_sparse_backprop_research/
├── algorithms/
│   └── ssb/
│       ├── registry.py
│       ├── v0/                  # historical instrumented backward-only SSB
│       ├── v1/                  # clean backward-only SSB
│       ├── v2/                  # inverse-probability-scaled backward-only SSB
│       ├── v3/                  # forward + backward neuron sparsity
│       └── block/               # block-structured backward sparsity
├── data/
│   ├── dataset_utils.py
│   └── loaders.py
├── models/
│   ├── common/
│   │   ├── mlp.py               # generic MLP components
│   │   ├── cnn.py               # generic dense-conv/classifier components
│   │   └── builders.py          # architecture selection only
│   ├── mnist/
│   │   ├── config.py            # MLP + CNN dataset architecture configuration
│   │   ├── dense/model.py
│   │   ├── sparse/model.py
│   │   ├── dropout/model.py
│   │   └── pruning/model.py
│   ├── fashion_mnist/
│   ├── kmnist/
│   ├── cifar10/
│   ├── cifar100/
│   └── svhn/
├── training/
│   └── runtime.py
├── results/
├── train.py                     # one experiment only
├── master_train.py              # grid orchestration only
├── summarize_results.py         # post-run means/std
├── visualize_results.py         # post-run plots
├── smoke_test.py
├── ROADMAP.md
├── PROJECT_STRUCTURE.md
└── README.md
```

## Rules
1. Dataset architecture configuration belongs under `models/<dataset>/`.
2. SSB algorithm mechanics belong under `algorithms/ssb/<variant>/`, never inside dataset model files.
3. `train.py` trains one configuration only.
4. `master_train.py` schedules configurations only; it does not implement models or calculate statistics.
5. Summary statistics and plots are post-processing steps.
6. V3 and block SSB remain separate experiments and separate registry names.
7. Preserve V0/V1/V2 semantics; do not silently rewrite earlier variants.
8. CNN experiments currently use dense convolution and SSB classifier layers only. Sparse Conv2d/channel SSB requires a separately defined algorithm.
9. New raw runs are separated by architecture and protocol under `results/<dataset>/<model>/...`; historical layouts remain readable by recursive summarization.
