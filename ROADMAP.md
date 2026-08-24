# Experiment Roadmap

## Stage 1 — scaling / larger datasets
- Preserve V1 as the primary clean backward-only SSB baseline.
- Test larger MLPs and larger datasets to look for a compute-size crossover.
- Add CNN workloads without changing the SSB rule: the initial CNN experiment uses a shared dense convolutional backbone and applies SSB only to classifier linear layers.
- Keep accuracy, forward/backward time, epoch time, and memory as primary systems metrics.

## Stage 2 — comparison baselines
- Dense baseline.
- Standard dropout at matching keep ratios (`dropout p = 1 - keep_ratio`).
- Random static neuron-pruning baseline at matching keep ratios.
- Add a stronger magnitude/structured pruning baseline after the simple control works.
- Be explicit that mask-based dropout/pruning may not produce physical speedups unless the actual compute is structurally reduced.

## Stage 3 — SSB V3
- Implemented as `ssb-v3`.
- One stochastic output-neuron mask is shared by forward and backward for each layer invocation.
- Only active output neurons are multiplied in the training forward pass; inactive neurons are scattered as zeros into the full-width activation tensor.
- Evaluation remains dense.
- Benchmark forward time explicitly in addition to backward/epoch time and memory.
- Current model replacement policy includes the final classifier layer, matching earlier SSB variants. A hidden-layers-only policy should be a separately named protocol if tested.

## Stage 4 — Block SSB
- Implemented as `ssb-v1-block`, `ssb-v2-block`, and `ssb-v3-block`.
- V1-block and V2-block keep the forward dense; V3-block performs block-sparse forward and backward using the same selected blocks.
- Output neurons are grouped into contiguous aligned blocks; each block is independently active with probability `keep_ratio`.
- Thus the requested neuron keep ratio is preserved in expectation, matching V1's Bernoulli interpretation while introducing block correlation.
- Sweep block sizes such as 8/16/32/64/128 at fixed keep ratios and compare each block variant against its neuron-level counterpart.

## Stage 5 — mask persistence
Reuse a selected mask for 2/4/8/16 batches and measure overhead/convergence.

## Stage 6 — gradient-aware stochastic masks
Use prior gradient magnitude/EMA plus random exploration and starvation prevention.

## Stage 7 — adaptive and layer-specific sparsity
Vary keep ratio by layer and over training time.

## Stage 8 — lower-level kernels
Only after profiling establishes a useful algorithm: consider `torch.compile`, Triton, CUDA/C++, or block-sparse kernels. Cython is not expected to remove the dominant tensor-kernel/indexing overhead.

## Separate future CNN algorithm experiment
The current CNN architecture intentionally keeps convolution dense. If results justify it, add channel-sparse SSB convolution as a new algorithm family rather than silently treating dense Conv2d + sparse classifier as sparse convolution.
