# Experiment Roadmap

## Stage 1 — scaling / larger datasets
- Preserve V1 as the primary clean SSB baseline.
- Test larger MLPs and larger datasets to look for a compute-size crossover.
- Keep accuracy, backward time, epoch time, and memory as the primary metrics.

## Stage 2 — comparison baselines
- Dense baseline.
- Standard dropout at matching keep ratios (`dropout p = 1 - keep_ratio`).
- Random static neuron-pruning baseline at matching keep ratios.
- Add a stronger magnitude/structured pruning baseline after the simple control works.
- Be explicit that mask-based dropout/pruning may not produce physical speedups unless the actual compute is structurally reduced.

## Stage 3 — SSB V3 (separate from block SSB)
- Apply one stochastic mask to both forward and backward.
- V3 must avoid computing inactive forward neurons rather than performing a full dense matmul and masking afterward.
- Benchmark forward time in addition to the standard metrics.

## Stage 4 — Block SSB (separate experiment)
- Keep forward dense initially.
- Select contiguous output-neuron blocks during backward.
- Sweep block sizes while keeping the same keep ratio.
- Compare against neuron-level V1.

## Stage 5 — mask persistence
Reuse a selected mask for 2/4/8/16 batches and measure overhead/convergence.

## Stage 6 — gradient-aware stochastic masks
Use prior gradient magnitude/EMA plus random exploration and starvation prevention.

## Stage 7 — adaptive and layer-specific sparsity
Vary keep ratio by layer and over training time.

## Stage 8 — lower-level kernels
Only after profiling establishes a useful algorithm: consider PyTorch compile/Triton/CUDA/block-sparse kernels.
