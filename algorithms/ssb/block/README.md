# Block SSB (planned separate experiment)

Block SSB is not V3. It changes **how the sparse subset is selected**.

Planned experiment:
- select contiguous blocks of output neurons rather than arbitrary individual neurons;
- sweep block sizes such as 8, 16, 32, 64, and 128;
- hold keep ratio constant while comparing block sizes;
- initially keep the forward pass dense so this isolates block-structured backward sparsity;
- compare accuracy, backward time, epoch time, and memory against neuron-level SSB.

Later, block selection may be combined with forward sparsity, but only after the separate block experiment is understood.
