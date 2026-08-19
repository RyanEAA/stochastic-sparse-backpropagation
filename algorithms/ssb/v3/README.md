# SSB V3 — forward + backward stochastic mask (planned)

V3 is intentionally **separate** from block SSB.

Planned definition:
- create a stochastic neuron mask before forward propagation;
- avoid computing inactive neurons in the forward pass (do not merely multiply a dense output by a mask);
- reuse the same mask during backward propagation;
- preserve a clearly defined keep-ratio interpretation;
- benchmark accuracy, forward time, backward time, total epoch time, and memory.

Do not register V3 in `algorithms/ssb/registry.py` until its implementation and tests are complete.
