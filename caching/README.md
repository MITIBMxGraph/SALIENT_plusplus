This directory contains modules for evaluating communication volume with
different remote-vertex caching schemes during GNN computations with node-wise
sampling.

- `vip.py`: Functions for estimating vertex inclusion probability (VIP) weights
  for GNN neighborhood expansion, and for identifying remote vertices to be
  cached based on the VIP weights.
  
- `experiment_communication_caching.py`: Driver script for running a set of
  simulation experiments to measure the inter-partition communication volume
  during GNN training with different caching parameters.

- `parse_communication_volume_results.py`: Utility script for printing the
  `.pobj` file output of `run_cache_simulation_experiments.py` as a CSV table.

- `util.py`: Utility functions used by the above modules.

For instructions on how to exercise the SALIENT++ artifact to run simulated
communication-volume experiments, see [the artifact repo README](../README.md).
