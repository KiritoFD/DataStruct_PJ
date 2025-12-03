# Ablation Experiment Instructions

This file describes how to run the ablation experiments using the `MySolution` implementation in this repository. The code has been modified to support runtime toggling of several fundamental optimizations.

## Overview
We added runtime ablation flags to `MySolution.cpp` with the following effects:
- `ABLATE_CSR`: If enabled, skip using the `FlatHNSW` (CSR read-only index) and use `SimpleHNSW` dynamic structure at query time.
- `ABLATE_PREFETCH`: If enabled, disable the software prefetches (calls to `_mm_prefetch` are ignored).
- `ABLATE_SIMD`: If enabled, force the scalar (non-SIMD) L2 distance computation.
- `ABLATE_PRUNING`: If enabled, disable the heuristic (diversity) pruning — use plain top-M neighbor selection.
- `ABLATE_HEAP`: If enabled, use `std::priority_queue` instead of sorted `vector` insertion in `SimpleHNSW::searchLayer` (construction-time variant).

All toggles are runtime flags controlled by the newly added functions:
- `set_ablation_flags(int csr, int prefetch, int simd, int pruning, int heap)`
- `get_ablation_flags(int* csr, int* prefetch, int* simd, int* pruning, int* heap)`
- Convenience functions:
  - `set_ablate_csr(int on)`
  - `set_ablate_prefetch(int on)`
  - `set_ablate_simd(int on)`
  - `set_ablate_pruning(int on)`
  - `set_ablate_heap(int on)`

The header `MySolution.h` declares these functions so they can be called by external test harness (C/C++ or FFI).

## Mapping to Experiments
- Experiment 1: CSR vs Pointer Graph
  - Baseline: `ABLATE_CSR = 0` (use `FlatHNSW`)
  - Variant: `ABLATE_CSR = 1` (use `SimpleHNSW` dynamic search)

- Experiment 2: Prefetching
  - Baseline: `ABLATE_PREFETCH = 0`
  - Variant: `ABLATE_PREFETCH = 1` (software prefetch suppressed)

- Experiment 3: SIMD vs Scalar
  - Baseline: `ABLATE_SIMD = 0` (SIMD distance functions used when available)
  - Variant: `ABLATE_SIMD = 1` (force scalar L2 distance computation)

- Experiment 4: Heuristic Pruning
  - Baseline: `ABLATE_PRUNING = 0` (full RNG-style selection for neighbors)
  - Variant: `ABLATE_PRUNING = 1` (pick top-M nearest only)

- Experiment 5: Vector insertion vs Heap at construction
  - Baseline: `ABLATE_HEAP = 0` (vector insertion)
  - Variant: `ABLATE_HEAP = 1` (priority_queue variant)

## Recommended Experiment Steps
1. Build/Build-From-Memory
   - Call `set_ablation_flags(...)` before building, if you want to ensure the ablation condition applies to build (e.g., `ABLATE_CSR` should be set before building so we choose dynamic vs flat behavior at build time).
   - Example: `set_ablation_flags(1 /*csr*/, 0, 0, 0, 0);` to test pointer graph.
   - Call `build_hnsw(d, base)` to build.

2. Warm-up / Single Query Bench
   - Run a few warm-up queries to stabilize caches:
     - `search_hnsw(query, k)`

3. Performance Collection
   - QPS & Latency: Run a large batch of queries (e.g., 10k) and measure total time & compute P99 latency.
   - Recall: Use ground-truth NN to compute recall@K.
   - Build Time: Measured by `get_last_build_time_ms`.
   - Memory: Monitor RSS peak using OS tools (e.g., `ps`, `top`, Windows Task Manager).

4. Compare Baseline vs Variant
   - Keep all parameters equal (same `M`, `efConstruction`, `efSearch`) except the ablation flag.
   - Keep random seed and the dataset identical.

## Examples
- Disable only CSR (use dynamic view):

    set_ablation_flags(1, 0, 0, 0, 0);
    build_hnsw(d, base);
    run queries...

- Remove both prefetching and SIMD to observe combined effect:

    set_ablation_flags(0, 1, 1, 0, 0);
    build_hnsw(d, base);
    run queries...

## Tips for Reliable Measures
- Run each configuration multiple times and average results.
- Use a fixed set of queries.
- Pin process to a dedicated CPU if possible to reduce noise.
- For P99 latency measurements, gather timestamps per query and compute percentiles.

---

If you want, I can also add a small C++ test harness to run automated experiments (e.g., run Baseline + 5 Variants automatically with logging to CSV). Would you like me to add that as well?

I added a script `run_ablation.sh` to automate compilation and running ablation experiments. Example usage:

```bash
# default run
./run_ablation.sh

# run with repeated runs per variant
REPEATS=3 ./run_ablation.sh

# override dataset and param values
BASE_FILE=data_o/glove/base.txt QUERY_FILE=data_o/glove/query.txt ./run_ablation.sh
```

The script will compile the binary, run the baseline & each ablation variant (configurable inside the script), save logs into `results/ablation_<timestamp>/*` and append parsed metrics to `results/ablation_<timestamp>/results.csv`.