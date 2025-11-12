---
layout: default
title: cuda learning log
date: 2025-11-12
---

learning cuda programming from scratch. tracking what works, what doesn't, and how fast things go.

## nov 11 2025: vec-add

basic vector addition. cpu vs gpu benchmark with 10M elements.

code: [00_vec_add.cu](https://github.com/niyarrbarman/cuda/tree/main/vec-add/00_vec_add.cu)

implemented:
- host and device memory management
- kernel launch with blocks and threads
- timing comparisons (20 iterations each)
- warm-up runs to eliminate cold start bias

build and run:

```bash
cd vec-add
make
make run
```

or:

```bash
make -C vec-add run
```

targets: `build`, `clean`, `run`

what i learned:
- `cudaMalloc` and `cudaMemcpy` for data transfer
- block/thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- `cudaDeviceSynchronize()` to wait for kernel completion
- gpu isn't always faster (memory transfer overhead matters)
