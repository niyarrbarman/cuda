---
layout: default
title: cuda learning log
date: 2025-11-12
---

learning cuda programming from scratch. tracking what works, what doesn't, and how fast things go.

resources:
 - [hpc computing with gpus - elliot arledge](https://www.youtube.com/watch?v=86FAWCzIe_4)
 - [gpu programming - simon oz](https://www.youtube.com/playlist?list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j)

---

## nov 11 2025: vec-add

basic vector addition. cpu vs gpu benchmark with 10M elements.

code: [00_vec_add.cu](https://github.com/niyarrbarman/cuda/tree/main/src/vec-add/00_vec_add.cu)

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

what i learned:
- `cudaMalloc` and `cudaMemcpy` for data transfer
- block/thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- `cudaDeviceSynchronize()` to wait for kernel completion
- gpu isn't always faster (memory transfer overhead matters)

---

## nov 15 2025: vec-add 3d

extended vec-add to compare 1d vs 3d kernel layouts. same operation, different thread organization.

code: [01_vec_add_3d.cu](https://github.com/niyarrbarman/cuda/tree/main/src/vec-add/01_vec_add_3d.cu)

implemented:
- 1d kernel with 1024 threads per block
- 3d kernel with 16x8x8 block shape
- verification against cpu results
- speedup calculations

what i learned:
- 1d and 3d kernels can perform the same work
- 3d layout useful when data has spatial structure
- block shape affects occupancy and performance
- always verify gpu results against cpu reference

---

## nov 22 2025: matmul

naive matrix multiplication on gpu. each thread computes one output element.

code: [matmul.cu](https://github.com/niyarrbarman/cuda/tree/main/src/matmul/matmul.cu)

implemented:
- 2d grid and block layout
- row-major indexing for matrices
- each thread does a dot product of row and column

what i learned:
- 2d thread indexing with `blockIdx.y * blockDim.y + threadIdx.y` for rows
- `dim3` for specifying block and grid dimensions
- naive approach is simple but not memory efficient (no tiling yet)

---

## nov 28 2025: add_broadcast

broadcasting addition across tensors of different shapes. adds a 3d tensor (X,Y,Z) with a 2d tensor (X,Y) and a 1d tensor (X).

code: [add_broadcast.cu](https://github.com/niyarrbarman/cuda/tree/main/src/add_broadcast/add_broadcast.cu)

implemented:
- 3d grid and block layout
- index calculation for different tensor ranks
- `__restrict__` pointers for compiler optimization hints

what i learned:
- how broadcasting works: expand smaller tensors to match larger ones
- 3d indexing: `x * (Y * Z) + y * Z + z` for the full tensor
- `dim3` can handle all three dimensions for blocks and grids

