# cuda learning log

learning cuda programming from scratch. tracking what works, what doesn't, and how fast things go.

## vec-add

basic vector addition. cpu vs gpu benchmark with 10M elements.

code: [00_vec_add.cu](vec-add/00_vec_add.cu)

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

---

_work in progress. adding more as i learn._
