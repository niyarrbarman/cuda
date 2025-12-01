---
layout: default
title: cuda
date: 2025-11-12
---

learning cuda (and now triton) from scratch. tracking what works, what doesn't, and how fast things go.

### resources:

<ul class="resource-list">
    <li>
        <a href="https://www.youtube.com/watch?v=86FAWCzIe_4">
            hpc computing with gpus — elliot arledge
        </a>
    </li>
    <li>
        <a href="https://www.youtube.com/playlist?list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j">
            gpu programming — simon oz
        </a>
    </li>
    <li>
        <a href="https://www.youtube.com/watch?v=DdTsX6DQk24">
            practitioner's guide to triton — umer h adil
        </a>
    </li>
</ul>

---

### contents

<div class="toc">
    <!-- <div class="toc-label">entries so far</div> -->
    <ul class="toc-list">
        <li><a href="#nov-11-2025-vec-add">vec-add</a><span class="toc-note">basic gpu vector addition</span></li>
        <li><a href="#nov-15-2025-vec-add-3d">vec-add 3d</a><span class="toc-note">1d vs 3d kernel layouts</span></li>
        <li><a href="#nov-16-2025-matmul">matmul</a><span class="toc-note">naive matrix multiplication</span></li>
        <li><a href="#nov-22-2025-triton-add-and-softmax">triton</a><span class="toc-note">writing gpu kernels in python</span></li>
        <li><a href="#nov-28-2025-add-broadcast">add_broadcast</a><span class="toc-note">broadcasting across tensor ranks</span></li>
        <li><a href="#nov-29-2025-nvtx-profiling">nvtx profiling</a><span class="toc-note">instrumenting cuda with nsight systems</span></li>
        <li><a href="#nov-30-2025-cublas-matmul">cublas matmul</a><span class="toc-note">sgemm and hgemm with cublas</span></li>
    </ul>
</div>

---

## nov 11 2025: vec-add

basic vector addition. cpu vs gpu benchmark with 10M elements.

this was day 1 stuff. i just wanted a kernel that ran end-to-end: allocate, copy, launch, time it, and see numbers move. most of the work here was convincing myself that the indexing math was actually correct and i wasn’t silently writing out of bounds.

code: [00_vec_add.cu](https://github.com/niyarrbarman/cuda/tree/main/src/vec-add/00_vec_add.cu)

the kernel is simple. each thread handles one element:

```cuda
__global__ void vector_add_gpu(float *a, float* b, float*c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}
```

the global thread index formula:

<p class="math">i = blockIdx.x × blockDim.x + threadIdx.x</p>

the `if` guard prevents out-of-bounds access when N isn't a perfect multiple of block size.

memory setup follows host → device → compute → device → host pattern:

```cuda
// allocate on device
cudaMalloc(&da, size);
cudaMalloc(&db, size);
cudaMalloc(&dc, size);

// copy input to device
cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

// launch kernel
int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(da, db, dc, N);
cudaDeviceSynchronize();
```

the `<<<num_blocks, BLOCK_SIZE>>>` syntax is cuda's way of specifying grid and block dimensions. number of blocks needed:

<p class="math">num_blocks = ⌈N / BLOCK_SIZE⌉</p>

`cudaDeviceSynchronize()` waits for the kernel to finish before timing.

implemented:
- host and device memory management
- kernel launch with blocks and threads
- timing comparisons (20 iterations each)
- warm-up runs to eliminate cold start bias

what i learned:
- `cudaMalloc` and `cudaMemcpy` for data transfer
- block/thread indexing: `blockIdx.x * blockDim.x + threadIdx.x`
- `cudaDeviceSynchronize()` to wait for kernel completion
- gpu isn't always faster (memory transfer overhead matters)

---

## nov 15 2025: vec-add 3d

extended vec-add to compare 1d vs 3d kernel layouts. same operation, different thread organization.

this was me coming back to the same problem but forcing it into a 3d shape. the goal was less “go faster” and more “do i really understand `dim3` and 3d indexing, or am i faking it?”. treating a flat array as (nx, ny, nz) made the indexing formulas click.

code: [01_vec_add_3d.cu](https://github.com/niyarrbarman/cuda/tree/main/src/vec-add/01_vec_add_3d.cu)

1d kernel is straightforward. one dimension, linear indexing:

```cuda
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

3d kernel treats the flat array as a 3d volume. each thread gets (i, j, k) coordinates. the linearized index:

<p class="math">idx = i + j·nₓ + k·nₓ·nᵧ</p>

```cuda
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}
```

launching 3d requires `dim3` for both blocks and grid:

```cuda
dim3 blockSize3D(16, 8, 8);  // 16*8*8 = 1024 threads per block
dim3 num_blocks_3d(
    (nx + blockSize3D.x - 1) / blockSize3D.x,
    (ny + blockSize3D.y - 1) / blockSize3D.y,
    (nz + blockSize3D.z - 1) / blockSize3D.z
);
vector_add_gpu_3d<<<num_blocks_3d, blockSize3D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
```

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

## nov 16 2025: matmul

naive matrix multiplication on gpu. each thread computes one output element.

after vec-add, matmul felt like the natural next thing. it shows up everywhere, and you can’t hide from the O(n³) cost. this is where i started thinking harder about memory access patterns, even though this version is still the simple, non-tiled one.

code: [matmul.cu](https://github.com/niyarrbarman/cuda/tree/main/src/matmul/matmul.cu)

each thread computes one element of the output matrix. row from A, column from B, dot product:

```cuda
__global__ void matmul(float* a, float* b, float* c, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n){
        float prod = 0.f;
        for (int i = 0; i < n; i++){
            prod += a[row * n + i] * b[i * n + col];
        }
        c[row*n + col] = prod;
    }
}
```

`row * n + i` walks along row of A. `i * n + col` walks down column of B. 

matrix multiplication: C = A × B where each element:

<p class="math">Cᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ</p>

complexity is O(n³) — each of n² threads does n multiply-adds. simple but not optimal — lots of global memory reads.

2d grid setup maps naturally to matrix structure:

```cuda
dim3 threadsPerBlock(16, 16);  // 256 threads per block
dim3 numBlocks(
    (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (n + threadsPerBlock.y - 1) / threadsPerBlock.y
);
matmul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
```

implemented:
- 2d grid and block layout
- row-major indexing for matrices
- each thread does a dot product of row and column

what i learned:
- 2d thread indexing with `blockIdx.y * blockDim.y + threadIdx.y` for rows
- `dim3` for specifying block and grid dimensions
- naive approach is simple but not memory efficient (no tiling yet)

---

## nov 22 2025: triton add and softmax

tried triton for the first time. writing gpu kernels in python feels weirdly nice.

this is where i temporarily stepped away from cuda c++ and rewrote the same ideas in triton. add first, then softmax. it was fun to see the same indexing and stability tricks (subtract max) but with nicer ergonomics and masking instead of manual `if (idx < n)` guards.

code: [add.py](https://github.com/niyarrbarman/cuda/tree/main/src/triton/add.py), [softmax.py](https://github.com/niyarrbarman/cuda/tree/main/src/triton/softmax.py)

triton add kernel looks like this:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
```

and the softmax kernel:

```python
@triton.jit
def _softmax_kernel(
    out_ptr,
    stride_out,
    x_ptr,
    stride_x,
    cols,
    block_size: tl.constexpr,
    num_warps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + (row_idx * stride_x)
    col_offsets = tl.arange(0, block_size)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < cols
    x = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    x_max = tl.max(x, axis=0)
    safe_x = x - x_max
    numerator = tl.exp(safe_x)
    denominator = tl.sum(numerator, axis=0)
    softmax = numerator / denominator
    tl.store(out_ptr + col_offsets + row_idx * stride_out, softmax, mask=mask)
```

what i learned:
- triton kernels feel like a higher-level version of the cuda grid/block mental model: `program_id` replaces manual `blockIdx * blockDim + threadIdx` math.
- masking is nice. no explicit `if (idx < n)` branches, just `mask` on loads/stores.
- writing softmax in triton made the usual "subtract max for stability" pattern really clear.

---

## nov 28 2025: add_broadcast

broadcasting addition across tensors of different shapes. adds a 3d tensor (X,Y,Z) with a 2d tensor (X,Y) and a 1d tensor (X).

this one was me coming back again to “shapes i actually see in pytorch”: 3d + 2d + 1d. i had to slow down, draw the shapes on paper, and write out the index formulas by hand before i trusted the kernel. it tied together the 3d indexing from earlier with the kind of broadcasting semantics i use all the time on the python side.

code: [add_broadcast.cu](https://github.com/niyarrbarman/cuda/tree/main/src/add_broadcast/add_broadcast.cu)

the trick is computing the right index for each tensor rank. for tensors with shapes (X, Y, Z), (X, Y), and (X):

<p class="math">idx₃ = x·(Y·Z) + y·Z + z</p>
<p class="math">idx₂ = x·Y + y</p>
<p class="math">idx₁ = x</p>

```cuda
__global__ void add_broadcast(
    const float* __restrict__ a,  // shape (X, Y, Z)
    const float* __restrict__ b,  // shape (X, Y)
    const float* __restrict__ c,  // shape (X,)
    float* out, 
    int X, int Y, int Z
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int z = blockIdx.z * blockDim.z + threadIdx.z;    

    if (x < X && y < Y && z < Z){
        int idx3 = x * (Y * Z) + y * Z + z;  // 3d index
        int idx2 = x * Y + y;                 // 2d index (ignores z)
        int idx1 = x;                         // 1d index (ignores y, z)

        out[idx3] = a[idx3] + b[idx2] + c[idx1];
    }
}
```

`__restrict__` tells the compiler these pointers don't alias, enabling better optimization.

3d grid launch to match the tensor shape:

```cuda
dim3 block(4, 4, 4);  // 64 threads per block
dim3 grid(
    (X + block.x - 1) / block.x,
    (Y + block.y - 1) / block.y,
    (Z + block.z - 1) / block.z
);
add_broadcast<<<grid, block>>>(d_a, d_b, d_c, d_o, X, Y, Z);
```

implemented:
- 3d grid and block layout
- index calculation for different tensor ranks
- `__restrict__` pointers for compiler optimization hints

what i learned:
- how broadcasting works: expand smaller tensors to match larger ones
- 3d indexing: `x * (Y * Z) + y * Z + z` for the full tensor
- `dim3` can handle all three dimensions for blocks and grids

---

## nov 29 2025: nvtx profiling

instrumenting cuda code with nvtx markers and profiling with nsight systems.

wanted to actually see where time goes in my matmul code. not just "gpu is fast" but "how much is allocation vs copy vs compute". nvtx lets you wrap sections of code with named ranges, then nsys picks them up and shows you a breakdown.

code: [nvtx_matmul.cu](https://github.com/niyarrbarman/cuda/tree/main/src/matmul/nvtx_matmul.cu)

the idea is simple: push a named range before a section, pop it after. nested ranges work too.

```cuda
#include <nvtx3/nvToolsExt.h>

void matrixMul(float* A, float* B, float* C, int N) {
    nvtxRangePush("Matrix Multiplication");
    
    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();  // end Matrix Multiplication
}
```

compile with nvtx library linked:

```bash
nvcc -o nvtx_matmul nvtx_matmul.cu -lnvToolsExt -lineinfo
```

then profile with nsys:

```bash
nsys profile --trace=cuda,nvtx --stats=true ./nvtx_matmul
```

the output shows exactly where time goes:

```
 Time (%)  Total Time (ns)  Instances   Avg (ns)         Range         
 --------  ---------------  ---------  -------------  ----------------------
     50.0      125,366,415          1  125,366,415.0  Matrix Multiplication
     50.0      125,357,659          1  125,357,659.0  Memory Allocation    
      0.0            3,652          1        3,652.0  Kernel Execution     
      0.0            3,330          1        3,330.0  Memory Copy H2D      
      0.0              140          1          140.0  Memory Deallocation  
      0.0              110          1          110.0  Memory Copy D2H      
```

almost all the time (~125ms) is in memory allocation. the first `cudaMalloc` triggers cuda context initialization — that's the cold start cost. actual kernel execution is just ~3.6µs.

implemented:
- nvtx range markers around each phase
- nested ranges (outer "Matrix Multiplication" contains inner phases)
- nsys profiling with `--trace=cuda,nvtx`

what i learned:
- `nvtxRangePush` / `nvtxRangePop` for instrumenting code sections
- first cuda call pays context initialization cost
- actual compute is often tiny compared to setup overhead
- profiling reveals where optimization effort should go

---

## nov 30 2025: cublas matmul

matrix multiplication using cublas instead of writing my own kernel. sgemm (float32) and hgemm (float16).

after writing naive matmul by hand, i wanted to see what the "real" way looks like. cublas is nvidia's optimized blas library — it's what pytorch uses under the hood. the tricky part is cublas assumes column-major (fortran style), but i store matrices row-major (c style). there's a neat trick to avoid transposing.

code: [matmul_cublas.cu](https://github.com/niyarrbarman/cuda/tree/main/src/cuBLAS/matmul_cublas.cu)

gemm computes:

<p class="math">C = αAB + βC</p>

with α = 1, β = 0 it's just C = AB.

the column-major workaround: instead of computing C = A × B directly, we compute C^T = B^T × A^T. since:

<p class="math">(AB)ᵀ = BᵀAᵀ</p>

and our row-major matrices look like their transposes to cublas, passing B first then A gives us the right answer in row-major C:

```cuda
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,           // dimensions
            &alpha,
            d_B, N,            // B first (!)
            d_A, K,            // then A
            &beta,
            d_C, N);           // output
```

the weird argument order (B before A) is the trick. leading dimensions (N, K, N) match the "inner" dimension of each matrix when viewed column-major.

hgemm is the same but with half precision. uses tensor cores on newer gpus. need to convert float32 inputs to fp16 first:

```cuda
// convert to half precision
half A_h[M * K], B_h[K * N];
for (int i = 0; i < M * K; i++) {
    A_h[i] = __float2half(A[i]);
}

// hgemm call — same pattern as sgemm
__half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha_h,
            d_B_h, N, d_A_h, K,
            &beta_h, d_C_h, N);

// convert result back
for (int i = 0; i < M * N; i++) {
    C_cublas_h[i] = __half2float(C_h[i]);
}
```

implemented:
- sgemm for float32 matmul
- hgemm for float16 matmul (tensor core eligible)
- column-major workaround by swapping A and B
- fp16 conversion with `__float2half` / `__half2float`

what i learned:
- cublas is column-major (fortran), not row-major (c)
- the transpose trick: pass B then A to get row-major output
- hgemm uses half precision — can leverage tensor cores
- `cuda_fp16.h` for half-precision intrinsics
- always wrap cublas/cuda calls in error-checking macros

---

