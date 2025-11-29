---
layout: default
title: cuda
date: 2025-11-12
---

learning cuda programming from scratch. tracking what works, what doesn't, and how fast things go.

### resources:
 - [hpc computing with gpus - elliot arledge](https://www.youtube.com/watch?v=86FAWCzIe_4)
 - [gpu programming - simon oz](https://www.youtube.com/playlist?list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j)

---

### contents

- [vec-add](#nov-11-2025-vec-add) — basic gpu vector addition
- [vec-add 3d](#nov-15-2025-vec-add-3d) — 1d vs 3d kernel layouts
- [matmul](#nov-22-2025-matmul) — naive matrix multiplication
- [add_broadcast](#nov-28-2025-add_broadcast) — broadcasting across tensor ranks

---

## nov 11 2025: vec-add

basic vector addition. cpu vs gpu benchmark with 10M elements.

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

## nov 22 2025: matmul

naive matrix multiplication on gpu. each thread computes one output element.

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

## nov 28 2025: add_broadcast

broadcasting addition across tensors of different shapes. adds a 3d tensor (X,Y,Z) with a 2d tensor (X,Y) and a 1d tensor (X).

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


