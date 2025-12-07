#include <climits>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N 10000000  // 10M
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

// cpu addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// 1d addition cuda kernel
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // one add, one multiply, one store
    if (i < n) {
        c[i] = a[i] + b[i];
        // one add, one store
    }
}

// 3d addition cuda kernel
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

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;

    size_t size = N * sizeof(float);

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c_cpu = (float *)malloc(size);
    h_c_gpu_1d = (float *)malloc(size);
    h_c_gpu_3d = (float *)malloc(size);
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c_1d, size);
    cudaMalloc((void **)&d_c_3d, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int numBlocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    int nx = 100;
    int ny = 100;
    int nz = N / (nx * ny);

    dim3 blockSize3D(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + blockSize3D.x - 1) / blockSize3D.x,
        (ny + blockSize3D.y - 1) / blockSize3D.y,
        (nz + blockSize3D.z - 1) / blockSize3D.z
    );

    printf("warm up run\n");
    for (int i = 0; i < 10; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<numBlocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, blockSize3D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    printf("cpu run\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total_time += (end - start);
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("gpu 1d run\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start = get_time();
        vector_add_gpu_1d<<<numBlocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_1d_total_time += (end - start);
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 20.0;

    // Verify 1D results immediately
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_1d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4) {
            correct_1d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }
    printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

    printf("gpu 3d run\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, blockSize3D>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_3d_total_time += (end - start);
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 20.0;


    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4) {
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}
