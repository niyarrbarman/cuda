#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s (%s:%d): %d\n", #call, __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.4f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");


void cpu_softmax(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) {
                max_val = input[i * cols + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val);
            sum += output[i * cols + j];
        }

        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum;
        }
    }
}


__global__ void gpu_softmax_naive(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = input[row * cols];
        for (int j = 1; j < cols; j++) {
            if (input[row * cols + j] > max_val) {
                max_val = input[row * cols + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val);
            sum += output[row * cols + j];
        }

        for (int j = 0; j < cols; j++) {
            output[row * cols + j] /= sum;
        }
    }
}


__global__ void gpu_softmax_optimized(float* input, float* output, int rows, int cols) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row >= rows) return;

    float max_val = -INFINITY;

    for (int j = tid; j < cols; j+= blockSize) {
        max_val = fmaxf(max_val, input[row * cols + j]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    for (int stride=blockSize/2; stride>0; stride>>=1){
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    max_val = sdata[0];
    __syncthreads();

    float thrd_sum = 0.0f;
    for (int j = tid; j < cols; j+= blockSize) {
        output[row * cols + j] = expf(input[row * cols + j] - max_val);
        thrd_sum += output[row * cols + j];
    }

    sdata[tid] = thrd_sum;
    __syncthreads();

    for (int stride = blockSize/2; stride > 0; stride >>= 1){
        if (tid < stride){
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float sum = sdata[0];
    __syncthreads();

    for (int j = tid; j < cols; j+= blockSize) {
        output[row * cols + j] /= sum;
    }
}

float random_float(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / (max - min));
}

int main() {
    const int rows = 8192;
    const int cols = 8192;
    const size_t size = rows * cols * sizeof(float);

    srand((unsigned int)time(NULL));

    float* h_input = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    float* h_output_gpu_naive = (float*)malloc(size);
    float* h_output_gpu_opt = (float*)malloc(size);

    printf("Initializing %d x %d matrix with random values...\n", rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = random_float(-10.0f, 10.0f);
    }

    printf("\nInput Matrix (first 5x5 corner):\n");
    for (int i = 0; i < 5 && i < rows; i++) {
        for (int j = 0; j < 5 && j < cols; j++)
            printf("%8.4f ", h_input[i * cols + j]);
        printf("\n");
    }
    printf("\n");

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // host softmax
    clock_t cpu_start = clock();
    cpu_softmax(h_input, h_output_cpu, rows, cols);
    clock_t cpu_end = clock();
    float cpu_time_ms = 1000.0f * (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // naive device softmax
    int blockSizeNaive = 256;
    int numBlocksNaive = (rows + blockSizeNaive - 1) / blockSizeNaive;

    CHECK_CUDA(cudaEventRecord(start));
    gpu_softmax_naive<<<numBlocksNaive, blockSizeNaive>>>(d_input, d_output, rows, cols);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float gpu_naive_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_naive_time_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_output_gpu_naive, d_output, size, cudaMemcpyDeviceToHost));

    float max_diff_naive = 0.0f;
    int mismatch_count_naive = 0;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(h_output_cpu[i] - h_output_gpu_naive[i]);
        if (diff > max_diff_naive) max_diff_naive = diff;
        if (diff > 1e-5) mismatch_count_naive++;
    }

    // optimized device softmax
    int blockSizeOpt = 256;
    int numBlocksOpt = rows;
    size_t sharedMemSize = blockSizeOpt * sizeof(float);

    CHECK_CUDA(cudaEventRecord(start));
    gpu_softmax_optimized<<<numBlocksOpt, blockSizeOpt, sharedMemSize>>>(d_input, d_output, rows, cols);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float gpu_opt_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_opt_time_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_output_gpu_opt, d_output, size, cudaMemcpyDeviceToHost));

    float max_diff_opt = 0.0f;
    int mismatch_count_opt = 0;
    for (int i = 0; i < rows * cols; i++) {
        float diff = fabsf(h_output_cpu[i] - h_output_gpu_opt[i]);
        if (diff > max_diff_opt) max_diff_opt = diff;
        if (diff > 1e-5) mismatch_count_opt++;
    }

    printf("\n==================== Results ====================\n");
    printf("Matrix size: %d x %d\n\n", rows, cols);

    printf("CPU time:                %.3f ms\n", cpu_time_ms);
    printf("GPU Naive time:          %.3f ms (Speedup: %.2fx)\n",
           gpu_naive_time_ms, cpu_time_ms / gpu_naive_time_ms);
    printf("GPU Optimized time:      %.3f ms (Speedup: %.2fx)\n",
           gpu_opt_time_ms, cpu_time_ms / gpu_opt_time_ms);
    printf("\nOptimized vs Naive:      %.2fx faster\n",
           gpu_naive_time_ms / gpu_opt_time_ms);

    printf("\n------------------ Verification ------------------\n");
    printf("Naive GPU - Max diff: %e, Mismatches: %d %s\n",
           max_diff_naive, mismatch_count_naive,
           mismatch_count_naive == 0 ? "(PASS)" : "(FAIL)");
    printf("Optimized GPU - Max diff: %e, Mismatches: %d %s\n",
           max_diff_opt, mismatch_count_opt,
           mismatch_count_opt == 0 ? "(PASS)" : "(FAIL)");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu_naive);
    free(h_output_gpu_opt);

    return 0;
}
