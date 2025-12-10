#include <cuda_device_runtime_api.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// softmax using cuBLAS — combines cuBLAS linear algebra with CUDA kernels for non-linear ops
// why cuBLAS? it's optimized for reductions (max, sum) but we need kernels for exp/division
// approach: per-row softmax with numerical stability (subtract max before exp)

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

#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

float random_float(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / (max - min));
}

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

__global__ void sub_max_exp(const float* input, float* output, float max_val, int length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length){
        float val = input[idx] - max_val;
        output[idx] = expf(val);
    }
}

__global__ void normalize(float* data, float sum, int length){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length){
        data[idx] = data[idx] / sum;
    }
}

void softmax_cuBlas(
    cublasHandle_t handle,
    float* d_input,
    float* d_output,
    int rows,
    int cols
) {
    const int THREADS = 256;
    const int BLOCKS = (cols + THREADS - 1) / THREADS;

    for (int row = 0; row < rows; row++) {
        const float *row_in = d_input + row * cols;
        float *row_out = d_output + row * cols;

        int max_idx = 0;
        CHECK_CUBLAS(cublasIsamax(handle, cols, row_in, 1, &max_idx));
        max_idx -= 1;

        float max_val;
        CHECK_CUDA(cudaMemcpy(&max_val, row_in + max_idx, sizeof(float), cudaMemcpyDeviceToHost));

        sub_max_exp<<<BLOCKS, THREADS>>>(row_in, row_out, max_val, cols);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        float sum_exp = 0.0f;
        CHECK_CUBLAS(cublasSasum(handle, cols, row_out, 1, &sum_exp));

        normalize<<<BLOCKS, THREADS>>>(row_out, sum_exp, cols);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}



int main() {

    const int rows = 128;
    const int cols = 128;
    const size_t size = rows * cols * sizeof(float);

    srand((unsigned int)time(NULL));

    float *h_input      = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);

    printf("Initializing %d x %d matrix with random values...\n", rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = random_float(-10.0f, 10.0f);
    }

    float *d_input  = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_input, size));
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    softmax_cuBlas(handle, d_input, d_output, rows, cols);
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));

    cpu_softmax(h_input, h_output_cpu, rows, cols);

    printf("Verifying results...\n");
    bool correct = true;

    for (int i = 0; i < rows * cols; ++i) {
        float diff = fabsf(h_output_cpu[i] - h_output_gpu[i]);
        if (diff > 1e-5f) {
            printf("Mismatch at %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n",
                   i, h_output_cpu[i], h_output_gpu[i], diff);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("Results match! Softmax implementation is correct.\n");
    }

    float sum_row = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum_row += h_output_gpu[col];
    }
    printf("Sum of first row softmax (GPU): %.6f\n", sum_row);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
