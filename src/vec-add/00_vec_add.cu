#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000 
#define BLOCK_SIZE 256

// function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

void vector_add_cpu(float *a, float *b, float*c, int n){
    for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float *a, float* b, float*c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

int main(){
    float *ha, *hb, *hc;    // initialize floats on host
    float *da, *db, *dc;    // initialize floats on device

    size_t size = N * sizeof(float);    // 10M * 4

    ha = (float*)malloc(size);
    hb = (float*)malloc(size);
    hc = (float*)malloc(size);
    
    
    srand(time(NULL));
    init_vector(ha, N);
    init_vector(hb, N);
    
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

    // int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; 
    int num_blocks = 4;

    printf("warm up run\n");
    for (int i=0; i<5; i++){
        vector_add_cpu(ha, hb, hc, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(da, db, dc, N);
        cudaDeviceSynchronize();
    }

    printf("cpu run\n");

    double hostTotalTime = 0.0;

    for (int i=0; i<20; i++){
        double start_time = get_time();
        vector_add_cpu(ha, hb, hc, N);
        double end_time = get_time();
        hostTotalTime += end_time - start_time;
    }
    double hostAvgTime = hostTotalTime / 20;

    printf("gpu run\n");

    double deviceTotalTime = 0.0;

    for (int i=0; i<20; i++){
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(da, db, dc, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        deviceTotalTime += end_time - start_time;
    }
    double deviceAvgTime = deviceTotalTime / 20;

    printf("host avg time = %fms\n", hostAvgTime * 1000);
    printf("device avg time = %fms\n", deviceAvgTime * 1000);

    printf("speedup = %fx\n", hostAvgTime / deviceAvgTime);

    free(ha);
    free(hb);
    free(hc);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    
    return 0;
}