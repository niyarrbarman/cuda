#include <iostream>
#include <cuda_runtime.h>
#include <vector>

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

int main(){
    int n = 4;  
    size_t bytes = n * n * sizeof(float);

    std::vector<float> h_a(n*n), h_b(n*n), h_c(n*n);

    for (int i=0; i<n*n; i++){
        h_a[i] = 1.f;
        h_b[i] = 2.f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    matmul<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);


    std::cout << "\nContents of A (h_a):\n";
    for (int i = 0; i < n*n; i++) {
        if ((i%n)==0){
            std::cout << "\n";
        }
        std::cout << h_a[i] << " ";
    }
    std::cout << "\n";

    std::cout << "\nContents of B (h_b):\n";
    for (int i = 0; i < n*n; i++) {
        if ((i%n)==0){
            std::cout << "\n";
        }
        std::cout << h_b[i] << " ";
    }
    std::cout << "\n";

    std::cout << "\nContents of C (h_c):\n";
    for (int i=0; i<n*n; i++){
        if ((i%n)==0){
            std::cout << "\n";
        }
        std::cout << h_c[i] << " ";
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

