#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void add_broadcast(
    const float* __restrict__ a, 
    const float* __restrict__ b, 
    const float* __restrict__ c, 
    float* out, 
    int X, 
    int Y, 
    int Z
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;    
    int z = blockIdx.z * blockDim.z + threadIdx.z;    

    if ( x < X && y < Y && z < Z){
        int idx3 = x * (Y * Z) + y * Z + z;
        int idx2 = x * Y + y;
        int idx1 = x;

        out[idx3] = a[idx3] + b[idx2] + c[idx1];
    }
}

int main(){

    int X=8, Y=8, Z=8;

    size_t bytes3 = X * Y * Z * sizeof(float);
    size_t bytes2 = X * Y * sizeof(float);
    size_t bytes1 = X * sizeof(float);

    std::vector<float> h_a(X * Y * Z), h_b(X * Y), h_c(X), h_o(X*Y*Z);

    for (int i = 0; i < X * Y * Z; i++){
        h_a[i] = 1.0f;
    }

    for (int i = 0; i < X * Y; i++){
        h_b[i] = 5.0f;
    }

    for (int i = 0; i < X; i++){
        h_c[i] = 100.0f;
    }

    float *d_a, *d_b, *d_c, *d_o;

    cudaMalloc(&d_a, bytes3);
    cudaMalloc(&d_b, bytes2);
    cudaMalloc(&d_c, bytes1);
    cudaMalloc(&d_o, bytes3);

    cudaMemcpy(d_a, h_a.data(), bytes3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c.data(), bytes1, cudaMemcpyHostToDevice);

    dim3 block(4, 4, 4);
    dim3 grid(
        (X + block.x - 1) / block.x,
        (Y + block.y - 1) / block.y,
        (Z + block.z - 1) / block.z
    );

    add_broadcast<<<grid, block>>>(d_a, d_b, d_c, d_o, X, Y, Z);
    cudaDeviceSynchronize();

    cudaMemcpy(h_o.data(), d_o, bytes3, cudaMemcpyDeviceToHost);

    for (int x = 0; x < X; ++x) {
        std::cout << "x = " << x << ":\n";
        for (int y = 0; y < Y; ++y) {
            for (int z = 0; z < Z; ++z) {
                int idx3 = (x * Y + y) * Z + z;
                std::cout << h_o[idx3] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }


    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_o);


    return 0;
}
