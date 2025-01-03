#include <cuda_runtime.h>
#include <math.h>

__global__ void choleskyKernel(double* A, double* L, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    L[k * n + k] = sqrt(A[k * n + k]);
    for (int i = k + 1; i < n; i++) {
        L[i * n + k] = A[i * n + k] / L[k * n + k];
    }
    for (int i = k + 1; i < n; i++) {
        for (int j = k + 1; j <= i; j++) {
            A[i * n + j] -= L[i * n + k] * L[j * n + k];
        }
    }
}

extern "C" void choleskyDecompositionCUDA(double* A, double* L, int n) {
    double* d_A;
    double* d_L;
    size_t size = n * n * sizeof(double);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_L, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    choleskyKernel<<<numBlocks, blockSize>>>(d_A, d_L, n);

    cudaMemcpy(L, d_L, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_L);
}