#include <iostream>
#include <cmath>

__device__ float f(float x)
{
    return pow(x, 2) + 4 * x + 2;
}

float F(float x)
{
    return (1.0/3.0) * pow(x, 3) + 2 * pow(x, 2) + 2 * x;
}

__global__ void integrate(float a, float b, float *result, int n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    float dx = (b - a) / n;
    float local_sum = 0.0;

    for (int i = idx; i < n; i += stride)
    {
        float x = a + i * dx;
        local_sum += f(x) * dx;
    }

    atomicAdd(result, local_sum);
}

int main()
{
    float a = 0.0f;
    float b = 2.0f;
    int n = 1000000;

    float h_result = 0.0f;
    float *d_result;

    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    integrate<<<numBlocks, blockSize>>>(a, b, d_result, n);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    std::cout << "The numerically computed integral result is : " << h_result << std::endl;
    std::cout << "The analitically computed integral result is: " << F(b) - F(a) << std::endl;

    return 0;
}
