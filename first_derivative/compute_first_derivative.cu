#include <iostream>
#include <cuda_runtime.h>

float f(float x)
{
    return 2 * powf(x, 3) - 5 * powf(x, 2) - 9 * x + 2;
}

float f_prime(float x)
{
    return 6 * powf(x, 2) - 10 * x - 9;
}

__global__ void compute_derivative(float *d_f, float *d_f_prime, float delta_x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n - 1)
    {
        d_f_prime[idx] = (d_f[idx + 1] - d_f[idx]) / delta_x;
    }
}

int main() 
{
    int n = 1000;
    float delta_x = 0.001f;

    float *h_f = new float[n];
    float *h_f_prime = new float[n - 1];

    for (int i = 0; i < n; i++)
    {
        float x = i * delta_x;
        h_f[i] = f(x);
    }

    float *d_f;
    float *d_f_prime;
    cudaMalloc(&d_f, n * sizeof(float));
    cudaMalloc(&d_f_prime, (n - 1) * sizeof(float));

    cudaMemcpy(d_f, h_f, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    compute_derivative<<<gridSize, blockSize>>>(d_f, d_f_prime, delta_x, n);
    
    cudaMemcpy(h_f_prime, d_f_prime, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0;
    for(int i = 0; i < n - 1; i++)
    {
        float x = i * delta_x;
        float error = fabs(f_prime(x) - h_f_prime[i]);
        if (error > max_error)
	        max_error = error;
    }
    std::cout << "Max error: " << max_error << std::endl;
    
    cudaFree(d_f);
    cudaFree(d_f_prime);
    delete[] h_f;
    delete[] h_f_prime;

    return 0;
}

