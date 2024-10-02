#include <iostream>
#include <cuda_runtime.h>

float f(float x, float y)
{
    return 2 * powf(x, 3) * y - 5 * powf(x, 2) * powf(y, 2) - 9 * x * powf(y, 3) + 1;
}

float f_prime_x(float x, float y)
{
    return 6 * powf(x, 2) * y - 10 * x * powf(y, 2) - 9 * powf(y, 3);
}

float f_prime_y(float x, float y)
{
    return 2 * powf(x, 3) - 10 * powf(x, 2) * y - 27 * x * powf(y, 2);
}

__global__ void compute_partial_derivatives(float *d_f, float *d_f_prime_x, float *d_f_prime_y, float delta_x, float delta_y, int nx, int ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nx - 1 && idy < ny - 1)
    {
        int index = idy * nx + idx;

        int idx_x_forward = idy * nx + (idx + 1);
        d_f_prime_x[index] = (d_f[idx_x_forward] - d_f[index]) / delta_x;

        int idx_y_forward = (idy + 1) * nx + idx;
        d_f_prime_y[index] = (d_f[idx_y_forward] - d_f[index]) / delta_y;
    }
}

int main()
{
    int nx = 2000;
    int ny = 500;
    float delta_x = 0.001f;
    float delta_y = 0.001f;

    int num_elements = nx * ny;

    float *h_f = new float[num_elements];
    float *h_f_prime_x = new float[num_elements - 1];
    float *h_f_prime_y = new float[num_elements -1];

    for (int j = 0; j < ny; ++j)
    {
        float y = j * delta_y;
        for (int i = 0; i < nx; ++i)
        {
            float x = i * delta_x;
            int idx = j * nx + i;
            h_f[idx] = f(x, y);
        }
    }

    float *d_f;
    float *d_f_prime_x;
    float *d_f_prime_y;
    cudaMalloc(&d_f, num_elements * sizeof(float));
    cudaMalloc(&d_f_prime_x, (num_elements - 1) * sizeof(float));
    cudaMalloc(&d_f_prime_y, (num_elements - 1) * sizeof(float));

    cudaMemcpy(d_f, h_f, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                  (ny + blockSize.y - 1) / blockSize.y);

    compute_partial_derivatives<<<gridSize, blockSize>>>(d_f, d_f_prime_x, d_f_prime_y, delta_x, delta_y, nx, ny);

    cudaMemcpy(h_f_prime_x, d_f_prime_x, (num_elements - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_prime_y, d_f_prime_y, (num_elements - 1) * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0;
    for (int j = 0; j < ny - 1; ++j)
    {
        float y = j * delta_y;
        for (int i = 0; i < nx - 1; ++i)
        {
            float x = i * delta_x;
            int idx = j * nx + i;
            float error_x = fabs(f_prime_x(x, y) - h_f_prime_x[idx]);
            if (error_x > max_error)
                max_error = error_x;
            float error_y = fabs(f_prime_y(x, y) - h_f_prime_y[idx]);
            if (error_y > max_error)
                max_error = error_y;
        }
    }
    std::cout << "Max error: " << max_error << std::endl;

    cudaFree(d_f);
    cudaFree(d_f_prime_x);
    cudaFree(d_f_prime_y);
    delete[] h_f;
    delete[] h_f_prime_x;
    delete[] h_f_prime_y;

    return 0;
}