#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename Type>
__global__ void negativeKernel(const Type* input, Type* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

template <typename Type>
__global__ void matrixMultKernel(const Type* A, const Type* B, Type* C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        Type sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

template <typename Type>
__global__ void elementWiseMultKernel(const Type* A, const Type* B, Type* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

template <typename Type>
__global__ void addKernel(const Type* A, const Type* B, Type* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

template <typename Type>
__global__ void scalarMultKernel(const Type* A, Type scalar, Type* C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * scalar;
    }
}

template <typename Type>
__global__ void transposeKernel(const Type* input, Type* output, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

template <typename Type>
class CudaMat2D
{
public:
    size_t _cols, _rows;
    Type* d_vals;

    CudaMat2D(size_t cols, size_t rows) : _cols(cols), _rows(rows), d_vals(nullptr)
    {
        cudaCheckError(cudaMalloc(&d_vals, rows * cols * sizeof(Type)));
        cudaCheckError(cudaMemset(d_vals, 0, rows * cols * sizeof(Type)));
    }

    CudaMat2D() : _cols(0), _rows(0), d_vals(nullptr) {}

    ~CudaMat2D()
    {
        if (d_vals) cudaFree(d_vals);
    }

    void copyToHost(Type* h_vals) const
    {
        cudaCheckError(cudaMemcpy(h_vals, d_vals, _rows * _cols * sizeof(Type), cudaMemcpyDeviceToHost));
    }

    void copyToDevice(const Type* h_vals)
    {
        cudaCheckError(cudaMemcpy(d_vals, h_vals, _rows * _cols * sizeof(Type), cudaMemcpyHostToDevice));
    }

    bool isSquare() const
    {
        return _rows == _cols;
    }

    CudaMat2D negative() const
    {
        CudaMat2D output(_cols, _rows);
        int size = _rows * _cols;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        negativeKernel<<<numBlocks, blockSize>>>(d_vals, output.d_vals, size);
        cudaCheckError(cudaDeviceSynchronize());

        return output;
    }

    CudaMat2D mult(const CudaMat2D& target) const
    {
        assert(_cols == target._rows);
        CudaMat2D output(target._cols, _rows);

        dim3 blockSize(16, 16);
        dim3 gridSize((target._cols + blockSize.x - 1) / blockSize.x, 
                      (_rows + blockSize.y - 1) / blockSize.y);

        matrixMultKernel<<<gridSize, blockSize>>>(d_vals, target.d_vals, output.d_vals, 
                                                  _rows, _cols, target._cols);
        
        cudaCheckError(cudaDeviceSynchronize());
        return output;
    }

    CudaMat2D multElem(const CudaMat2D& target) const
    {
        assert(_rows == target._rows && _cols == target._cols);
        CudaMat2D output(_cols, _rows);

        int size = _rows * _cols;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        elementWiseMultKernel<<<numBlocks, blockSize>>>(d_vals, target.d_vals, output.d_vals, size);
        cudaCheckError(cudaDeviceSynchronize());

        return output;
    }

    CudaMat2D add(const CudaMat2D& target) const
    {
        assert(_rows == target._rows && _cols == target._cols);
        CudaMat2D output(_cols, _rows);

        int size = _rows * _cols;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        addKernel<<<numBlocks, blockSize>>>(d_vals, target.d_vals, output.d_vals, size);
        cudaCheckError(cudaDeviceSynchronize());

        return output;
    }

    CudaMat2D multScalar(Type s) const
    {
        CudaMat2D output(_cols, _rows);

        int size = _rows * _cols;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        scalarMultKernel<<<numBlocks, blockSize>>>(d_vals, s, output.d_vals, size);
        cudaCheckError(cudaDeviceSynchronize());

        return output;
    }

    CudaMat2D transpose() const
    {
        CudaMat2D output(_rows, _cols);

        dim3 blockSize(16, 16);
        dim3 gridSize((_cols + blockSize.x - 1) / blockSize.x, 
                      (_rows + blockSize.y - 1) / blockSize.y);

        transposeKernel<<<gridSize, blockSize>>>(d_vals, output.d_vals, _rows, _cols);
        cudaCheckError(cudaDeviceSynchronize());

        return output;
    }

    void print() const {
        Type* h_vals = new Type[_rows * _cols];
        copyToHost(h_vals);

        for (size_t y = 0; y < _rows; y++) {
            for (size_t x = 0; x < _cols; x++)
                std::cout << std::setw(10) << h_vals[y * _cols + x] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        delete[] h_vals;
    }
};