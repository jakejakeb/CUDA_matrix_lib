#include "../include/Mat2d.cuh"
#include <iostream>

int main() {
    CudaMat2D<float> mat1(3, 3);
    CudaMat2D<float> mat2(3, 3);

    float h_vals1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float h_vals2[] = {9, 8, 7, 6, 5, 4, 3, 2, 1};

    mat1.copyToDevice(h_vals1);
    mat2.copyToDevice(h_vals2);

    std::cout << "Matrix 1:" << std::endl;
    mat1.print();

    std::cout << "Matrix 2:" << std::endl;
    mat2.print();

    CudaMat2D<float> result = mat1.mult(mat2);

    std::cout << "Result of multiplication:" << std::endl;
    result.print();

    return 0;
}