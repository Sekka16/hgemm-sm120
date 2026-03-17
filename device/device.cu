#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << std::endl;

    return 0;
}