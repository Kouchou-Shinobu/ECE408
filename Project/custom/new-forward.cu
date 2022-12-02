#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 32
// Mask is essentially dynamically allocated. The size of a constant memory, though, must be
// determined at compile time.
// So
#define MASK_WIDTH 16384

//__constant__ float Mask[MASK_WIDTH];

__global__ void
conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out,
                    const int Channel, const int Height, const int Width, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    /*
    Recall that the process of a single convolution (for LeNet-CNN Network) is:
    We have some input images, with a number of `Batch` in total. Each of image has `Channel` channels, and has a dimension
     of `Height` * `Width`

    Along with the images come weights, with a total number of `Map-out`, `Channel` channels per map, and `K` * `K` entries
     per map.

    We convolve the input images with weights, and sum over the channels. Eventually we get some outputs, with a total
     number of `Batch`, `Map-out` features per output, and dimensions of `Height_out` * `Width_out`.
     */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const unsigned int Width_grid = ceil((1.0 * Width_out) / TILE_WIDTH);

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

#define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
#define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // As defined in conv_forward_gpu, the 3 dimensions of a block represents Map, Grid dimensions, and Batch,
    // respectively.

    // Current index of batch.
    unsigned int batch = blockIdx.z;
    // Current index of map.
    unsigned int map = blockIdx.x;
    unsigned int h = (blockIdx.y / Width_grid) * TILE_WIDTH + threadIdx.y;
    unsigned int w = (blockIdx.y % Width_grid) * TILE_WIDTH + threadIdx.x;

    // Do the calculation.
    // Data of input matrix is read from global.
    float Pvalue = 0.0f;
    if (h < Height_out && w < Width_out) {
        for (unsigned c = 0; c < Channel; ++c) {    // For every feature map.
            for (unsigned p = 0; p < K; ++p) {      // For every entry in K by K kernel.
                for (unsigned q = 0; q < K; ++q) {
                    Pvalue += mask_4d(map, c, p, q) * in_4d(batch, c, h + p, w + q);
                }
            }
        }
        out_4d(batch, map, h, w) = Pvalue;
    }

#undef out_4d
#undef in_4d
#undef mask_4d
}


__host__ void
GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                      float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                      const int Batch, const int Map_out, const int Channel, const int Height,
                                      const int Width, const int K) {
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaError_t e;

    const unsigned int Height_out = Height - K + 1;
    const unsigned int Width_out = Width - K + 1;
    const unsigned int size_in = Batch * Channel * Height * Width * sizeof(float);
    const unsigned int size_out = Batch * Map_out * Height_out * Width_out * sizeof(float);
    const unsigned int size_mask = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void **) device_input_ptr, size_in);
    cudaMalloc((void **) device_output_ptr, size_out);
    cudaMalloc((void **) device_mask_ptr, size_mask);
    // Recall that double pointers are given for device pointers.
    // Need to dereference first.
    cudaMemcpy(*device_input_ptr, host_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, size_mask, cudaMemcpyHostToDevice);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "CUDA error while copying data to device: " << cudaGetErrorString(e) << std::endl;
        exit(-1);
    }

    // Useful snippet for error checking
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::cout << "CUDA error while performing pre-kernel set-ups: " << cudaGetErrorString(e) << std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel, const int Height,
                                             const int Width, const int K) {
    // Set the kernel dimensions and call the kernel
    const unsigned int Height_out = Height - K + 1;
    const unsigned int Width_out = Width - K + 1;
    const unsigned int Height_grid = ceil((1.0 * Height_out) / TILE_WIDTH);
    const unsigned int Width_grid = ceil((1.0 * Width_out) / TILE_WIDTH);

    dim3 blockd(TILE_WIDTH, TILE_WIDTH, 1);
    // For 4D convolution, we flatten the dimensions of height and width.
    dim3 gridd(Map_out, Height_grid * Width_grid, Batch);
    conv_forward_kernel<<<gridd, blockd>>>(
            device_output,
            device_input,
            device_mask,
            Batch, Map_out, Channel, Height, Width, K);
}


__host__ void
GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                      const int Batch, const int Map_out, const int Channel, const int Height,
                                      const int Width, const int K) {
    const unsigned int Height_out = Height - K + 1;
    const unsigned int Width_out = Width - K + 1;
    // Copy the output back to host
    cudaMemcpy(host_output, device_output,
               Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
