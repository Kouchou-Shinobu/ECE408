#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define MASK_WIDTH 3
#define RADIUS (MASK_WIDTH / 2)
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)
//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];


__global__ void conv3d(const float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
    //@@ Insert kernel code here
    // Keep in mind: width == col == x, height == row == y, depth == z !

    __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

    // Thread number. [0, BLOCK_WIDTH)
    int tx = (int) threadIdx.x;
    int ty = (int) threadIdx.y;
    int tz = (int) threadIdx.z;
    int ix_out = (int) blockIdx.x * TILE_WIDTH + tx;
    int iy_out = (int) blockIdx.y * TILE_WIDTH + ty;  // 0 ~ y_size - 1
    int iz_out = (int) blockIdx.z * TILE_WIDTH + tz;
    int ix_in = ix_out - RADIUS;
    int iy_in = iy_out - RADIUS;
    int iz_in = iz_out - RADIUS;

#define in_3d(i3, i2, i1) input[(x_size * y_size * (i3) + x_size * (i2) + (i1))]
#define out_3d(i3, i2, i1) output[(x_size * y_size * (i3) + x_size * (i2) + (i1))]
#define within_bound(i3, i2, i1) ((i3) >= 0 && (i3) < z_size && \
(i2) >= 0 && (i2) < y_size && \
(i1) >= 0 && (i1) < x_size)

    // Preload tiles.
    if (within_bound(iz_in, iy_in, ix_in)) {
        tile[tz][ty][tx] = in_3d(iz_in, iy_in, ix_in);
    } else {
        tile[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    // Do the calculation.
    float Pvalue = 0.0f;
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH) {
        for (int z = 0; z < MASK_WIDTH; ++z) {
            for (int y = 0; y < MASK_WIDTH; ++y) {
                for (int x = 0; x < MASK_WIDTH; ++x) {
                    Pvalue += tile[z + tz][y + ty][x + tx] * M[z][y][x];
                }
            }
        }
        // Assign results.
        // Note that threads indices must be within the scope of tiles.
        if (ix_out >= 0 && iy_out >= 0 && iz_out >= 0 &&
            ix_out < x_size && iy_out < y_size && iz_out < z_size) {
            out_3d(iz_out, iy_out, ix_out) = Pvalue;
        }
    }

#undef in_3d
#undef out_3d
#undef within_bound
}


int main(int argc, char *argv[]) {
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength;
    float *hostInput;
    float *hostKernel;
    float *hostOutput;
    float *deviceInput;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel =
            (float *) wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input mat_data_size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);

    assert(RADIUS == 1);
    assert(BLOCK_WIDTH == 10);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    size_t mat_data_size = (inputLength - 3) * sizeof(float);
    size_t mask_data_size = MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float);
    cudaMalloc((void **) &deviceInput, mat_data_size);
    cudaMalloc((void **) &deviceOutput, mat_data_size);
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do not need to be copied to the gpu
    cudaMemcpy(deviceInput, hostInput + 3, mat_data_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M, hostKernel, mask_data_size);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Initialize grid and block dimensions here
    dim3 blockd(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridd(ceil((1.0 * x_size) / TILE_WIDTH),
               ceil((1.0 * y_size) / TILE_WIDTH),
               ceil((1.0 * z_size) / TILE_WIDTH));
#ifdef __DEBUG__
    printf("Configurations:\n");
    printf("\t- BLOCK_WIDTH: %d\n\t- TILE_WIDTH: %d\n\t- RADIUS: %d\n",
           BLOCK_WIDTH, TILE_WIDTH, RADIUS);
    printf("Using block(%d, %d, %d), grid(%d, %d, %d)\n",
           blockd.x, blockd.y, blockd.z,
           gridd.x, gridd.y, gridd.z);
#endif

    //@@ Launch the GPU kernel here
    conv3d<<<gridd, blockd>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    cudaMemcpy(hostOutput + 3, deviceOutput, mat_data_size, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
