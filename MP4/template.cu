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

__device__ int indexof(int x, int y, int z, int x_max, int y_max) {
  return x_max * y_max * z + x_max * y + x;
}

/**
 * Check whether a number is within [0, boundary).
 */
__device__ bool within_bound(int n, int bound) {
  return 0 <= n && n < bound;
}

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // Keep in mind: width == col == x, height == row == y, depth == z !

  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1]
                       [TILE_WIDTH + MASK_WIDTH - 1]
                       [TILE_WIDTH + MASK_WIDTH - 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;   // 0 ~ TILE_WIDTH + MASK_WIDTH - 2
  int tz = threadIdx.z;
  int ix_out = blockIdx.x * TILE_WIDTH + tx;
  int iy_out = blockIdx.y * TILE_WIDTH + ty;  // 0 ~ y_size - 1
  int iz_out = blockIdx.z * TILE_WIDTH + tz;
  int ix_in = ix_out - RADIUS;
  int iy_in = iy_out - RADIUS;  // -RADIUS ~ y_size - 1 - RADIUS
  int iz_in = iz_out - RADIUS;

  // //debug
  // printf("input[%d][%d][%d] == %4.2f\n",
  //         ix_out, iy_out, iz_out, 
  //         input[indexof(ix_out, iy_out, iz_out, x_size, y_size)]);
  // //end of debug

  // Preload tiles.
  if (within_bound(ix_in, x_size) &&
      within_bound(iy_in, y_size) &&
      within_bound(iz_in, z_size)) {
    // // debug
    // if (ix_in == 0 && iy_in == 0 && iz_in == 0)
    //   printf("tile[%d][%d][%d] == input[%d] == input[%d][%d][%d] == %4.2f\n",
    //           tz, ty, tx, 
    //           indexof(ix_in, iy_in, iz_in, x_size, y_size),
    //           iz_in, iy_in, ix_in,
    //           input[indexof(ix_in, iy_in, iz_in, x_size, y_size)]);
    // // end of debug
    tile[tz][ty][tx] = input[indexof(ix_in, iy_in, iz_in, x_size, y_size)];
  }
  else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  // Do the calculation.
  float Pvalue = 0.0f;
  if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH) {
    for (int z = 0; z < MASK_WIDTH; ++z) {
      for (int y = 0; y < MASK_WIDTH; ++y) {
        for (int x = 0; x < MASK_WIDTH; ++x) {
          // //debug
          // if (z+tz > 3 || y+ty > 3 || x+tx > 3)
          //   printf("tile[%d][%d][%d] = %4.2f  |  mask[%d][%d][%d] == %4.2f\n",
          //           z+tz, y+ty, x+tx, tile[z+tz][y+ty][x+tx],
          //           z, y, x, M[z][y][x]);
          // //end of debug
          Pvalue += tile[z+tz][y+ty][x+tx] * M[z][y][x];
        }
      }
    }
  }
  // Assign results.
  if (within_bound(ix_out, x_size) && 
      within_bound(iy_out, y_size) &&
      within_bound(iz_out, z_size)) {
    // //debug
    // if (Pvalue != 1.0f)
    //   printf("Thread(x=%d, y=%d, z=%d), Block(x=%d, y=%d, z=%d)\n",
    //           tx, ty, tz,
    //           blockIdx.x, blockIdx.y, blockIdx.z);
    // //end of debug
    // // debug
    // if (blockIdx.x == 1 && blockIdx.y == 0 && blockIdx.z == 0)
    //   printf("(%dx%dx%d): output[%d][%d][%d] == output[%d] == %4.2f\n",
    //           x_size, y_size, z_size,
    //           iz_out, iy_out, ix_out, 
    //           indexof(ix_out, iy_out, iz_out, x_size, y_size),
    //           Pvalue);
    // //end of debug
    output[indexof(ix_out, iy_out, iz_out, x_size, y_size)] = Pvalue;
  }
}

// __global__ void conv3d(float *input, float *output, const int z_size,
//                        const int y_size, const int x_size) {
//   //...
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int tz = threadIdx.z;
//   int iy_out = blockIdx.y * TILE_WIDTH + ty;
//   int ix_out = blockIdx.x * TILE_WIDTH + tx;
//   int iz_out = blockIdx.z * TILE_WIDTH + tz;

//   float Pvalue = 0.0f;
//   for (int z = 0; z < MASK_WIDTH; ++z) {
//     for (int y = 0; y < MASK_WIDTH; ++y) {
//       for (int x = 0; x < MASK_WIDTH; ++x) {
//         if (within_bound(ix_out - RADIUS, x_size) &&
//             within_bound(iy_out - RADIUS, y_size) &&
//             within_bound(iz_out - RADIUS, z_size)) {
//           Pvalue += M[z][y][x] * input[
//               indexof(ix_out - RADIUS, iy_out - RADIUS, iz_out - RADIUS, x_size, y_size)];
//         }
//         else {
//           Pvalue += 0;
//         }
//       }
//     }
//   }
//   if (within_bound(ix_out, x_size) &&
//       within_bound(iy_out, y_size) &&
//       within_bound(iz_out, z_size)) {
//     output[indexof(ix_out, iy_out, iz_out, x_size, y_size)] = Pvalue;
//   }
// }

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
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  size_t size = x_size * y_size * z_size * sizeof(float);
  cudaMalloc((void**) &deviceInput, size);
  cudaMalloc((void**) &deviceOutput, size);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 gridDim(ceil((1.0 * x_size) / TILE_WIDTH), 
               ceil((1.0 * y_size) / TILE_WIDTH),
               ceil((1.0 * z_size) / TILE_WIDTH));
  dim3 blockDim(BLOCK_WIDTH, 
                BLOCK_WIDTH, 
                BLOCK_WIDTH);

  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaError_t kernelStatus = cudaDeviceSynchronize();
  if (cudaSuccess != kernelStatus)
    printf("kernel launch failed with error \"%s\".\n",
            cudaGetErrorString(kernelStatus));
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, size, cudaMemcpyDeviceToHost);
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
