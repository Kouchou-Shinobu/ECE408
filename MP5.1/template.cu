// MP 5.1 Reduction
// Given a list of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void total(const float *input, float *output, int len) {
    __shared__ float Segment[2 * BLOCK_SIZE];
    unsigned int tx = threadIdx.x;
    // Each thread is responsible for two elements. So the offset is twice as large.
    unsigned int start_idx = 2 * blockIdx.x * blockDim.x;
    //@@ Load a segment of the input vector into shared memory
    Segment[tx] = start_idx + tx < len ? input[start_idx + tx] : 0.0f;
    Segment[BLOCK_SIZE + tx] = BLOCK_SIZE + start_idx + tx < len ? input[BLOCK_SIZE + start_idx + tx] : 0.0f;

    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx < stride)
            Segment[tx] += Segment[stride + tx];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (tx == 0)    // Only the first thread in each block does the output assignment.
        output[blockIdx.x] = Segment[0];
}

int main(int argc, char **argv) {
    int ii;
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput =
            (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = (numInputElements - 1) / (BLOCK_SIZE << 1) + 1;
    hostOutput = (float *) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ",
          numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
    cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 blockd(BLOCK_SIZE, 1, 1);
    dim3 gridd(numOutputElements, 1, 1);
#ifdef __DEBUG__
    printf("Using block(%d, %d, %d)  |  grid(%d, %d, %d)\n",
           blockd.x, blockd.y, blockd.z,
           gridd.x, gridd.y, gridd.z);
#endif

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<gridd, blockd>>>(deviceInput, deviceOutput, numInputElements);

    wbCheck(cudaDeviceSynchronize());

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /***********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input.
     * For simplicity, we do not require that for this lab!
     ***********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
#ifdef __DEBUG__
        printf("%4.2f, ", hostOutput[ii]);
    }
    printf("\n");
    printf("First element: %4.2f\n", hostOutput[0]);
#else
    }
#endif

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}
