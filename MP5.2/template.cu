// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

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


__global__ void unit_scan(const float *input, float *output, unsigned len) {
    __shared__ float tile[2 * BLOCK_SIZE];
    unsigned tx = threadIdx.x;

    // Populate the shared memory.
    unsigned aidx = 2 * blockIdx.x * blockDim.x + tx;
    if (aidx < len)
        tile[tx] = input[aidx];
    else
        tile[tx] = 0.0f;
    if (aidx + BLOCK_SIZE < len)
        tile[tx + BLOCK_SIZE] = input[aidx + BLOCK_SIZE];
    else
        tile[tx + BLOCK_SIZE] = 0.0f;
    // Need a sync to ensure every thread had finished loading.
    __syncthreads();

    // Pre-scan.
    int stride = 1;
    while (stride < 2 * BLOCK_SIZE) {
        // Index to which threads are writing. Following a pattern like (1, 3, 7, 11,...)
        int write_idx = ((int) tx + 1) * 2 * stride - 1;
        int read_idx = write_idx - stride;
        if (write_idx < 2 * BLOCK_SIZE && read_idx >= 0) {
            tile[write_idx] += tile[read_idx];
        }
        stride *= 2;
        __syncthreads();
    }

    // Post-scan.
    stride = BLOCK_SIZE / 2;
    while (stride > 0) {
        // Much like the writing index, but this time threads are reading from it.
        int read_idx = ((int) tx + 1) * 2 * stride - 1;
        int write_idx = read_idx + stride;
        if (write_idx < 2 * BLOCK_SIZE) {
            tile[write_idx] += tile[read_idx];
        }
        stride /= 2;
        __syncthreads();
    }

    // Output
    if (aidx < len)
        output[aidx] = tile[tx];
    if (aidx + BLOCK_SIZE < len)
        output[aidx + BLOCK_SIZE] = tile[tx + BLOCK_SIZE];
}

__global__ void integrative_scan(const float *input, float *output, unsigned len, float *sums, unsigned len_sums) {
    unsigned tx = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + tx;
    unsigned aidx = idx * BLOCK_SIZE * 2;

    if (idx < len_sums) {
        float p = 0;
        if (idx > 0)
            p = sums[idx - 1];
        for (unsigned i = aidx; i < aidx + BLOCK_SIZE * 2; ++i) {
            output[i] = input[i] + p;
        }
    }

    // Make sure every thread has finished its work before exiting.
    __syncthreads();
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void **) &deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **) &deviceOutput, numElements * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 blockd(BLOCK_SIZE, 1, 1);
    dim3 gridd(ceil((float) numElements / BLOCK_SIZE), 1, 1);
#ifdef __DEBUG__
    printf("Grid(%d, %d, %d)  |  Block(%d, %d, %d)\n",
           gridd.x, gridd.y, gridd.z,
           blockd.x, blockd.y, blockd.z);
#endif

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    unit_scan<<<gridd, blockd>>>(deviceInput, deviceOutput, numElements);
    wbCheck(cudaDeviceSynchronize());
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    // Do extra work for arrays exceeds one block.
    if (gridd.x > 1) {
        // Auxiliary array storing the summations.
        float *aux;
        aux = (float *) malloc(gridd.x * sizeof(float));
        int start_idx = 0;
        for (size_t i = 0; i < gridd.x; ++i) {
            start_idx += BLOCK_SIZE * 2;
            if (start_idx <= numElements)
                aux[i] = hostOutput[start_idx - 1];
            else
                aux[i] = hostOutput[numElements - 1];
        }

        // Second unit scan.
        wbCheck(cudaMemcpy(deviceInput, aux, gridd.x * sizeof(float), cudaMemcpyHostToDevice));
        dim3 gridd_s(ceil((float) gridd.x / (BLOCK_SIZE * 2)), 1, 1);
        dim3 blockd_s(BLOCK_SIZE, 1, 1);
        unit_scan<<<gridd_s, blockd_s>>>(deviceInput, deviceOutput, gridd.x);
        wbCheck(cudaDeviceSynchronize());
        wbCheck(cudaMemcpy(aux, deviceOutput, gridd.x * sizeof(float), cudaMemcpyDeviceToHost));

        // Integrative scan.
        float *device_aux;
        cudaMalloc((void **) &device_aux, gridd.x * sizeof(float));
        wbCheck(cudaMemcpy(deviceInput, hostOutput, numElements * sizeof(float), cudaMemcpyHostToDevice));
        wbCheck(cudaMemcpy(device_aux, aux, gridd.x * sizeof(float), cudaMemcpyHostToDevice));
        dim3 gridd_intg_scan(ceil((float) (gridd.x - 1) / BLOCK_SIZE), 1, 1);
        dim3 blockd_intg_scan(BLOCK_SIZE, 1, 1);
        integrative_scan<<<gridd_intg_scan, blockd_intg_scan>>>(deviceInput, deviceOutput, numElements, device_aux,
                                                                gridd.x);
        wbCheck(cudaDeviceSynchronize());
        cudaFree(device_aux);
        free(aux);
    }

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
