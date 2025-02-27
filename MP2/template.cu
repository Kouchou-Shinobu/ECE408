
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row < numCRows) && (col < numCColumns)) {
        float tmp = 0.0;
        for (int i = 0; i < numAColumns; ++i) {
            tmp += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numCColumns + col] = tmp;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set
    // this)
    const int BLOCK_WIDTH = 8;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows,
                               &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows,
                               &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    size_t sizeA = numAColumns * numARows * sizeof(float);
    size_t sizeB = numBColumns * numBRows * sizeof(float);
    size_t sizeC = numCColumns * numCRows * sizeof(float);
    cudaMalloc((void **) &deviceA, sizeA);
    cudaMalloc((void **) &deviceB, sizeB);
    cudaMalloc((void **) &deviceC, sizeC);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 gridDim(ceil(1.0 * numCColumns / BLOCK_WIDTH), ceil(1.0 * numCRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    wbLog(TRACE, "Grid R: ",
          ceil(1.0 * numCRows / BLOCK_WIDTH));
    wbLog(TRACE, "Grid C: ",
          ceil(1.0 * numCColumns / BLOCK_WIDTH));

    matrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostA, deviceA, sizeA, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostB, deviceB, sizeB, cudaMemcpyDeviceToHost);
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
