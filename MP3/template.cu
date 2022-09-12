
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

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    // Sub-tiles
    __shared__ float subA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;

    // row and column of the result matrix (aka. `C`).
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float c = 0.0;

    for (int q = 0; q < ceil(1.0 * numAColumns / TILE_WIDTH); ++q) {
        // two conditions correspond to boundary check on Column and Row, respectively.
        if ((q * TILE_WIDTH + tx < numAColumns) && (row < numARows))
            subA[ty][tx] = A[row * numAColumns + q * TILE_WIDTH + tx];
        if ((col < numCColumns) && (q * TILE_WIDTH + ty < numBRows))
            subB[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + col];
        __syncthreads();
        
        for (int i = 0; i < TILE_WIDTH && i < numAColumns; ++i) {
            if ((q * TILE_WIDTH + tx < numAColumns)
                && (q * TILE_WIDTH + ty < numARows))
                c += subA[ty][i] * subB[i][tx];
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns)
      C[row * numCColumns + col] = c;
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
    wbTime_stop(Generic, "Importing data and creating memory on host");
    size_t sizeA = numARows * numAColumns * sizeof(float);
    size_t sizeB = numBRows * numBColumns * sizeof(float);
    size_t sizeC = numCRows * numCColumns * sizeof(float);
    hostC = (float*) malloc(sizeC);

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceA, sizeA);
    cudaMalloc((void**) &deviceB, sizeB);
    cudaMalloc((void**) &deviceC, sizeC);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int gridCol = ceil(1.0 * numCColumns / TILE_WIDTH);
    int gridRow = ceil(1.0 * numCRows / TILE_WIDTH);
    dim3 gridDim(gridCol, gridRow, 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    wbLog(TRACE, "Grid column: ", gridCol);
    wbLog(TRACE, "Grid row: ", gridRow);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);
    // Print error msg if kernel failed to launch.
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
    cudaFree(deviceA);  cudaFree(deviceB);  cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
