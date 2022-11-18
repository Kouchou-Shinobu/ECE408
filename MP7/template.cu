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

#define TILE_WIDTH 16

/**
 * Sparse matrix vector multiplication kernel for JDS compression.
 *
 * @param out The output vector.
 * @param matColStart Column start indices. Indicates where each column starts.
 * @param matCols Column indices. Flattened vertically.
 * @param matRowPerm Row permutation vector.
 * @param matRows Number of non-zero elements for each row.
 * @param matData Non-zero elements. Flattened vertically.
 * @param vec The input vector.
 * @param dim The number of rows.
 */
__global__ void spmvJDSKernel(float *out, const int *matColStart, const int *matCols,
                              const int *matRowPerm, const int *matRows,
                              const float *matData, const float *vec, int dim) {
    unsigned row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < dim) {    // To Ensure we are operating within the problem size.
        float dot = 0.0f;
        // We are working with transposed matrix.
        for (unsigned sec = 0; sec < matRows[row]; ++sec) {
            /*
             * Recall that `matColStart` indicates the starting index of each column.
             * Here we divide `matData` into several sections, each denoting a column. In each section the elements
             *  composes a row.
             * The goal is to map from `row` to "column" (that's what matrix-vector multiplication do), and by sectioning
             *  we can easily obtain the target index (consider `row` as the offset for each section.)
             */
            unsigned tar_idx = matColStart[sec] + row;  // The target index to work with.
            dot += matData[tar_idx] * vec[matCols[tar_idx]];
        }
        out[matRowPerm[row]] = dot;
    }
}

static void spmvJDS(float *out, int *matColStart, int *matCols,
                    int *matRowPerm, int *matRows, float *matData,
                    float *vec, int dim) {
    dim3 blockd(TILE_WIDTH, 1, 1);
    dim3 gridd(ceil((float) dim / TILE_WIDTH), 1, 1);
    spmvJDSKernel<<<gridd, blockd>>>(out, matColStart, matCols, matRowPerm, matRows, matData, vec, dim);
}

int main(int argc, char **argv) {
    wbArg_t args;
    int *hostCSRCols;
    int *hostCSRRows;
    float *hostCSRData;
    int *hostJDSColStart;
    int *hostJDSCols;
    int *hostJDSRowPerm;
    int *hostJDSRows;
    float *hostJDSData;
    float *hostVector;
    float *hostOutput;
    int *deviceJDSColStart;
    int *deviceJDSCols;
    int *deviceJDSRowPerm;
    int *deviceJDSRows;
    float *deviceJDSData;
    float *deviceVector;
    float *deviceOutput;
    int dim, ncols, nrows, ndata;
    int maxRowNNZ;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostCSRCols = (int *) wbImport(wbArg_getInputFile(args, 0), &ncols, "Integer");
    hostCSRRows = (int *) wbImport(wbArg_getInputFile(args, 1), &nrows, "Integer");
    hostCSRData = (float *) wbImport(wbArg_getInputFile(args, 2), &ndata, "Real");
    hostVector = (float *) wbImport(wbArg_getInputFile(args, 3), &dim, "Real");

    hostOutput = (float *) malloc(sizeof(float) * dim);

    wbTime_stop(Generic, "Importing data and creating memory on host");

    CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm, &hostJDSRows,
             &hostJDSColStart, &hostJDSCols, &hostJDSData);
    maxRowNNZ = hostJDSRows[0];

    wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc((void **) &deviceJDSColStart, sizeof(int) * maxRowNNZ);
    cudaMalloc((void **) &deviceJDSCols, sizeof(int) * ndata);
    cudaMalloc((void **) &deviceJDSRowPerm, sizeof(int) * dim);
    cudaMalloc((void **) &deviceJDSRows, sizeof(int) * dim);
    cudaMalloc((void **) &deviceJDSData, sizeof(float) * ndata);

    cudaMalloc((void **) &deviceVector, sizeof(float) * dim);
    cudaMalloc((void **) &deviceOutput, sizeof(float) * dim);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");
    spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm, deviceJDSRows,
            deviceJDSData, deviceVector, dim);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceVector);
    cudaFree(deviceOutput);
    cudaFree(deviceJDSColStart);
    cudaFree(deviceJDSCols);
    cudaFree(deviceJDSRowPerm);
    cudaFree(deviceJDSRows);
    cudaFree(deviceJDSData);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, dim);

    free(hostCSRCols);
    free(hostCSRRows);
    free(hostCSRData);
    free(hostVector);
    free(hostOutput);
    free(hostJDSColStart);
    free(hostJDSCols);
    free(hostJDSRowPerm);
    free(hostJDSRows);
    free(hostJDSData);

    return 0;
}
