#include<cstdio>
#include<cuda.h>

#define BLOCK_DIM 256
#define WARP_SIZE 32

__global__ void cuChamfer(float* A, int nA, float* B, int nB, int dim, float* total_dist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float min_dist = 0.00000f;

    if (i < nA) {
        min_dist = __FLT_MAX__;
        for (int j = 0; j < nB; j++) {
            float diff_sum = 0.0f;
            for (int d = 0; d < dim; d++) {
                float val = A[i + d * nA] - B[j + d * nB];
                diff_sum += val * val;
            }
            min_dist = fminf(min_dist, diff_sum);
        }
    }

    unsigned mask = __activemask();
    for(int offset = WARP_SIZE>>1; offset > 0; offset >>=1) {
        min_dist += __shfl_down_sync(mask, min_dist, offset);
    }

    __shared__ float warp_sums[BLOCK_DIM / WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = min_dist;
    }
    __syncthreads();
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        mask = __activemask();
        for(int offset = WARP_SIZE>>1; offset > 0; offset >>=1) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(total_dist, block_sum);
        }
    }
}

__global__ void cuFinalCalculation(const float* col_val, int col_size, 
                                   const float* row_val, int row_size, float* v) {
    float a = *col_val / (float)col_size;
    float b = *row_val / (float)row_size;
    *v = a + b;
}

void chamfer_distance_device(float* pc1_dev, int pc1_nb, float* pc2_dev, int pc2_nb, 
    int dim, float* result_dev) {
    float *result1, *result2;
    cudaMalloc((void**)&result1, sizeof(float));
    cudaMalloc((void**)&result2, sizeof(float));
    dim3 block_256(BLOCK_DIM, 1, 1);
    dim3 grid_256_1((pc1_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    dim3 grid_256_2((pc2_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
    
    cudaMemset(result1, 0, sizeof(float));
    cuChamfer<<<grid_256_1, block_256, 0>>>(pc1_dev, pc1_nb, pc2_dev, pc2_nb, dim, result1);

    cudaMemset(result2, 0, sizeof(float));
    cuChamfer<<<grid_256_2, block_256, 0>>>(pc2_dev, pc2_nb, pc1_dev, pc1_nb, dim, result2);

    cuFinalCalculation<<<1,1,0>>>(result1, pc1_nb, result2, pc2_nb, result_dev);

    cudaFree(result1);
    cudaFree(result2);
}