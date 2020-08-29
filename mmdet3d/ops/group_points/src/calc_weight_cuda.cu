// Modified from
// group_points_gpu.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS_PER_BLOCK 128 //grids->block->threads
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))// round-up division
#define LAMBDA 0.5 //


__global__ void calc_weight_kernel(int b, int npoints, int nsample, 
                                    const float *__restrict__ grouped_point,
                                    const float *__restrict__ center,  
                                    float *__restrict__ weight){
    // grouped_point: (B,npoints,nsample,3)
    // centers: (B,npoints,3)
    // output:
    // weight: (B,npoint,nsample)
    int bs_idx = blockIdx.y;// batch->y
    int p_idx = blockIdx.x * blockDim.x + threadIdx.x;// sampling point->x
    if (bs_idx >= b || p_idx >= npoints) return; // check the size
    // spawn batch_size and npoints along the x-y dims in gpu
    grouped_point += bs_idx * npoints * nsample * 3 + p_idx * nsample * 3;
    center += bs_idx * npoints * 3 + p_idx * 3;
    weight += bs_idx * npoints * nsample + p_idx * nsample;
    float center_x = center[0];
    float center_y = center[1];
    float center_z = center[2];
    for(int i = 0; i < nsample; ++i)// iterate each sample in (B,npoints,:,:)
    {
        float sel_point_x = grouped_point[i*3+0];
        float sel_point_y = grouped_point[i*3+1];
        float sel_point_z = grouped_point[i*3+2];
        float offset = sqrt((sel_point_x-center_x)*(sel_point_x-center_x) + (sel_point_y-center_y)*(sel_point_y-center_y)
                        + (sel_point_z-center_z)*(sel_point_z-center_z));
        float min_dist = 1e40;
        for(int j = 0; j < nsample; ++j)
        {
            float point_x = grouped_point[j*3+0];
            float point_y = grouped_point[j*3+1];
            float point_z = grouped_point[j*3+2];
            float dist = sqrt((sel_point_x-point_x)*(sel_point_x-point_x) + (sel_point_y-point_y)*(sel_point_y-point_y)
            + (sel_point_z-point_z)*(sel_point_z-point_z));
            if ((dist>1e-5) && (dist<min_dist))
            {
                min_dist = dist;
            }
        }
        if (min_dist>1e39)
        {
            weight[i] = weight[0];
        }
        else
        {
            weight[i] = offset - LAMBDA*min_dist;
        }
    }

    }

void calc_weight_kernel_launcher(int b, int npoints, int nsample, const float *grouped_point,
                                const float *center, float *weight, cudaStream_t stream){
    // grouped_point (B,npoints,nsample,3)
    // center (B,npoints,3)
    // output:
    // weight (B,npoints,nsample)
    cudaError_t err;

    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK),
                b);  // blockIdx.x(col), blockIdx.y(row),for input(16,256,16),block:(x=2,y=16),with 128 threads in each block
    dim3 threads(THREADS_PER_BLOCK);
  
    calc_weight_kernel<<<blocks, threads, 0, stream>>>(b, npoints, nsample, grouped_point, center, weight);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}