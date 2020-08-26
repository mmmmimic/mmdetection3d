// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 256 //每个block包含线程数：256，GPU包含多个grid，每个grid包含多个block，每个block包含多个threads
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

__global__ void ball_query_kernel(int b, int n, int m, float radius,
                                  int nsample,
                                  const float *__restrict__ new_xyz,
                                  const float *__restrict__ xyz,
                                  int *__restrict__ idx) {
  // __global__修饰说明这个函数是个核函数kernel，运行在GPU(device)上，从CPU(host)上获取数据，核函数类型只能是void
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)
  int bs_idx = blockIdx.y;// blockIdx指grid中block的坐标，二维，包括x和y
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;// threadIdx指block中thread的坐标，二维，包括x和y
  if (bs_idx >= b || pt_idx >= m) return;

  new_xyz += bs_idx * m * 3 + pt_idx * 3;//这里的idx的计算暂时还没搞懂
  xyz += bs_idx * n * 3;
  idx += bs_idx * m * nsample + pt_idx * nsample;

  float radius2 = radius * radius;// 球半径的平方，点到球中心的距离的平方的上限
  float new_x = new_xyz[0];// new_xyz在输入中是下采样点的坐标，(B, M, 3)
  float new_y = new_xyz[1];
  float new_z = new_xyz[2];

  int cnt = 0; //counter
  for (int k = 0; k < n; ++k) { // 迭代点云中的每个点
    float x = xyz[k * 3 + 0]; // 对于batch中的每个样本，都放在不同的threads上并行处理，所以这里只需要考虑(M,3)
    float y = xyz[k * 3 + 1];
    float z = xyz[k * 3 + 2];
    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
               (new_z - z) * (new_z - z);// 点到下采样点的距离，对于每个下采样点，也是放在不同的threads上处理的，避免多重循环嵌套
    if (d2 < radius2) { // 当点在球中时
      if (cnt == 0) {
        for (int l = 0; l < nsample; ++l) {//先用第一个点的index填满nsample，之后再去替换，因为有的球中可能没有nsample个近邻点
          idx[l] = k;
        }
      }
      idx[cnt] = k;
      ++cnt;
      if (cnt >= nsample) break;
    }
  }
}

void ball_query_kernel_launcher(int b, int n, int m, float radius, int nsample,
                                const float *new_xyz, const float *xyz,
                                int *idx, cudaStream_t stream) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  cudaError_t err;

  dim3 blocks(DIVUP(m, THREADS_PER_BLOCK),
              b);  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);
  // blocks, threads和grids都是dim3类型
  ball_query_kernel<<<blocks, threads, 0, stream>>>(b, n, m, radius, nsample,
                                                    new_xyz, xyz, idx);
  //调用kernel时要使用<<<grid, block>>>来指定kernel要执行的线程数量
  // cudaDeviceSynchronize();  // for using printf in kernel function
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
