// Modified from
// group_points.cpp, ball_query.cpp

#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
extern THCState *state;

int calc_weight_wrapper(int b, int npoints, int nsample, at::Tensor grouped_xyz_tensor, 
                        at::Tensor centers_tensor, at::Tensor weights_tensor);

void calc_weight_kernel_launcher(int b, int npoints, int nsample, 
                                const float* grouped_point, 
                                const float* center, 
                                float* weight, 
                                cudaStream_t stream);

int calc_weight_wrapper(int b, int npoints, int nsample, at::Tensor grouped_xyz_tensor, 
                        at::Tensor centers_tensor, at::Tensor weights_tensor)
{
    CHECK_INPUT(grouped_xyz_tensor);
    CHECK_INPUT(centers_tensor);
    CHECK_INPUT(weights_tensor);
    const float *grouped_point = grouped_xyz_tensor.data_ptr<float>();
    const float *center = centers_tensor.data_ptr<float>();
    float *weight = weights_tensor.data_ptr<float>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    calc_weight_kernel_launcher(b,npoints,nsample,grouped_point,center,weight,stream);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("calc_weight", &calc_weight_wrapper, "calc_weight_wrapper"); 
}
