#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "constants.cuh"

// hybrid csr/coo representation
struct Graph
{
  thrust::device_vector<int> v1s;     // v1s and v2s together form the coo representation
  thrust::device_vector<int> v2s;     // this is the equivalent of adjacency lists, part of both csr and coo
  thrust::device_vector<int> row_ptr; // this is the standard row pointer from csr
};

template <typename T>
struct mod_mul
{
  /// Boolean sum operator, returns <tt>a + b</tt>

  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
  {
    return (a * b) % ((T)g_const::mod);
  }
};

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define X_TID (blockIdx.x * blockDim.x + threadIdx.x)
#define Y_TID (blockIdx.y * blockDim.y + threadIdx.y)
#define Z_TID (blockIdx.z * blockDim.z + threadIdx.z)

#define GRID_STRIDE_LOOP(var, n) for (int var = blockIdx.x * blockDim.x + threadIdx.x; var < n; var += blockDim.x * gridDim.x)

// source: https://stackoverflow.com/questions/34073315/removing-elements-from-cuda-array?rq=2
// Used during dev only, unless I forgot to remove it somewhere
#define PRINTER(name) print(#name, (name))
template <template <typename...> class V, typename T, typename... Args>
void print(const char *name, const V<T, Args...> &v)
{
  std::cout << name << ":\t";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, "\t"));
  std::cout << std::endl;
}

template <typename T>
struct plus_mod
{
  __host__ __device__
      T
      operator()(T lhs, T rhs) const
  {
    return ((lhs % g_const::mod) + (rhs % g_const::mod)) % g_const::mod;
  }
};
