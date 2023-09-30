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

// https://github.com/NVIDIA/thrust/blob/master/thrust/detail/range/head_flags.h
// copied from ^here because an implementation change in thrust might break result

#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

template <typename RandomAccessIterator,
          typename BinaryPredicate = thrust::equal_to<typename thrust::iterator_value<RandomAccessIterator>::type>,
          typename ValueType = bool,
          typename IndexType = typename thrust::iterator_difference<RandomAccessIterator>::type>
class head_flags
{
  // XXX WAR cudafe issue
  // private:
public:
  struct head_flag_functor
  {
    BinaryPredicate binary_pred; // this must be the first member for performance reasons
    IndexType n;

    typedef ValueType result_type;

    __host__ __device__
    head_flag_functor(IndexType n)
        : binary_pred(), n(n)
    {
    }

    __host__ __device__
    head_flag_functor(IndexType n, BinaryPredicate binary_pred)
        : binary_pred(binary_pred), n(n)
    {
    }

    template <typename Tuple>
    __host__ __device__ __thrust_forceinline__
        result_type
        operator()(const Tuple &t)
    {
      const IndexType i = thrust::get<0>(t);

      // note that we do not dereference the tuple's 2nd element when i <= 0
      // and therefore do not dereference a bad location at the boundary
      return (i == 0 || !binary_pred(thrust::get<1>(t), thrust::get<2>(t)));
    }
  };

  typedef thrust::counting_iterator<IndexType> counting_iterator;

public:
  typedef thrust::transform_iterator<
      head_flag_functor,
      thrust::zip_iterator<thrust::tuple<counting_iterator, RandomAccessIterator, RandomAccessIterator>>>
      iterator;

  __host__ __device__
  head_flags(RandomAccessIterator first, RandomAccessIterator last)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(last - first))),
        m_end(m_begin + (last - first))
  {
  }

  __host__ __device__
  head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
      : m_begin(thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), first, first - 1)),
                                                head_flag_functor(last - first, binary_pred))),
        m_end(m_begin + (last - first))
  {
  }

  __host__ __device__
      iterator
      begin() const
  {
    return m_begin;
  }

  __host__ __device__
      iterator
      end() const
  {
    return m_end;
  }

  template <typename OtherIndex>
  __host__ __device__
      typename iterator::reference
      operator[](OtherIndex i)
  {
    return *(begin() + i);
  }

private:
  iterator m_begin, m_end;
};

template <typename RandomAccessIterator>
__host__ __device__
    head_flags<RandomAccessIterator>
    make_head_flags(RandomAccessIterator first, RandomAccessIterator last)
{
  return head_flags<RandomAccessIterator>(first, last);
}

template <typename RandomAccessIterator, typename BinaryPredicate>
__host__ __device__
    head_flags<RandomAccessIterator, BinaryPredicate>
    make_head_flags(RandomAccessIterator first, RandomAccessIterator last, BinaryPredicate binary_pred)
{
  return head_flags<RandomAccessIterator, BinaryPredicate>(first, last, binary_pred);
}
