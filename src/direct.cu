#include "direct.cuh"
#include "input.cuh"
#include "util.cuh"
#include "constants.cuh"
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>

__global__ void directKern(int *v_degrees, int *offset, int *v1s, int *v2s)
{
    int v1;
    GRID_STRIDE_LOOP(ii, g_const::num_edges_dev)
    {
        v1 = v1s[ii];
        int deg1 = v_degrees[v1];
        int deg2 = v_degrees[v2s[ii]];
        offset[ii] = -1 ? deg1 > deg2 : 0;
    }
    __syncthreads();
    GRID_STRIDE_LOOP(ii, g_const::num_vertices_dev)
    {
        v_degrees[ii] = 0;
    }
    __syncthreads();
    GRID_STRIDE_LOOP(ii, g_const::num_vertices_dev)
    {
        atomicAdd(&v_degrees[v1], offset[ii]);
    }
}

int directGraph(Graph g)
{
    // ***********************
    // direct graph by degree
    // ************************

    // Add an actual histogram to store undirected degrees
    thrust::device_vector<int> v_degrees(g_const::num_vertices_host + 1, 0);
    // thrust::device_vector<int> v_degrees(num_vertices_host);
    thrust::adjacent_difference(g.row_ptr.begin() + 1, g.row_ptr.end(), v_degrees.begin()); // todo maybe not in place would be faster, subtraction instead
    thrust::device_vector<int> pointer_offset_sparse(g_const::num_edges_host);
    // thrust::device_vector<int> pointer_offset_reduced(num_vertices_host + 1);
    directKern<<<g_const::blocks_per_grid, g_const::threads_per_block>>>(
        thrust::raw_pointer_cast(v_degrees.data()),
        thrust::raw_pointer_cast(pointer_offset_sparse.data()),
        thrust::raw_pointer_cast(g.v1s.data()),
        thrust::raw_pointer_cast(g.v2s.data()));

    // thrust::reduce_by_key(v1s.begin(), v1s.end(), pointer_offset_sparse.begin(), v_degrees.begin(), pointer_offset_reduced.begin()); // v_degrees saves on an alloc
    // thrust::exclusive_scan(pointer_offset_reduced.begin(), pointer_offset_reduced.end(), pointer_offset_reduced.begin(), 0);
    thrust::exclusive_scan(v_degrees.begin(), v_degrees.end(), v_degrees.begin(), 0);
    thrust::transform(g.row_ptr.begin(), g.row_ptr.end(), v_degrees.begin(), g.row_ptr.begin(), thrust::plus<int>());
    using namespace thrust::placeholders;
    auto es = thrust::make_zip_iterator(thrust::make_tuple(g.v1s.begin(), g.v2s.begin()));
    auto end = thrust::remove_if(es, es + g_const::num_edges_host, pointer_offset_sparse.begin(), !_1);
    assert(thrust::get<0>(end.get_iterator_tuple()) - g.v1s.begin() == g_const::num_edges_host / 2);
    g_const::num_edges_host /= 2;
    cudaMemcpyToSymbol(&g_const::num_edges_dev, &g_const::num_edges_host, sizeof g_const::num_edges_host);
    return 0;
}