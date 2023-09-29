#include "direct.cuh"
#include "input.cuh"
#include "util.cuh"
#include "constants.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>

__global__ void directKern(int *v_degrees, int *offset, int *v1s, int *v2s)
{
    GRID_STRIDE_LOOP(ii, g_const::num_edges_dev)
    {
        int v1;
        v1 = v1s[ii];
        int deg1 = v_degrees[v1];
        int v2 = v2s[ii];
        int deg2 = v_degrees[v2];
        offset[ii] = -((deg1 > deg2) || ((deg1 == deg2) && (v1 > v2)));
    }
}

int directGraph(Graph &g)
{
    // ***********************
    // direct graph by degree
    // ************************

    // Add an actual histogram to store undirected degrees
    thrust::device_vector<int> v_degrees(g_const::num_vertices_host, 0);
    // thrust::device_vector<int> v_degrees(num_vertices_host);
    thrust::adjacent_difference(g.row_ptr.begin() + 1, g.row_ptr.end(), v_degrees.begin()); // todo maybe not in place would be faster, subtraction instead
    thrust::device_vector<int> pointer_offset_sparse(g_const::num_edges_host);
    // thrust::device_vector<int> pointer_offset_reduced(num_vertices_host + 1);
    directKern<<<g_const::blocks_per_grid_host, g_const::threads_per_block>>>(
        thrust::raw_pointer_cast(v_degrees.data()),
        thrust::raw_pointer_cast(pointer_offset_sparse.data()),
        thrust::raw_pointer_cast(g.v1s.data()),
        thrust::raw_pointer_cast(g.v2s.data()));
    thrust::device_vector<int> offset_reduced(g_const::num_vertices_host + 1);
    thrust::reduce_by_key(g.v1s.cbegin(), g.v1s.cend(), pointer_offset_sparse.begin(), v_degrees.begin(), offset_reduced.begin());
    thrust::exclusive_scan(offset_reduced.begin(), offset_reduced.end(), offset_reduced.begin(), 0);
    thrust::transform(g.row_ptr.begin(), g.row_ptr.end(), offset_reduced.begin(), g.row_ptr.begin(), thrust::plus<int>());
    using namespace thrust::placeholders;
    auto es = thrust::make_zip_iterator(thrust::make_tuple(g.v1s.begin(), g.v2s.begin()));
    auto end = thrust::remove_if(es, es + g_const::num_edges_host, pointer_offset_sparse.begin(), _1);
    assert(thrust::get<0>(end.get_iterator_tuple()) - g.v1s.begin() == g_const::num_edges_host / 2);
    g_const::num_edges_host /= 2;
    HANDLE_ERROR(cudaMemcpyToSymbol(g_const::num_edges_dev, &g_const::num_edges_host, sizeof g_const::num_edges_host));
    g.v1s.resize(g_const::num_edges_host);
    g.v2s.resize(g_const::num_edges_host);
    std::cout << "direct done" << std::endl;
    return 0;
}