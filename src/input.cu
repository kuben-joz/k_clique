#include <iostream>
#include <map>
#include <algorithm>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>

#include "constants.cuh"
#include "input.cuh"

using namespace std;

Graph getGraph()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    map<uint32_t, int> idxs;
    uint32_t a, b;
    thrust::host_vector<int> v1s_host; // starting node i.e. v1 -> v2
    thrust::host_vector<int> v2s_host; // end node
    thrust::host_vector<int> temp_edge(2);
    int idx = 0;
    int clique_size;
    cin >> clique_size;
    while (cin >> a)
    {
        cin >> b;
        auto temp = idxs.try_emplace(a, idx);
        temp_edge[0] = temp.first->second; // idx if new, a's already assigned idx otherwise
        if (temp.second)
        { // if insertion took place increment
            idx++;
        }
        // do same for b
        temp = idxs.try_emplace(b, idx);
        temp_edge[1] = temp.first->second;
        if (temp.second)
        {
            idx++;
        }
        v1s_host.insert(v1s_host.end(), temp_edge.cbegin(), temp_edge.cend());
        v2s_host.insert(v2s_host.end(), temp_edge.crbegin(), temp_edge.crend());
    }
    g_const::num_edges_host = v1s_host.size();
    g_const::num_vertices_host = idx;
    cudaMemcpyToSymbol(&g_const::num_edges_dev, &g_const::num_edges_host, sizeof g_const::num_edges_host);
    cudaMemcpyToSymbol(&g_const::num_vertices_dev, &g_const::num_vertices_host, sizeof g_const::num_vertices_host);
    thrust::device_vector<int> v1s_dev(v1s_host);
    thrust::device_vector<int> v2s_dev(v2s_host);
    auto edge_it = thrust::make_zip_iterator(thrust::make_tuple(v1s_dev.begin(), v2s_dev.begin()));
    thrust::sort(edge_it, edge_it + g_const::num_edges_host);

    // adding row pointer array using dense cummulative histogram,
    // based on https://github.com/NVIDIA/thrust/blob/master/examples/histogram.cu
    thrust::device_vector<int> row_pointer(g_const::num_vertices_host, 0);
    thrust::counting_iterator<int> idx_counter(0);
    thrust::upper_bound(v1s_dev.begin(), v1s_dev.end(),
                        idx_counter, idx_counter + idx,
                        row_pointer.begin() + 1);
    PRINTER(row_pointer);
    Graph res;
    res.v1s = v1s_dev;
    res.v2s = v2s_dev;
    res.row_ptr = row_pointer;
    return res;
}