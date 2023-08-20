#pragma once

namespace g_const
{
    __constant__ int num_edges_dev;
    __constant__ int num_vertices_dev;
    __constant__ int mod = 1000000000;
    __constant__ int max_clique = 5;
    static int num_edges_host;
    static int num_vertices_host;
    const int N = 30 * 1024;
    const int max_deg = 1024;
    const int max_clique_size = 12;
    const int threads_per_block = 256;
    const int blocks_per_grid = min(32, (N + threads_per_block - 1) / threads_per_block);
}