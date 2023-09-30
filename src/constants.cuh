#pragma once

namespace g_const
{
    extern __constant__ int num_edges_dev;
    extern __constant__ int num_vertices_dev;
    const int mod = 1000000000;
    extern int num_edges_host;
    extern int num_vertices_host;
    const int max_deg = 1024;
    const int max_clique_size = 12;
    const int threads_per_block = 128;
    const int blocks_per_grid = 1944; // max for a100
    extern int blocks_per_grid_host;
    extern __constant__ int blocks_per_grid_dev;
}