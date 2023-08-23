#pragma once

namespace g_const
{
    extern __constant__ int num_edges_dev;
    extern __constant__ int num_vertices_dev;
    extern __constant__ int mod;
    extern __constant__ int max_clique;
    extern int num_edges_host;
    extern int num_vertices_host;
    const int max_deg = 1024;
    const int max_clique_size = 12;
    const int threads_per_block = 128; // todo orginally 256
    const int blocks_per_grid = 64;    // todo originally 32
}