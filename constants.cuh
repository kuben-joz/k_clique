#pragma once


__constant__ int num_edges_dev;
__constant__ int num_vertices_dev;
__constant__ int mod = 1000000000;
__constant__ int max_clique = 5;
static int num_edges_host;
static int num_vertices_host;
const int N = 30 * 1024;
const int max_deg = 1024;
const int max_clique_size = 12;
