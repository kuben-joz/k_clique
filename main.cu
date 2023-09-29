#include <iostream>
#include <string>

#include "input.cuh"
#include "direct.cuh"
#include "orient.cuh"
#include "pivot.cuh"

namespace g_const
{
    __constant__ int num_edges_dev = 0;
    __constant__ int num_vertices_dev = 0;
    int num_edges_host = 0;
    int num_vertices_host = 0;
    int blocks_per_grid_host = 0;
    __constant__ int blocks_per_grid_dev = 0;
}

int main(int argc, char *argv[])
{
    int max_clique = atoi(argv[2]);
    std::string output_path = std::string(argv[3]);
    Graph g = getGraph(argv[1]);
    directGraph(g);
    if (max_clique < 5)
    {
        countCliquesOrient(g, max_clique, output_path);
    }
    else
    {
        countCliquesPivot(g, max_clique, output_path);
    }
    return 0;
}
