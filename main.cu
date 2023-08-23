#include <iostream>
#include <string>

#include "input.cuh"
#include "direct.cuh"
#include "count_cliques.cuh"

int main(int argc, char *argv[])
{
    int max_clique = atoi(argv[2]);
    std::string output_path = std::string(argv[3]);
    Graph g = getGraph(argv[1]);
    directGraph(g);
    countCliques(g, max_clique, output_path);
    return 0;
}
