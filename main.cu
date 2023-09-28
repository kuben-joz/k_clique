#include <iostream>
#include <string>

#include "input.cuh"
#include "direct.cuh"
#include "orient.cuh"
#include "pivot.cuh"

int main(int argc, char *argv[])
{
    int max_clique = atoi(argv[2]);
    std::string output_path = std::string(argv[3]);
    Graph g = getGraph(argv[1]);
    directGraph(g);
    if (max_clique < 7)
    {
        countCliquesOrient(g, max_clique, output_path);
    }
    else
    {
        countCliquesPivot(g, max_clique, output_path);
    }
    return 0;
}
