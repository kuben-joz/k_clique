#include "direct.cuh"
#include "input.cuh"
#include "util.cuh"

const int threadsPerBlock = 256;
const int blocksPerGrid = 32;

__constant__ int g_params[CONST_NUM];

__global__ void directKern() {
    unsigned tid = X_TID;

}

int directGraph(Graph& g) {
    int *cpu_g_params = static_cast<int *>(malloc(sizeof(int) * CONST_NUM));
    cpu_g_params[V_IDX] = g.vs.size();
    cpu_g_params[E_IDX] = g.es.size();

    cudaMemcpyToSymbol(g_params, cpu_g_params, sizeof(int) * CONST_NUM);



    return 0;
}