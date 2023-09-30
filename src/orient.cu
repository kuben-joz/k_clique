#include <thrust/device_vector.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector_types.h>
#include <string>
#include <fstream>

#include "constants.cuh"
#include "util.cuh"

namespace cg = cooperative_groups;

const int tile_size_orient = 4;

__device__ unsigned int neigh_bitmap_orient[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int lvl_bitmap_orient[g_const::blocks_per_grid][g_const::threads_per_block / tile_size_orient][g_const::max_clique_size - 4][(g_const::max_deg + 31) / 32];

inline __device__ void moduloAdd(cuda::atomic<int, cuda::thread_scope_block> &res, int val)
{
    val = val % g_const::mod;
    int expected = res.load(cuda::memory_order_relaxed);
    int desired;
    do
    {
        desired = (expected + val) % g_const::mod;
    } while (!res.compare_exchange_weak(expected, desired, cuda::memory_order_acq_rel));
}

static const int tile_sz = tile_size_orient;
__device__ void calculateIntersectsOrient(int v_idx, int *row_ptrs, int *v1s, int *v2s, int clique_size, int *res)
{
    static_assert(g_const::max_deg == 1024); // this assumption is made for optimisations here
    cg::thread_group block = cg::this_thread_block();
    __shared__ int neigh_ids[g_const::max_deg];
    int start = row_ptrs[v_idx];
    int end = row_ptrs[v_idx + 1];
    const int num_neighs = end - start;
    if (num_neighs == 0)
    {
        if(block.thread_rank() == 0) {
            int val = 1;
            int expected = 0;
            int desired = val;
            int old = atomicCAS(&res[0], expected, desired);
            while (old != expected)
            {
                expected = old;
                desired = (old + val) % g_const::mod;
                old = atomicCAS(&res[0], expected, desired);
            }
        }
        return;
    }
 
    const int num_neighs_bitmap = (num_neighs + 31) / 32;
    for (int ii = block.thread_rank(); ii < num_neighs; ii += g_const::threads_per_block)
    {
        neigh_ids[ii] = v2s[ii + start];
    }
    block.sync();
    cg::thread_block_tile<tile_sz> tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    assert(tile.size() == tile_sz);
    const int tile_idx = block.thread_rank() / tile_sz;
    const int num_tiles = g_const::threads_per_block / tile_sz;
    for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
    {
        unsigned int current_neigh[(g_const::max_deg + 31) / 32]; // 32*32 = 1024 i.e. max num neighs
        for (int jj = 0; jj < num_neighs_bitmap; jj++)            // this is supposedly faster than cudamemset
        {
            current_neigh[jj] = 0;
        }
        int neigh_v1 = neigh_ids[ii];
        int neigh_start = row_ptrs[neigh_v1];
        int neigh_end = row_ptrs[neigh_v1 + 1];
        for (int jj = tile.thread_rank(); jj < neigh_end - neigh_start; jj += tile_sz)
        {
            int neigh_v2 = v2s[jj + neigh_start];
            for (int kk = 0; kk < num_neighs; kk++)
            {
                unsigned int temp = neigh_ids[kk] == neigh_v2;
                current_neigh[kk / 32] |= (temp) << (kk % 32);
            }
        }
        for (int jj = 0; jj < num_neighs_bitmap; jj++)
        {
            current_neigh[jj] = cg::reduce(tile, current_neigh[jj], cg::bit_or<unsigned int>());
        }
        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
        {
            neigh_bitmap_orient[blockIdx.x][ii][jj] = current_neigh[jj];
        }
        // tile.sync(); //todo I dont think this is neccesary
    }
    // block.sync(); //block sync below is sufficent

    __shared__ cuda::atomic<int, cuda::thread_scope_block> num_cliques_total[g_const::max_clique_size]; // todo add up into global
    for (int ii = block.thread_rank(); ii < clique_size; ii += g_const::threads_per_block)
    {
        num_cliques_total[ii] = ii == 0;
    }

    block.sync();
    __shared__ unsigned int current_lvl_bitmap[num_tiles][(g_const::max_deg + 31) / 32]; // 32 kibibytes so should fit, todo check
    int lvl_idx[g_const::max_clique_size - 1];
    int lvl_num_neighs[g_const::max_clique_size - 2];
    int current_level; // current level is counting cliques of size +3
    for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
    {
        // *************  first level ***********************

        lvl_idx[0] = 0;
        current_level = 0;
        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
        {
            current_lvl_bitmap[tile_idx][jj] = neigh_bitmap_orient[blockIdx.x][ii][jj];
        }
        if (tile.thread_rank() == 0)
        {
            moduloAdd(num_cliques_total[1], 1);
        }
        tile.sync();
        // ******************* implicit stack used from here on out *********************
        do
        {
            while (lvl_idx[current_level] < num_neighs)
            {
                int idx = lvl_idx[current_level]++;
                if (idx == 0)
                {
                    lvl_idx[current_level + 1] = 0;
                    lvl_num_neighs[current_level] = 0;
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        lvl_num_neighs[current_level] += __popc(current_lvl_bitmap[tile_idx][jj]);
                    }
                    lvl_num_neighs[current_level] = cg::reduce(tile, lvl_num_neighs[current_level], cg::plus<int>());
                    if (lvl_num_neighs[current_level] > 0)
                    {
                        if (tile.thread_rank() == 0)
                        {
                            moduloAdd(num_cliques_total[current_level + 2], lvl_num_neighs[current_level]);
                        }
                        tile.sync();
                    }
                    if (current_level == clique_size - 3)
                    { // max depth reached
                        break;
                    }
                }

                if (lvl_num_neighs[current_level] == 0)
                {
                    break;
                }
                unsigned int temp = current_lvl_bitmap[tile_idx][idx / 32] & (1U << (idx % 32));
                if (temp > 0)
                {
                    lvl_num_neighs[current_level]--;
                    if (current_level > 0) // saves one transfer into global
                    {
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            lvl_bitmap_orient[blockIdx.x][tile_idx][current_level - 1][jj] = current_lvl_bitmap[tile_idx][jj];
                            current_lvl_bitmap[tile_idx][jj] = current_lvl_bitmap[tile_idx][jj] & neigh_bitmap_orient[blockIdx.x][idx][jj];
                        }
                    }
                    else
                    {
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            current_lvl_bitmap[tile_idx][jj] = current_lvl_bitmap[tile_idx][jj] & neigh_bitmap_orient[blockIdx.x][idx][jj];
                        }
                    }
                    temp = 0;
                    idx = 0;
                    current_level++;
                }
            }
            do
            {
                current_level--;
            } while (current_level >= 0 && lvl_num_neighs[current_level] == 0);
            if (current_level >= 0) // putting this here saves clique_size transfers
            {
                lvl_idx[current_level + 1] = 0;
                if (current_level > 0)
                {
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        current_lvl_bitmap[tile_idx][jj] = lvl_bitmap_orient[blockIdx.x][tile_idx][current_level - 1][jj]; // -1 as we only store levels greater than 0
                    }
                }
                else if (current_level == 0)
                { // doing it this way saves one transfer into global
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        current_lvl_bitmap[tile_idx][jj] = neigh_bitmap_orient[blockIdx.x][ii][jj];
                    }
                }
            }

        } while (current_level >= 0);
        tile.sync();
    }
    block.sync();
    for (int ii = block.thread_rank(); ii < clique_size; ii += block.size())
    {
        int val = num_cliques_total[ii];
        int expected = 0;
        int desired = val;
        int old = atomicCAS(&res[ii], expected, desired);
        while (old != expected)
        {
            expected = old;
            desired = (old + val) % g_const::mod;
            old = atomicCAS(&res[ii], expected, desired);
        }
    }
}

__global__ void countCliquesKernOrient(int *row_ptrs, int *v1s, int *v2s, int *res, int clique_size)
{
    for (int ii = blockIdx.x; ii < g_const::num_vertices_dev; ii += gridDim.x)
    {
        calculateIntersectsOrient(ii, row_ptrs, v1s, v2s, clique_size, res);
        __syncthreads();
    }
}

void countCliquesOrient(Graph &g, int clique_size, std::string output_path)
{
    thrust::device_vector<int> res_dev(clique_size, 0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
    int maxActiveBlocks;
    HANDLE_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                               countCliquesKernOrient, g_const::threads_per_block,
                                                               0));
    g_const::blocks_per_grid_host = (maxActiveBlocks + 1) * deviceProp.multiProcessorCount * 2;
    g_const::blocks_per_grid_host = g_const::blocks_per_grid_host > g_const::blocks_per_grid ? g_const::blocks_per_grid : g_const::blocks_per_grid_host;
    std::cout << "'Optimal' by formula is " << g_const::blocks_per_grid_host << " blocks\n";
    std::cout << "Using " << g_const::blocks_per_grid_host << " blocks of size " << g_const::threads_per_block << " for calculation\n";
    HANDLE_ERROR(cudaMemcpyToSymbol(g_const::blocks_per_grid_dev, &g_const::blocks_per_grid_host, sizeof g_const::blocks_per_grid_host));
    assert(g_const::blocks_per_grid_host > 0);
    countCliquesKernOrient<<<g_const::blocks_per_grid_host, g_const::threads_per_block>>>(
        thrust::raw_pointer_cast(g.row_ptr.data()),
        thrust::raw_pointer_cast(g.v1s.data()),
        thrust::raw_pointer_cast(g.v2s.data()),
        thrust::raw_pointer_cast(res_dev.data()),
        clique_size);
    PRINTER(res_dev);
    thrust::host_vector<int> res_host(res_dev);
    std::ofstream outfile;
    outfile.open(output_path);
    for (int ii = 0; ii < res_host.size(); ii++)
    {
        if (ii)
        {
            outfile << ' ';
        }
        outfile << res_host[ii];
    }
    outfile.close();
}