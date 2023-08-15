#include <thrust/device_vector.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector_types.h>

#include "constants.cuh"
#include "util.cuh"

namespace cg = cooperative_groups;

const int tile_size = 32;
const int threads_per_block = 256;
const int blocks_per_grid = min(32, (N + threads_per_block - 1) / threads_per_block);
const unsigned int FULL_MASK = 0xffffffff;

class Bitmap
{
private:
    int len;
    __device__ char *data;
};

__device__ int countSubtreeCliques(int level, int *num_cliques, int clique_size,
                                   int neigh_ids[max_deg], int neigh_bitmasks[max_deg][32],
                                   int v_idx, int v_neigh_idx, int *row_ptrs, int *v1s, int *v2s)
{
}

template <int tile_sz>
__device__ Bitmap calculateIntersects(int v_idx, int *row_ptrs, int *v1s, int *v2s, int clique_size)
{
    static_assert(max_deg == 1024); // this assumption is made for optimisations here
    // static_assert(threads_per_block == 256); // this assumption is made for optimisations here
    cg::thread_group block = cg::this_thread_block();
    __shared__ int neigh_ids[max_deg];
    __global__ unsigned int neigh_bitmasks[max_deg][32];
    int start = row_ptrs[v_idx];
    int end = row_ptrs[v_idx + 1];
    const int num_neighs = end - start;
    const int num_neighs_bitmap = (num_neighs + 31) / 32 for (int ii = block.thread_rank(); ii < num_neighs; ii += blockDim.x)
    {
        neigh_ids[ii + start] = v2s[ii + start];
    }
    block.sync();
    cg::thread_block_tile<tile_sz> tile = cg::tiled_partition<tile_sz>(block);
    const int tile_idx = block.thread_rank() / tile_sz;
    const int num_tiles = threads_per_block / tile_sz;
    for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
    {
        unsigned int current_neigh[32];                // 32*32 = 1024 i.e. max num neighs
        for (int jj = 0; jj < num_neighs_bitmap; jj++) // this is supposedly faster than cudamemset
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
                current_neigh[kk / 32] &= ((unsigned int)(neigh_ids[kk] == neigh_v2)) << (kk % 32);
            }
        }
        for (int jj = 0; jj < num_neighs_bitmap; jj++)
        {
            current_neigh[jj] = cg::reduce(tile, current_neigh[jj], cg::bit_and<unsigned int>());
        }
        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
        {
            neigh_bitmasks[neigh_v1][jj] = current_neigh[jj];
        }
        // tile.sync(); //todo I dont think htis is neccesary
    }
    // block.sync(); //block sync below is sufficent

    __shared__ cuda::atomic<int, cuda::thread_scope_block> num_cliques_total[max_clique_size];
    for (int ii = block.thread_rank(); ii < clique_size; ii += threads_per_block)
    {
        num_cliques_total[ii] = ii == 0;
    }
    block.sync();
    __shared__ unsigned int current_lvl_bitmap[num_tiles][32];              // 32 kibibytes so should fit, todo check
    __global__ unsigned int lvl_bitmap[num_tiles][max_clique_size - 4][32]; // todo this might be +1, will be buggy regardless if I set max_clqiue_size ot 3 during testing
    unsigned int lvl_idx[max_clique_size - 1];                              // todo this mioght be +1
    unsigned int lvl_num_neighs[max_clique_size - 2];                       // todo this might be + 1
    int current_level;                                                      // current level is counting cliques of size +3
    for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
    {
        // *************  first level ***********************
        int start_neigh = ii;
        lvl_idx[0] = 0;
        current_level = 0;
        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
        {
            current_lvl_bitmap[tile_idx][jj] = neigh_bitmasks[ii][jj];
        }
        if (tile.thread_rank() == 0)
        {
            num_cliques_total[1].fetch_add(1, cuda::memory_order_relaxed); // can be relaxed as the value is not needed elsewhere
        }
        // ******************* implicit stack used from here on out *********************
        do
        {
            while (lvl_idx[current_level] < num_neighs)
            {
                int idx = lvl_idx[current_level]++;
                if (idx == 0)
                {
                    level_idx[current_level + 1] = 0;
                    lvl_num_neighs[current_level] = 0;
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        lvl_num_neighs[current_level] += __popc(current_lvl_bitmap[tile_idx][jj]);
                    }
                    cg::reduce(tile, lvl_num_neighs[current_level], cd::plus<int>());
                    if (lvl_num_neighs[current_level] > 0 && tile.thread_rank() == 0)
                    {
                        num_cliques_total[level + 2].fetch_add(lvl_num_neighs[current_level], cuda::memory_order_relaxed);
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
                if (current_lvl_bitmap[tile_idx][idx / 32] & (0UL << (idx % 32)))
                {
                    lvl_num_neighs[current_level]--;
                    if (current_level > 0) // saves one transfer into global
                    {
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            lvl_bitmap[tile_idx][current_level - 1][jj] = current_lvl_bitmap[tile_idx][jj];
                            current_lvl_bitmap[tile_idx][jj] = current_lvl_bitmap[tile_idx][jj] & neigh_bitmasks[idx][jj];
                        }
                    }
                    current_level++;
                }
            }
            current_level--;
            if (current_level >= 0 && lvl_idx[current_level] < num_neighs) // putting this here saves clique_size transfers
            {
                if (current_level > 0)
                {
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        current_lvl_bitmap[tile_idx][jj] = lvl_bitmap[tile_idx][current_level - 1][jj];
                    }
                }
                else if (current_level == 0)
                { // doing it this way saves one transfer into global
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        current_lvl_bitmap[tile_idx][current_level][jj] = neigh_bitmasks[ii][jj];
                    }
                }
            }
        } while (current_level >= 0);
    }

    /*
     cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<int>());

    // synchronize the block, to ensure all async reductions are ready
    block.sync();
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tiled-partition
    */
}

__global__ void countCliquesKern(int *row_ptrs, int *v1s, int *v2s, int *res, int clique_size)
{
    __shared__ int block_res = 0;
    for (int ii = blockIdx.x; ii < num_vertices_dev; ii += blockDim.x * gridDim.x)
    {
        Bitmap bt = calculateIntersects<tile_size>(ii, row_ptrs, v1s, v2s, clique_size);
    }
}

void countCliques(thrust::device_vector<int> row_pointer, thrust::device_vector<int> v1s, thrust::device_vector<int> v2s)
{
}