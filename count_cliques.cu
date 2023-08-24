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

const int tile_size = 4;

__device__ unsigned int neigh_bitmap_lvl1[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int neigh_bitmaps[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int lvl_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int lvl_pruned_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ int lvl_idx[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_num_neighs[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_num_pivots[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_pivot[g_const::blocks_per_grid][g_const::max_deg];

inline __device__ void modulo_add(cuda::atomic<int, cuda::thread_scope_block> &res, int val)
{
    val = val % g_const::mod;
    int expected = res;
    int desired;
    do
    {
        desired = (expected + val) % g_const::mod;
    } while (!res.compare_exchange_weak(expected, desired, cuda::memory_order_relaxed));
}

const int tile_sz = tile_size;
// todo change to template, if I forgot it's because I had to change for intellisensen to work during dev, sorry
__device__ void calculateIntersects(int v_idx, int *row_ptrs, int *v1s, int *v2s, int clique_size, int *res)
{
    static_assert(g_const::max_deg == 1024); // this assumption is made for optimisations here
    cg::thread_group block = cg::this_thread_block();
    __shared__ int neigh_ids_lvl1[g_const::max_deg];

    int start = row_ptrs[v_idx];
    int end = row_ptrs[v_idx + 1];
    const int num_neighs_lvl1 = end - start;
    const int num_neighs_bitmap_lvl1 = (num_neighs_lvl1 + 31) / 32;
    for (int ii = block.thread_rank(); ii < num_neighs_lvl1; ii += g_const::threads_per_block)
    {
        neigh_ids_lvl1[ii] = v2s[ii + start];
    }
    block.sync();
    cg::thread_block_tile<tile_sz> tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
    assert(tile.size() == tile_sz);
    const int tile_idx = block.thread_rank() / tile_sz;
    const int num_tiles = g_const::threads_per_block / tile_sz;
    for (int ii = tile_idx; ii < num_neighs_lvl1; ii += num_tiles)
    {
        unsigned int current_neigh[(g_const::max_deg + 31) / 32]; // 32*32 = 1024 i.e. max num neighs
        for (int jj = 0; jj < num_neighs_bitmap_lvl1; jj++)       // this is supposedly faster than cudamemset
        {
            current_neigh[jj] = 0;
        }
        int neigh_v1 = neigh_ids_lvl1[ii];
        int neigh_start = row_ptrs[neigh_v1];
        int neigh_end = row_ptrs[neigh_v1 + 1];
        for (int jj = tile.thread_rank(); jj < neigh_end - neigh_start; jj += tile_sz)
        {
            int neigh_v2 = v2s[jj + neigh_start];
            for (int kk = 0; kk < num_neighs_lvl1; kk++)
            {
                unsigned int temp = neigh_ids_lvl1[kk] == neigh_v2;
                current_neigh[kk / 32] |= (temp) << (kk % 32);
            }
        }
        for (int jj = 0; jj < num_neighs_bitmap_lvl1; jj++)
        {
            current_neigh[jj] = cg::reduce(tile, current_neigh[jj], cg::bit_or<unsigned int>());
        }
        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap_lvl1; jj += tile_sz)
        {
            neigh_bitmap_lvl1[blockIdx.x][ii][jj] = current_neigh[jj];
        }
    }
    //###################################################################################################################
    for (int bb = blockIdx.x; bb < num_neighs_lvl1; bb += g_const::blocks_per_grid)
    {
        __shared__ int neigh_ids[g_const::max_deg];
        __shared__ int neigh_idxs[g_const::max_deg];
        __shared__ unsigned int current_lvl_bitmap[num_tiles][(g_const::max_deg + 31) / 32];        // reused later to save shared mem
        __shared__ unsigned int current_lvl_pruned_bitmap[num_tiles][(g_const::max_deg + 31) / 32]; // reused later to save shared mem
        __shared__ unsigned int root_bitmap[(g_const::max_deg + 31) / 32];
        for (int ii = block.thread_rank(); ii < num_neighs_bitmap_lvl1; ii += g_const::threads_per_block)
        {
            root_bitmap[ii] = neigh_bitmap_lvl1[blockIdx.x][bb][ii];
        }
        // todo what if only cliques sof size 3?
        int num_neighs;
        __shared__ cuda::atomic<int, cuda::thread_scope_block> index;
        cg::invoke_one(block, [&]
                       { index = 0; });
        block.sync();
        for (int ii = block.thread_rank(); ii < num_neighs_lvl1; ii += g_const::blocks_per_grid)
        {
            unsigned int temp = root_bitmap[ii / 32] & (1U << (ii % 32));
            if (temp)
            {
                int temp_index = index.fetch_add(1, cuda::memory_order_relaxed); // todo chekc if this conflicts with the setting of 0 above
                neigh_ids[temp_index] = neigh_ids_lvl1[ii];
                neigh_idxs[temp_index] = ii;
            }
            else {
                neigh_idxs[ii] = -1;
            }
        }
        block.sync();
        num_neighs = index;
        int num_neighs_bitmap = (num_neighs + 31) / 32;
        for (int ii = tile_idx; ii < num_neighs_lvl1; ii += num_tiles)
        {
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap_lvl1; jj += tile_sz)
            {
                current_lvl_bitmap[tile_idx][jj] = neigh_bitmap_lvl1[blockIdx.x][ii][jj] & root_bitmap[jj];
            }
            unsigned int current_neigh[(g_const::max_deg + 31) / 32];
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                current_neigh[jj] = 0;
            }
            tile.sync(); // todo maybe not neccesary
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap_lvl1; jj += tile_sz)
            {
                unsigned int current = current_lvl_bitmap[tile_idx][jj];
                int kk = 0;
                kk = 0;
                for (; current > 0; current >>= 1)
                {
                    if (current % 2)
                    {
                        int temp_index = neigh_idxs[jj * 32 + kk];
                        current_neigh[temp_index / 32] |= 1U << (temp_index % 32);
                    }
                    kk++;
                }
            }
            //todo current_neigh or current_neigh_pruned bitmap?
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                current_neigh[jj] = cg::reduce(tile, current_neigh[jj], cg::bit_or<unsigned int>());
            }
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
            {
                lvl_bitmap[blockIdx.x][]
            }
        }

        //###############################################################################################################

        int start = row_ptrs[v_idx];
        int end = row_ptrs[v_idx + 1];
        const int num_neighs = end - start;
        const int num_neighs_bitmap = (num_neighs + 31) / 32;
        for (int ii = block.thread_rank(); ii < num_neighs; ii += g_const::threads_per_block)
        {
            neigh_ids[ii] = v2s[ii + start];
        }
        block.sync();
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
                neigh_bitmap[blockIdx.x][ii][jj] = current_neigh[jj];
            }
        }

        __shared__ cuda::atomic<int, cuda::thread_scope_block> num_cliques_total[g_const::max_clique_size]; // todo add up into global
        for (int ii = block.thread_rank(); ii < clique_size; ii += g_const::threads_per_block)
        {
            num_cliques_total[ii] = ii == 0;
        }
        block.sync();

        int current_level; // current level is counting cliques of size +3
        int current_idx;
        int current_num_neighs;
        int current_num_pivots;
        int current_pivot;
        for (int ii = blockIdx.x; ii < num_neighs; ii += gridDim.x) // todo this is wrong
        {
            // *************  first level ***********************
            cg::invoke_one(block, [&]
                           { lvl_idx[blockIdx.x][tile_idx][0] = 0; });
            current_level = 0;
            current_num_pivots = 1; // todo chekc if 0
            current_idx = -1;       // will be incrmeneted immediatelly to 0

            for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::blocks_per_grid)
            {
                current_lvl_bitmap[tile_idx][jj] = neigh_bitmaps[blockIdx.x][ii][jj];
            }
            // ******************* implicit stack used from here on out *********************
            do
            {
                while (current_idx < num_neighs)
                {
                    current_idx++;
                    if (current_idx == 0)
                    {
                        cg::invoke_one(block, [&]
                                       { lvl_idx[blockIdx.x][tile_idx][current_level + 1] = 0; });
                        int pivot;
                        int pivot_overlap;
                        for (int jj = 0; jj < num_neighs; jj++)
                        {
                            unsigned int temp = current_lvl_bitmap[tile_idx][jj / 32] & (1U << (jj % 32));
                            if (!temp)
                            {
                                continue;
                            }
                            int overlap = 0;
                            for (int kk = tile.thread_rank(); kk < num_neighs_bitmap; kk += tile_sz)
                            {
                                overlap += __popc(current_lvl_bitmap[tile_idx][kk] & neigh_bitmaps[blockIdx.x][ii][kk]);
                            }
                            overlap = cg::reduce(tile, overlap, cg::plus<int>());
                            if (overlap > pivot_overlap)
                            {
                                pivot_overlap = overlap;
                                pivot = jj;
                            }
                        }
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            current_lvl_pruned_bitmap[tile_idx][jj] &= ~neigh_bitmaps[blockIdx.x][pivot][jj];
                        }
                        current_num_neighs = pivot_overlap;
                        current_pivot = pivot;
                    }
                    if (current_num_neighs == 0)
                    {
                        break;
                    }
                    unsigned int temp = current_lvl_pruned_bitmap[tile_idx][current_idx / 32] & (1U << (current_idx % 32));
                    if (temp)
                    {
                        current_num_neighs--;
                        int new_num_pivots = current_idx == current_pivot ? current_num_pivots + 1 : current_num_pivots;
                        if (current_level + 4 - 3 <= new_num_pivots)
                        {
                            if (current_level > 0) // saves one transfer into global
                            {
                                for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                                {
                                    lvl_bitmap[blockIdx.x][tile_idx][current_level - 1][jj] = current_lvl_bitmap[tile_idx][jj];
                                }
                            }
                            int next_num_neighs = 0;
                            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                            {
                                current_lvl_bitmap[tile_idx][jj] &= neigh_bitmaps[blockIdx.x][current_idx][jj];
                                temp = current_idx / 32 == jj ? 1 : 0; // bit twiddling to ignore vertices that we covered earlier line 8 of pivoter alg in paper
                                temp = (temp << (current_idx % 32)) - 1;
                                temp = current_idx / 32 > jj ? 0 : temp;
                                temp = temp & current_lvl_pruned_bitmap[tile_idx][jj];
                                current_lvl_bitmap[tile_idx][jj] &= ~temp;
                                next_num_neighs += __popc(current_lvl_bitmap[tile_idx][jj]);
                            }
                            cg::reduce(tile, next_num_neighs, cg::plus<int>());
                            if (next_num_neighs)
                            {
                                for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                                {
                                    lvl_pruned_bitmap[blockIdx.x][tile_idx][current_level][jj] = current_lvl_pruned_bitmap[tile_idx][jj];
                                }
                                cg::invoke_one(tile, [&]
                                               { lvl_idx[blockIdx.x][tile_idx][current_level] = current_idx; });
                                cg::invoke_one(tile, [&]
                                               { lvl_num_neighs[blockIdx.x][tile_idx][current_level] = current_num_neighs; });
                                cg::invoke_one(tile, [&]
                                               { lvl_num_pivots[blockIdx.x][tile_idx][current_level] = current_num_pivots; });
                                cg::invoke_one(tile, [&]
                                               { lvl_pivot[blockIdx.x][tile_idx][current_level] = current_pivot; });
                                current_level++;
                            }
                            else
                            {
                                int lim = current_level + 3 < clique_size ? current_level + 3 : clique_size;
                                for (int jj = tile.thread_rank(); jj < lim; jj += tile_sz)
                                {
                                }
                            }
                        }
                    }
                }
                do
                {
                    current_level--;
                } while (current_level >= 0 && lvl_num_neighs[blockIdx.x][tile_idx][current_level] == 0);
                if (current_level >= 0) // putting this here saves clique_size transfers
                {
                    cg::invoke_one(tile, [&]
                                   { lvl_idx[blockIdx.x][tile_idx][current_level + 1] = 0; });
                    current_num_neighs = lvl_num_neighs[blockIdx.x][tile_idx][current_level];
                    if (current_level > 0)
                    {
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            current_lvl_bitmap[tile_idx][jj] = lvl_bitmap[blockIdx.x][tile_idx][current_level - 1][jj]; // -1 as we only store levels greater than 0
                        }
                    }
                    else
                    { // doing it this way saves one transfer into global
                        for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                        {
                            current_lvl_bitmap[tile_idx][jj] = neigh_bitmaps[blockIdx.x][ii][jj];
                        }
                    }
                    for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
                    {
                        current_lvl_pruned_bitmap[tile_idx][jj] = lvl_pruned_bitmap[blockIdx.x][tile_idx][current_level][jj];
                    }
                    current_idx = lvl_idx[blockIdx.x][tile_idx][current_level];
                    current_num_neighs = lvl_num_neighs[blockIdx.x][tile_idx][current_level];
                    current_num_pivots = lvl_num_pivots[blockIdx.x][tile_idx][current_level];
                    current_pivot = lvl_pivot[blockIdx.x][tile_idx][current_level];
                }
            } while (current_level >= 0);
            tile.sync();
        }
        block.sync();
    }
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

__global__ void countCliquesKern(int *row_ptrs, int *v1s, int *v2s, int *res, int clique_size)
{
    for (int ii = 0; ii < g_const::num_vertices_dev; ii++)
    {
        calculateIntersects(ii, row_ptrs, v1s, v2s, clique_size, res);
        __syncthreads();
    }
}

void countCliques(Graph &g, int clique_size, std::string output_path)
{
    thrust::device_vector<int> res_dev(clique_size, 0);
    countCliquesKern<<<g_const::blocks_per_grid, g_const::threads_per_block>>>(
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