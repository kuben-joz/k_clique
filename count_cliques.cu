#include <thrust/device_vector.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector_types.h>
#include <string>
#include <fstream>
#include <cub/cub.cuh>

#include "constants.cuh"
#include "util.cuh"

namespace cg = cooperative_groups;
namespace stdc = cuda::std;

const int tile_size = 4;

__device__ unsigned int neigh_bitmap_lvl1[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int neigh_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int inv_neigh_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int lvl_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ unsigned int lvl_pruned_bitmap[g_const::blocks_per_grid][g_const::max_deg][(g_const::max_deg + 31) / 32];
__device__ int lvl_idx[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_num_neighs[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_num_pivots[g_const::blocks_per_grid][g_const::max_deg];
__device__ int lvl_pivot[g_const::blocks_per_grid][g_const::max_deg];

inline __device__ void moduloAdd(cuda::atomic<int, cuda::thread_scope_block> &res, int val)
{
    val = val % g_const::mod;
    int expected = res.load(cuda::memory_order_relaxed);
    int desired;
    do
    {
        desired = (expected + val) % g_const::mod;
    } while (!res.compare_exchange_weak(expected, desired, cuda::memory_order_relaxed));
}

// computes number of factors of p in n choose k
inline __device__ int numFactors(int n, int p)
{
    int s_n;
    if (p == 2)
    {
        s_n = __popc(n);
    }
    else
    {
        s_n = 0;
        for (; n > 0; n /= p)
        {
            s_n += n % p;
        }
    }

    return s_n;
}

const int tile_sz = tile_size;
// todo change to template, if I forgot it's because I had to change for intellisensen to work during dev, sorry
__device__ void calculateIntersects(const int v_idx, const int *__restrict__ row_ptrs, const int *__restrict__ v1s, const int *__restrict__ v2s, const int clique_size, int *__restrict__ res, int &block_idx)
{
    static_assert(g_const::max_deg == 1024); // this assumption is made for optimisations here
    auto block = cg::this_thread_block();
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
    auto tile = cg::tiled_partition<tile_sz>(cg::this_thread_block());
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
            neigh_bitmap_lvl1[block.group_index().x][ii][jj] = current_neigh[jj];
        }
    }
    // ###################################################################################################################
    for (int bb = block_idx; bb < num_neighs_lvl1; bb += g_const::blocks_per_grid)
    {
        __shared__ int neigh_ids[g_const::max_deg];

        __shared__ union SharedStorage1
        {
            unsigned int current_lvl_bitmap[2][(g_const::max_deg + 31) / 32];
            unsigned int temp_lvl1_bitmap[num_tiles][(g_const::max_deg + 31) / 32];
        } shr_str1;

        __shared__ union SharedStorage2
        {
            unsigned int current_lvl_pruned_bitmap[(g_const::max_deg + 31) / 32];
            unsigned int root_neigh_bitmap[(g_const::max_deg + 31) / 32];
        } shr_str2;

        typedef cub::BlockScan<int, g_const::threads_per_block, cub::BLOCK_SCAN_RAKING> BlockScanT32; // todo try others https://github.com/dmlc/cub/blob/master/cub/block/block_scan.cuh
        typedef cub::BlockScan<long long, g_const::threads_per_block, cub::BLOCK_SCAN_RAKING> BlockScanT64;

        __shared__ union SharedStorage3
        {
            int neigh_ids_rev[g_const::max_deg];
            // int n_C_k_factors[2][g_const::max_deg];
            typename BlockScanT32::TempStorage scan32;
            typename BlockScanT64::TempStorage scan64;
        } shr_str3;

        for (int ii = block.thread_rank(); ii < num_neighs_bitmap_lvl1; ii += g_const::threads_per_block)
        {
            shr_str2.root_neigh_bitmap[ii] = neigh_bitmap_lvl1[block.group_index().x][bb][ii];
        }
        // todo what if only cliques sof size 3?
        int num_neighs;
        __shared__ cuda::atomic<int, cuda::thread_scope_block> tmp_atomic; // reused later
        cg::invoke_one(block, [&]
                       { tmp_atomic.store(0, cuda::memory_order_relaxed); });
        block.sync();
        for (int ii = block.thread_rank(); ii < num_neighs_lvl1; ii += g_const::blocks_per_grid)
        {
            unsigned int temp = shr_str2.root_neigh_bitmap[ii / 32] & (1U << (ii % 32));
            if (temp)
            {
                int temp_index = tmp_atomic.fetch_add(1, cuda::memory_order_relaxed); // todo chekc if this conflicts with the setting of 0 above
                neigh_ids[temp_index] = ii;
                shr_str3.neigh_ids_rev[ii] = temp_index;
            }
        }
        block.sync();
        num_neighs = tmp_atomic.load(cuda::memory_order_relaxed);
        int num_neighs_bitmap = (num_neighs + 31) / 32;
        for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
        {
            int old_idx = neigh_ids[ii];
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap_lvl1; jj += tile_sz)
            {
                shr_str1.temp_lvl1_bitmap[tile_idx][jj] = neigh_bitmap_lvl1[block.group_index().x][old_idx][jj] & shr_str2.current_lvl_pruned_bitmap[jj];
            }
            unsigned int current_neigh[(g_const::max_deg + 31) / 32];
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                current_neigh[jj] = 0;
            }
            tile.sync(); // todo maybe not neccesary
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap_lvl1; jj += tile_sz)
            {
                unsigned int current = shr_str1.temp_lvl1_bitmap[tile_idx][jj];
                int kk = 0;
                for (; current > 0; current >>= 1)
                {
                    if (current % 2)
                    {
                        int temp_index = shr_str3.neigh_ids_rev[jj * 32 + kk];
                        current_neigh[temp_index / 32] |= 1U << (temp_index % 32);
                    }
                    kk++;
                }
            }
            // todo current_neigh or current_neigh_pruned bitmap?
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                current_neigh[jj] = cg::reduce(tile, current_neigh[jj], cg::bit_or<unsigned int>());
            }
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
            {
                neigh_bitmap[block.group_index().x][ii][jj] = current_neigh[jj];
            }
        }

        unsigned int temp_bitmap[(g_const::max_deg + 31) / 32];
        /*
        int lim = num_neighs_bitmap % num_tiles;
        lim = num_neighs_bitmap + (num_tiles - lim); // todo change all of this to 32
        for (int ii = tile_idx; ii < lim; ii += num_tiles)
        {
            for (int jj = 0; jj < 32; jj++)
            {
                if (ii < num_neighs_bitmap)
                {
                    for (int kk = 0; kk < num_neighs_bitmap; kk++)
                    {
                        temp_bitmap[kk] = 0;
                    }
                    for (int kk = tile.thread_rank(); kk < num_neighs; kk += tile_sz)
                    {
                        unsigned int temp = neigh_bitmap[block.group_index().x][kk][ii] & (1U << (jj));
                        if (temp)
                        {
                            temp_bitmap[kk / 32] |= 1U << (kk % 32);
                        }
                    }
                    for (int kk = 0; kk < num_neighs_bitmap; kk++)
                    {
                        temp_bitmap[kk] = cg::reduce(tile, temp_bitmap[kk], cg::bit_or<unsigned int>());
                    }
                }
                block.sync();
                if (ii < num_neighs_bitmap)
                {
                    for (int kk = tile.thread_rank(); kk < num_neighs; kk += tile_sz)
                    {
                        neigh_bitmap[block.group_index().x][ii * 32 + jj][kk] = temp_bitmap[kk];
                    }
                }
                block.sync();
            }
        }
        */

        for (int ii = tile_idx; ii < num_neighs; ii += num_tiles)
        {
            int v1 = ((ii * 32) % num_neighs) + ((ii * 32) / num_neighs); // avoid bank conflicts
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                temp_bitmap[jj] = 0;
            }
            for (int jj = tile.thread_rank() + 1; jj < num_neighs; jj += tile_sz)
            {
                int v2 = (v1 + jj) % num_neighs;
                unsigned int temp = neigh_bitmap[block.group_index().x][v2][v1 / 32] & (1U << (v1 % 32));
                if (temp)
                {
                    temp_bitmap[v2 / 32] |= 1U << (v2 % 32);
                }
            }
            for (int jj = 0; jj < num_neighs_bitmap; jj++)
            {
                temp_bitmap[jj] = cg::reduce(tile, temp_bitmap[jj], cg::bit_or<unsigned int>());
            }
            for (int jj = tile.thread_rank(); jj < num_neighs_bitmap; jj += tile_sz)
            {
                inv_neigh_bitmap[block.group_index().x][v1][jj] = temp_bitmap[jj];
            }
        }
        block.sync();
        for (int ii = block.thread_rank(); ii < num_neighs * 32; ii += g_const::threads_per_block)
        {
            neigh_bitmap[block.group_index().x][ii / 32][ii % 32] |= inv_neigh_bitmap[block.group_index().x][ii / 32][ii % 32];
        }

        // ###############################################################################################################

        __shared__ cuda::atomic<int, cuda::thread_scope_block> num_cliques_total[g_const::max_clique_size - 4]; // todo add up into global
        for (int ii = block.thread_rank(); ii < clique_size; ii += g_const::threads_per_block)
        {
            num_cliques_total[ii].store(ii == 3 ? num_neighs : 0, cuda::memory_order_relaxed);
        }
        block.sync();

        int current_level; // current level is counting cliques of size +2
        int current_idx;
        int current_num_neighs;
        int current_num_pivots;
        int current_pivot;
        int parity = 0;
        // *************  first level ***********************
        cg::invoke_one(block, [&]
                       { lvl_idx[block.group_index().x][0] = 0; });
        current_level = 0;
        current_num_pivots = 1; // todo chekc if 0
        current_idx = -1;       // will be incrmeneted immediatelly to 0

        for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::blocks_per_grid)
        {
            shr_str1.current_lvl_bitmap[parity][jj] = ~0U;
        }
        // ******************* implicit stack used from here on out *********************
        do
        {
            while (current_idx < num_neighs)
            {
                current_idx++;
                if (current_idx == 0)
                {
                    // cg::invoke_one(block, [&]
                    //                { lvl_idx[block.group_index().x][current_level + 1] = 0; }); // todo I think we can remove
                    int pivot;
                    int pivot_overlap;
                    for (int jj = tile_idx; jj < num_neighs; jj += num_tiles)
                    {
                        unsigned int temp = shr_str1.current_lvl_bitmap[parity][jj / 32] & (1U << (jj % 32));
                        if (!temp)
                        {
                            continue;
                        }
                        int overlap = 0;
                        for (int kk = tile.thread_rank(); kk < num_neighs_bitmap; kk += tile_sz)
                        {
                            overlap += __popc(shr_str1.current_lvl_bitmap[parity][kk] & neigh_bitmap[block.group_index().x][jj][kk]);
                        }
                        overlap = cg::reduce(tile, overlap, cg::plus<int>());
                        if (overlap > pivot_overlap)
                        {
                            pivot_overlap = overlap;
                            pivot = jj;
                        }
                    }
                    int temp_pivot_overlap = cg::reduce(block, pivot_overlap, cg::greater<int>());
                    if (temp_pivot_overlap == pivot_overlap)
                    { // if two pivots have the same overlap we choose one at random
                        cg::invoke_one(tile, [&]
                                       { tmp_atomic.store(pivot, cuda::memory_order_relaxed); });
                    }
                    block.sync();
                    pivot = cg::invoke_one_broadcast(tile, [&]()
                                                     { return tmp_atomic.load(cuda::memory_order_relaxed); }); // todo try to get it to work with block, I don't think it's possible
                    pivot_overlap = temp_pivot_overlap;

                    for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                    {
                        shr_str2.current_lvl_pruned_bitmap[jj] &= ~neigh_bitmap[block.group_index().x][pivot][jj];
                    }
                    current_num_neighs = pivot_overlap;
                    current_pivot = pivot;
                }
                if (current_num_neighs == 0)
                {
                    break;
                }
                unsigned int temp = shr_str2.current_lvl_pruned_bitmap[current_idx / 32] & (1U << (current_idx % 32));
                if (temp)
                {
                    current_num_neighs--;
                    int new_num_pivots = current_idx == current_pivot ? current_num_pivots + 1 : current_num_pivots;
                    // +4 because thats the clique size the next level repesents
                    if (current_level + 4 - clique_size <= new_num_pivots)
                    {
                        int next_num_neighs = 0;
                        for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                        { // current_lvl_bitmap[1] not zero to compute temprarily the I' in line 8 of figure 3 in paper
                            shr_str1.current_lvl_bitmap[!parity][jj] = shr_str1.current_lvl_bitmap[parity][jj] & neigh_bitmap[block.group_index().x][current_idx][jj];
                            temp = (current_idx / 32) == jj ? 1 : 0; // bit twiddling to ignore vertices that we covered earlier line 8 of pivoter alg in paper
                            temp = (temp << (current_idx % 32)) - 1;
                            temp = (current_idx / 32) > jj ? 0 : temp;
                            temp = temp & shr_str2.current_lvl_pruned_bitmap[jj];
                            shr_str1.current_lvl_bitmap[!parity][jj] &= ~temp;
                            next_num_neighs += __popc(shr_str1.current_lvl_bitmap[!parity][jj]);
                        }
                        next_num_neighs = cg::reduce(tile, next_num_neighs, cg::plus<int>());
                        if (next_num_neighs)
                        {
                            if (current_level > 0) // saves one transfer into global
                            {
                                for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                                {
                                    lvl_bitmap[block.group_index().x][current_level - 1][jj] = shr_str1.current_lvl_bitmap[parity][jj];
                                }
                            }
                            for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                            {
                                lvl_pruned_bitmap[block.group_index().x][current_level][jj] = shr_str2.current_lvl_pruned_bitmap[jj];
                            }
                            parity = !parity;
                            cg::invoke_one(tile, [&]
                                           { lvl_idx[block.group_index().x][current_level] = current_idx; });
                            cg::invoke_one(tile, [&]
                                           { lvl_num_neighs[block.group_index().x][current_level] = current_num_neighs; });
                            cg::invoke_one(tile, [&]
                                           { lvl_num_pivots[block.group_index().x][current_level] = current_num_pivots; });
                            cg::invoke_one(tile, [&]
                                           { lvl_pivot[block.group_index().x][current_level] = current_pivot; });
                            current_num_neighs = next_num_neighs;
                            current_idx = 0;
                            current_level++;
                        }
                        else // level + 1 is always countign at least 4 clqiues
                        {
                            // reusing neigh_ids_rev to store partial results
                            // https://math.stackexchange.com/questions/70125/calculating-n-choose-k-mod-one-million
                            int n = new_num_pivots;
                            const int per_thread = (g_const::max_deg + g_const::threads_per_block - 1) / g_const::threads_per_block;
                            int temp_twos[per_thread];
                            int temp_fives[per_thread];
                            long long temp_res[per_thread];
                            for (int ii = block.thread_rank(); ii <= g_const::max_deg; ii += g_const::threads_per_block)
                            {
                                int k = (per_thread * block.thread_rank() + (ii / g_const::threads_per_block));
                                // n_C_k[k] = k == 0; // n choose 0 is 1
                                temp_twos[k % per_thread] = 0;
                                temp_fives[k % per_thread] = 0;
                                temp_res[k % per_thread] = k == 0;

                                if (k > 0 && k <= current_level + 4 - 4)
                                {
                                    int numer = n - k + 1; // todo was it +2
                                    int denom = k;
                                    int num_fives = 0;
                                    int num_twos = 0;
                                    while (!(numer % 2))
                                    {
                                        numer /= 2;
                                        num_twos++;
                                    }
                                    while (!(numer % 5))
                                    {
                                        numer /= 5;
                                        num_fives++;
                                    }
                                    while (!(denom % 2))
                                    {
                                        denom /= 2;
                                        num_twos--;
                                    }
                                    while (!(denom % 2))
                                    {
                                        denom /= 5;
                                        num_fives--;
                                    }
                                    temp_twos[k % per_thread] = num_twos;
                                    temp_fives[k % per_thread] = num_fives;
                                    // find inverse mod 10^9 of denominator
                                    int x = 1;
                                    int y = 0;
                                    int x1 = 0;
                                    int y1 = 1;
                                    int a = denom;
                                    int b = g_const::mod;
                                    while (b)
                                    {
                                        int q = a / b;
                                        stdc::tie(x, x1) = stdc::make_tuple(x1, x - q * x1);
                                        stdc::tie(y, y1) = stdc::make_tuple(y1, y - q * y1);
                                        stdc::tie(a, b) = stdc::make_tuple(b, a - q * b);
                                    }
                                    assert(a != 1);
                                    unsigned long long res = (x % g_const::mod + g_const::mod) % g_const::mod;
                                    unsigned long long temp = numer % g_const::mod; // todo I dont think mod is neccesary here numer % mod
                                    res = (res * temp) % g_const::mod;
                                    temp_res[k % per_thread] = res;
                                }
                            }

                            block.sync(); // todo check if neccesary
                            BlockScanT32(shr_str3.scan32).InclusiveSum(temp_twos, temp_twos);
                            block.sync();
                            BlockScanT32(shr_str3.scan32).InclusiveSum(temp_fives, temp_fives);
                            block.sync(); // todo check if neccesary
                            for (int ii = block.thread_rank(); ii <= g_const::max_deg; ii += g_const::threads_per_block)
                            {
                                int k = (per_thread * block.thread_rank() + (ii / g_const::threads_per_block));
                                for (int jj = 0; jj < temp_twos[k % per_thread]; jj++)
                                {
                                    temp_res[k % per_thread] = (temp_res[k % per_thread] * 2ULL) % g_const::mod;
                                }
                                for (int jj = 0; jj < temp_fives[k % per_thread]; jj++)
                                {
                                    temp_res[k % per_thread] = (temp_res[k % per_thread] * 5ULL) % g_const::mod;
                                }
                            }
                            block.sync();
                            BlockScanT64(shr_str3.scan64).InclusiveScan(temp_res, temp_res, mod_mul<long long>());
                            block.sync(); // todo chekc if neccesary

                            for (int ii = block.thread_rank(); ii <= g_const::max_deg; ii += g_const::threads_per_block)
                            {
                                int k = (per_thread * block.thread_rank() + (ii / g_const::threads_per_block));
                                if (k >= 4 && k <= clique_size)
                                {
                                    moduloAdd(num_cliques_total[k - 4], temp_twos[k % per_thread]);
                                }
                            }
                        }
                    }
                }
            }
            int prev_level = current_level;
            do
            {
                current_level--;
            } while (current_level >= 0 && lvl_num_neighs[block.group_index().x][current_level] == 0);
            if (current_level >= 0) // putting this here saves clique_size transfers
            {
                // cg::invoke_one(tile, [&]
                //                { lvl_idx[block.group_index().x][current_level + 1] = 0; });
                current_num_neighs = lvl_num_neighs[block.group_index().x][current_level];
                if (current_level > 0 && prev_level - 1 != current_level)
                {
                    for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                    {
                        shr_str1.current_lvl_bitmap[parity][jj] = lvl_bitmap[block.group_index().x][current_level - 1][jj]; // -1 as we only store levels greater than 0
                    }
                }
                else if (prev_level - 1 == current_level)
                {
                    parity = !parity;
                }
                else
                { // doing it this way saves one transfer into global
                    for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                    {
                        shr_str1.current_lvl_bitmap[parity][jj] = ~0U;
                    }
                }
                for (int jj = block.thread_rank(); jj < num_neighs_bitmap; jj += g_const::threads_per_block)
                {
                    shr_str2.current_lvl_pruned_bitmap[jj] = lvl_pruned_bitmap[block.group_index().x][current_level][jj];
                }
                current_idx = lvl_idx[block.group_index().x][current_level];
                current_num_neighs = lvl_num_neighs[block.group_index().x][current_level];
                current_num_pivots = lvl_num_pivots[block.group_index().x][current_level];
                current_pivot = lvl_pivot[block.group_index().x][current_level];
            }
        } while (current_level >= 0);

        block.sync();

        for (int ii = block.thread_rank(); ii < clique_size; ii += block.size())
        {
            int val = num_cliques_total[ii].load(cuda::memory_order_relaxed);
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
    block_idx = (block_idx + num_neighs_bitmap_lvl1) % g_const::blocks_per_grid; // so spare blocks can start calculating next vertex
    // todo chekc this doesnt brake stuff
}

__global__ void countCliquesKern(const int *__restrict__ row_ptrs, const int *__restrict__ v1s, const int *__restrict__ v2s, int *__restrict__ res, const int clique_size)
{
    int block_idx = blockIdx.x;
    for (int ii = 0; ii < g_const::num_vertices_dev; ii++)
    {
        calculateIntersects(ii, row_ptrs, v1s, v2s, clique_size, res, block_idx);
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