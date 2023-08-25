- maybe dont renumber vertices in input.cu
  - remeber to change to uint32_t then
  - sort on gpu
  - scan operation to recompute csr pointers
  - check by key stuff on thrust as that is kidna csr
  - learn hsitogram sparse vs dense
  - add ifddef for execution policy on device for debugging vs gpu
  - add sub threadgroup sync
  - template for printing vector

  - cliques of size 3 with normal algorithm for orientation



  ## tests
  - empty graph
  - dense histogram works for isolated vertices

  ## optim
  - don't calculate result for cliques size 1 and 2
  - consts for numer of edges etc.
  - texture memory for edges that are close 
  - for each with atomics instead of reduce by eky
  - host malloc to not page memory upon reading
  - maybe even jsut gather straight into pointer array
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters
  - --use_fast_math
  - --compiler-options "-fopenmp -fexceptions -ffast-math -g -Wall"

  - parity optim for non pivot version
  - parity optim for pruned bitmap for pivot version
  - extend it for as many levels as possible, at least tile size for non pruned bitmap, also only save to global once required
  - same can be done for rest of stack variables

  - move bloocks and tiles to main level so we dont re init them every time we enter new vertex

## CRITICAL
  - don't calculate result for cliques size 1 and 2


## report
didnt use grid synchronisation as it seemed slower


![Alt text](image.png)