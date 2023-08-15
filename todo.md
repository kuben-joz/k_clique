- maybe dont renumber vertices in input.cu
  - remeber to change to uint32_t then
  - sort on gpu
  - scan operation to recompute csr pointers
  - check by key stuff on thrust as that is kidna csr
  - learn hsitogram sparse vs dense
  - add ifddef for execution policy on device for debugging vs gpu
  - add sub threadgroup sync
  - template for printing vector



  ## tests
  - empty graph

  ## optim
  - consts for numer of edges etc.
  - texture memory for edges that are close 
  - for each with atomics instead of reduce by eky
  - host malloc to not page memory upon reading
  - maybe even jsut gather straight into pointer array
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters