# Cuda assingment report

## Implementation

### Libraries and CUDA compute capabilities
The implementation makes use of the followoing "newer" compute capability features..
- Tiled partitions
    - Compile time templated size for better code optimisation
- Cooperative groups
    - `invoke_one` instead of choosing thread by idx
    - `reduce` within a tile partition

And the libraries:
- Thrust for orientation and passing data to kernel
- cub for `BlockScan` and `BlockReduce` within the kernel

### Algorithms
The implementation uses:
- Vertex centric graph orientation for queries of $k < 8$, for $7$ this is about 2x faster with skitter as input
- Edge centric pivoting for larger size clique searches

Both make use of the degree edge orientation criteria.


### Optimisations
The implementation of vertex centric graph orientation and edge centric pivoting are two completelely seprate kernels. The choice of which one to run is decided upon by the input parameters. The edge centric and pivoting algorithm applicability almost overlapped in the publication (being the choice at $k=6$ and $k=7$ respectivelly). So I decided to have just the two implementations instead of all 4 combinations.

The graph orientation is done using thrust operations on GPU. Chosen mostly as an educational exercise in using the library. I believe cub would be faster, but the time save is negligible in comparison to the actual clique search for high k. It would probably improve the througput for many small graphs, but at that point I think it would be better to merge the graphs into one problem.

Both implementations use bitmaps as decribed in the publication.. Wherether possible I interlaced the threads to access separate ints in the bitmaps so the threads $0,1,2,3...$ would look at bits $0,32,64,...1,33,65$ to avoid memory access collisions as much as possible.

I tried to maximise usage of `__shared__` memory by using unions of shared storage for different things. One optimisation I didn't do was remember the current level bitmap for n levels instead of just 2 to reduce the number of transfers into the artifical recursion stack, which has to be in global memory. This applies only to the pivot kernel.

### Parameter Selection
Seems like 128 threads per block are optimal

Blocks per grid are `cudaOccupancyMaxActiveBlocksPerMultiprocessor*deviceProp.multiProcessorCount*2` this seems close to optimal for the GPUs I tried, but I didn't have access to entropy in the end. The makefile generates code for all GPUs that this will likely be run on.

