#pragma once

#define X_TID (blockIdx.x*blockDim.x+threadIdx.x)
#define Y_TID (blockIdx.y*blockDim.y+threadIdx.y)
#define Z_TID (blockIdx.z*blockDim.z+threadIdx.z)

#define V_IDX 0
#define E_IDX 1
#define CONST_NUM 2