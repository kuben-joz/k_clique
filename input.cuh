#pragma once

#include <vector>
#include <utility>

typedef struct Graph{
  std::vector<uint32_t> vs; // CSR rep
  std::vector<int> v_ptrs;
  std::vector<uint32_t> es;
  std::vector<std::pair<uint32_t, uint32_t>> coo; // COO rep
} Graph;


Graph getGraph();
