#include <iostream>
#include <map>
#include <vector>
#include <algorithm>

#include "input.h"


using namespace std;

struct Graph{
  vector<int> vs;
  vector<int> v_ptrs;
  vector<int> es;
  vector<pair<int, int>> coo;
};

void getGraph() {
    Graph g = Graph();
    uint32_t a, b;
    int idx = 0;
    map<uint32_t, int> idxs;
    // renumber and put into COO
    while(cin >> a) {
        cin >> b;
        int a_idx = idxs[a];
        if(!a_idx) {
            a_idx = idx;
            idxs[a] = idx++;
        }
        int b_idx = idxs[b];
        if(!b_idx) {
            b_idx = idx;
            idxs[b] = idx++;
        }
        g.coo.emplace_back(a_idx, b_idx);
        g.coo.emplace_back(b_idx, a_idx);
    }
    sort(g.coo.begin(), g.coo.end(), [](auto &left, auto &right) {
        return left.first == right.first ? left.second < right.second : left.first < right.first;
    });
    // add CSR representation
    g.es.resize(g.coo.size());
    g.vs.resize(idx);
    g.v_ptrs.resize(idx+1);
    int v_idx = 0;
    int e_idx = 0;
    int v_prev = -1;
    for(const auto &e : g.coo) {
        int v = e.first;
    }






}