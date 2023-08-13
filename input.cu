#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

#include "input.cuh"


using namespace std;

Graph getGraph() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    Graph g = Graph();
    uint32_t a, b;
    map<uint32_t, vector<uint32_t>> v_neighs;
    int num_es = 0;
    while(cin >> a) {
        cin >> b;
        v_neighs[a].emplace_back(b);
        v_neighs[b].emplace_back(a);
        num_es += 2;
    }
    /*
    sort(g.coo.begin(), g.coo.end(), [](auto &left, auto &right) {
        return left.first == right.first ? left.second < right.second : left.first < right.first;
    });
     */
    g.es.resize(num_es);
    g.vs.resize(v_neighs.size());
    g.v_ptrs.resize(v_neighs.size()+1);
    g.coo.resize(num_es);
    auto vs_it = g.vs.begin();
    auto v_ptrs_it = g.v_ptrs.begin();
    *v_ptrs_it = 0;
    v_ptrs_it++;
    int v_ptr_prev = 0;
    auto es_it = g.es.begin();
    auto coo_it = g.coo.begin();
    for(const auto &v : v_neighs) {
        uint32_t v_id = v.first;
        vector<uint32_t> v_neigh = v.second;
        sort(v_neigh.begin(), v_neigh.end());
        // add COO
        for(auto n : v_neigh) {
            *coo_it = make_pair(v_id, n);
            coo_it++;
        }
        // add CSR
        *vs_it = v_id;
        vs_it++;
        *v_ptrs_it = v_ptr_prev + v_neigh.size();
        v_ptr_prev = *v_ptrs_it;
        v_ptrs_it++;
        for(auto n : v_neigh) {
            *es_it = n;
            es_it++;
        }
    }
    *v_ptrs_it = num_es;
    assert(v_ptrs_it == g.v_ptrs.end());
    assert(vs_it == g.vs.end());
    assert(es_it == g.es.end());
    assert(coo_it == g.coo.end());
    return g;
}