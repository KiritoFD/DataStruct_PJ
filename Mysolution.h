#pragma once
#include <vector>
#include <cstdint>
#include <cstring>

// 默认参数常量
constexpr int HNSW_DEFAULT_M = 43;
constexpr int HNSW_DEFAULT_MAX_LAYER = 10;
constexpr int HNSW_DEFAULT_EF_CONSTRUCTION = 735;
constexpr int HNSW_DEFAULT_EF_SEARCH = 425;


// 使用宏守卫避免与 visited_list.h 重复定义
#ifndef VISITED_LIST_DEFINED
#define VISITED_LIST_DEFINED

class VisitedList {
public:
    std::vector<uint8_t> arr;
    uint8_t cur_tag = 1;
    int cap = 0;

    void init(int n) {
        if (n > cap) { arr.assign(n, 0); cap = n; cur_tag = 1; }
    }
    void advance() {
        if (++cur_tag == 0) { std::memset(arr.data(), 0, cap); cur_tag = 1; }
    }
    bool isVisited(int id) const { return arr[id] == cur_tag; }
    void mark(int id) { arr[id] = cur_tag; }
};

class TagVisitedList {
public:
    std::vector<uint16_t> arr;
    uint16_t cur_tag = 1;
    int cap = 0;

    void init(int n) {
        if (n > cap) { arr.assign(n, 0); cap = n; cur_tag = 1; }
    }
    void advance() {
        if (++cur_tag == 0) { std::memset(arr.data(), 0, cap * sizeof(uint16_t)); cur_tag = 1; }
    }
    bool isVisited(int id) const { return arr[id] == cur_tag; }
    void mark(int id) { arr[id] = cur_tag; }
    const uint16_t* data() const { return arr.data(); }
    uint16_t currentTag() const { return cur_tag; }
};

#endif // VISITED_LIST_DEFINED

class Solution {
public:
    Solution();
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* result);
    void setK(int k) { k_ = k; }
private:
    int k_;
};

void build_hnsw(int d, const std::vector<float>& base);
std::vector<std::pair<int, float>> search_hnsw(const std::vector<float>& query, int k);

extern "C" {
    void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads);
    void set_hnsw_debug(int dbg);
    void set_ablation_flags(int csr, int prefetch, int simd, int pruning, int heap);
    void get_ablation_flags(int* csr, int* prefetch, int* simd, int* pruning, int* heap);
    void set_ablate_csr(int on);
    void set_ablate_prefetch(int on);
    void set_ablate_simd(int on);
    void set_ablate_pruning(int on);
    void set_ablate_heap(int on);
    uint64_t get_total_queries();
    double get_avg_dists_per_query();
    uint64_t get_last_query_dists();
    void reset_dist_counters();
    double get_last_build_time_ms();
    int get_graph_max_level();
    int get_graph_num_nodes();
    double get_graph_avg_degree_l0();
    int get_graph_actual_max_layer();
    int get_graph_nodes_at_level(int level);
    double get_graph_avg_degree_upper();
}
