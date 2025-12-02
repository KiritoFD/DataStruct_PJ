#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <vector>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <xmmintrin.h>

// ---------------------------------------------------------
// HNSW 默认参数常量 (单一定义)
// ---------------------------------------------------------
constexpr int HNSW_DEFAULT_M = 40;
constexpr int HNSW_DEFAULT_MAX_LAYER = 17;
constexpr int HNSW_DEFAULT_EF_CONSTRUCTION = 648;
constexpr int HNSW_DEFAULT_EF_SEARCH = 457;

// ---------------------------------------------------------
// 预取宏定义
// ---------------------------------------------------------
#define PREFETCH_L1(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#define PREFETCH_L2(ptr) _mm_prefetch((const char*)(ptr), _MM_HINT_T1)

// ---------------------------------------------------------
// VisitedList (兼容旧接口)
// ---------------------------------------------------------
class VisitedList {
public:
    std::vector<unsigned short> visited_tags;
    unsigned short curr_tag;
    int capacity;

    VisitedList() : curr_tag(0), capacity(0) {}

    inline void init(int size) {
        if (size > capacity) {
            visited_tags.resize(size, 0);
            capacity = size;
            curr_tag = 0;
        }
    }

    inline void advance() {
        if (++curr_tag == 0) {
            std::memset(visited_tags.data(), 0, capacity * sizeof(unsigned short));
            curr_tag = 1;
        }
    }

    inline bool isVisited(int id) const { return visited_tags[id] == curr_tag; }
    inline void mark(int id) { visited_tags[id] = curr_tag; }
    
    // 暴露 data() 用于预取
    inline const unsigned short* data() const { return visited_tags.data(); }
    inline unsigned short currentTag() const { return curr_tag; }
};

// ---------------------------------------------------------
// TagVisitedList (高性能版本，用于 FlatHNSW 搜索)
// 优点：
// 1. 避免每次搜索 memset 清零 (O(N) -> O(1))
// 2. 暴露 .data() 指针用于 SIMD 预取
// 3. 使用 uint16_t 节省带宽
// ---------------------------------------------------------
class TagVisitedList {
public:
    std::vector<uint16_t> tags;
    uint16_t current_tag;
    int capacity;

    TagVisitedList() : current_tag(0), capacity(0) {}

    inline void init(int n) {
        if (n > capacity) {
            tags.resize(n, 0);
            capacity = n;
            current_tag = 0;
        }
    }

    inline void advance() {
        if (++current_tag == 0) {
            std::fill(tags.begin(), tags.end(), (uint16_t)0);
            current_tag = 1;
        }
    }

    inline bool isVisited(int id) const { return tags[id] == current_tag; }
    inline void mark(int id) { tags[id] = current_tag; }
    inline const uint16_t* data() const { return tags.data(); }
    inline uint16_t currentTag() const { return current_tag; }
};

// ---------------------------------------------------------
// 对外接口类
// ---------------------------------------------------------
class Solution {
public:
    Solution();
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
private:
    int k_ = 10;
};

// ---------------------------------------------------------
// C 接口 (参数设置、统计、图质量)
// ---------------------------------------------------------
extern "C" {
    void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads);
    void set_hnsw_debug(int dbg);
    uint64_t get_total_queries();
    double get_avg_dists_per_query();
    uint64_t get_last_query_dists();
    void reset_dist_counters();
    double get_last_build_time_ms();

    // 图质量统计
    int get_graph_max_level();
    int get_graph_num_nodes();
    double get_graph_avg_degree_l0();
    int get_graph_actual_max_layer();
    int get_graph_nodes_at_level(int level);
    double get_graph_avg_degree_upper();
}

#endif // MYSOLUTION_H
