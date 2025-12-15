#include "MySolution.h"
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ---------------------------------------------------------
// 32字节对齐内存分配工具
// ---------------------------------------------------------
static constexpr size_t ALIGN_SIZE = 32;

static inline void* aligned_alloc_32(size_t size) {
    if (size == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(size, ALIGN_SIZE);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) return nullptr;
    return ptr;
#endif
}

static inline void aligned_free_32(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

struct AlignedFloatArray {
    float* ptr = nullptr;
    size_t size_ = 0;
    
    AlignedFloatArray() = default;
    ~AlignedFloatArray() { clear(); }
    AlignedFloatArray(const AlignedFloatArray&) = delete;
    AlignedFloatArray& operator=(const AlignedFloatArray&) = delete;
    AlignedFloatArray(AlignedFloatArray&& o) noexcept : ptr(o.ptr), size_(o.size_) { o.ptr = nullptr; o.size_ = 0; }
    AlignedFloatArray& operator=(AlignedFloatArray&& o) noexcept {
        if (this != &o) { clear(); ptr = o.ptr; size_ = o.size_; o.ptr = nullptr; o.size_ = 0; }
        return *this;
    }
    void resize(size_t n) { if (n == size_) return; clear(); if (n > 0) { ptr = (float*)aligned_alloc_32(n * sizeof(float)); size_ = n; } }
    void clear() { if (ptr) { aligned_free_32(ptr); ptr = nullptr; } size_ = 0; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    float* data() { return ptr; }
    const float* data() const { return ptr; }
};

// ---------------------------------------------------------
// 全局变量和消融标志
// ---------------------------------------------------------
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
static thread_local uint64_t tl_dist_counter = 0;
static std::atomic<double> g_last_build_ms{0.0};

// 消融标志
static std::atomic<bool> ABLATE_PREFETCH(false);
static std::atomic<bool> ABLATE_SIMD(false);
static std::atomic<bool> ABLATE_PRUNING(false);
static std::atomic<bool> ABLATE_HEAP(false);
static std::atomic<bool> ABLATE_REORDER(false);
static std::atomic<bool> ABLATE_ADAPTIVE_EP(false);  // 自适应起点消融开关

// 自适应起点参数
static std::atomic<int> ADAPTIVE_EP_K{256};           // 聚类数量
static std::atomic<int> KMEANS_ITERATIONS{20};        // k-means迭代次数
static std::atomic<int> ADAPTIVE_EP_NUM_PROBES{3};    // 多起点探测数量

static std::atomic<bool> ENABLE_RUNTIME_DIST_COUNTING(true);

#ifndef ENABLE_DIST_COUNTING
#define ENABLE_DIST_COUNTING 1
#endif

#include "distance.h"
#include "Threadf.h"

static std::atomic<int> g_HNSW_M{HNSW_DEFAULT_M};
static std::atomic<int> g_HNSW_MAX_LAYER{HNSW_DEFAULT_MAX_LAYER};
static std::atomic<int> g_HNSW_EF_CONSTRUCTION{HNSW_DEFAULT_EF_CONSTRUCTION};
static std::atomic<int> g_HNSW_EF_SEARCH{HNSW_DEFAULT_EF_SEARCH};

static std::atomic<int> HNSW_BUILD_THREADS = [](){
    unsigned t = std::thread::hardware_concurrency();
    return t ? (int)t : 8;
}();

static ThreadPool* g_thread_pool = nullptr;
static std::mutex g_pool_mutex;

static ThreadPool* getThreadPool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (!g_thread_pool) g_thread_pool = new ThreadPool(HNSW_BUILD_THREADS);
    return g_thread_pool;
}

static bool DEBUG_TIMING = true;

static inline void my_prefetch_l1(const void* ptr) {
#ifdef __GNUC__
    if (!ABLATE_PREFETCH.load(std::memory_order_relaxed))
        _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

static constexpr int PREFETCH_AHEAD = 4;

// ---------------------------------------------------------
// FlatHNSW - 只读优化的扁平化索引
// ---------------------------------------------------------
class FlatHNSW {
public:
    int dim, max_m, max_m_upper, enter_point, num_nodes, max_level;
    AlignedFloatArray data;
    std::vector<uint64_t> l0_offsets;
    std::vector<int> l0_links;
    std::vector<int> node_levels;
    std::vector<int> upper_link_offsets;
    std::vector<int> upper_link_storage;
    std::vector<int> label_lookup;
    
    // 自适应起点数据
    std::vector<int> entry_candidates;
    AlignedFloatArray cluster_centers;
    int num_clusters = 0;

    FlatHNSW(int d) : dim(d), max_m(0), max_m_upper(0), enter_point(-1), num_nodes(0), max_level(0), num_clusters(0) {}

    inline const int* get_l0_links(int id, int& count) const {
        uint64_t start = l0_offsets[id], end = l0_offsets[id + 1];
        count = (int)(end - start);
        return l0_links.data() + start;
    }
    
    inline const int* get_upper_links(int id, int level, int& count) const {
        if (level <= 0 || level > node_levels[id]) { count = 0; return nullptr; }
        int offset = upper_link_offsets[(size_t)id * max_level + level];
        if (offset < 0) { count = 0; return nullptr; }
        const int* base = upper_link_storage.data() + offset;
        count = *base;
        return base + 1;
    }
    
    inline const float* get_vec(int id) const { return data.data() + (size_t)id * dim; }
    inline float dist(int id, const float* q) const { return l2sq_100d(get_vec(id), q); }
    
    // 上层贪婪搜索
    int greedySearchUpper(int ep, const float* q, int level) const {
        if (ep < 0 || ep >= num_nodes) return -1;
        float best_d = dist(ep, q);
        int best_node = ep;
        bool changed = true;
        while (changed) {
            changed = false;
            int count;
            const int* links = get_upper_links(best_node, level, count);
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                float d = dist(nb, q);
                if (d < best_d) { best_d = d; best_node = nb; changed = true; }
            }
        }
        return best_node;
    }
    
    // L0层搜索
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        using Pair = std::pair<float, int>;
        static thread_local std::vector<Pair> candidates, top_results;
        static thread_local TagVisitedList visited;
        
        candidates.clear(); top_results.clear();
        candidates.reserve(ef * 2); top_results.reserve(ef + 1);
        visited.init(num_nodes); visited.advance();
        
        const uint16_t* visited_ptr = visited.data();
        uint16_t cur_tag = visited.currentTag();
        
        float d0 = dist(ep, q);
        visited.mark(ep);
        candidates.push_back({d0, ep});
        top_results.push_back({d0, ep});
        float worst_dist = d0;
        
        auto min_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto max_comp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        
        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), min_comp);
            Pair curr = candidates.back();
            candidates.pop_back();
            
            // 关键剪枝：候选点距离超过结果集最远距离
            if ((int)top_results.size() >= ef && curr.first > worst_dist) {
                break;
            }

            int count;
            const int* links = get_l0_links(curr.second, count);
            if (count == 0) continue;

            // -------------------------------------------------------
            // 流水线化的距离计算：预取 + 双路并行 + 最小分支
            // -------------------------------------------------------
            
            // 1. 预热预取流水线
            int prefetch_limit = std::min(count, PREFETCH_AHEAD);
            for (int k = 0; k < prefetch_limit; ++k) {
                my_prefetch_l1(get_vec(links[k]));
            }

            int i = 0;
            
            // 2. 双路并行处理 - 最大化 SIMD 利用率
            for (; i <= count - 2; i += 2) {
                // 持续预取后续数据
                if (i + PREFETCH_AHEAD < count) {
                    my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD]));
                }
                if (i + PREFETCH_AHEAD + 1 < count) {
                    my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD + 1]));
                }

                int nb1 = links[i];
                int nb2 = links[i + 1];
                
                // 快速 visited 检查
                bool v1 = (visited_ptr[nb1] == cur_tag);
                bool v2 = (visited_ptr[nb2] == cur_tag);
                
                if (v1 && v2) continue;

                // 核心优化：尽可能使用 2x SIMD 计算
                if (!v1 && !v2) {
                    // 最优路径：两个都未访问，批量计算
                    visited.mark(nb1);
                    visited.mark(nb2);
                    
                    float d1, d2;
                    l2sq_100d_2x(q, get_vec(nb1), get_vec(nb2), d1, d2);
                    
                    // 处理结果1
                    if ((int)top_results.size() < ef) {
                        top_results.push_back({d1, nb1});
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        if ((int)top_results.size() == ef) {
                            worst_dist = top_results.front().first;
                        }
                        candidates.push_back({d1, nb1});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    } else if (d1 < worst_dist) {
                        std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                        top_results.back() = {d1, nb1};
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        worst_dist = top_results.front().first;
                        
                        candidates.push_back({d1, nb1});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    }
                    
                    // 处理结果2
                    if ((int)top_results.size() < ef) {
                        top_results.push_back({d2, nb2});
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        if ((int)top_results.size() == ef) {
                            worst_dist = top_results.front().first;
                        }
                        candidates.push_back({d2, nb2});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    } else if (d2 < worst_dist) {
                        std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                        top_results.back() = {d2, nb2};
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        worst_dist = top_results.front().first;
                        
                        candidates.push_back({d2, nb2});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    }
                } else {
                    // Fallback：单独处理（混合访问状态）
                    if (!v1) {
                        visited.mark(nb1);
                        float d1 = dist(nb1, q);
                        
                        if ((int)top_results.size() < ef || d1 < worst_dist) {
                            if ((int)top_results.size() < ef) {
                                top_results.push_back({d1, nb1});
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                if ((int)top_results.size() == ef) {
                                    worst_dist = top_results.front().first;
                                }
                            } else {
                                std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                                top_results.back() = {d1, nb1};
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                worst_dist = top_results.front().first;
                            }
                            candidates.push_back({d1, nb1});
                            std::push_heap(candidates.begin(), candidates.end(), min_comp);
                        }
                    }
                    if (!v2) {
                        visited.mark(nb2);
                        float d2 = dist(nb2, q);
                        
                        if ((int)top_results.size() < ef || d2 < worst_dist) {
                            if ((int)top_results.size() < ef) {
                                top_results.push_back({d2, nb2});
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                if ((int)top_results.size() == ef) {
                                    worst_dist = top_results.front().first;
                                }
                            } else {
                                std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                                top_results.back() = {d2, nb2};
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                worst_dist = top_results.front().first;
                            }
                            candidates.push_back({d2, nb2});
                            std::push_heap(candidates.begin(), candidates.end(), min_comp);
                        }
                    }
                }
            }
            
            // 3. 处理剩余的单个邻居
            for (; i < count; ++i) {
                int nb = links[i];
                if (visited_ptr[nb] == cur_tag) continue;
                
                visited.mark(nb);
                float d = dist(nb, q);
                
                if ((int)top_results.size() < ef) {
                    top_results.push_back({d, nb});
                    std::push_heap(top_results.begin(), top_results.end(), max_comp);
                    if ((int)top_results.size() == ef) {
                        worst_dist = top_results.front().first;
                    }
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_comp);
                } else if (d < worst_dist) {
                    std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                    top_results.back() = {d, nb};
                    std::push_heap(top_results.begin(), top_results.end(), max_comp);
                    worst_dist = top_results.front().first;
                    
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_comp);
                }
            }
        }

        std::sort_heap(top_results.begin(), top_results.end(), max_comp);
        return top_results;
    }
    
    // =====================================================
    // 【核心】多起点搜索 - 自适应起点的关键实现
    // =====================================================
    std::vector<std::pair<float, int>> searchL0MultiEntry(const float* q, int ef) const {
        // 如果没有聚类信息，回退到单起点
        if (num_clusters == 0 || entry_candidates.empty() || cluster_centers.empty()) {
            return searchL0(q, enter_point, ef);
        }
        
        const int num_probes = std::min(ADAPTIVE_EP_NUM_PROBES.load(std::memory_order_relaxed), num_clusters);
        
        // 找最近的num_probes个聚类中心
        std::vector<std::pair<float, int>> nearest_clusters;
        nearest_clusters.reserve(num_probes + 1);
        auto heap_comp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first < b.first; };
        
        for (int i = 0; i < num_clusters; ++i) {
            const float* center = cluster_centers.data() + (size_t)i * dim;
            float d = l2sq_100d(center, q);
            if ((int)nearest_clusters.size() < num_probes) {
                nearest_clusters.push_back({d, i});
                std::push_heap(nearest_clusters.begin(), nearest_clusters.end(), heap_comp);
            } else if (d < nearest_clusters.front().first) {
                std::pop_heap(nearest_clusters.begin(), nearest_clusters.end(), heap_comp);
                nearest_clusters.back() = {d, i};
                std::push_heap(nearest_clusters.begin(), nearest_clusters.end(), heap_comp);
            }
        }
        
        // 收集起点
        std::vector<int> entry_points;
        entry_points.reserve(num_probes);
        for (const auto& p : nearest_clusters) {
            int idx = p.second;
            if (idx >= 0 && idx < (int)entry_candidates.size()) {
                int ep = entry_candidates[idx];
                if (ep >= 0 && ep < num_nodes) entry_points.push_back(ep);
            }
        }
        if (entry_points.empty()) return searchL0(q, enter_point, ef);
        
        // 多起点搜索
        using Pair = std::pair<float, int>;
        static thread_local std::vector<Pair> candidates, top_results;
        static thread_local TagVisitedList visited;
        
        candidates.clear(); top_results.clear();
        candidates.reserve(ef * 2); top_results.reserve(ef + 1);
        visited.init(num_nodes); visited.advance();
        
        const uint16_t* visited_ptr = visited.data();
        uint16_t cur_tag = visited.currentTag();
        auto min_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto max_comp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        float worst_dist = std::numeric_limits<float>::max();
        
        // 初始化所有起点
        for (int ep : entry_points) {
            if (visited_ptr[ep] == cur_tag) continue;
            visited.mark(ep);
            float d = dist(ep, q);
            candidates.push_back({d, ep});
            std::push_heap(candidates.begin(), candidates.end(), min_comp);
            if ((int)top_results.size() < ef) {
                top_results.push_back({d, ep});
                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                if ((int)top_results.size() == ef) worst_dist = top_results.front().first;
            } else if (d < worst_dist) {
                std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                top_results.back() = {d, ep};
                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                worst_dist = top_results.front().first;
            }
        }
        
        // 搜索循环
        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), min_comp);
            Pair curr = candidates.back();
            candidates.pop_back();
            if ((int)top_results.size() >= ef && curr.first > worst_dist) break;
            
            int count;
            const int* links = get_l0_links(curr.second, count);
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                if (visited_ptr[nb] == cur_tag) continue;
                visited.mark(nb);
                float d = dist(nb, q);
                if ((int)top_results.size() < ef) {
                    top_results.push_back({d, nb});
                    std::push_heap(top_results.begin(), top_results.end(), max_comp);
                    if ((int)top_results.size() == ef) worst_dist = top_results.front().first;
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_comp);
                } else if (d < worst_dist) {
                    std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                    top_results.back() = {d, nb};
                    std::push_heap(top_results.begin(), top_results.end(), max_comp);
                    worst_dist = top_results.front().first;
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_comp);
                }
            }
        }
        std::sort_heap(top_results.begin(), top_results.end(), max_comp);
        return top_results;
    }
};

// ---------------------------------------------------------
// HNSWNode 和 SimpleHNSW - 构建用
// ---------------------------------------------------------
struct HNSWNode {
    std::vector<std::vector<int>> links;
    mutable std::shared_mutex lock;
    HNSWNode(int max_level, int M) { links.resize(max_level + 1); for(auto& l : links) l.reserve(M * 2 + 1); }
};

class SimpleHNSW {
public:
    int dim, M, maxLayer, enter_point = -1;
    std::vector<float> data_flat;
    std::vector<HNSWNode*> nodes;
    std::shared_mutex global_mutex;

    SimpleHNSW(int d, int m, int ml) : dim(d), M(m), maxLayer(ml) {}
    ~SimpleHNSW() { for (auto p : nodes) delete p; }

    int size() const { return (int)nodes.size(); }
    const float* getVec(int id) const { return data_flat.data() + (size_t)id * dim; }
    float dist(int id, const float* q) const { return l2sq_100d(getVec(id), q); }
    float distNodes(int a, int b) const { return l2sq_100d(getVec(a), getVec(b)); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        return (int)(-std::log(ud(rng)) * (1.0 / std::log((float)M)));
    }

    template<bool UseLock>
    int greedySearch(int ep, const float* q, int l) const {
        if (ep < 0 || ep >= size()) return -1;
        float curd = dist(ep, q);
        bool changed = true;
        while (changed) {
            changed = false;
            std::shared_lock<std::shared_mutex> lock_guard;
            if constexpr (UseLock) lock_guard = std::shared_lock<std::shared_mutex>(nodes[ep]->lock);
            for (int nb : nodes[ep]->links[l]) {
                float nd = dist(nb, q);
                if (nd < curd) { curd = nd; ep = nb; changed = true; }
            }
        }
        return ep;
    }

    template<bool UseLock>
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (ep < 0 || ep >= size()) return {};
        using Pair = std::pair<float, int>;
        static thread_local std::vector<Pair> top_candidates, candidate_queue;
        static thread_local VisitedList visited_list;
        
        top_candidates.clear(); candidate_queue.clear();
        visited_list.init(size()); visited_list.advance();
        auto greater_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };

        float d0 = dist(ep, q);
        visited_list.mark(ep);
        top_candidates.push_back({d0, ep});
        candidate_queue.push_back({d0, ep});
        std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
        float lower_bound = d0;

        while (!candidate_queue.empty()) {
            std::pop_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
            auto curr = candidate_queue.back();
            candidate_queue.pop_back();
            if (curr.first > lower_bound && (int)top_candidates.size() >= ef) break;

            std::shared_lock<std::shared_mutex> lock_guard;
            if constexpr (UseLock) lock_guard = std::shared_lock<std::shared_mutex>(nodes[curr.second]->lock);
            for (int nb : nodes[curr.second]->links[l]) {
                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float d_nb = dist(nb, q);
                    if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                        auto it = std::upper_bound(top_candidates.begin(), top_candidates.end(),
                            Pair{d_nb, nb}, [](const Pair& a, const Pair& b) { return a.first < b.first; });
                        top_candidates.insert(it, {d_nb, nb});
                        if ((int)top_candidates.size() > ef) top_candidates.pop_back();
                        lower_bound = top_candidates.back().first;
                    }
                    candidate_queue.push_back({d_nb, nb});
                    std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
                }
            }
        }
        return top_candidates;
    }

    void connectNodeHeuristic(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;
        std::vector<std::pair<float, int>> all_candidates = candidates;
        {
            std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
            for (int old_nb : nodes[id]->links[l])
                if (old_nb >= 0 && old_nb < size()) all_candidates.push_back({distNodes(id, old_nb), old_nb});
        }
        std::sort(all_candidates.begin(), all_candidates.end());
        all_candidates.erase(std::unique(all_candidates.begin(), all_candidates.end(),
            [](const auto& a, const auto& b) { return a.second == b.second; }), all_candidates.end());

        std::vector<int> result_links;
        for (const auto& cand : all_candidates) {
            if ((int)result_links.size() >= m_max) break;
            if (cand.second == id) continue;
            bool keep = true;
            for (int sel : result_links) {
                if (distNodes(cand.second, sel) < cand.first) { keep = false; break; }
            }
            if (keep) result_links.push_back(cand.second);
        }
        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            nodes[id]->links[l] = std::move(result_links);
        }
    }

    void insertPointParallel(int id, int level) {
        int ep_curr;
        { std::shared_lock<std::shared_mutex> lock(global_mutex); ep_curr = enter_point; }
        if (ep_curr != -1) {
            int max_l = (int)nodes[ep_curr]->links.size() - 1;
            int curr = ep_curr;
            for (int l = max_l; l > level; l--) curr = greedySearch<true>(curr, getVec(id), l);
            for (int l = std::min(level, max_l); l >= 0; l--) {
                auto top = searchLayer<true>(getVec(id), curr, l, g_HNSW_EF_CONSTRUCTION.load());
                if (!top.empty()) curr = top[0].second;
                connectNodeHeuristic(id, top, l);
            }
        }
        {
            std::unique_lock<std::shared_mutex> lock(global_mutex);
            if (enter_point == -1 || level > (int)nodes[enter_point]->links.size() - 1) enter_point = id;
        }
    }
};

// ---------------------------------------------------------
// 转换和辅助函数的前向声明
// ---------------------------------------------------------
static FlatHNSW* convert_to_flat(SimpleHNSW* src);

// ---------------------------------------------------------
// K-Means聚类
// ---------------------------------------------------------
static void kmeans_clustering(const float* data, int n, int dim, int K, int max_iter,
                              std::vector<float>& centers, std::vector<int>& assignments) {
    if (n <= 0 || K <= 0 || K > n) { centers.clear(); assignments.clear(); return; }
    centers.resize((size_t)K * dim);
    assignments.resize(n, 0);
    
    std::minstd_rand rng(42);
    std::uniform_int_distribution<int> dist_init(0, n - 1);
    
    int first = dist_init(rng);
    std::memcpy(centers.data(), data + (size_t)first * dim, dim * sizeof(float));
    if (K == 1) { std::fill(assignments.begin(), assignments.end(), 0); return; }
    
    std::vector<float> min_dists(n, std::numeric_limits<float>::max());
    for (int c = 1; c < K; ++c) {
        const float* last_center = centers.data() + (size_t)(c - 1) * dim;
        double total_dist = 0.0;
        for (int i = 0; i < n; ++i) {
            float d = l2sq_100d(data + (size_t)i * dim, last_center);
            if (d < min_dists[i]) min_dists[i] = d;
            total_dist += min_dists[i];
        }
        if (total_dist <= 0) {
            std::memcpy(centers.data() + (size_t)c * dim, data + (size_t)dist_init(rng) * dim, dim * sizeof(float));
            continue;
        }
        std::uniform_real_distribution<double> dist_prob(0.0, total_dist);
        double threshold = dist_prob(rng), cumsum = 0.0;
        int next_center = n - 1;
        for (int i = 0; i < n; ++i) { cumsum += min_dists[i]; if (cumsum >= threshold) { next_center = i; break; } }
        std::memcpy(centers.data() + (size_t)c * dim, data + (size_t)next_center * dim, dim * sizeof(float));
    }
    
    std::vector<float> new_centers((size_t)K * dim);
    std::vector<int64_t> cluster_sizes(K);
    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            const float* vec = data + (size_t)i * dim;
            int best_c = 0; float best_d = std::numeric_limits<float>::max();
            for (int c = 0; c < K; ++c) {
                float d = l2sq_100d(vec, centers.data() + (size_t)c * dim);
                if (d < best_d) { best_d = d; best_c = c; }
            }
            assignments[i] = best_c;
        }
        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
        for (int i = 0; i < n; ++i) {
            int c = assignments[i]; cluster_sizes[c]++;
            for (int d = 0; d < dim; ++d) new_centers[(size_t)c * dim + d] += data[(size_t)i * dim + d];
        }
        for (int c = 0; c < K; ++c) {
            if (cluster_sizes[c] > 0) {
                float inv = 1.0f / cluster_sizes[c];
                for (int d = 0; d < dim; ++d) new_centers[(size_t)c * dim + d] *= inv;
            } else {
                std::memcpy(new_centers.data() + (size_t)c * dim, data + (size_t)(rng() % n) * dim, dim * sizeof(float));
            }
        }
        centers = new_centers;
    }
}

// ---------------------------------------------------------
// 构建自适应起点
// ---------------------------------------------------------
static void build_adaptive_entry_points(FlatHNSW* flat, int K) {
    if (!flat || flat->num_nodes == 0) return;
    K = std::min(K, flat->num_nodes);
    if (K <= 0) K = 256;
    
    auto start = std::chrono::high_resolution_clock::now();
    if (DEBUG_TIMING) std::cout << "[Adaptive EP] Building K=" << K << " clusters..." << std::endl;
    
    std::vector<float> centers;
    std::vector<int> assignments;
    kmeans_clustering(flat->data.data(), flat->num_nodes, flat->dim, K, KMEANS_ITERATIONS.load(), centers, assignments);
    
    if (centers.empty()) { if (DEBUG_TIMING) std::cout << "[Adaptive EP] Clustering failed" << std::endl; return; }
    
    flat->entry_candidates.resize(K);
    flat->cluster_centers.resize((size_t)K * flat->dim);
    std::memcpy(flat->cluster_centers.data(), centers.data(), (size_t)K * flat->dim * sizeof(float));
    flat->num_clusters = K;
    
    // 为每个簇选择离中心最近的点作为入口
    for (int c = 0; c < K; ++c) {
        const float* center = centers.data() + (size_t)c * flat->dim;
        int best_id = -1; float best_dist = std::numeric_limits<float>::max();
        for (int i = 0; i < flat->num_nodes; ++i) {
            if (assignments[i] != c) continue;
            float d = l2sq_100d(flat->get_vec(i), center);
            if (d < best_dist) { best_dist = d; best_id = i; }
        }
        if (best_id < 0) { // 空簇回退
            for (int i = 0; i < flat->num_nodes; ++i) {
                float d = l2sq_100d(flat->get_vec(i), center);
                if (d < best_dist) { best_dist = d; best_id = i; }
            }
        }
        flat->entry_candidates[c] = (best_id >= 0) ? best_id : flat->enter_point;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    if (DEBUG_TIMING) std::cout << "[Adaptive EP] Done in " << std::chrono::duration<double,std::milli>(end-start).count() << " ms" << std::endl;
}

#include "cache.h"

// ---------------------------------------------------------
// HnswSolutionParallel - 主要封装类
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    FlatHNSW* flat_index = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { delete hnsw; delete flat_index; }

    void build_from_memory(int d, const float* data, int n) {
        delete flat_index; flat_index = nullptr;
        delete hnsw; hnsw = nullptr;
        
        int M = g_HNSW_M.load(), max_layer = g_HNSW_MAX_LAYER.load(), efc = g_HNSW_EF_CONSTRUCTION.load();
        std::string cache_path = get_index_cache_path(n, d, M, max_layer, efc);
        
        #ifdef _WIN32
        _mkdir("cache");
        #else
        mkdir("cache", 0755);
        #endif
        
        auto cache_start = std::chrono::high_resolution_clock::now();
        flat_index = load_flat_index(cache_path);
        
        if (flat_index) {
            auto cache_end = std::chrono::high_resolution_clock::now();
            g_last_build_ms.store(std::chrono::duration<double,std::milli>(cache_end-cache_start).count());
            if (DEBUG_TIMING) std::cout << "[Cache] Loaded from " << cache_path << std::endl;
            point_ids.resize(n);
            for (int i = 0; i < n; ++i) point_ids[i] = i;
            
            // 按需构建聚类
            if (!ABLATE_ADAPTIVE_EP.load() && flat_index->num_clusters == 0) {
                build_adaptive_entry_points(flat_index, ADAPTIVE_EP_K.load());
            }
            return;
        }

        // 构建索引
        hnsw = new SimpleHNSW(d, M, max_layer);
        hnsw->data_flat.resize((size_t)n * d);
        std::memcpy(hnsw->data_flat.data(), data, (size_t)n * d * sizeof(float));
        
        std::vector<int> levels(n);
        for (int i = 0; i < n; ++i) {
            levels[i] = std::min(hnsw->randomLevel(), max_layer);
            hnsw->nodes.push_back(new HNSWNode(levels[i], M));
        }
        point_ids.resize(n);
        for (int i = 0; i < n; ++i) point_ids[i] = i;
        if (n > 0) hnsw->enter_point = 0;

        auto build_start = std::chrono::high_resolution_clock::now();
        ThreadPool* pool = getThreadPool();
        std::atomic<int> processed(1);
        
        for (int i = 1; i < n; i += 1000) {
            int end = std::min(i + 1000, n);
            pool->enqueue([this, i, end, &levels, &processed]() {
                for (int j = i; j < end; ++j) hnsw->insertPointParallel(j, levels[j]);
                processed.fetch_add(end - i);
            });
        }
        while (processed.load() < n) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        auto build_end = std::chrono::high_resolution_clock::now();
        g_last_build_ms.store(std::chrono::duration<double,std::milli>(build_end-build_start).count());
        if (DEBUG_TIMING) std::cout << "[Build] " << g_last_build_ms.load() << " ms" << std::endl;

        flat_index = convert_to_flat(hnsw);
        
        if (!ABLATE_ADAPTIVE_EP.load()) build_adaptive_entry_points(flat_index, ADAPTIVE_EP_K.load());
        
        save_flat_index(flat_index, cache_path);
        delete hnsw; hnsw = nullptr;
    }

    // =====================================================
    // 【核心】搜索方法 - 消融实验关键分支
    // =====================================================
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;
        if (!flat_index || flat_index->enter_point < 0) return {};

        std::vector<std::pair<float, int>> top;
        
        // =====================================================
        // 消融实验分支：
        // ABLATE_ADAPTIVE_EP = false (默认): 使用自适应多起点搜索
        // ABLATE_ADAPTIVE_EP = true (消融): 使用标准HNSW单起点搜索
        // =====================================================
        bool use_adaptive = !ABLATE_ADAPTIVE_EP.load(std::memory_order_relaxed);
        
        if (use_adaptive && flat_index->num_clusters > 0) {
            // 【方案A】自适应起点：直接从多个几何最近点开始L0搜索
            top = flat_index->searchL0MultiEntry(query.data(), g_HNSW_EF_SEARCH.load());
        } else {
            // 【方案B】标准HNSW：从固定入口点逐层向下搜索
            int ep = flat_index->enter_point;
            int curr = ep;
            // 上层导航
            for (int l = flat_index->node_levels[ep]; l > 0; l--) {
                curr = flat_index->greedySearchUpper(curr, query.data(), l);
            }
            // L0搜索
            top = flat_index->searchL0(query.data(), curr, g_HNSW_EF_SEARCH.load());
        }
        
        // 结果转换
        std::vector<std::pair<int, float>> out;
        int cnt = std::min(k, (int)top.size());
        for (int i = 0; i < cnt; ++i) {
            int idx = flat_index->label_lookup.empty() ? top[i].second : flat_index->label_lookup[top[i].second];
            if (idx >= 0 && idx < (int)point_ids.size()) out.push_back({point_ids[idx], top[i].first});
        }

        g_last_query_dist.store(tl_dist_counter);
        g_total_dist_count.fetch_add(tl_dist_counter);
        g_total_query_count.fetch_add(1);
        tl_dist_counter = 0;
        return out;
    }
};

// ---------------------------------------------------------
// 全局实例和对外接口
// ---------------------------------------------------------
static HnswSolutionParallel* g_impl = nullptr;

void build_hnsw(int d, const std::vector<float>& base) {
    int n = (int)base.size() / d;
    delete g_impl;
    g_impl = new HnswSolutionParallel();
    g_impl->build_from_memory(d, base.data(), n);
}

std::vector<std::pair<int, float>> search_hnsw(const std::vector<float>& query, int k) {
    if (!g_impl) return {};
    return g_impl->search(query, k);
}

Solution::Solution() : k_(10) {}

void Solution::build(int d, const std::vector<float>& base) {
    build_hnsw(d, base);
}

void Solution::search(const std::vector<float>& query, int* result) {
    std::fill(result, result + k_, -1);
    auto res = search_hnsw(query, k_);
    for (size_t i = 0; i < res.size() && i < (size_t)k_; ++i) result[i] = res[i].first;
}

// 在g_impl定义之后包含extern.h
#include "extern.h"
