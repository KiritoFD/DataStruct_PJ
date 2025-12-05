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
// 距离统计相关全局变量
// ---------------------------------------------------------
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
static thread_local uint64_t tl_dist_counter = 0;
static std::atomic<double> g_last_build_ms{0.0};

// ---------------------------------------------------------
// Ablation flags (runtime toggles for experiments) - 提前声明
// ---------------------------------------------------------
static std::atomic<bool> ABLATE_CSR(false);
static std::atomic<bool> ABLATE_PREFETCH(false);
static std::atomic<bool> ABLATE_SIMD(false);
static std::atomic<bool> ABLATE_PRUNING(false);
static std::atomic<bool> ABLATE_HEAP(false);
static std::atomic<bool> ABLATE_FLAT_INDEX(true);  // 新增：强制使用动态索引而不转换为扁平化

// 新增：runtime toggle to enable/disable distance counting (avoid TLS increment when disabled)
static std::atomic<bool> ENABLE_RUNTIME_DIST_COUNTING(true);

// ---------------------------------------------------------
// SIMD 距离计算 (带/不带计数)
// ---------------------------------------------------------
// 通过宏 ENABLE_DIST_COUNTING 控制是否统计距离计算次数以避免 TLS 开销（默认关闭）
#ifndef ENABLE_DIST_COUNTING
#define ENABLE_DIST_COUNTING 1
#endif

#include "distance.h"
// ---------------------------------------------------------
// 全局线程池
// ---------------------------------------------------------
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

static bool DEBUG_TIMING = true;  // 改为 false，关闭调试输出

// Prefetch wrapper (runtime toggle)
static inline void my_prefetch_l1(const void* ptr) {
#ifdef __GNUC__
    if (ABLATE_PREFETCH.load(std::memory_order_relaxed)) return;
    _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#else
    (void)ptr;
#endif
}

// ---------------------------------------------------------
// 扁平化 HNSW 索引 (Read-Only Optimized) - CSR 格式
// ---------------------------------------------------------
class FlatHNSW {
public:
    int dim;
    int max_m;       // 第0层的最大连接数 (通常是 2*M)
    int max_m_upper; // 上层的最大连接数 (M)
    int enter_point;
    int num_nodes;
    int max_level;   // 入口点的最大层级
    
    // 数据存储
    std::vector<float> data; 
    
    // [优化] CSR 存储结构 - 消除 Padding，高度紧凑
    std::vector<uint64_t> l0_offsets;
    std::vector<int> l0_links;
    
    // 上层图结构 (扁平化存储)
    std::vector<int> node_levels;
    std::vector<int> upper_link_offsets;
    std::vector<int> upper_link_storage;

    FlatHNSW(int d) : dim(d), max_m(0), max_m_upper(0), enter_point(-1), num_nodes(0), max_level(0) {}

    inline int size() const { return num_nodes; }

    inline const int* get_l0_links(int id, int& count) const {
        uint64_t start = l0_offsets[id];
        uint64_t end = l0_offsets[id + 1];
        count = (int)(end - start);
        return l0_links.data() + start;
    }
    
    inline const int* get_upper_links(int id, int level, int& count) const {
        if (level <= 0 || level > node_levels[id]) {
            count = 0;
            return nullptr;
        }
        int offset = upper_link_offsets[(size_t)id * max_level + level];
        if (offset < 0) {
            count = 0;
            return nullptr;
        }
        const int* base = upper_link_storage.data() + offset;
        count = *base;
        return base + 1;
    }
    
    inline const float* get_vec(int id) const {
        return data.data() + (size_t)id * dim;
    }

    inline float dist(int id, const float* q) const {
        return l2sq_100d(get_vec(id), q);
    }
    
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_100d(get_vec(id_a), get_vec(id_b));
    }
    
    // 上层贪婪搜索 (无锁)
    int greedySearchUpper(int ep, const float* q, int level) const {
        if (ep < 0 || ep >= num_nodes) return -1;
        
        float curd = dist(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            int count;
            const int* links = get_upper_links(ep, level, count);
            
            if (count > 0) {
                my_prefetch_l1(get_vec(links[0]));
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                if (i + 1 < count) {
                    my_prefetch_l1(get_vec(links[i+1]));
                }
                float nd = dist(nb, q);
                if (nd < best_d) {
                    best_d = nd;
                    best_nb = nb;
                }
            }
            
            if (best_nb >= 0) {
                curd = best_d;
                ep = best_nb;
                changed = true;
            }
        }
        return ep;
    }
    
    // -------------------------------------------------------------
    // 【简化版】L0 搜索 - 高效紧凑实现
    // -------------------------------------------------------------
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        
        using Pair = std::pair<float, int>;
        
        // 使用简单的 vector + heap，避免过度工程化
        static thread_local std::vector<Pair> candidates;  // 最小堆（待扩展）
        static thread_local std::vector<Pair> results;     // 最大堆（结果集）
        static thread_local TagVisitedList visited;
        
        candidates.clear();
        results.clear();
        candidates.reserve(ef * 4);
        results.reserve(ef + 1);
        
        visited.init(num_nodes);
        visited.advance();
        
        const uint16_t* visited_ptr = visited.data();
        uint16_t cur_tag = visited.currentTag();

        float d0 = dist(ep, q);
        visited.mark(ep);
        
        candidates.push_back({d0, ep});
        results.push_back({d0, ep});
        
        // 比较器
        auto min_cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto max_cmp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        
        float worst_dist = d0;

        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), min_cmp);
            Pair curr = candidates.back();
            candidates.pop_back();

            // 剪枝
            if ((int)results.size() >= ef && curr.first > worst_dist) {
                break;
            }

            int u = curr.id;
            int count;
            const int* links = get_l0_links(u, count);

            // 预取前几个邻居
            constexpr int PF_AHEAD = 4;
            for (int i = 0; i < std::min(count, PF_AHEAD); ++i) {
                my_prefetch_l1(get_vec(links[i]));
            }

            for (int i = 0; i < count; ++i) {
                // 预取后续
                if (i + PF_AHEAD < count) {
                    my_prefetch_l1(get_vec(links[i + PF_AHEAD]));
                }
                
                int nb = links[i];
                if (visited_ptr[nb] == cur_tag) continue;
                visited.mark(nb);
                
                float d = dist(nb, q);
                
                // 更新结果集
                if ((int)results.size() < ef) {
                    results.push_back({d, nb});
                    std::push_heap(results.begin(), results.end(), max_cmp);
                    if ((int)results.size() == ef) {
                        worst_dist = results.front().first;
                    }
                    // 加入候选队列
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_cmp);
                } else if (d < worst_dist) {
                    // 替换最远结果
                    std::pop_heap(results.begin(), results.end(), max_cmp);
                    results.back() = {d, nb};
                    std::push_heap(results.begin(), results.end(), max_cmp);
                    worst_dist = results.front().first;
                    
                    // 加入候选队列
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_cmp);
                }
            }
        }
        
        // 按距离排序输出
        std::sort_heap(results.begin(), results.end(), max_cmp);
        return results;
    }
};

// ---------------------------------------------------------
// Node Structure
// ---------------------------------------------------------
struct HNSWNode {
    std::vector<std::vector<int>> links;
    mutable std::shared_mutex lock;
    
    HNSWNode(int max_level, int M) {
        links.resize(max_level + 1);
        for(auto& l : links) l.reserve(M * 2 + 1);
    }
};

// ---------------------------------------------------------
// SimpleHNSW - 优化版本
// ---------------------------------------------------------
class SimpleHNSW {
public:
    int dim;
    int M;
    int maxLayer;
    
    // 扁平化数据存储：连续内存，Cache友好
    std::vector<float> data_flat;  // size = n * dim
    std::vector<HNSWNode*> nodes;
    
    int enter_point;
    std::shared_mutex global_mutex;

    SimpleHNSW(int d, int m = 16, int ml = 16)
        : dim(d), M(m), maxLayer(ml), enter_point(-1) {}

    ~SimpleHNSW() { for (auto p : nodes) delete p; }

    inline int size() const { return (int)nodes.size(); }
    
    // 获取第 id 个向量的指针
    inline const float* getVec(int id) const {
        return data_flat.data() + (size_t)id * dim;
    }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    // 计算 query 到节点 id 的距离
    inline float dist(int id, const float* q) const {
        return l2sq_100d(getVec(id), q);
    }
    
    // 计算两个节点之间的距离
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_100d(getVec(id_a), getVec(id_b));
    }

    // -------------------------------------------------------
    // greedySearch
    // -------------------------------------------------------
    template<bool UseLock>
    int greedySearch(int ep, const float* q, int l) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return -1;
        
        float curd = dist(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            
            const std::vector<int>* neighbors_ptr;
            std::shared_lock<std::shared_mutex> lock_guard;
            
            if constexpr (UseLock) {
                lock_guard = std::shared_lock<std::shared_mutex>(nodes[ep]->lock);
            }
            neighbors_ptr = &nodes[ep]->links[l];
            
            const auto& neighbors = *neighbors_ptr;
            const int nsize = (int)neighbors.size();
            
            if (nsize > 0) {
                my_prefetch_l1(getVec(neighbors[0]));
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    my_prefetch_l1(getVec(neighbors[i+1]));
                }
                float nd = dist(nb, q);
                if (nd < best_d) {
                    best_d = nd;
                    best_nb = nb;
                }
            }
            
            if (best_nb >= 0) {
                curd = best_d;
                ep = best_nb;
                changed = true;
            }
        }
        return ep;
    }

    // -------------------------------------------------------
    // 优化后的 searchLayer：使用排序数组代替 priority_queue
    // -------------------------------------------------------
    template<bool UseLock>
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return {};
        
        using Pair = std::pair<float, int>;
        
        static thread_local std::vector<Pair> top_candidates;
        static thread_local std::vector<Pair> candidate_queue;
        static thread_local std::priority_queue<Pair, std::vector<Pair>, std::function<bool(const Pair&, const Pair&)>> top_heap(
            std::function<bool(const Pair&, const Pair&)>([](const Pair& a, const Pair& b) { return a.first < b.first; })
        );
        static thread_local VisitedList visited_list;
        
        top_candidates.clear();
        candidate_queue.clear();
        top_candidates.reserve(ef + 1);
        candidate_queue.reserve(ef * 2);
        
        visited_list.init(size());
        visited_list.advance();

        // 最小堆比较器
        auto greater_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };

        float d0 = dist(ep, q);
        visited_list.mark(ep);
        
        if (ABLATE_HEAP.load(std::memory_order_relaxed)) {
            while (!top_heap.empty()) top_heap.pop();
            top_heap.push({d0, ep});
        } else {
            top_candidates.push_back({d0, ep});
        }
        candidate_queue.push_back({d0, ep});
        std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);

        float lower_bound = d0;

        while (!candidate_queue.empty()) {
            std::pop_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
            auto curr = candidate_queue.back();
            candidate_queue.pop_back();

            // 关键剪枝：当前最近候选已超过结果集最远距离
            if (curr.first > lower_bound && ((ABLATE_HEAP.load(std::memory_order_relaxed) && (int)top_heap.size() >= ef) || (!ABLATE_HEAP.load(std::memory_order_relaxed) && (int)top_candidates.size() >= ef))) {
                break;
            }

            const std::vector<int>* neighbors_ptr;
            std::shared_lock<std::shared_mutex> lock_guard;
            
            if constexpr (UseLock) {
                lock_guard = std::shared_lock<std::shared_mutex>(nodes[curr.second]->lock);
            }
            neighbors_ptr = &nodes[curr.second]->links[l];
            
            const auto& neighbors = *neighbors_ptr;
            const int nsize = (int)neighbors.size();

            if (nsize > 0) {
                my_prefetch_l1(getVec(neighbors[0]));
            }

            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    my_prefetch_l1(getVec(neighbors[i+1]));
                }

                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float d_nb = dist(nb, q);

                    if (ABLATE_HEAP.load(std::memory_order_relaxed)) {
                        if ((int)top_heap.size() < ef) {
                            top_heap.push({d_nb, nb});
                        } else if (d_nb < top_heap.top().first) {
                            top_heap.pop();
                            top_heap.push({d_nb, nb});
                        }
                        // recompute lower bound
                        if (!top_heap.empty()) lower_bound = top_heap.top().first;
                    } else {
                        if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                            // 二分查找插入位置（保持升序）
                            auto it = std::upper_bound(top_candidates.begin(), top_candidates.end(),
                                Pair{d_nb, nb}, [](const Pair& a, const Pair& b) { return a.first < b.first; });
                            top_candidates.insert(it, {d_nb, nb});

                            if ((int)top_candidates.size() > ef) {
                                top_candidates.pop_back();
                            }
                            lower_bound = top_candidates.back().first;
                        }
                    }

                    candidate_queue.push_back({d_nb, nb});
                    std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
                }
            }
        }

        if (ABLATE_HEAP.load(std::memory_order_relaxed)) {
            top_candidates.clear();
            while (!top_heap.empty()) {
                top_candidates.push_back(top_heap.top());
                top_heap.pop();
            }
            std::sort(top_candidates.begin(), top_candidates.end(), [](const Pair& a, const Pair& b){ return a.first < b.first; });
            return top_candidates;
        }

        return top_candidates;
    }

    // -------------------------------------------------------
    // Robust Pruning (启发式选边) - 核心优化
    // -------------------------------------------------------
    void connectNodeHeuristic(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;

        std::vector<std::pair<float, int>> all_candidates;
        all_candidates.reserve(candidates.size() + m_max);
        
        for (const auto& p : candidates) {
            all_candidates.push_back(p);
        }

        {
            std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
            const auto& old_links = nodes[id]->links[l];
            for (int old_nb : old_links) {
                if (old_nb >= 0 && old_nb < size()) {
                    all_candidates.push_back({distNodes(id, old_nb), old_nb});
                }
            }
        }

        std::sort(all_candidates.begin(), all_candidates.end());
        all_candidates.erase(
            std::unique(all_candidates.begin(), all_candidates.end(),
                [](const auto& a, const auto& b) { return a.second == b.second; }),
            all_candidates.end()
        );

        std::vector<int> result_links;
        result_links.reserve(m_max);

        if (ABLATE_PRUNING.load(std::memory_order_relaxed)) {
            int taken = 0;
            for (const auto& cand : all_candidates) {
                if (taken >= m_max) break;
                if (cand.second == id) continue;
                result_links.push_back(cand.second);
                ++taken;
            }
        } else {
            for (const auto& cand : all_candidates) {
                if ((int)result_links.size() >= m_max) break;

                float d_cand_to_curr = cand.first;
                int cand_id = cand.second;
                
                if (cand_id == id) continue;

                bool keep = true;
                for (int selected_nbr : result_links) {
                    float d_cand_to_selected = distNodes(cand_id, selected_nbr);
                    if (d_cand_to_selected < d_cand_to_curr) {
                        keep = false;
                        break;
                    }
                }

                if (keep) {
                    result_links.push_back(cand_id);
                }
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            nodes[id]->links[l] = std::move(result_links);
        }

        for (const auto& p : all_candidates) {
            int nb = p.second;
            if (nb < 0 || nb >= size() || nb == id) continue;
            
            bool in_result = false;
            {
                std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
                for (int r : nodes[id]->links[l]) {
                    if (r == nb) { in_result = true; break; }
                }
            }
            if (!in_result) continue;

            std::vector<std::pair<float, int>> nb_candidates;
            {
                std::shared_lock<std::shared_mutex> lock(nodes[nb]->lock);
                const auto& nb_links = nodes[nb]->links[l];
                nb_candidates.reserve(nb_links.size() + 1);
                for (int x : nb_links) {
                    if (x >= 0 && x < size()) {
                        nb_candidates.push_back({distNodes(nb, x), x});
                    }
                }
            }
            nb_candidates.push_back({distNodes(nb, id), id});

            std::sort(nb_candidates.begin(), nb_candidates.end());
            nb_candidates.erase(
                std::unique(nb_candidates.begin(), nb_candidates.end(),
                    [](const auto& a, const auto& b) { return a.second == b.second; }),
                nb_candidates.end()
            );

            std::vector<int> nb_result;
            nb_result.reserve(m_max);

            if (ABLATE_PRUNING.load(std::memory_order_relaxed)) {
                int taken = 0;
                for (const auto& c : nb_candidates) {
                    if (taken >= m_max) break;
                    if (c.second == nb) continue;
                    nb_result.push_back(c.second);
                    ++taken;
                }
            } else {
                for (const auto& c : nb_candidates) {
                    if ((int)nb_result.size() >= m_max) break;
                    if (c.second == nb) continue;

                    bool keep = true;
                    for (int sel : nb_result) {
                        if (distNodes(c.second, sel) < c.first) {
                            keep = false;
                            break;
                        }
                    }
                    if (keep) nb_result.push_back(c.second);
                }
            }

            {
                std::unique_lock<std::shared_mutex> lock(nodes[nb]->lock);
                nodes[nb]->links[l] = std::move(nb_result);
            }
        }
    }

    void insertPointParallel(int id, int level) {
        int ep_curr;
        {
            std::shared_lock<std::shared_mutex> lock(global_mutex);
            ep_curr = enter_point;
        }

        if (ep_curr != -1) {
            int max_l = (int)nodes[ep_curr]->links.size() - 1;
            int curr = ep_curr;
            
            for (int l = max_l; l > level; l--) {
                curr = greedySearch<true>(curr, getVec(id), l);
            }

            for (int l = std::min(level, max_l); l >= 0; l--) {
                auto top = searchLayer<true>(getVec(id), curr, l, g_HNSW_EF_CONSTRUCTION.load());
                if (!top.empty()) curr = top[0].second;
                connectNodeHeuristic(id, top, l);
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(global_mutex);
            if (enter_point == -1 || level > (int)nodes[enter_point]->links.size() - 1) {
                enter_point = id;
            }
        }
    }
};

// ---------------------------------------------------------
// 转换函数：将动态图转换为静态扁平化图 - CSR 格式
// ---------------------------------------------------------
static FlatHNSW* convert_to_flat(SimpleHNSW* src) {
    FlatHNSW* flat = new FlatHNSW(src->dim);
    flat->data = std::move(src->data_flat);
    flat->enter_point = src->enter_point;
    
    int N = src->size();
    int M = src->M;
    flat->max_m = M * 2;
    flat->max_m_upper = M;
    flat->num_nodes = N;

    if (src->enter_point >= 0 && src->enter_point < N) {
        flat->max_level = (int)src->nodes[src->enter_point]->links.size() - 1;
    } else {
        flat->max_level = 0;
    }

    // 邻居排序：对每个节点的邻居按距离升序排序
    for (int i = 0; i < N; ++i) {
        const float* node_vec = flat->data.data() + (size_t)i * flat->dim;
        
        auto& l0_links = src->nodes[i]->links[0];
        std::vector<std::pair<float, int>> sorted_neighbors;
        sorted_neighbors.reserve(l0_links.size());
        for (int nb : l0_links) {
            const float* nb_vec = flat->data.data() + (size_t)nb * flat->dim;
            float d = l2sq_100d(node_vec, nb_vec);
            sorted_neighbors.push_back({d, nb});
        }
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
        for (size_t j = 0; j < l0_links.size(); ++j) {
            l0_links[j] = sorted_neighbors[j].second;
        }
        
        // 上层邻居也排序
        for (size_t l = 1; l < src->nodes[i]->links.size(); ++l) {
            auto& upper_links = src->nodes[i]->links[l];
            if (upper_links.empty()) continue;
            sorted_neighbors.clear();
            for (int nb : upper_links) {
                const float* nb_vec = flat->data.data() + (size_t)nb * flat->dim;
                float d = l2sq_100d(node_vec, nb_vec);
                sorted_neighbors.push_back({d, nb});
            }
            std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
            for (size_t j = 0; j < upper_links.size(); ++j) {
                upper_links[j] = sorted_neighbors[j].second;
            }
        }
    }

    // CSR 格式构建 Level 0
    flat->l0_offsets.resize(N + 1);
    uint64_t total_links = 0;
    for (int i = 0; i < N; ++i) {
        total_links += src->nodes[i]->links[0].size();
    }
    flat->l0_links.resize(total_links);
    
    uint64_t current_offset = 0;
    for (int i = 0; i < N; ++i) {
        flat->l0_offsets[i] = current_offset;
        const auto& src_links = src->nodes[i]->links[0];
        if (!src_links.empty()) {
            std::memcpy(flat->l0_links.data() + current_offset, 
                       src_links.data(), 
                       src_links.size() * sizeof(int));
            current_offset += src_links.size();
        }
    }
    flat->l0_offsets[N] = current_offset;

    // 上层图结构
    flat->node_levels.resize(N);
    flat->upper_link_offsets.resize((size_t)N * (flat->max_level + 1), -1);
    flat->upper_link_storage.reserve(N * M);

    for (int i = 0; i < N; ++i) {
        int level = (int)src->nodes[i]->links.size() - 1;
        flat->node_levels[i] = level;
        
        for (int l = 1; l <= level; ++l) {
            auto& links = src->nodes[i]->links[l];
            int cnt = (int)links.size();
            
            int start_idx = (int)flat->upper_link_storage.size();
            flat->upper_link_offsets[(size_t)i * (flat->max_level + 1) + l] = start_idx;
            
            flat->upper_link_storage.push_back(cnt);
            for (int nb : links) {
                flat->upper_link_storage.push_back(nb);
            }
        }
    }

    if (DEBUG_TIMING) {
        std::cout << "[FlatHNSW-CSR] Converted " << N << " nodes (with neighbor reordering)" << std::endl;
    }
    
    return flat;
}

// ---------------------------------------------------------
// 反序列化函数：将 FlatHNSW 转换回 SimpleHNSW (用于消融实验)
// ---------------------------------------------------------
static SimpleHNSW* convert_flat_to_simple(FlatHNSW* flat) {
    SimpleHNSW* simple = new SimpleHNSW(flat->dim, flat->max_m_upper, flat->max_level);
    simple->data_flat = flat->data;
    simple->enter_point = flat->enter_point;
    
    int N = flat->num_nodes;
    simple->nodes.reserve(N);
    
    // 重建 HNSWNode 结构
    for (int i = 0; i < N; ++i) {
        int level = flat->node_levels[i];
        HNSWNode* node = new HNSWNode(level, flat->max_m_upper);
        
        // 重建 L0 链接
        {
            int cnt;
            const int* links = flat->get_l0_links(i, cnt);
            node->links[0].assign(links, links + cnt);
        }
        
        // 重建上层链接
        for (int l = 1; l <= level; ++l) {
            int cnt;
            const int* links = flat->get_upper_links(i, l, cnt);
            if (links) {
                node->links[l].assign(links, links + cnt);
            }
        }
        
        simple->nodes.push_back(node);
    }
    
    return simple;
}


#include "cache.h"

// ---------------------------------------------------------
// 并行包装类 - 修改以支持缓存
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    FlatHNSW* flat_index = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { 
        delete hnsw; 
        delete flat_index;
    }

    void build_from_memory(int d, const float* data, int n) {
        delete flat_index;
        flat_index = nullptr;
        delete hnsw;
        hnsw = nullptr;
        
        int M = g_HNSW_M.load();
        int max_layer = g_HNSW_MAX_LAYER.load();
        int efc = g_HNSW_EF_CONSTRUCTION.load();
        
        // 尝试从缓存加载
        std::string cache_path = get_index_cache_path(n, d, M, max_layer, efc);
        
        // 创建缓存目录
        #ifdef _WIN32
        _mkdir("cache");
        #else
        mkdir("cache", 0755);
        #endif
        
        auto cache_start = std::chrono::high_resolution_clock::now();
        // 任何情况下都尝试加载缓存（包括消融）
        FlatHNSW* cached_flat = load_flat_index(cache_path);
        auto cache_end = std::chrono::high_resolution_clock::now();
        
        if (cached_flat != nullptr) {
            // 缓存命中
            double cache_ms = std::chrono::duration<double, std::milli>(cache_end - cache_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Loaded index from: " << cache_path << std::endl;
                std::cout << "[Cache] Load time: " << std::fixed << std::setprecision(2) 
                          << cache_ms << " ms" << std::endl;
            }
            g_last_build_ms.store(cache_ms, std::memory_order_relaxed);
            
            // 初始化 point_ids
            point_ids.resize(n);
            for (int i = 0; i < n; ++i) point_ids[i] = i;
            
            // 根据消融标志决定使用哪个结构
            if (ABLATE_CSR.load(std::memory_order_relaxed) || ABLATE_FLAT_INDEX.load(std::memory_order_relaxed)) {
                // 消融：反序列化回动态结构
                hnsw = convert_flat_to_simple(cached_flat);
                delete cached_flat;
                if (DEBUG_TIMING) {
                    std::cout << "[Ablation] Converted cached FlatHNSW to SimpleHNSW for dynamic queries" << std::endl;
                }
            } else {
                // 正常：使用扁平化结构
                flat_index = cached_flat;
            }
            
            return;
        }
        
        if (DEBUG_TIMING) {
            std::cout << "[Cache] Cache miss, building index..." << std::endl;
        }
        
        // 缓存未命中，正常构建
        hnsw = new SimpleHNSW(d, M, max_layer);
        
        hnsw->data_flat.resize((size_t)n * d);
        std::memcpy(hnsw->data_flat.data(), data, (size_t)n * d * sizeof(float));
        
        hnsw->nodes.reserve(n);
        
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
        int chunk_size = 1000;

        for (int i = 1; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            pool->enqueue([this, i, end, &levels, &processed]() {
                for (int j = i; j < end; ++j) {
                    hnsw->insertPointParallel(j, levels[j]);
                }
                processed.fetch_add(end - i, std::memory_order_release);
            });
        }

        // 进度监控
        std::thread progress_thread([&processed, n, &build_start]() {
            int last_reported = 0;
            while (processed.load(std::memory_order_acquire) < n) {
                int curr = processed.load(std::memory_order_acquire);
                if (curr - last_reported >= std::max(50000, n / 10)) {
                    double pct = 100.0 * curr / n;
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(now - build_start).count();
                    if (DEBUG_TIMING) {
                        std::cout << "[Progress] " << curr << "/" << n 
                                  << " (" << std::fixed << std::setprecision(1) << pct << "%) "
                                  << "Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
                        std::cout.flush();
                    }
                    last_reported = curr;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });

        while (processed.load(std::memory_order_acquire) < n) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        progress_thread.join();

        auto build_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Parallel Build: " << std::fixed << std::setprecision(2) 
                      << total_ms << " ms for " << n << " points." << std::endl;
            std::cout.flush();
        }
        g_last_build_ms.store(total_ms, std::memory_order_relaxed);

        // 转换为扁平化结构并缓存
        auto convert_start = std::chrono::high_resolution_clock::now();
        flat_index = convert_to_flat(hnsw);
        auto convert_end = std::chrono::high_resolution_clock::now();
        double convert_ms = std::chrono::duration<double, std::milli>(convert_end - convert_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Flat Conversion: " << std::fixed << std::setprecision(2) 
                      << convert_ms << " ms" << std::endl;
            std::cout.flush();
        }

        // 保存到缓存
        auto save_start = std::chrono::high_resolution_clock::now();
        if (save_flat_index(flat_index, cache_path)) {
            auto save_end = std::chrono::high_resolution_clock::now();
            double save_ms = std::chrono::duration<double, std::milli>(save_end - save_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Saved index to: " << cache_path << std::endl;
                std::cout << "[Cache] Save time: " << std::fixed << std::setprecision(2) 
                          << save_ms << " ms" << std::endl;
            }
        }

        // 根据消融标志决定保留哪个结构
        if (ABLATE_CSR.load(std::memory_order_relaxed) || ABLATE_FLAT_INDEX.load(std::memory_order_relaxed)) {
            // 消融：保留动态结构，删除扁平化索引
            delete flat_index;
            flat_index = nullptr;
            if (DEBUG_TIMING) {
                std::cout << "[Ablation] Keeping SimpleHNSW for dynamic queries" << std::endl;
            }
        } else {
            // 正常：删除动态结构，保留扁平化索引
            delete hnsw;
            hnsw = nullptr;
        }
    }

    // -------------------------------------------------------
    // search 方法 - 【优化3】multi-queue + 【优化4】skip-layer
    // -------------------------------------------------------
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

        if (ABLATE_CSR.load(std::memory_order_relaxed) && hnsw != nullptr) {
            int ep = hnsw->enter_point;
            if (ep < 0 || ep >= (int)hnsw->nodes.size()) return {};
            int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
            int curr = ep;
            int start_l = std::min(max_l, 4);
            for (int l = start_l; l > 0; l--) {
                curr = hnsw->greedySearch<true>(curr, query.data(), l);
            }
            auto top = hnsw->searchLayer<true>(query.data(), curr, 0, g_HNSW_EF_SEARCH.load());
            std::vector<std::pair<int, float>> out;
            int cnt = std::min(k, (int)top.size());
            out.reserve(cnt);
            for (int i = 0; i < cnt; ++i) {
                int idx = top[i].second;
                if (idx >= 0 && idx < (int)point_ids.size()) {
                    out.push_back({point_ids[idx], top[i].first});
                }
            }
            uint64_t last = tl_dist_counter;
            tl_dist_counter = 0;
            g_last_query_dist.store(last, std::memory_order_relaxed);
            g_total_dist_count.fetch_add(last, std::memory_order_relaxed);
            g_total_query_count.fetch_add(1, std::memory_order_relaxed);
            return out;
        }

        if (!flat_index || flat_index->enter_point < 0) return {};

        int ep = flat_index->enter_point;
        int max_l = flat_index->node_levels[ep];
        int curr = ep;

        // 【优化4】skip-layer greedy：跳层下降，减少上层距离计算
        int l = std::min(max_l, 4);
        while (l > 0) {
            curr = flat_index->greedySearchUpper(curr, query.data(), l);
            // 跳层：l > 1 时跳2层，否则跳1层
            l = (l > 1) ? (l - 2) : (l - 1);
        }
    
        auto top = flat_index->searchL0(query.data(), curr, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int, float>> out;
        int cnt = std::min(k, (int)top.size());
        out.reserve(cnt);
        for (int i = 0; i < cnt; ++i) {
            int idx = top[i].second;
            if (idx >= 0 && idx < (int)point_ids.size()) {
                out.push_back({point_ids[idx], top[i].first});
            }
        }

        uint64_t last = tl_dist_counter;
        tl_dist_counter = 0;
        g_last_query_dist.store(last, std::memory_order_relaxed);
        g_total_dist_count.fetch_add(last, std::memory_order_relaxed);
        g_total_query_count.fetch_add(1, std::memory_order_relaxed);

        return out;
    }
};

// ---------------------------------------------------------
// 对外接口
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
    for (size_t i = 0; i < res.size() && i < static_cast<size_t>(k_); ++i) {
        result[i] = res[i].first;
    }
}

// ---------------------------------------------------------
// 设置参数接口
// ---------------------------------------------------------
#include "extern.h"