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
// 32字节对齐内存分配工具 (必须在 FlatHNSW 之前定义)
// ---------------------------------------------------------
static constexpr size_t ALIGN_SIZE = 32;

static inline void* aligned_alloc_32(size_t size) {
    if (size == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(size, ALIGN_SIZE);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN_SIZE, size) != 0) {
        return nullptr;
    }
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

// 对齐的 float 数组包装器
struct AlignedFloatArray {
    float* ptr = nullptr;
    size_t size_ = 0;
    
    AlignedFloatArray() = default;
    ~AlignedFloatArray() { clear(); }
    
    // 禁止拷贝
    AlignedFloatArray(const AlignedFloatArray&) = delete;
    AlignedFloatArray& operator=(const AlignedFloatArray&) = delete;
    
    // 允许移动
    AlignedFloatArray(AlignedFloatArray&& other) noexcept 
        : ptr(other.ptr), size_(other.size_) {
        other.ptr = nullptr;
        other.size_ = 0;
    }
    
    AlignedFloatArray& operator=(AlignedFloatArray&& other) noexcept {
        if (this != &other) {
            clear();
            ptr = other.ptr;
            size_ = other.size_;
            other.ptr = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void resize(size_t n) {
        if (n == size_) return;
        clear();
        if (n > 0) {
            ptr = (float*)aligned_alloc_32(n * sizeof(float));
            size_ = n;
        }
    }
    
    void clear() {
        if (ptr) {
            aligned_free_32(ptr);
            ptr = nullptr;
        }
        size_ = 0;
    }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    float* data() { return ptr; }
    const float* data() const { return ptr; }
    float* begin() { return ptr; }
    float* end() { return ptr + size_; }
    const float* begin() const { return ptr; }
    const float* end() const { return ptr + size_; }
    float& operator[](size_t i) { return ptr[i]; }
    const float& operator[](size_t i) const { return ptr[i]; }
};

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
static std::atomic<bool> ABLATE_FLAT_INDEX(false);
static std::atomic<bool> ABLATE_REORDER(false);

// 【新增】三角不等式剪枝开关 (默认开启)
static std::atomic<bool> ENABLE_TRIANGLE_PRUNING(true);

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

// 新增：统一的 prefetch 跳跃距离
static constexpr int PREFETCH_AHEAD = 7;

// ---------------------------------------------------------
// 扁平化 HNSW 索引 (Read-Only Optimized) - CSR 格式
// ---------------------------------------------------------
class FlatHNSW {
public:
    int dim;
    int max_m;
    int max_m_upper;
    int enter_point;
    int num_nodes;
    int max_level;
    
    // 数据存储 - 32字节对齐
    AlignedFloatArray data; 
    
    // [优化] CSR 存储结构 - 消除 Padding，高度紧凑
    std::vector<uint64_t> l0_offsets;
    std::vector<int> l0_links;
    
    // 上层图结构 (扁平化存储)
    std::vector<int> node_levels;
    std::vector<int> upper_link_offsets;
    std::vector<int> upper_link_storage;
    
    // [新增] ID 映射表：New ID -> Old ID (原始 index)
    std::vector<int> label_lookup;

    // 【新增】存储每个节点到 Pivot (Entry Point) 的 L2 距离（开方后）
    // 用于三角不等式剪枝
    AlignedFloatArray pivot_dists;

    FlatHNSW(int d) : dim(d), max_m(0), max_m_upper(0), enter_point(-1), num_nodes(0), max_level(0) {}

    inline int size() const { return num_nodes; }

    // [优化] CSR 格式获取第0层邻居 - 无 Padding，全是有效数据
    inline const int* get_l0_links(int id, int& count) const {
        uint64_t start = l0_offsets[id];
        uint64_t end = l0_offsets[id + 1];
        count = (int)(end - start);
        return l0_links.data() + start;
    }
    
    // 获取上层邻居
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
    // 【优化】带三角不等式剪枝的 L0 搜索
    // -------------------------------------------------------------
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        
        using Pair = std::pair<float, int>;
        
        static thread_local std::vector<Pair> candidates;
        static thread_local std::vector<Pair> top_results;
        static thread_local std::vector<int> expand_queue;
        static thread_local TagVisitedList visited;
        
        candidates.clear(); 
        top_results.clear();
        expand_queue.clear();
        
        candidates.reserve(ef * 2);
        top_results.reserve(ef + 1);
        expand_queue.reserve(64);

        visited.init(num_nodes);
        visited.advance();
        
        const uint16_t* visited_ptr = visited.data();
        uint16_t cur_tag = visited.currentTag();

        // 【三角不等式】预计算 Query 到 Pivot 的距离
        float d_q_pivot = 0.0f;
        bool use_triangle = ENABLE_TRIANGLE_PRUNING.load(std::memory_order_relaxed) 
                           && (enter_point >= 0) 
                           && (pivot_dists.size() > 0);
        if (use_triangle) {
            float sq = dist(enter_point, q);
            d_q_pivot = std::sqrt(sq);
        }

        float d0 = dist(ep, q);
        visited.mark(ep);
        
        candidates.push_back({d0, ep});
        top_results.push_back({d0, ep});

        float worst_dist = d0;

        auto min_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto max_comp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        
        // 内联处理候选点的 lambda
        auto process_candidate = [&](float d, int nb) {
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
        };

        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), min_comp);
            Pair curr = candidates.back();
            candidates.pop_back();

            if ((int)top_results.size() >= ef && curr.first > worst_dist) {
                break;
            }

            expand_queue.clear();
            expand_queue.push_back(curr.second);

            for (size_t qi = 0; qi < expand_queue.size(); ++qi) {
                int x = expand_queue[qi];
                int count;
                const int* links = get_l0_links(x, count);

                // 预取
                if (count > 0) {
                    int pf_idx = std::min(count - 1, PREFETCH_AHEAD);
                    my_prefetch_l1(get_vec(links[pf_idx]));
                }

                int i = 0;
                
                // --- 2-way Batching Loop with Triangle Pruning ---
                for (; i <= count - 2; i += 2) {
                    if (i + PREFETCH_AHEAD < count) {
                        my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD]));
                    }
                    if (i + PREFETCH_AHEAD + 1 < count) {
                        my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD + 1]));
                    }

                    int nb1 = links[i];
                    int nb2 = links[i + 1];
                    
                    bool v1 = (visited_ptr[nb1] == cur_tag);
                    bool v2 = (visited_ptr[nb2] == cur_tag);
                    
                    if (v1 && v2) continue;
                    
                    if (!v1) visited.mark(nb1);
                    if (!v2) visited.mark(nb2);

                    // 【三角不等式剪枝】
                    bool skip1 = false, skip2 = false;
                    if (use_triangle && (int)top_results.size() >= ef) {
                        float sqrt_worst = std::sqrt(worst_dist);
                        if (!v1) {
                            float d_n1_pivot = pivot_dists.ptr[nb1];
                            float diff1 = d_q_pivot - d_n1_pivot;
                            if (diff1 < 0) diff1 = -diff1;
                            if (diff1 > sqrt_worst * 1.0001f) skip1 = true;
                        }
                        if (!v2) {
                            float d_n2_pivot = pivot_dists.ptr[nb2];
                            float diff2 = d_q_pivot - d_n2_pivot;
                            if (diff2 < 0) diff2 = -diff2;
                            if (diff2 > sqrt_worst * 1.0001f) skip2 = true;
                        }
                    }
                    
                    // 根据剪枝结果决定是否计算距离
                    if (skip1 && skip2) continue;
                    
                    float d1 = 0, d2 = 0;
                    
                    if (!v1 && !skip1 && !v2 && !skip2) {
                        l2sq_100d_2x(q, get_vec(nb1), get_vec(nb2), d1, d2);
                    } else if (!v1 && !skip1) {
                        d1 = dist(nb1, q);
                    } else if (!v2 && !skip2) {
                        d2 = dist(nb2, q);
                    }

                    if (!v1 && !skip1) process_candidate(d1, nb1);
                    if (!v2 && !skip2) process_candidate(d2, nb2);
                }
                
                // 处理剩余的 1 个
                for (; i < count; ++i) {
                    if (i + PREFETCH_AHEAD < count) {
                        my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD]));
                    }
                    
                    int nb = links[i];
                    if (visited_ptr[nb] == cur_tag) continue;
                    visited.mark(nb);
                    
                    // 【三角不等式剪枝】
                    if (use_triangle && (int)top_results.size() >= ef) {
                        float d_n_pivot = pivot_dists.ptr[nb];
                        float diff = d_q_pivot - d_n_pivot;
                        if (diff < 0) diff = -diff;
                        float sqrt_worst = std::sqrt(worst_dist);
                        if (diff > sqrt_worst * 1.0001f) continue;
                    }
                    
                    float d = dist(nb, q);
                    process_candidate(d, nb);
                }
            }
        }

        std::sort_heap(top_results.begin(), top_results.end(), max_comp);
        return top_results;
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

// Forward declarations so functions are visible before usage
static FlatHNSW* convert_to_flat(SimpleHNSW* src);

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

        // no_csr / flat_index ablation: 禁用 flat 转换与缓存
        bool disable_flat = ABLATE_CSR.load(std::memory_order_relaxed) ||
                            ABLATE_FLAT_INDEX.load(std::memory_order_relaxed);

        FlatHNSW* cached_flat = nullptr;
        std::string cache_path;
        if (!disable_flat) {
            cache_path = get_index_cache_path(n, d, M, max_layer, efc);
            #ifdef _WIN32
            _mkdir("cache");
            #else
            mkdir("cache", 0755);
            #endif
            auto cache_start = std::chrono::high_resolution_clock::now();
            cached_flat = load_flat_index(cache_path);
            auto cache_end = std::chrono::high_resolution_clock::now();
            if (cached_flat != nullptr) {
                double cache_ms = std::chrono::duration<double, std::milli>(cache_end - cache_start).count();
                if (DEBUG_TIMING) {
                    std::cout << "[Cache] Loaded index from: " << cache_path << std::endl;
                    std::cout << "[Cache] Load time: " << std::fixed << std::setprecision(2) 
                              << cache_ms << " ms" << std::endl;
                }
                g_last_build_ms.store(cache_ms, std::memory_order_relaxed);
                point_ids.resize(n);
                for (int i = 0; i < n; ++i) point_ids[i] = i;
                flat_index = cached_flat;
                return;
            }
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

        if (!disable_flat) {
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

        int l = std::min(max_l, 4);
        while (l > 0) {
            curr = flat_index->greedySearchUpper(curr, query.data(), l);
            l = (l > 1) ? (l - 2) : (l - 1);
        }
    
        auto top = flat_index->searchL0(query.data(), curr, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int, float>> out;
        int cnt = std::min(k, (int)top.size());
        out.reserve(cnt);
        for (int i = 0; i < cnt; ++i) {
            int internal_id = top[i].second;
            float dist = top[i].first;
            
            // 转换 ID：如果使用了重排，需要映射回原始 ID
            int original_idx;
            if (!flat_index->label_lookup.empty()) {
                original_idx = flat_index->label_lookup[internal_id];
            } else {
                original_idx = internal_id;
            }
            
            if (original_idx >= 0 && original_idx < (int)point_ids.size()) {
                out.push_back({point_ids[original_idx], dist});
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
static void generate_dfs_reordering(SimpleHNSW* src, std::vector<int>& old_to_new, std::vector<int>& new_to_old) {
    int N = src->size();
    // 初始化映射表：old_to_new 初始化为 -1 表示未访问
    old_to_new.assign(N, -1);
    new_to_old.resize(N);
    
    int new_id_counter = 0;
    std::vector<int> stack;
    // 预分配内存以避免频繁扩容，N 为节点总数
    stack.reserve(N);
    
    // 优先从入口点开始遍历，保证搜索起点的局部性
    if (src->enter_point != -1 && src->enter_point < N) {
        stack.push_back(src->enter_point);
    }
    
    // 用于处理非连通图或初始点的扫描指针
    int scan_idx = 0;
    
    while (new_id_counter < N) {
        // 如果栈空了，说明当前连通分量遍历完毕，或者刚开始
        if (stack.empty()) {
            // 寻找下一个未访问的节点
            while (scan_idx < N && old_to_new[scan_idx] != -1) {
                scan_idx++;
            }
            if (scan_idx < N) {
                stack.push_back(scan_idx);
            } else {
                // 所有节点都处理完毕
                break;
            }
        }
        
        // 弹出栈顶元素
        int u = stack.back();
        stack.pop_back();
        
        // 如果已访问过，跳过
        if (old_to_new[u] != -1) continue;
        
        // 建立映射关系：旧ID -> 新ID (连续递增)
        old_to_new[u] = new_id_counter;
        new_to_old[new_id_counter] = u;
        new_id_counter++;
        
        // 将邻居入栈
        // 使用 Layer 0 的连接，因为这层最密集，对缓存影响最大
        if (u >= 0 && u < (int)src->nodes.size()) {
            const auto& links = src->nodes[u]->links[0];
            // 倒序遍历邻居并入栈，这样出栈顺序（即访问顺序）就是正序的
            // 这有助于保持与原始构建顺序或启发式选边顺序的一致性
            for (auto it = links.rbegin(); it != links.rend(); ++it) {
                int v = *it;
                // 只将合法且未访问的邻居入栈
                if (v >= 0 && v < N && old_to_new[v] == -1) {
                    stack.push_back(v);
                }
            }
        }
    }
}
// ---------------------------------------------------------
// 设置参数接口
// ---------------------------------------------------------
#include "extern.h"

static FlatHNSW* convert_to_flat(SimpleHNSW* src) {
    FlatHNSW* flat = new FlatHNSW(src->dim);

    int N = src->size();
    int M = src->M;
    flat->num_nodes = N;
    flat->max_m = M * 2;
    flat->max_m_upper = M;

    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Starting conversion for " << N << " nodes, dim=" << src->dim << std::endl;
        std::cout.flush();
    }

    // 判断是否启用重排优化
    bool do_reorder = !ABLATE_REORDER.load(std::memory_order_relaxed);

    std::vector<int> old_to_new, new_to_old;

    if (do_reorder) {
        // 1. 生成 ID 映射（使用 DFS 重排）
        generate_dfs_reordering(src, old_to_new, new_to_old);

        // 更新入口点
        flat->enter_point = (src->enter_point == -1) ? -1 : old_to_new[src->enter_point];

        // 计算 max_level
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) flat->max_level = level;
        }

        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder enabled, enter_point=" << flat->enter_point
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }

        flat->label_lookup = new_to_old;
    } else {
        flat->enter_point = src->enter_point;
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) flat->max_level = level;
        }

        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder disabled, enter_point=" << flat->enter_point
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }

        old_to_new.resize(N);
        new_to_old.resize(N);
        for (int i = 0; i < N; ++i) { old_to_new[i] = i; new_to_old[i] = i; }
        // label_lookup left empty => identity
    }

    // 2. 重排/复制向量数据
    flat->data.resize((size_t)N * flat->dim);
    if (flat->data.data() == nullptr) {
        std::cerr << "[FlatConvert] ERROR: Failed to allocate data array!" << std::endl;
        delete flat;
        return nullptr;
    }

    for (int new_id = 0; new_id < N; ++new_id) {
        int old_id = new_to_old[new_id];
        const float* src_vec = src->getVec(old_id);
        float* dst_vec = flat->data.data() + (size_t)new_id * flat->dim;
        std::memcpy(dst_vec, src_vec, flat->dim * sizeof(float));
    }

    // 预计算三角不等式 pivot 距离（如果开启）
    if (ENABLE_TRIANGLE_PRUNING.load(std::memory_order_relaxed) && flat->enter_point >= 0) {
        if (DEBUG_TIMING) std::cout << "[FlatConvert] Precomputing pivot distances..." << std::endl;
        flat->pivot_dists.resize(N);
        int pivot_id = flat->enter_point;
        const float* pivot_vec = flat->data.data() + (size_t)pivot_id * flat->dim;
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static, 2048)
        #endif
        for (int i = 0; i < N; ++i) {
            const float* vec_i = flat->data.data() + (size_t)i * flat->dim;
            float sq_d = l2sq_100d(pivot_vec, vec_i);
            flat->pivot_dists.ptr[i] = std::sqrt(sq_d);
        }
        if (DEBUG_TIMING) std::cout << "[FlatConvert] Pivot distances computed." << std::endl;
    }

    // 3. 构建 L0 CSR
    flat->l0_offsets.resize(N + 1);
    uint64_t total_l0_links = 0;
    for (int i = 0; i < N; ++i) total_l0_links += src->nodes[i]->links[0].size();
    flat->l0_links.resize(total_l0_links);
    uint64_t current_offset = 0;

    std::vector<std::pair<float,int>> temp_neighbors;
    temp_neighbors.reserve(M * 2);

    for (int new_id = 0; new_id < N; ++new_id) {
        flat->l0_offsets[new_id] = current_offset;
        int old_id = new_to_old[new_id];
        const auto& src_links = src->nodes[old_id]->links[0];
        temp_neighbors.clear();
        const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;

        for (int old_nb : src_links) {
            if (old_nb < 0 || old_nb >= N) continue;
            int new_nb = old_to_new[old_nb];
            if (new_nb < 0 || new_nb >= N) continue;
            const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
            float d = l2sq_100d(vec_u, vec_v);
            temp_neighbors.emplace_back(d, new_nb);
        }
        std::sort(temp_neighbors.begin(), temp_neighbors.end());
        for (const auto &p : temp_neighbors) {
            flat->l0_links[current_offset++] = p.second;
        }
    }
    flat->l0_offsets[N] = current_offset;

    // 4. 构建上层结构
    flat->node_levels.resize(N);
    if (flat->max_level < 0) flat->max_level = 0;
    if (flat->max_level > 100) flat->max_level = 100;
    flat->upper_link_offsets.assign((size_t)N * (flat->max_level + 1), -1);
    flat->upper_link_storage.reserve(N * M);

    for (int new_id = 0; new_id < N; ++new_id) {
        int old_id = new_to_old[new_id];
        int level = (int)src->nodes[old_id]->links.size() - 1;
        flat->node_levels[new_id] = level;
        for (int l = 1; l <= level && l <= flat->max_level; ++l) {
            const auto& src_links = src->nodes[old_id]->links[l];
            int storage_idx = (int)flat->upper_link_storage.size();
            size_t offset_idx = (size_t)new_id * (flat->max_level + 1) + l;
            if (offset_idx < flat->upper_link_offsets.size()) flat->upper_link_offsets[offset_idx] = storage_idx;
            flat->upper_link_storage.push_back((int)src_links.size());

            temp_neighbors.clear();
            const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;
            for (int old_nb : src_links) {
                if (old_nb < 0 || old_nb >= N) continue;
                int new_nb = old_to_new[old_nb];
                if (new_nb < 0 || new_nb >= N) continue;
                const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
                float d = l2sq_100d(vec_u, vec_v);
                temp_neighbors.emplace_back(d, new_nb);
            }
            std::sort(temp_neighbors.begin(), temp_neighbors.end());
            for (const auto &p : temp_neighbors) flat->upper_link_storage.push_back(p.second);
        }
    }

    if (DEBUG_TIMING) {
        std::cout << "[FlatHNSW-CSR] Converted " << N << " nodes with triangle pruning support." << std::endl;
    }

    return flat;
}