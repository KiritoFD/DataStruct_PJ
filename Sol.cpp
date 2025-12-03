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
static std::atomic<bool> ABLATE_FLAT_INDEX(false);  // 新增：强制使用动态索引而不转换为扁平化
static std::atomic<bool> ABLATE_REORDER(false);     // 新增：禁用图拓扑重排优化

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
    // 【优化3】multi-queue L0 搜索 - 使用 2-way Batching
    // -------------------------------------------------------------
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        
        using Pair = std::pair<float, int>;
        
        static thread_local std::vector<Pair> candidates;      // min-heap
        static thread_local std::vector<Pair> top_results;     // max-heap
        static thread_local std::vector<int> expand_queue;     // FIFO扩张队列
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
                constexpr int pf_lookahead = 6;
                for (int i = 0; i < count && i < pf_lookahead; ++i) {
                    my_prefetch_l1(get_vec(links[i]));
                }

                int i = 0;
                
                // --- 2-way Batching Loop ---
                for (; i <= count - 2; i += 2) {
                    // 预取未来的向量
                    if (i + pf_lookahead < count) {
                        my_prefetch_l1(get_vec(links[i + pf_lookahead]));
                    }
                    if (i + pf_lookahead + 1 < count) {
                        my_prefetch_l1(get_vec(links[i + pf_lookahead + 1]));
                    }

                    int nb1 = links[i];
                    int nb2 = links[i + 1];
                    
                    bool v1 = (visited_ptr[nb1] == cur_tag);
                    bool v2 = (visited_ptr[nb2] == cur_tag);
                    
                    if (v1 && v2) continue; // 都访问过
                    
                    if (!v1) visited.mark(nb1);
                    if (!v2) visited.mark(nb2);
                    
                    float d1 = 0, d2 = 0;
                    
                    // 如果都没访问过，并行计算
                    if (!v1 && !v2) {
                        l2sq_100d_2x(q, get_vec(nb1), get_vec(nb2), d1, d2);
                    } else if (!v1) {
                        d1 = dist(nb1, q);
                    } else { // !v2
                        d2 = dist(nb2, q);
                    }

                    // 处理 nb1
                    if (!v1) {
                        process_candidate(d1, nb1);
                    }
                    // 处理 nb2
                    if (!v2) {
                        process_candidate(d2, nb2);
                    }
                }
                
                // 处理剩余的 1 个
                for (; i < count; ++i) {
                    if (i + pf_lookahead < count) {
                        my_prefetch_l1(get_vec(links[i + pf_lookahead]));
                    }
                    
                    int nb = links[i];
                    if (visited_ptr[nb] == cur_tag) continue;
                    visited.mark(nb);
                    
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

// ---------------------------------------------------------
// 生成基于 BFS 顺序的 ID 映射表
// 使得在图上遍历时，内存访问更连续
// ---------------------------------------------------------
static void generate_reordering(SimpleHNSW* src, std::vector<int>& old_to_new, std::vector<int>& new_to_old) {
    int N = src->size();
    old_to_new.assign(N, -1);
    new_to_old.resize(N);
    
    int ep = src->enter_point;
    int current_new_id = 0;
    
    std::vector<int> queue;
    queue.reserve(N);
    
    // 1. 先把入口点放入
    if (ep >= 0 && ep < N) {
        old_to_new[ep] = current_new_id++;
        new_to_old[0] = ep;
        queue.push_back(ep);
    }
    
    // 2. BFS 遍历（只用 L0 层邻居，因为 L0 最密集）
    size_t head = 0;
    while (head < queue.size()) {
        int u = queue[head++];
        const auto& links = src->nodes[u]->links[0];
        for (int v : links) {
            if (v >= 0 && v < N && old_to_new[v] == -1) {
                old_to_new[v] = current_new_id++;
                new_to_old[old_to_new[v]] = v;
                queue.push_back(v);
            }
        }
    }
    
    // 3. 处理不可达节点（孤立点）
    for (int i = 0; i < N; ++i) {
        if (old_to_new[i] == -1) {
            old_to_new[i] = current_new_id++;
            new_to_old[old_to_new[i]] = i;
        }
    }
    
    if (DEBUG_TIMING) {
        std::cout << "[Reorder] Generated BFS mapping for " << N << " nodes." << std::endl;
    }
}

// ---------------------------------------------------------
// 生成基于 DFS 顺序的 ID 映射表
// DFS 更能模拟贪婪搜索的"钻井"轨迹，使搜索路径上的节点在内存中更靠近
// ---------------------------------------------------------
static void generate_dfs_reordering(SimpleHNSW* src, std::vector<int>& old_to_new, std::vector<int>& new_to_old) {
    int N = src->size();
    old_to_new.assign(N, -1);
    new_to_old.resize(N);
    
    int ep = src->enter_point;
    int current_new_id = 0;
    
    // 使用栈代替队列
    std::vector<int> stack;
    stack.reserve(N);
    
    // 临时 visited 数组，防止环路导致的重复入栈
    std::vector<bool> visited_temp(N, false);
    
    if (ep >= 0 && ep < N) {
        stack.push_back(ep);
        visited_temp[ep] = true;
    }

    while (!stack.empty()) {
        int u = stack.back();
        stack.pop_back();
        
        // 只有出栈时才真正分配 ID，保证 DFS 序
        if (old_to_new[u] == -1) {
            old_to_new[u] = current_new_id;
            new_to_old[current_new_id] = u;
            current_new_id++;
        }

        const auto& links = src->nodes[u]->links[0];
        // 倒序压栈，这样出栈顺序就是正序的（优先访问距离近的邻居）
        for (auto it = links.rbegin(); it != links.rend(); ++it) {
            int v = *it;
            if (v >= 0 && v < N && !visited_temp[v]) {
                visited_temp[v] = true;
                stack.push_back(v);
            }
        }
    }
    
    // 处理孤立点
    for (int i = 0; i < N; ++i) {
        if (old_to_new[i] == -1) {
            old_to_new[i] = current_new_id;
            new_to_old[current_new_id] = i;
            current_new_id++;
        }
    }
    
    if (DEBUG_TIMING) {
        std::cout << "[Reorder] Generated DFS mapping for " << N << " nodes." << std::endl;
    }
}

// ---------------------------------------------------------
// 转换函数：将动态图转换为静态扁平化图 - CSR 格式
// ---------------------------------------------------------
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
        // 1. 生成 ID 映射
        generate_dfs_reordering(src, old_to_new, new_to_old);
        
        // 更新入口点
        flat->enter_point = (src->enter_point == -1) ? -1 : old_to_new[src->enter_point];
        
        // 计算 max_level - 需要遍历所有节点找最大层级
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) {
                flat->max_level = level;
            }
        }
        
        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder enabled, enter_point=" << flat->enter_point 
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }
        
        // 存储映射表供查询时使用
        flat->label_lookup = new_to_old;  // 复制而不是移动，因为后面还要用
        
    } else {
        // 不重排：identity mapping
        flat->enter_point = src->enter_point;
        
        // 计算 max_level
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) {
                flat->max_level = level;
            }
        }
        
        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder disabled, enter_point=" << flat->enter_point 
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }
        
        old_to_new.resize(N);
        new_to_old.resize(N);
        for (int i = 0; i < N; ++i) {
            old_to_new[i] = i;
            new_to_old[i] = i;
        }
        // label_lookup 留空表示 identity
    }

    // 2. 重排 Vector Data（使用对齐内存）
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Allocating data array: " << (size_t)N * flat->dim << " floats ("
                  << ((size_t)N * flat->dim * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        std::cout.flush();
    }
    
    flat->data.resize((size_t)N * flat->dim);
    
    if (flat->data.data() == nullptr) {
        std::cerr << "[FlatConvert] ERROR: Failed to allocate data array!" << std::endl;
        delete flat;
        return nullptr;
    }
    
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Copying vector data..." << std::endl;
        std::cout.flush();
    }
    
    for (int new_id = 0; new_id < N; ++new_id) {
        int old_id = new_to_old[new_id];
        if (old_id < 0 || old_id >= N) {
            std::cerr << "[FlatConvert] ERROR: Invalid old_id=" << old_id << " for new_id=" << new_id << std::endl;
            continue;
        }
        const float* src_vec = src->getVec(old_id);
        float* dst_vec = flat->data.data() + (size_t)new_id * flat->dim;
        std::memcpy(dst_vec, src_vec, flat->dim * sizeof(float));
    }
    
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Vector data copied." << std::endl;
        std::cout.flush();
    }

    // 3. 构建 L0 CSR 结构
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Building L0 CSR structure..." << std::endl;
        std::cout.flush();
    }
    
    flat->l0_offsets.resize(N + 1);
    
    // 预计算总大小
    uint64_t total_l0_links = 0;
    for (int i = 0; i < N; ++i) {
        total_l0_links += src->nodes[i]->links[0].size();
    }
    
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Total L0 links: " << total_l0_links << std::endl;
        std::cout.flush();
    }
    
    flat->l0_links.resize(total_l0_links);
    uint64_t current_offset = 0;

    // 临时缓冲区用于排序
    std::vector<std::pair<float, int>> temp_neighbors;
    temp_neighbors.reserve(M * 2);

    for (int new_id = 0; new_id < N; ++new_id) {
        flat->l0_offsets[new_id] = current_offset;
        
        int old_id = new_to_old[new_id];
        const auto& src_links = src->nodes[old_id]->links[0];
        
        // 收集邻居并转换 ID
        temp_neighbors.clear();
        const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;

        for (int old_nb : src_links) {
            if (old_nb < 0 || old_nb >= N) continue;  // 跳过无效邻居
            int new_nb = old_to_new[old_nb];
            if (new_nb < 0 || new_nb >= N) continue;  // 跳过无效映射
            const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
            float d = l2sq_100d(vec_u, vec_v);
            temp_neighbors.push_back({d, new_nb});
        }
        
        // 按距离排序
        std::sort(temp_neighbors.begin(), temp_neighbors.end());
        
        // 填入 l0_links
        for (const auto& p : temp_neighbors) {
            if (current_offset < total_l0_links) {
                flat->l0_links[current_offset++] = p.second;
            }
        }
    }
    flat->l0_offsets[N] = current_offset;
    
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] L0 CSR built, actual links: " << current_offset << std::endl;
        std::cout.flush();
    }

    // 4. 构建上层结构
    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Building upper layer structure, max_level=" << flat->max_level << std::endl;
        std::cout.flush();
    }
    
    flat->node_levels.resize(N);
    
    // 安全检查：确保 max_level 合理
    if (flat->max_level < 0) flat->max_level = 0;
    if (flat->max_level > 100) {
        std::cerr << "[FlatConvert] WARNING: max_level=" << flat->max_level << " seems too large, capping at 100" << std::endl;
        flat->max_level = 100;
    }
    
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
            if (offset_idx < flat->upper_link_offsets.size()) {
                flat->upper_link_offsets[offset_idx] = storage_idx;
            }
            
            flat->upper_link_storage.push_back((int)src_links.size());
            
            // 转换 ID 并排序
            temp_neighbors.clear();
            const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;

            for (int old_nb : src_links) {
                if (old_nb < 0 || old_nb >= N) continue;
                int new_nb = old_to_new[old_nb];
                if (new_nb < 0 || new_nb >= N) continue;
                const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
                float d = l2sq_100d(vec_u, vec_v);
                temp_neighbors.push_back({d, new_nb});
            }
            std::sort(temp_neighbors.begin(), temp_neighbors.end());

            for (const auto& p : temp_neighbors) {
                flat->upper_link_storage.push_back(p.second);
            }
        }
    }

    if (DEBUG_TIMING) {
        if (do_reorder) {
            std::cout << "[FlatHNSW-CSR] Converted & Reordered " << N << " nodes." << std::endl;
        } else {
            std::cout << "[FlatHNSW-CSR] Converted " << N << " nodes (reorder disabled)." << std::endl;
        }
        std::cout.flush();
    }
    
    return flat;
}

// ---------------------------------------------------------
// 反序列化函数：将 FlatHNSW 转换回 SimpleHNSW (用于消融实验)
// ---------------------------------------------------------
static SimpleHNSW* convert_flat_to_simple(FlatHNSW* flat) {
    SimpleHNSW* simple = new SimpleHNSW(flat->dim, flat->max_m_upper, flat->max_level);
    
    // 从 AlignedFloatArray 复制到 std::vector<float>
    simple->data_flat.resize(flat->data.size());
    if (flat->data.size() > 0) {
        std::memcpy(simple->data_flat.data(), flat->data.data(), flat->data.size() * sizeof(float));
    }
    
    simple->enter_point = flat->enter_point;
    
    int N = flat->num_nodes;
    simple->nodes.reserve(N);
    
    for (int i = 0; i < N; ++i) {
        int level = flat->node_levels[i];
        HNSWNode* node = new HNSWNode(level, flat->max_m_upper);
        
        {
            int cnt;
            const int* links = flat->get_l0_links(i, cnt);
            node->links[0].assign(links, links + cnt);
        }
        
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

// ---------------------------------------------------------
// 设置参数接口
// ---------------------------------------------------------
#include "extern.h"