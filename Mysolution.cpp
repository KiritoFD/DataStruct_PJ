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
// SIMD 距离计算 (带/不带计数)
// ---------------------------------------------------------
// 通过宏 ENABLE_DIST_COUNTING 控制是否统计距离计算次数以避免 TLS 开销（默认关闭）
#ifndef ENABLE_DIST_COUNTING
#define ENABLE_DIST_COUNTING 0
#endif

#if defined(__AVX512F__)
static inline float l2sq_counted(const float* __restrict a, const float* __restrict b, int dim) {
#if ENABLE_DIST_COUNTING
    ++tl_dist_counter;
#endif
    int i = 0;
    __m512 sumv = _mm512_setzero_ps();
    for (; i <= dim - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d = _mm512_sub_ps(va, vb);
        sumv = _mm512_fmadd_ps(d, d, sumv);
    }
    float s = _mm512_reduce_add_ps(sumv);
    for (; i < dim; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}
#elif defined(__AVX2__)
static inline float l2sq_counted(const float* __restrict a, const float* __restrict b, int dim) {
#if ENABLE_DIST_COUNTING
    ++tl_dist_counter;
#endif
    int i = 0;
    __m256 sumv = _mm256_setzero_ps();
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        sumv = _mm256_fmadd_ps(d, d, sumv);
    }
    __m128 lo = _mm256_castps256_ps128(sumv);
    __m128 hi = _mm256_extractf128_ps(sumv, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float s = _mm_cvtss_f32(sum128);
    for (; i < dim; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}
#else
static inline float l2sq_counted(const float* __restrict a, const float* __restrict b, int dim) {
#if ENABLE_DIST_COUNTING
    ++tl_dist_counter;
#endif
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}
#endif

// ---------------------------------------------------------
// 全局线程池
// ---------------------------------------------------------
class ThreadPool {
public:
    ThreadPool(int n) : stop(false) {
        for (int i = 0; i < n; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(qmtx);
                        cv.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(qmtx); stop = true; }
        cv.notify_all();
        for (auto &w : workers) w.join();
    }
    template<class F> void enqueue(F&& f) {
        { std::unique_lock<std::mutex> lock(qmtx); tasks.emplace(std::forward<F>(f)); }
        cv.notify_one();
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex qmtx;
    std::condition_variable cv;
    bool stop;
};

static std::atomic<int> g_HNSW_M{HNSW_DEFAULT_M};                    // 改为使用头文件常量
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

// ---------------------------------------------------------
// 扁平化 HNSW 索引 (Read-Only Optimized)
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
    
    // 图存储 - 彻底移除 vector<vector>
    // graph_l0: 存储第0层的所有连接
    // 内存布局: [Count, nb1, nb2, ..., Padding] * N
    std::vector<int> graph_l0; 
    
    // 上层图结构 (扁平化存储)
    std::vector<int> node_levels;        // 每个节点的层级
    std::vector<int> upper_link_offsets; // 每个节点上层链接在storage中的偏移 (N * max_layer)
    std::vector<int> upper_link_storage; // 存储格式: [count, nb1, nb2, ...]

    FlatHNSW(int d) : dim(d), enter_point(-1), num_nodes(0), max_level(0), max_m(0), max_m_upper(0) {}

    inline int size() const { return num_nodes; }

    // 获取第0层邻居的指针 (极速路径)
    inline const int* get_l0_links(int id, int& count) const {
        const int* base = graph_l0.data() + (size_t)id * (max_m + 1);
        count = *base;
        return base + 1;
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
        return l2sq_counted(get_vec(id), q, dim);
    }
    
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_counted(get_vec(id_a), get_vec(id_b), dim);
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
                PREFETCH_L1(get_vec(links[0]));
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                if (i + 1 < count) {
                    PREFETCH_L1(get_vec(links[i+1]));
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
    // 极速 Level 0 搜索 (针对大 ef_search 优化)
    // 使用 Max-Heap 维护结果集，避免线性插入开销
    // -------------------------------------------------------------
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        
        using Pair = std::pair<float, int>;
        
        // 1. 线程局部缓存 (避免反复分配内存)
        static thread_local std::vector<Pair> candidates;     // Min-Heap (待探索任务队列)
        static thread_local std::vector<Pair> top_results;    // Max-Heap (当前找到的最优结果集)
        static thread_local TagVisitedList visited;
        static thread_local std::vector<int> process_queue;   // 待计算距离的候选点队列
        
        candidates.clear(); 
        top_results.clear();
        process_queue.clear();
        
        // 预分配内存，防止动态扩容
        candidates.reserve(ef * 3);
        top_results.reserve(ef + 1);
        process_queue.reserve(max_m + 16);

        // 初始化 Visited
        visited.init(num_nodes);
        visited.advance();
        
        // 获取原始指针避免边界检查开销
        const uint16_t* visited_ptr = visited.data();
        uint16_t cur_tag = visited.currentTag();

        // 2. 初始化入口点
        float d0 = dist(ep, q);
        visited.mark(ep);
        
        candidates.push_back({d0, ep});     // Min-Heap
        top_results.push_back({d0, ep});    // Max-Heap

        // lower_bound 是当前结果集中"最差"(最远)的距离
        float lower_bound = d0;

        // 比较器定义
        auto min_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; }; // 小顶堆
        auto max_comp = [](const Pair& a, const Pair& b) { return a.first < b.first; }; // 大顶堆

        while (!candidates.empty()) {
            // 2.1 取出当前最近的候选点
            std::pop_heap(candidates.begin(), candidates.end(), min_comp);
            Pair curr = candidates.back();
            candidates.pop_back();

            // 2.2 剪枝条件 (Strict Pruning)
            if (curr.first > lower_bound && (int)top_results.size() >= ef) {
                break;
            }

            // 2.3 获取邻居
            int count;
            const int* links = get_l0_links(curr.second, count);

            // 2.4 Stage 1: 快速过滤 (Filter) 
            // 紧凑循环，只做位运算/比较，无浮点计算
            process_queue.clear();
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                if (visited_ptr[nb] != cur_tag) {
                    visited.mark(nb);
                    process_queue.push_back(nb);
                }
            }

            int q_size = (int)process_queue.size();
            if (q_size == 0) continue;

            // 2.5 Stage 2: 预取与计算 (Prefetch & Compute)
            const int* p_queue = process_queue.data();

            // >>> 预取流水线启动 <<<
            // 提前预取前 2 个点的向量数据
            constexpr int pf_lookahead = 2; 
            for (int i = 0; i < q_size && i < pf_lookahead; ++i) {
                PREFETCH_L1(get_vec(p_queue[i]));
            }

            for (int i = 0; i < q_size; ++i) {
                // 流水线预取：在计算 i 的同时，预取 i + lookahead
                if (i + pf_lookahead < q_size) {
                    PREFETCH_L1(get_vec(p_queue[i + pf_lookahead]));
                }

                int nb = p_queue[i];
                float d = dist(nb, q);

                // 2.6 结果集维护 (Heap Operations)
                if ((int)top_results.size() < ef || d < lower_bound) {
                    // 加入 Max-Heap
                    top_results.push_back({d, nb});
                    std::push_heap(top_results.begin(), top_results.end(), max_comp);

                    // 如果超限，弹出最远的
                    if ((int)top_results.size() > ef) {
                        std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                        top_results.pop_back();
                    }
                    
                    // 更新门槛值 (Max-Heap 的堆顶是最远的)
                    lower_bound = top_results.front().first;

                    // 加入 Min-Heap 继续探索
                    candidates.push_back({d, nb});
                    std::push_heap(candidates.begin(), candidates.end(), min_comp);
                }
            }
        }

        // 3. 最终排序 - sort_heap 将堆转为升序数组
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
        return l2sq_counted(getVec(id), q, dim);
    }
    
    // 计算两个节点之间的距离
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_counted(getVec(id_a), getVec(id_b), dim);
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
                _mm_prefetch((const char*)getVec(neighbors[0]), _MM_HINT_T0);
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    _mm_prefetch((const char*)getVec(neighbors[i+1]), _MM_HINT_T0);
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
        
        // thread_local 缓存避免频繁分配
        static thread_local std::vector<Pair> top_candidates;
        static thread_local std::vector<Pair> candidate_queue;
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
        
        top_candidates.push_back({d0, ep});
        candidate_queue.push_back({d0, ep});
        std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);

        float lower_bound = d0;

        while (!candidate_queue.empty()) {
            std::pop_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
            auto curr = candidate_queue.back();
            candidate_queue.pop_back();

            // 关键剪枝：当前最近候选已超过结果集最远距离
            if (curr.first > lower_bound && (int)top_candidates.size() >= ef) {
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
                _mm_prefetch((const char*)getVec(neighbors[0]), _MM_HINT_T0);
            }

            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    _mm_prefetch((const char*)getVec(neighbors[i+1]), _MM_HINT_T0);
                }

                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float d_nb = dist(nb, q);

                    if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                        // 二分查找插入位置（保持升序）
                        auto it = std::upper_bound(top_candidates.begin(), top_candidates.end(),
                            Pair{d_nb, nb}, [](const Pair& a, const Pair& b) { return a.first < b.first; });
                        top_candidates.insert(it, {d_nb, nb});

                        if ((int)top_candidates.size() > ef) {
                            top_candidates.pop_back();
                        }
                        lower_bound = top_candidates.back().first;

                        candidate_queue.push_back({d_nb, nb});
                        std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
                    }
                }
            }
        }

        return top_candidates;
    }

    // -------------------------------------------------------
    // Robust Pruning (启发式选边) - 核心优化
    // -------------------------------------------------------
    void connectNodeHeuristic(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;

        // 1. 收集所有候选点（新搜索到的 + 原有邻居）
        std::vector<std::pair<float, int>> all_candidates;
        all_candidates.reserve(candidates.size() + m_max);
        
        for (const auto& p : candidates) {
            all_candidates.push_back(p);
        }

        // 读取旧邻居
        {
            std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
            const auto& old_links = nodes[id]->links[l];
            for (int old_nb : old_links) {
                if (old_nb >= 0 && old_nb < size()) {
                    all_candidates.push_back({distNodes(id, old_nb), old_nb});
                }
            }
        }

        // 2. 按距离排序并去重
        std::sort(all_candidates.begin(), all_candidates.end());
        all_candidates.erase(
            std::unique(all_candidates.begin(), all_candidates.end(),
                [](const auto& a, const auto& b) { return a.second == b.second; }),
            all_candidates.end()
        );

        // 3. 启发式筛选 (Robust Pruning / RNG 变体)
        std::vector<int> result_links;
        result_links.reserve(m_max);

        for (const auto& cand : all_candidates) {
            if ((int)result_links.size() >= m_max) break;

            float d_cand_to_curr = cand.first;
            int cand_id = cand.second;
            
            if (cand_id == id) continue;  // 排除自己

            // 多样性检查：新节点必须比"已选邻居"更接近当前点
            bool keep = true;
            for (int selected_nbr : result_links) {
                float d_cand_to_selected = distNodes(cand_id, selected_nbr);
                
                // 如果 cand 到已选邻居的距离 < cand 到当前点的距离
                // 说明 cand 应该被 selected 覆盖，不保留
                if (d_cand_to_selected < d_cand_to_curr) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                result_links.push_back(cand_id);
            }
        }

        // 4. 写回链接
        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            nodes[id]->links[l] = std::move(result_links);
        }

        // 5. 双向连接（对反向邻居也应用启发式）
        for (const auto& p : all_candidates) {
            int nb = p.second;
            if (nb < 0 || nb >= size() || nb == id) continue;
            
            // 检查结果链接中是否包含此邻居
            bool in_result = false;
            {
                std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
                for (int r : nodes[id]->links[l]) {
                    if (r == nb) { in_result = true; break; }
                }
            }
            if (!in_result) continue;

            // 收集反向邻居的候选
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

            // 排序去重
            std::sort(nb_candidates.begin(), nb_candidates.end());
            nb_candidates.erase(
                std::unique(nb_candidates.begin(), nb_candidates.end(),
                    [](const auto& a, const auto& b) { return a.second == b.second; }),
                nb_candidates.end()
            );

            // 启发式筛选
            std::vector<int> nb_result;
            nb_result.reserve(m_max);

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

            // 写回
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
// 转换函数：将动态图转换为静态扁平化图
// ---------------------------------------------------------
static FlatHNSW* convert_to_flat(SimpleHNSW* src) {
    FlatHNSW* flat = new FlatHNSW(src->dim);
    flat->data = std::move(src->data_flat);  // 移动数据避免拷贝
    flat->enter_point = src->enter_point;
    
    int N = src->size();
    int M = src->M;
    flat->max_m = M * 2;
    flat->max_m_upper = M;
    flat->num_nodes = N;

    // 计算入口点的最大层级
    if (src->enter_point >= 0 && src->enter_point < N) {
        flat->max_level = (int)src->nodes[src->enter_point]->links.size() - 1;
    } else {
        flat->max_level = 0;
    }

    // 1. 扁平化 Level 0 (最热路径)
    // 内存布局: [Count, nb1, nb2, ..., Padding] * N
    flat->graph_l0.resize((size_t)N * (flat->max_m + 1), 0);
    
    for (int i = 0; i < N; ++i) {
        auto& links = src->nodes[i]->links[0];
        int cnt = (int)links.size();
        size_t offset = (size_t)i * (flat->max_m + 1);
        
        flat->graph_l0[offset] = cnt;
        for (int j = 0; j < cnt; ++j) {
            flat->graph_l0[offset + 1 + j] = links[j];
        }
    }

    // 2. 扁平化 Upper Layers
    flat->node_levels.resize(N);
    flat->upper_link_offsets.resize((size_t)N * (flat->max_level + 1), -1);
    flat->upper_link_storage.reserve(N * M);  // 预估大小

    for (int i = 0; i < N; ++i) {
        int level = (int)src->nodes[i]->links.size() - 1;
        flat->node_levels[i] = level;
        
        for (int l = 1; l <= level; ++l) {
            auto& links = src->nodes[i]->links[l];
            int cnt = (int)links.size();
            
            // 记录偏移
            int start_idx = (int)flat->upper_link_storage.size();
            flat->upper_link_offsets[(size_t)i * (flat->max_level + 1) + l] = start_idx;
            
            // 存储数据: [count, nb1, nb2, ...]
            flat->upper_link_storage.push_back(cnt);
            for (int nb : links) {
                flat->upper_link_storage.push_back(nb);
            }
        }
    }

    if (DEBUG_TIMING) {
        std::cout << "[FlatHNSW] Converted " << N << " nodes, L0 size: " 
                  << flat->graph_l0.size() * sizeof(int) / 1024 << " KB, "
                  << "Upper size: " << flat->upper_link_storage.size() * sizeof(int) / 1024 << " KB" << std::endl;
    }
    
    return flat;
}

// ---------------------------------------------------------
// 并行包装类
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    FlatHNSW* flat_index = nullptr;  // 扁平化只读索引
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { 
        delete hnsw; 
        delete flat_index;
    }

    void build_from_memory(int d, const float* data, int n) {
        delete hnsw;
        hnsw = new SimpleHNSW(d, g_HNSW_M.load(), g_HNSW_MAX_LAYER.load());
        
        // 扁平化存储
        hnsw->data_flat.resize((size_t)n * d);
        std::memcpy(hnsw->data_flat.data(), data, (size_t)n * d * sizeof(float));
        
        hnsw->nodes.reserve(n);
        
        std::vector<int> levels(n);
        for (int i = 0; i < n; ++i) {
            levels[i] = std::min(hnsw->randomLevel(), g_HNSW_MAX_LAYER.load());
            hnsw->nodes.push_back(new HNSWNode(levels[i], g_HNSW_M.load()));
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

        // 转换为扁平化只读索引
        auto convert_start = std::chrono::high_resolution_clock::now();
        flat_index = convert_to_flat(hnsw);
        auto convert_end = std::chrono::high_resolution_clock::now();
        double convert_ms = std::chrono::duration<double, std::milli>(convert_end - convert_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Flat Conversion: " << std::fixed << std::setprecision(2) 
                      << convert_ms << " ms" << std::endl;
            std::cout.flush();
        }

        // 释放动态图以节省内存
        delete hnsw;
        hnsw = nullptr;
    }

    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

        if (!flat_index || flat_index->enter_point < 0) return {};

        int ep = flat_index->enter_point;
        int max_l = flat_index->node_levels[ep];
        int curr = ep;

        // 优化：限制上层搜索的起始层级
        // 对于高质量图，从较低层级开始可以减少不必要的跳跃
        int start_l = std::min(max_l, 4);  // 最多从第4层开始
        
        // 上层贪婪搜索
        for (int l = start_l; l > 0; l--) {
            curr = flat_index->greedySearchUpper(curr, query.data(), l);
        }
        
        // Level 0 极速搜索
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
extern "C" {

void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads) {
    if (M > 0) g_HNSW_M.store(M);
    if (max_layer > 0) g_HNSW_MAX_LAYER.store(max_layer);
    if (ef_construction > 0) g_HNSW_EF_CONSTRUCTION.store(ef_construction);
    if (ef_search > 0) g_HNSW_EF_SEARCH.store(ef_search);

    if (build_threads > 0) {
        int old = HNSW_BUILD_THREADS.load();
        if (build_threads != old) {
            std::lock_guard<std::mutex> lock(g_pool_mutex);
            HNSW_BUILD_THREADS.store(build_threads);
            if (g_thread_pool) {
                delete g_thread_pool;
                g_thread_pool = new ThreadPool(build_threads);
            }
        }
    }
}

void set_hnsw_debug(int dbg) { DEBUG_TIMING = (dbg != 0); }

uint64_t get_total_queries() { return g_total_query_count.load(std::memory_order_relaxed); }
double get_avg_dists_per_query() {
    uint64_t q = g_total_query_count.load(std::memory_order_relaxed);
    if (q == 0) return 0.0;
    return double(g_total_dist_count.load(std::memory_order_relaxed)) / double(q);
}
uint64_t get_last_query_dists() { return g_last_query_dist.load(std::memory_order_relaxed); }
void reset_dist_counters() {
    g_total_dist_count.store(0, std::memory_order_relaxed);
    g_total_query_count.store(0, std::memory_order_relaxed);
    g_last_query_dist.store(0, std::memory_order_relaxed);
}
double get_last_build_time_ms() { return g_last_build_ms.load(std::memory_order_relaxed); }

// 图质量统计函数
int get_graph_max_level() {
    if (!g_impl || !g_impl->flat_index) return 0;
    return g_impl->flat_index->max_level;
}

int get_graph_num_nodes() {
    if (!g_impl || !g_impl->flat_index) return 0;
    return g_impl->flat_index->num_nodes;
}

double get_graph_avg_degree_l0() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0) return 0.0;
    
    uint64_t total_degree = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        int count;
        idx->get_l0_links(i, count);
        total_degree += count;
    }
    return double(total_degree) / double(idx->num_nodes);
}

int get_graph_actual_max_layer() {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int max_lv = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        if (idx->node_levels[i] > max_lv) {
            max_lv = idx->node_levels[i];
        }
    }
    return max_lv;
}

// 获取各层级节点数量分布 (返回层级 l 的节点数)
int get_graph_nodes_at_level(int level) {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int count = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        if (idx->node_levels[i] >= level) {
            ++count;
        }
    }
    return count;
}

// 获取上层平均度数
double get_graph_avg_degree_upper() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0 || idx->max_level == 0) return 0.0;
    
    uint64_t total_degree = 0;
    int total_upper_nodes = 0;
    
    for (int i = 0; i < idx->num_nodes; ++i) {
        for (int l = 1; l <= idx->node_levels[i]; ++l) {
            int count;
            idx->get_upper_links(i, l, count);
            total_degree += count;
            ++total_upper_nodes;
        }
    }
    
    return total_upper_nodes > 0 ? double(total_degree) / double(total_upper_nodes) : 0.0;
}

} // extern "C"