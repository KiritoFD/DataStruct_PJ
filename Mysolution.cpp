#include "MySolution.h"
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <xmmintrin.h>
#include <cstdint>
#include <iomanip>
#include <chrono>

// 距离统计相关全局变量
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
static thread_local uint64_t tl_dist_counter = 0;
static std::atomic<double> g_last_build_ms{0.0};

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

static std::atomic<int> g_HNSW_M{32};
static std::atomic<int> g_HNSW_MAX_LAYER{16};
static std::atomic<int> g_HNSW_EF_CONSTRUCTION{400};
static std::atomic<int> g_HNSW_EF_SEARCH{80};

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
// SIMD 距离计算（扁平化数据版本）
// ---------------------------------------------------------
#if defined(__AVX512F__)
#include <immintrin.h>
static inline float l2sq_flat(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;
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
#include <immintrin.h>
static inline float l2sq_flat(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;
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
static inline float l2sq_flat(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}
#endif

// ---------------------------------------------------------
// Visited List (位图优化)
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
        return l2sq_flat(getVec(id), q, dim);
    }
    
    // 计算两个节点之间的距离
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_flat(getVec(id_a), getVec(id_b), dim);
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
// 并行包装类
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { delete hnsw; }

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
    }

    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

        int ep = hnsw->enter_point;
        if (ep < 0 || ep >= hnsw->size()) return {};

        int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
        int curr = ep;

        for (int l = max_l; l > 0; l--) {
            curr = hnsw->greedySearch<false>(curr, query.data(), l);
        }
        
        auto top = hnsw->searchLayer<false>(query.data(), curr, 0, g_HNSW_EF_SEARCH.load());
        
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

} // extern "C"