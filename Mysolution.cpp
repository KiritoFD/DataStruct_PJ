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
#include <xmmintrin.h> // _mm_prefetch
#include <cstdint>
#include <iomanip>
#include <chrono>

// 距离统计相关全局变量（简化后）
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};

// 线程本地计数器（单次查询使用）
static thread_local uint64_t tl_dist_counter = 0;

// 记录最近一次 build 耗时（毫秒）
static std::atomic<double> g_last_build_ms{0.0};

// ---------------------------------------------------------
// 全局线程池 (保持不变)
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
        {
            std::unique_lock<std::mutex> lock(qmtx);
            stop = true;
        }
        cv.notify_all();
        for (auto &w : workers) w.join();
    }
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(qmtx);
            tasks.emplace(std::forward<F>(f));
        }
        cv.notify_one();
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex qmtx;
    std::condition_variable cv;
    bool stop;
};

static std::atomic<int> g_HNSW_M{36};
static std::atomic<int> g_HNSW_MAX_LAYER{16};
static std::atomic<int> g_HNSW_EF_CONSTRUCTION{371};
static std::atomic<int> g_HNSW_EF_SEARCH{35};

// 把线程计数也改为atomic，允许在运行时重建线程池
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

// ---------------------------------------------------------
// 参数设置
// ---------------------------------------------------------
static bool DEBUG_TIMING = true;

// ---------------------------------------------------------
// SIMD 距离计算：AVX-512 > AVX2 > 标量
// ---------------------------------------------------------
#if defined(__AVX512F__)
#include <immintrin.h>
static inline float l2sq_dense(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;

    int i = 0;
    __m512 sumv = _mm512_setzero_ps();
    
    // 每次处理 16 个 float
    for (; i <= dim - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 d = _mm512_sub_ps(va, vb);
        sumv = _mm512_fmadd_ps(d, d, sumv);  // FMA: sumv += d * d
    }
    
    float s = _mm512_reduce_add_ps(sumv);
    
    // 处理剩余（可用 AVX2 或标量）
    for (; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

#elif defined(__AVX2__)
#include <immintrin.h>
static inline float l2sq_dense(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;

    int i = 0;
    __m256 sumv = _mm256_setzero_ps();
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        sumv = _mm256_fmadd_ps(d, d, sumv);  // FMA 替代 mul + add
    }
    
    // 水平求和
    __m128 lo = _mm256_castps256_ps128(sumv);
    __m128 hi = _mm256_extractf128_ps(sumv, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float s = _mm_cvtss_f32(sum128);
    
    for (; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

#else
static inline float l2sq_dense(const float* __restrict a, const float* __restrict b, int dim) {
    ++tl_dist_counter;

    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}
#endif

// ---------------------------------------------------------
// Visited List（优化：避免频繁 resize 检查）
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

    inline bool isVisited(int id) const {
        return visited_tags[id] == curr_tag;
    }

    inline void mark(int id) {
        visited_tags[id] = curr_tag;
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
        for(auto& l : links) l.reserve(M + 1); 
    }
};

class SimpleHNSW {
 public:
    int dim;
    int M;
    int maxLayer;
    
    std::vector<std::vector<float>> data_vecs; 
    std::vector<HNSWNode*> nodes; 
    
    int enter_point; 
    std::shared_mutex global_mutex; 

    // 使用编译时常量作为默认值（运行时参数通过 g_HNSW_* 传入）
    SimpleHNSW(int d, int m = 16, int ml = 16)
        : dim(d), M(m), maxLayer(ml), enter_point(-1) {}

    ~SimpleHNSW() {
        for (auto p : nodes) delete p;
    }

    inline int size() const { return (int)nodes.size(); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    inline float dist(int id, const float* v) const {
        return l2sq_dense(data_vecs[id].data(), v, dim);
    }

    // -------------------------------------------------------
    // 优化后的 greedySearch：减少内存分配
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
                _mm_prefetch((const char*)data_vecs[neighbors[0]].data(), _MM_HINT_T0);
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    _mm_prefetch((const char*)data_vecs[neighbors[i+1]].data(), _MM_HINT_T0);
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
    // 优化后的 searchLayer：减少拷贝，预分配
    // -------------------------------------------------------
    template<bool UseLock>
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return {};
        
        using Pair = std::pair<float, int>;
        
        // 使用 thread_local 避免频繁堆分配
        static thread_local std::vector<Pair> candidates_vec;
        static thread_local std::vector<Pair> results_vec;
        candidates_vec.clear();
        results_vec.clear();
        candidates_vec.reserve(ef * 2);
        results_vec.reserve(ef + 1);

        auto cmp_min = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto cmp_max = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> candidates(cmp_min, std::move(candidates_vec));
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> top_results(cmp_max, std::move(results_vec));

        static thread_local VisitedList visited_list;
        visited_list.init(size());
        visited_list.advance();

        float d0 = dist(ep, q);
        candidates.push({d0, ep});
        top_results.push({d0, ep});
        visited_list.mark(ep);

        float lowerBound = d0;

        while (!candidates.empty()) {
            auto cur = candidates.top();
            candidates.pop();
            
            if (cur.first > lowerBound) break;

            const std::vector<int>* neighbors_ptr;
            std::shared_lock<std::shared_mutex> lock_guard;
            
            if constexpr (UseLock) {
                lock_guard = std::shared_lock<std::shared_mutex>(nodes[cur.second]->lock);
            }
            neighbors_ptr = &nodes[cur.second]->links[l];
            
            const auto& neighbors = *neighbors_ptr;
            const int nsize = (int)neighbors.size();

            if (nsize > 0) {
                _mm_prefetch((const char*)data_vecs[neighbors[0]].data(), _MM_HINT_T0);
            }

            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    _mm_prefetch((const char*)data_vecs[neighbors[i+1]].data(), _MM_HINT_T0);
                }

                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float nd = dist(nb, q);
                    if (top_results.size() < (size_t)ef || nd < lowerBound) {
                        candidates.push({nd, nb});
                        top_results.push({nd, nb});
                        if (top_results.size() > (size_t)ef) {
                            top_results.pop();
                        }
                        lowerBound = top_results.top().first;
                    }
                }
            }
        }

        std::vector<Pair> res;
        res.reserve(top_results.size());
        while (!top_results.empty()) {
            res.push_back(top_results.top());
            top_results.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    // -------------------------------------------------------
    // 优化后的 connectNode：距离计算和排序在锁外执行
    // -------------------------------------------------------
    void connectNode(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;
        
        std::vector<std::pair<float, int>> selected = candidates;
        if ((int)selected.size() > m_max) selected.resize(m_max);

        // 1. 为当前节点添加邻居（锁内仅执行链接操作）
        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            auto& links = nodes[id]->links[l];
            for (auto& p : selected) links.push_back(p.second);
        }

        // 2. 为每个邻居添加反向链接
        for (auto& p : selected) {
            int nb = p.second;
            if (nb < 0 || nb >= size()) continue;

            // 2a. 在读锁下快速拷贝当前邻居列表
            std::vector<int> current_links;
            {
                std::shared_lock<std::shared_mutex> read_lock(nodes[nb]->lock);
                current_links = nodes[nb]->links[l];
            }

            // 2b. 检查是否已存在
            bool exists = false;
            for (int x : current_links) {
                if (x == id) { exists = true; break; }
            }

            // 2c. 在锁外计算所有距离并排序
            std::vector<std::pair<float, int>> nbr_dists;
            nbr_dists.reserve(current_links.size() + 1);
            
            for (int n_nb : current_links) {
                if (n_nb < 0 || n_nb >= size()) continue;
                nbr_dists.push_back({dist(n_nb, data_vecs[nb].data()), n_nb});
            }
            
            // 添加新节点
            if (!exists) {
                nbr_dists.push_back({dist(id, data_vecs[nb].data()), id});
            }

            // 仅在需要裁剪时才排序
            bool need_prune = ((int)nbr_dists.size() > m_max);
            if (need_prune) {
                std::sort(nbr_dists.begin(), nbr_dists.end());
            }

            // 2d. 在写锁内执行最小化更新
            {
                std::unique_lock<std::shared_mutex> write_lock(nodes[nb]->lock);
                auto& links = nodes[nb]->links[l];
                
                if (need_prune) {
                    links.clear();
                    for (int i = 0; i < m_max && i < (int)nbr_dists.size(); ++i) {
                        links.push_back(nbr_dists[i].second);
                    }
                } else if (!exists) {
                    links.push_back(id);
                }
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
            
            // 【构建阶段】：UseLock = true
            for (int l = max_l; l > level; l--) {
                curr = greedySearch<true>(curr, data_vecs[id].data(), l);
            }

            for (int l = std::min(level, max_l); l >= 0; l--) {
                auto top = searchLayer<true>(data_vecs[id].data(), curr, l, g_HNSW_EF_CONSTRUCTION.load());
                if (!top.empty()) curr = top[0].second;
                connectNode(id, top, l);
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

    ~HnswSolutionParallel(){ delete hnsw; }

    void build_from_memory(int d, const std::vector<std::vector<float>>& data) {
        int n = (int)data.size();
        delete hnsw;
        hnsw = new SimpleHNSW(d, g_HNSW_M.load(), g_HNSW_MAX_LAYER.load());
        
        hnsw->data_vecs = data; 
        hnsw->nodes.reserve(n);
        
        std::vector<int> levels(n);
        for(int i=0; i<n; ++i) {
            levels[i] = hnsw->randomLevel();
            if(levels[i] > g_HNSW_MAX_LAYER) levels[i] = g_HNSW_MAX_LAYER;
            hnsw->nodes.push_back(new HNSWNode(levels[i], g_HNSW_M.load()));
        }
        
        point_ids.resize(n);
        for(int i=0; i<n; ++i) point_ids[i] = i;

        if (n > 0) {
            hnsw->enter_point = 0;
        }

        auto build_start = std::chrono::high_resolution_clock::now();

        ThreadPool* pool = getThreadPool();
        std::atomic<int> processed(1); 

        int start_idx = 1;
        int chunk_size = 1000; 

        for (int i = start_idx; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            pool->enqueue([this, i, end, &levels, &processed, n]() {
                for (int j = i; j < end; ++j) {
                    hnsw->insertPointParallel(j, levels[j]);
                }
                // 批量增加计数，减少原子操作争用
                processed.fetch_add(end - i, std::memory_order_release);
            });
        }
        
        // 修正等待逻辑：processed 初始为 1，因为 0 号点已处理
        // 添加进度监控线程
        std::thread progress_thread([&processed, n, &build_start]() {
            int last_reported = 0;
            while (processed.load(std::memory_order_acquire) < n) {
                int curr = processed.load(std::memory_order_acquire);
                // 每处理 10% 或至少 50000 个点输出一次
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

    std::vector<std::pair<int,float>> search(const std::vector<float>& query, int k) {
        auto search_start = std::chrono::high_resolution_clock::now();
        
        // 清零线程本地计数器
        tl_dist_counter = 0;

        int ep = hnsw->enter_point;
        if (ep < 0 || ep >= hnsw->size()) {
            tl_dist_counter = 0;
            return {};
        }

        int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
        int curr = ep;

        for (int l = max_l; l > 0; l--) {
            curr = hnsw->greedySearch<false>(curr, query.data(), l);
        }
        
        auto top = hnsw->searchLayer<false>(query.data(), curr, 0, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int,float>> out;
        if ((int)top.size() > k) top.resize(k);
        out.reserve(top.size());
        for (const auto& p : top) {
            if (p.second < 0 || p.second >= (int)point_ids.size()) continue;
            out.push_back({point_ids[p.second], p.first});
        }

        // 结束时聚合统计 (必须先于时间输出)
        uint64_t last = tl_dist_counter;
        tl_dist_counter = 0;
        g_last_query_dist.store(last, std::memory_order_relaxed);
        g_total_dist_count.fetch_add(last, std::memory_order_relaxed);
        g_total_query_count.fetch_add(1, std::memory_order_relaxed);

        auto search_end = std::chrono::high_resolution_clock::now();
        double search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Search: " << search_ms << " ms, dist_count=" << last << std::endl;
            std::cout.flush();
        }

        return out;
    }
};

// ---------------------------------------------------------
// 对外接口
// ---------------------------------------------------------
static HnswSolutionParallel* g_impl = nullptr;

void build_hnsw(int d, const std::vector<float>& base) {
    int n = (int)base.size() / d;
    std::vector<std::vector<float>> data(n, std::vector<float>(d));
    #pragma omp parallel for if(n > 10000)
    for(int i = 0; i < n; i++) {
        std::memcpy(data[i].data(), &base[i*d], d * sizeof(float));
    }

    delete g_impl;
    g_impl = new HnswSolutionParallel();
    g_impl->build_from_memory(d, data);
}

std::vector<std::pair<int,float>> search_hnsw(const std::vector<float>& query, int k) {
    if(!g_impl) return {};
    return g_impl->search(query, k);
}

// Implementations for Solution declared in MySolution.h
Solution::Solution() : k_(10) {}

void Solution::build(int d, const std::vector<float>& base) {
    build_hnsw(d, base);
}

void Solution::search(const std::vector<float>& query, int* result) {
    // 保证返回数组长度为 k_ (10)，未填满项用 -1 填充
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

// 运行时批量设置（支持部分设置)
void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads) {
    if (M > 0) g_HNSW_M.store(M);
    if (max_layer > 0) g_HNSW_MAX_LAYER.store(max_layer);
    if (ef_construction > 0) g_HNSW_EF_CONSTRUCTION.store(ef_construction);
    if (ef_search > 0) g_HNSW_EF_SEARCH.store(ef_search);

    if (build_threads > 0) {
        int old = HNSW_BUILD_THREADS.load();
        if (build_threads != old) {
            // 保护并重建线程池
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

// 新增：对外查询接口
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
 // 新增：获取最近一次 build 耗时（ms）
 double get_last_build_time_ms() { return g_last_build_ms.load(std::memory_order_relaxed); }

} // extern "C"