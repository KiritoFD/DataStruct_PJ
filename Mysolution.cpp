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
#include <cstdint> // 新增：距离统计相关全局变量

// 新增：距离统计相关全局变量
static std::atomic<uint64_t> g_dist_count{0};
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
// 默认关闭计数，仅在搜索时显式开启
static std::atomic<bool>   g_count_dist_enabled{false};

// 新增：线程本地累积，减少原子争用，并保证最后强制 flush
static thread_local uint64_t tl_dist_counter = 0;
static const uint64_t DIST_FLUSH_THRESHOLD = 1024ULL;

// 新增：记录最近一次 build 耗时（毫秒）
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
static bool DEBUG_TIMING = true; // 默认关闭调试输出，运行时可通过接口打开

// ---------------------------------------------------------
// SIMD 距离计算
// ---------------------------------------------------------
#if defined(__AVX2__)
#include <immintrin.h>
static inline float l2sq_dense(const float* a, const float* b, int dim) {
    // 采用线程本地统计并批量刷新到全局计数，减少原子操作并避免丢计数
    if (g_count_dist_enabled.load(std::memory_order_relaxed)) {
        ++tl_dist_counter;
        if (tl_dist_counter >= DIST_FLUSH_THRESHOLD) {
            g_dist_count.fetch_add(tl_dist_counter, std::memory_order_relaxed);
            tl_dist_counter = 0;
        }
    }

    int i = 0;
    __m256 sumv = _mm256_setzero_ps();
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        sumv = _mm256_add_ps(sumv, _mm256_mul_ps(d, d));
    }
    alignas(32) float tmp[8];
    // 使用非对齐存储，避免对齐不满足时导致未定义行为/栈破坏
    _mm256_storeu_ps(tmp, sumv);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}
#else
static inline float l2sq_dense(const float* a, const float* b, int dim) {
    if (g_count_dist_enabled.load(std::memory_order_relaxed)) {
        ++tl_dist_counter;
        if (tl_dist_counter >= DIST_FLUSH_THRESHOLD) {
            g_dist_count.fetch_add(tl_dist_counter, std::memory_order_relaxed);
            tl_dist_counter = 0;
        }
    }

    float s = 0.0f;
    const float* end = a + dim;
    while (a < end) {
        float t = *a++ - *b++;
        s += t * t;
    }
    return s;
}
#endif

// ---------------------------------------------------------
// Visited List
// ---------------------------------------------------------
class VisitedList {
public:
    std::vector<unsigned short> visited_tags;
    unsigned short curr_tag;

    VisitedList() : curr_tag(0) {}

    void init(int size) {
        if (visited_tags.size() < (size_t)size) {
            visited_tags.resize(size, 0);
            curr_tag = 0;
        }
    }

    void advance() {
        curr_tag++;
        if (curr_tag == 0) {
            std::fill(visited_tags.begin(), visited_tags.end(), 0);
            curr_tag = 1;
        }
    }

    inline bool isVisited(int id) const {
        if (id < 0 || (size_t)id >= visited_tags.size()) return false;
        return visited_tags[id] == curr_tag;
    }

    inline void mark(int id) {
        if (id < 0 || (size_t)id >= visited_tags.size()) return;
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

    int size() const { return (int)nodes.size(); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    inline float dist(int id, const float* v) const {
        if (id < 0 || id >= (int)data_vecs.size()) return std::numeric_limits<float>::infinity();
        return l2sq_dense(data_vecs[id].data(), v, dim);
    }

    // -------------------------------------------------------
    // 关键修复：模板化锁控制
    // UseLock = true  -> 构建时使用，安全但稍慢
    // UseLock = false -> 搜索时使用，极速但非线程安全(不可写)
    // -------------------------------------------------------
    
    template<bool UseLock>
    int greedySearch(int ep, const float* q, int l) const {
        if (ep < 0 || ep >= size()) return -1;
        float curd = dist(ep, q);
        bool changed = true;
        while (changed) {
            changed = false;
            {
                std::shared_lock<std::shared_mutex> lock_guard;
                const std::vector<int>* neighbors_ptr;
                if constexpr (UseLock) {
                    lock_guard = std::shared_lock<std::shared_mutex>(nodes[ep]->lock);
                    neighbors_ptr = &nodes[ep]->links[l];
                } else {
                    neighbors_ptr = &nodes[ep]->links[l];
                }
                const auto& neighbors = *neighbors_ptr;
                if (!neighbors.empty()) {
                    _mm_prefetch((const char*)data_vecs[neighbors[0]].data(), _MM_HINT_T0);
                }
                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int nb = neighbors[i];
                    if (nb < 0 || nb >= size()) continue; // 防止越界访问导致崩溃/栈破坏
                    if (i + 1 < neighbors.size()) {
                        _mm_prefetch((const char*)data_vecs[neighbors[i+1]].data(), _MM_HINT_T0);
                    }
                    float nd = dist(nb, q);
                    if (nd < curd) {
                        curd = nd;
                        ep = nb;
                        changed = true;
                    }
                }
            }
        }
        return ep;
    }

    template<bool UseLock>
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (ep < 0 || ep >= size()) return {};
        
        using Pair = std::pair<float, int>;
        auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; }; 
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> candidates(cmp);
        std::priority_queue<Pair> top_results; 

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
            float curd = cur.first;
            int curid = cur.second;

            if (curd > lowerBound) break;

            {
                std::shared_lock<std::shared_mutex> lock_guard;
                const std::vector<int>* neighbors_ptr;

                if constexpr (UseLock) {
                    lock_guard = std::shared_lock<std::shared_mutex>(nodes[curid]->lock);
                    neighbors_ptr = &nodes[curid]->links[l];
                } else {
                    neighbors_ptr = &nodes[curid]->links[l];
                }

                const auto& neighbors = *neighbors_ptr;

                if (!neighbors.empty()) {
                    _mm_prefetch((const char*)data_vecs[neighbors[0]].data(), _MM_HINT_T0);
                }

                for (size_t i = 0; i < neighbors.size(); ++i) {
                    int nb = neighbors[i];
                    if (nb < 0 || nb >= size()) continue; // 防止越界
                    if (i + 1 < neighbors.size()) {
                        _mm_prefetch((const char*)data_vecs[neighbors[i+1]].data(), _MM_HINT_T0);
                    }

                    if (!visited_list.isVisited(nb)) {
                        visited_list.mark(nb);
                        float nd = dist(nb, q);
                        if (top_results.size() < ef || nd < lowerBound) {
                            candidates.push({nd, nb});
                            top_results.push({nd, nb});
                            if (top_results.size() > ef) top_results.pop();
                            lowerBound = top_results.top().first;
                        }
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

    // ConnectNode 始终需要写锁
    void connectNode(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
         int m_max = (l == 0) ? M * 2 : M; 
         std::vector<std::pair<float, int>> selected = candidates;
         if ((int)selected.size() > m_max) selected.resize(m_max);

         {
             std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
             auto& links = nodes[id]->links[l];
             for (auto& p : selected) links.push_back(p.second);
         }

         for (auto& p : selected) {
             int nb = p.second;
             if (nb < 0 || nb >= size()) continue;
             std::unique_lock<std::shared_mutex> lock(nodes[nb]->lock);
             auto& links = nodes[nb]->links[l];
             bool exists = false;
             for(int x : links) if(x == id) { exists=true; break; }
             if(!exists) links.push_back(id);

             if ((int)links.size() > m_max) {
                 std::vector<std::pair<float, int>> nbr_dists;
                 nbr_dists.reserve(links.size());
                 for (int n_nb : links) {
                    if (n_nb < 0 || n_nb >= size()) continue;
                     nbr_dists.push_back({dist(n_nb, data_vecs[nb].data()), n_nb});
                 }
                 std::sort(nbr_dists.begin(), nbr_dists.end());
                 
                 links.clear();
                 for(int i=0; i<m_max; ++i) links.push_back(nbr_dists[i].second);
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
                    
                    // 可选：调试输出进度，如果不需要可注释掉
                    // if (DEBUG_TIMING) {
                    //    int p = processed.fetch_add(1, std::memory_order_relaxed);
                    // } else {
                       // processed.fetch_add(1, std::memory_order_relaxed);
                    // }
                }
                // 批量增加计数，减少原子操作争用
                processed.fetch_add(end - i, std::memory_order_release);
            });
        }
        
        // 修正等待逻辑：processed 初始为 1，因为 0 号点已处理
        while (processed.load(std::memory_order_acquire) < n) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto build_end = std::chrono::high_resolution_clock::now();
        if (DEBUG_TIMING) {
            double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            std::cout << "[Timing] Parallel Build: " << total_ms << " ms for " << n << " points." << std::endl;
        }
        // 记录最近一次 build 耗时（ms）
        {
            double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            g_last_build_ms.store(total_ms, std::memory_order_relaxed);
        }
    }

    std::vector<std::pair<int,float>> search(const std::vector<float>& query, int k) {
        // 开启距离计数（仅对本次搜索计数）
        g_dist_count.store(0, std::memory_order_relaxed);
        g_count_dist_enabled.store(true, std::memory_order_release);

        // 原有搜索逻辑
        int ep = hnsw->enter_point;
        if (ep < 0 || ep >= hnsw->size()) {
            // 在返回之前关闭并 flush 本地计数（防止泄漏）
            if (tl_dist_counter) {
                g_dist_count.fetch_add(tl_dist_counter, std::memory_order_relaxed);
                tl_dist_counter = 0;
            }
            g_count_dist_enabled.store(false, std::memory_order_release);
            return {}; // 防御性检查
        }

        int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
        int curr = ep;

        // 【搜索阶段】：UseLock = false (极速模式)
        for(int l = max_l; l > 0; l--) curr = hnsw->greedySearch<false>(curr, query.data(), l);
        
        auto top = hnsw->searchLayer<false>(query.data(), curr, 0, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int,float>> out;
        if((int)top.size() > k) top.resize(k);
        out.reserve(top.size());
        for(const auto &p: top){
            if (p.second < 0 || p.second >= (int)point_ids.size()) continue;
            out.push_back({point_ids[p.second], p.first});
        }

        // 关闭计数并强制 flush 线程本地计数，然后记录
        if (tl_dist_counter) {
            g_dist_count.fetch_add(tl_dist_counter, std::memory_order_relaxed);
            tl_dist_counter = 0;
        }
        g_count_dist_enabled.store(false, std::memory_order_release);

        uint64_t last = g_dist_count.load(std::memory_order_relaxed);
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