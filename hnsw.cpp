// hnsw_solution_parallel.cpp
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
// timing
#include <chrono>
// 并行辅助
#include <atomic>
#include <condition_variable>
#include <functional>

// 全局线程池（避免反复创建销毁线程）
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
    void wait() {
        // simple barrier: wait until all tasks done
        std::unique_lock<std::mutex> lock(qmtx);
        cv.wait(lock, [this]() { return tasks.empty(); });
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex qmtx;
    std::condition_variable cv;
    bool stop;
};

// 构建与搜索中用于并行距离计算的线程数（可调）
static int HNSW_BUILD_THREADS = [](){
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

constexpr int HNSW_M = 16;                     // 恢复为较高 M 提升召回率（牺牲部分速度）
constexpr int HNSW_MAX_LAYER = 6;
constexpr int HNSW_EF_CONSTRUCTION = 200;     // 恢复构建时的 ef，提升邻居质量
constexpr int HNSW_EF_SEARCH = 256;          // 增大搜索时的 ef 提高 recall（可根据时间/精度折中）
static bool DEBUG_TIMING = true;              // 默认为 false，遇到问题可打开

// 尝试使用 AVX2 加速（如果可用），否则回退到标量实现
#if defined(__AVX2__)
#include <immintrin.h>
static inline float l2sq_dense(const float* a, const float* b, int dim) {
    int i = 0;
    __m256 sumv = _mm256_setzero_ps();
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d = _mm256_sub_ps(va, vb);
        sumv = _mm256_add_ps(sumv, _mm256_mul_ps(d, d));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sumv);
    float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    for (; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}
#else
static inline float l2sq_dense(const float* a, const float* b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}
#endif

// ---------------- Simple HNSW Node ----------------
struct HNSWNode {
    std::vector<float> vec;
    std::vector<std::vector<int>> links;
};

// ---------------- Simple HNSW for batch ----------------
class SimpleHNSW {
public:
    int dim;
    int M;
    int maxLayer;
    std::vector<HNSWNode> nodes;
    int enter_point;

    SimpleHNSW(int d, int m = HNSW_M, int ml = HNSW_MAX_LAYER)
        : dim(d), M(m), maxLayer(ml), enter_point(-1) {}

    int size() const { return (int)nodes.size(); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f,1.f);
        int lvl = 0; float p = 0.5f;
        while (ud(rng) < p && lvl < maxLayer) lvl++;
        return lvl;
    }

    int greedySearch(int ep, const float* q, int l) const {
        if (ep < 0) return -1;
        float curd = l2sq_dense(nodes[ep].vec.data(), q, dim);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int nb : nodes[ep].links[l]) {
                float nd = l2sq_dense(nodes[nb].vec.data(), q, dim);
                if (nd < curd) { curd = nd; ep = nb; changed = true; }
            }
        }
        return ep;
    }

    // 使用线程池并行批量距离计算（提高批处理阈值、减少线程开销）
    void batchDistances(const float* q, const std::vector<int>& ids, std::vector<float>& out) const {
        out.resize(ids.size());
        int n = (int)ids.size();
        // 提高阈值到 128，小批量不值得并行
        if (n < 128) {
            for (int i = 0; i < n; ++i) {
                int id = ids[i];
                out[i] = l2sq_dense(nodes[id].vec.data(), q, dim);
            }
            return;
        }
        ThreadPool* pool = getThreadPool();
        int T = HNSW_BUILD_THREADS;
        int chunk = (n + T - 1) / T;
        std::atomic<int> done_count(0);
        for (int t = 0; t < T; ++t) {
            int begin = t * chunk;
            int end = std::min(begin + chunk, n);
            if (begin >= end) break;
            pool->enqueue([&, begin, end](){
                for (int i = begin; i < end; ++i) {
                    int id = ids[i];
                    out[i] = l2sq_dense(nodes[id].vec.data(), q, dim);
                }
                done_count.fetch_add(1, std::memory_order_release);
            });
        }
        // spin wait (更高效简单)
        int expected = (n + chunk - 1) / chunk;
        while (done_count.load(std::memory_order_acquire) < expected) {
            std::this_thread::yield();
        }
    }

    std::vector<std::pair<float,int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (ep < 0) return {};
        // candidate: min-heap (distance ascending)
        using Pair = std::pair<float,int>;
        struct MinCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first > b.first; } };
        // top: max-heap (largest distance on top) to track worst among kept ef
        struct MaxCmp { bool operator()(const Pair& a, const Pair& b) const { return a.first < b.first; } };
        std::priority_queue<Pair, std::vector<Pair>, MinCmp> candidate;
        std::priority_queue<Pair, std::vector<Pair>, MaxCmp> top;
        std::vector<char> visited(size(), 0);

        float d0 = l2sq_dense(nodes[ep].vec.data(), q, dim);
        candidate.push({d0, ep});
        top.push({d0, ep});
        visited[ep] = 1;

        float worstDist = top.top().first;

        while (!candidate.empty()) {
            auto cur = candidate.top(); candidate.pop();
            float curd = cur.first;
            int curid = cur.second;
            // termination condition
            if ((int)top.size() >= ef && curd > worstDist) break;

            // 收集未访问邻居
            const auto &nbrs = nodes[curid].links[l];
            std::vector<int> batch_ids;
            batch_ids.reserve(nbrs.size());
            for (int nb : nbrs) {
                if (nb < 0 || nb >= size()) continue;
                if (visited[nb]) continue;
                batch_ids.push_back(nb);
            }
            if (!batch_ids.empty()) {
                std::vector<float> dists;
                batchDistances(q, batch_ids, dists);
                for (size_t bi = 0; bi < batch_ids.size(); ++bi) {
                    int nb = batch_ids[bi];
                    float nd = dists[bi];
                    visited[nb] = 1;
                    if ((int)top.size() < ef || nd < worstDist) {
                        candidate.push({nd, nb});
                        top.push({nd, nb});
                        if ((int)top.size() > ef) top.pop();
                        worstDist = top.top().first;
                    }
                }
            }
        }
        // extract results (ascending)
        std::vector<Pair> res;
        res.reserve(top.size());
        while (!top.empty()) { res.push_back(top.top()); top.pop(); }
        std::sort(res.begin(), res.end(), [](const Pair& a, const Pair& b){ return a.first < b.first; });
        return res;
    }

    void connectNode(int id, const std::vector<std::pair<float,int>>& candidates, int l) {
        if ((int)nodes.size() <= id) return;
        int m = std::min(M, (int)candidates.size());
        if ((int)nodes[id].links.size() <= l) nodes[id].links.resize(l + 1);

        // helper: add neighbor b into a's level l list (no-dup), prune if > M*2
        // 并行化修剪时的距离计算
        auto addNeighbor = [&](int a, int b) {
            if (a < 0 || a >= (int)nodes.size() || b < 0 || b >= (int)nodes.size()) return;
            auto &lst = nodes[a].links[l];
            // check duplicate
            bool found = false;
            for (int x : lst) if (x == b) { found = true; break; }
            if (!found) lst.push_back(b);
            // prune if too large: keep closest M*2 neighbors
            int cap = M * 2;
            if ((int)lst.size() > cap) {
                // 并行计算修剪距离
                std::vector<int> ids(lst.begin(), lst.end());
                std::vector<float> dists;
                batchDistances(nodes[a].vec.data(), ids, dists);
                std::vector<std::pair<float,int>> tmp;
                tmp.reserve(ids.size());
                for (size_t i = 0; i < ids.size(); ++i) tmp.emplace_back(dists[i], ids[i]);
                std::sort(tmp.begin(), tmp.end(), [](auto &x, auto &y){ return x.first < y.first; });
                lst.clear();
                for (int i = 0; i < cap; ++i) lst.push_back(tmp[i].second);
            }
        };

        // Ensure candidates are considered in ascending distance order (safety)
        std::vector<std::pair<float,int>> cand = candidates;
        std::sort(cand.begin(), cand.end(), [](auto &a, auto &b){ return a.first < b.first; });

        for (int i = 0; i < m; ++i) {
            int nb = cand[i].second;
            if (nb < 0 || nb >= (int)nodes.size()) continue;
            addNeighbor(id, nb);
            // ensure neighbor node has at least l+1 levels before symmetric push
            if ((int)nodes[nb].links.size() <= l) nodes[nb].links.resize(l + 1);
            addNeighbor(nb, id);
        }
    }

    int addPoint(const float* v) {
        int id = (int)nodes.size();
        nodes.emplace_back();
        nodes[id].vec.assign(v, v+dim);
        int level = randomLevel();
        nodes[id].links.resize(level+1);

        if(enter_point<0){enter_point=id; return id;}

        int ep = enter_point;
        int entry_level = (int)nodes[enter_point].links.size()-1;
        for(int l=entry_level;l>level;l--) ep = greedySearch(ep, v, l);
        for(int l=std::min(level,entry_level);l>=0;l--){
            auto top = searchLayer(v, ep, l, HNSW_EF_CONSTRUCTION);
            connectNode(id, top, l);
            if(!top.empty()) ep = top[0].second;
        }
        if((int)nodes[enter_point].links.size()-1<level) enter_point=id;
        return id;
    }

    std::vector<std::pair<float,int>> searchKNN(const float* q, int k, int ef=HNSW_EF_SEARCH) const {
        if(nodes.empty()) return {};
        int ep = enter_point;
        int entry_level = (int)nodes[enter_point].links.size()-1;
        for(int l=entry_level;l>0;l--) ep=greedySearch(ep,q,l);
        auto top=searchLayer(q,ep,0,ef);
        if((int)top.size()>k) top.resize(k);
        return top;
    }
};

// ---------------- Parallel HNSW Wrapper ----------------
class HnswSolutionParallel {
public:
    int dim;
    std::vector<float> point_data;
    std::vector<int> point_ids;
    SimpleHNSW* hnsw = nullptr;

    ~HnswSolutionParallel(){delete hnsw;}

    float* point_ptr(int idx){ return point_data.data() + (size_t)idx*dim; }

    void build_from_memory(int d, std::vector<std::vector<float>> data){
        dim=d;
        int n=(int)data.size();
        point_ids.resize(n);
        point_data.assign(n*dim,0.f);
        for(int i=0;i<n;i++){
            point_ids[i]=i;
            float* dst=point_ptr(i);
            for(int j=0;j<dim;j++) dst[j]=data[i][j];
        }

        // Build HNSW by inserting points sequentially (正确但耗时)。
        // 优化点：提前 reserve 节点容量，避免 vector 频繁重分配，使用更快的距离函数
        delete hnsw;
        hnsw = new SimpleHNSW(d, HNSW_M, HNSW_MAX_LAYER);
        hnsw->nodes.reserve(n + 16);
        auto build_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; ++i) {
            hnsw->addPoint(point_ptr(i));
            if (DEBUG_TIMING && (i % 10000 == 0)) {
                std::cerr << "[Timing] inserted " << i << " / " << n << " points\n";
            }
        }
        auto build_end = std::chrono::high_resolution_clock::now();
        if (DEBUG_TIMING) {
            double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            std::cerr << "[Timing] build parallel-distance total: " << total_ms << " ms (threads=" << HNSW_BUILD_THREADS << ")\n";
        }
    }

    std::vector<std::pair<int,float>> search(const std::vector<float>& query, int k){
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<int,float>> out;
        if(!hnsw) return out;
        auto top=hnsw->searchKNN(query.data(),k,HNSW_EF_SEARCH);
        out.reserve(top.size());
        for(auto &p: top){
            int nid=p.second;
            float dist=p.first;
            int orig_id = (nid>=0 && nid<(int)point_ids.size())?point_ids[nid]:nid;
            out.push_back({orig_id, dist});
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (DEBUG_TIMING) {
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cerr << "[Timing] HnswSolutionParallel::search k=" << k << " took " << ms << " ms\n";
        }
        return out;
    }
};

// ---------------- global wrapper ----------------
static HnswSolutionParallel* g_impl = nullptr;

Solution::Solution(int num_centroid, int kmean_iter, int nprob)
    : num_centroid_(num_centroid), kmean_iter_(kmean_iter), nprob_(nprob) {}

void Solution::build(int d, const std::vector<float>& base){
    auto t0 = std::chrono::high_resolution_clock::now();
    int n=(int)base.size()/d;
    std::vector<std::vector<float>> data(n,std::vector<float>(d));
    for(int i=0;i<n;i++)
        for(int j=0;j<d;j++) data[i][j]=base[i*d+j];
    delete g_impl;
    g_impl=new HnswSolutionParallel();
    g_impl->build_from_memory(d,std::move(data));
    if (DEBUG_TIMING) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[Timing] Solution::build total: " << ms << " ms\n";
    }
}

void Solution::search(const std::vector<float>& query,int* res){
    if(!g_impl){
        for(int i=0;i<10;i++) res[i]=-1;
        return;
    }
    auto ans=g_impl->search(query,10);
    if (DEBUG_TIMING) {
        // search timing printed inside g_impl->search already; keep a small marker here
        std::cerr << "[Timing] Solution::search got " << ans.size() << " results\n";
    }
    int idx=0;
    for(;idx<(int)ans.size() && idx<10;idx++) res[idx]=ans[idx].first;
    for(;idx<10;idx++) res[idx]=-1;
}
