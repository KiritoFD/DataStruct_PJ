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
// 内存对齐工具
// ---------------------------------------------------------
static constexpr size_t ALIGN_SIZE = 32;

static inline void* aligned_alloc_32(size_t size) {
    if (size == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(size, ALIGN_SIZE);
#else
    void* ptr = nullptr;
    return (posix_memalign(&ptr, ALIGN_SIZE, size) == 0) ? ptr : nullptr;
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
    void resize(size_t n) { if (n != size_) { clear(); if (n > 0) { ptr = (float*)aligned_alloc_32(n * sizeof(float)); size_ = n; } } }
    void clear() { if (ptr) { aligned_free_32(ptr); ptr = nullptr; } size_ = 0; }
    size_t size() const { return size_; }
    float* data() { return ptr; }
    const float* data() const { return ptr; }
};

// ---------------------------------------------------------
// 全局配置
// ---------------------------------------------------------
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
static thread_local uint64_t tl_dist_counter = 0;
static std::atomic<double> g_last_build_ms{0.0};

// 消融标签
static std::atomic<bool> ABLATE_CSR(false);
static std::atomic<bool> ABLATE_PREFETCH(false);
static std::atomic<bool> ABLATE_SIMD(false);
static std::atomic<bool> ABLATE_PRUNING(false);
static std::atomic<bool> ABLATE_HEAP(false);
static std::atomic<bool> ABLATE_FLAT_INDEX(false);
static std::atomic<bool> ABLATE_REORDER(false);

// 优化配置
std::atomic<bool> ENABLE_POST_OPTIMIZATION(true);
std::atomic<int> POST_OPT_M(32);
std::atomic<float> PRUNING_ALPHA(1.0f);

static std::atomic<bool> ENABLE_RUNTIME_DIST_COUNTING(false);
static bool DEBUG_TIMING = true;

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
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    if (!g_thread_pool) g_thread_pool = new ThreadPool(HNSW_BUILD_THREADS);
    return g_thread_pool;
}

static inline void my_prefetch_l1(const void* ptr) {
#ifdef __GNUC__
    if (!ABLATE_PREFETCH.load(std::memory_order_relaxed))
        _mm_prefetch((const char*)ptr, _MM_HINT_T0);
#endif
}

// =========================================================
// Part 1: SimpleHNSW (动态构建结构)
// =========================================================
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

    SimpleHNSW(int d, int m = 16, int ml = 16) : dim(d), M(m), maxLayer(ml) {}
    ~SimpleHNSW() { for (auto p : nodes) delete p; }
    
    inline int size() const { return (int)nodes.size(); }
    inline const float* getVec(int id) const { return data_flat.data() + (size_t)id * dim; }
    inline float dist(int id, const float* q) const { return l2sq_100d(getVec(id), q); }
    inline float distNodes(int a, int b) const { return l2sq_100d(getVec(a), getVec(b)); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        return (int)(-std::log(ud(rng)) / std::log((float)M));
    }

    template<bool UseLock>
    int greedySearch(int ep, const float* q, int l) const {
        if (ep < 0 || ep >= size()) return -1;
        float curd = dist(ep, q);
        bool changed = true;
        while (changed) {
            changed = false;
            std::shared_lock<std::shared_mutex> lk;
            if constexpr (UseLock) lk = std::shared_lock(nodes[ep]->lock);
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
        static thread_local std::vector<Pair> top, cand;
        static thread_local VisitedList vis;
        
        top.clear(); cand.clear();
        vis.init(size()); vis.advance();
        
        float d0 = dist(ep, q);
        vis.mark(ep);
        top.push_back({d0, ep});
        cand.push_back({d0, ep});
        auto gt = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        std::push_heap(cand.begin(), cand.end(), gt);
        float lb = d0;
        
        while (!cand.empty()) {
            std::pop_heap(cand.begin(), cand.end(), gt);
            auto curr = cand.back(); cand.pop_back();
            if (curr.first > lb && (int)top.size() >= ef) break;
            
            std::shared_lock<std::shared_mutex> lk;
            if constexpr (UseLock) lk = std::shared_lock(nodes[curr.second]->lock);
            
            for (int nb : nodes[curr.second]->links[l]) {
                if (vis.isVisited(nb)) continue;
                vis.mark(nb);
                float d = dist(nb, q);
                if ((int)top.size() < ef || d < lb) {
                    auto it = std::upper_bound(top.begin(), top.end(), Pair{d, nb}, 
                        [](const Pair& a, const Pair& b) { return a.first < b.first; });
                    top.insert(it, {d, nb});
                    if ((int)top.size() > ef) top.pop_back();
                    lb = top.back().first;
                }
                cand.push_back({d, nb}); std::push_heap(cand.begin(), cand.end(), gt);
            }
        }
        return top;
    }

    void connectNodeHeuristic(int id, const std::vector<std::pair<float, int>>& candidates, int l, float alpha = 1.0f) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;

        std::vector<std::pair<float, int>> all;
        all.reserve(candidates.size() + m_max);
        for (const auto& p : candidates) all.push_back(p);
        {
            std::shared_lock lk(nodes[id]->lock);
            for (int nb : nodes[id]->links[l])
                if (nb >= 0 && nb < size()) all.push_back({distNodes(id, nb), nb});
        }

        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end(), [](auto& a, auto& b) { return a.second == b.second; }), all.end());

        std::vector<int> result;
        result.reserve(m_max);
        bool skip_pruning = ABLATE_PRUNING.load(std::memory_order_relaxed);
        
        for (const auto& c : all) {
            if ((int)result.size() >= m_max) break;
            if (c.second == id) continue;
            bool keep = true;
            if (!skip_pruning) {
                for (int sel : result)
                    if (distNodes(c.second, sel) < alpha * c.first) { keep = false; break; }
            }
            if (keep) result.push_back(c.second);
        }
        { std::unique_lock lk(nodes[id]->lock); nodes[id]->links[l] = std::move(result); }

        for (const auto& p : all) {
            int nb = p.second;
            if (nb < 0 || nb >= size() || nb == id) continue;
            bool in_result = false;
            { std::shared_lock lk(nodes[id]->lock); for (int r : nodes[id]->links[l]) if (r == nb) { in_result = true; break; } }
            if (!in_result) continue;

            std::vector<std::pair<float, int>> nb_all;
            { std::shared_lock lk(nodes[nb]->lock); for (int x : nodes[nb]->links[l]) if (x >= 0 && x < size()) nb_all.push_back({distNodes(nb, x), x}); }
            nb_all.push_back({distNodes(nb, id), id});
            std::sort(nb_all.begin(), nb_all.end());
            nb_all.erase(std::unique(nb_all.begin(), nb_all.end(), [](auto& a, auto& b) { return a.second == b.second; }), nb_all.end());

            std::vector<int> nb_result;
            nb_result.reserve(m_max);
            for (const auto& c : nb_all) {
                if ((int)nb_result.size() >= m_max) break;
                if (c.second == nb) continue;
                bool keep = true;
                if (!skip_pruning) {
                    for (int sel : nb_result) if (distNodes(c.second, sel) < alpha * c.first) { keep = false; break; }
                }
                if (keep) nb_result.push_back(c.second);
            }
            { std::unique_lock lk(nodes[nb]->lock); nodes[nb]->links[l] = std::move(nb_result); }
        }
    }

    void insertPointParallel(int id, int level) {
        int ep_curr;
        { std::shared_lock lk(global_mutex); ep_curr = enter_point; }
        
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
        { std::unique_lock lk(global_mutex); if (enter_point == -1 || level > (int)nodes[enter_point]->links.size() - 1) enter_point = id; }
    }
};

// =========================================================
// Part 2: 缓存系统
// =========================================================
#include "cache.h"

// =========================================================
// Part 3: FlatHNSW (只读查询结构)
// =========================================================
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

    FlatHNSW(int d) : dim(d), max_m(0), max_m_upper(0), enter_point(-1), num_nodes(0), max_level(0) {}
    
    inline const int* get_l0_links(int id, int& count) const {
        count = (int)(l0_offsets[id + 1] - l0_offsets[id]);
        return l0_links.data() + l0_offsets[id];
    }
    
    inline const int* get_upper_links(int id, int level, int& count) const {
        if (level <= 0 || level > node_levels[id]) { count = 0; return nullptr; }
        int off = upper_link_offsets[(size_t)id * max_level + level];
        if (off < 0) { count = 0; return nullptr; }
        count = upper_link_storage[off];
        return upper_link_storage.data() + off + 1;
    }
    
    inline const float* get_vec(int id) const { return data.data() + (size_t)id * dim; }
    inline float dist(int id, const float* q) const { return l2sq_100d(get_vec(id), q); }
    
    int greedySearchUpper(int ep, const float* q, int level) const {
        if (ep < 0 || ep >= num_nodes) return -1;
        float curd = dist(ep, q);
        bool changed = true;
        while (changed) {
            changed = false;
            int count; const int* links = get_upper_links(ep, level, count);
            for (int i = 0; i < count; ++i) {
                float nd = dist(links[i], q);
                if (nd < curd) { curd = nd; ep = links[i]; changed = true; }
            }
        }
        return ep;
    }
    
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        using Pair = std::pair<float, int>;
        static thread_local std::vector<Pair> top_candidates;
        static thread_local std::vector<Pair> search_queue;
        static thread_local TagVisitedList visited;
        
        top_candidates.clear(); search_queue.clear();
        top_candidates.reserve(ef + 1);
        search_queue.reserve(ef * 2);
        visited.init(num_nodes); visited.advance();
        
        float d0 = dist(ep, q);
        visited.mark(ep);
        top_candidates.push_back({d0, ep});
        search_queue.push_back({d0, ep});
        float lower_bound = d0;
        
        auto min_cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
        auto max_cmp = [](const Pair& a, const Pair& b) { return a.first < b.first; };
        
        while (!search_queue.empty()) {
            std::pop_heap(search_queue.begin(), search_queue.end(), min_cmp);
            Pair curr = search_queue.back(); search_queue.pop_back();
            if (curr.first > lower_bound && (int)top_candidates.size() >= ef) break;
            
            int count; const int* links = get_l0_links(curr.second, count);
            if (count > 0) my_prefetch_l1(get_vec(links[0]));
            
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                if (i + 2 < count) my_prefetch_l1(get_vec(links[i+2]));
                if (visited.isVisited(nb)) continue;
                visited.mark(nb);
                float d = dist(nb, q);
                
                if ((int)top_candidates.size() < ef || d < lower_bound) {
                    top_candidates.push_back({d, nb});
                    std::push_heap(top_candidates.begin(), top_candidates.end(), max_cmp);
                    if ((int)top_candidates.size() > ef) {
                        std::pop_heap(top_candidates.begin(), top_candidates.end(), max_cmp);
                        top_candidates.pop_back();
                    }
                    lower_bound = top_candidates.front().first;
                    search_queue.push_back({d, nb});
                    std::push_heap(search_queue.begin(), search_queue.end(), min_cmp);
                }
            }
        }
        std::sort_heap(top_candidates.begin(), top_candidates.end(), max_cmp);
        return top_candidates;
    }
};

// ---------------------------------------------------------
// 优化后的启发式剪枝 (RNG) - 增加最小边数保证
// ---------------------------------------------------------
static void heuristic_prune(
    const float* curr_vec,
    const std::vector<std::pair<float, int>>& candidates,
    std::vector<int>& result,
    int max_m,
    const float* all_data,
    int dim,
    float alpha = 1.0f,
    int min_edges = 2  // 新增：最小边数保证
) {
    result.clear();
    result.reserve(max_m);
    if (candidates.empty()) return;
    
    // Phase 1: RNG pruning
    std::vector<bool> selected(candidates.size(), false);
    selected[0] = true;
    result.push_back(candidates[0].second);
    
    int size = (int)candidates.size();
    for (int i = 1; i < size && (int)result.size() < max_m; ++i) {
        const auto& cand = candidates[i];
        int cand_id = cand.second;
        float dist_to_curr = cand.first;
        const float* cand_vec = all_data + (size_t)cand_id * dim;
        bool keep = true;
        for (int existing_id : result) {
            const float* exist_vec = all_data + (size_t)existing_id * dim;
            float dist_to_exist = l2sq_100d(cand_vec, exist_vec);
            if (dist_to_exist < alpha * dist_to_curr) { keep = false; break; }
        }
        if (keep) {
            result.push_back(cand_id);
            selected[i] = true;
        }
    }
    
    // Phase 2: Connectivity enforcement - ensure minimum edges
    if ((int)result.size() < min_edges) {
        for (int i = 0; i < size && (int)result.size() < min_edges; ++i) {
            if (!selected[i]) {
                result.push_back(candidates[i].second);
                selected[i] = true;
            }
        }
    }
}

// =========================================================
// Part 4: 图重排 + 转换函数
// =========================================================


// DFS 重排：模拟贪婪路径，优化 Cache 布局
static void generate_dfs_reordering(SimpleHNSW* src, std::vector<int>& old2new, std::vector<int>& new2old) {
    int N = src->size();
    old2new.assign(N, -1);
    new2old.resize(N);
    int current_id = 0;
    std::vector<int> stack;
    stack.reserve(N);
    std::vector<bool> visited(N, false);

    if (src->enter_point >= 0 && src->enter_point < N) {
        stack.push_back(src->enter_point);
        visited[src->enter_point] = true;
    }
    for (int root = 0; root < N; ++root) {
        if (old2new[root] != -1) continue;
        if (stack.empty()) { stack.push_back(root); visited[root] = true; }
        while (!stack.empty()) {
            int u = stack.back(); stack.pop_back();
            if (old2new[u] == -1) { old2new[u] = current_id; new2old[current_id++] = u; }
            std::shared_lock<std::shared_mutex> lk(src->nodes[u]->lock);
            const auto& links = src->nodes[u]->links[0];
            for (auto it = links.rbegin(); it != links.rend(); ++it) {
                int v = *it;
                if (v >= 0 && v < N && !visited[v]) { visited[v] = true; stack.push_back(v); }
            }
        }
    }
}

struct Candidate {
    int id;
    float dist;
    bool operator<(const Candidate& o) const { return dist < o.dist; }
};

[[maybe_unused]] static void reorder_ids_by_coordinate(SimpleHNSW* src, std::vector<int>& old2new, std::vector<int>& new2old, bool debug) {
    int N = src->size();
    old2new.assign(N, -1);
    new2old.resize(N);
    struct NodeInfo { int id; float key; };
    std::vector<NodeInfo> nodes;
    nodes.reserve(N);
    for (int i = 0; i < N; ++i) {
        const float* vec = src->getVec(i);
        float key = 0.f;
        int dims = std::min(src->dim, 4);
        for (int d = 0; d < dims; ++d) key += vec[d];
        nodes.push_back({i, key});
    }
    std::sort(nodes.begin(), nodes.end(), [](const NodeInfo& a, const NodeInfo& b) { return a.key < b.key; });
    for (int rank = 0; rank < N; ++rank) {
        int orig = nodes[rank].id;
        new2old[rank] = orig;
        old2new[orig] = rank;
    }
    if (debug) std::cout << "[Reorder] Coordinate mapping done for " << N << " nodes" << std::endl;
}

[[maybe_unused]] static void rewire_and_prune(SimpleHNSW* src, int target_M, float prune_alpha = 1.0f) {
    int N = src->size();
    if (N <= 0 || target_M <= 0) return;
    std::vector<std::vector<int>> reverse_links(N);
    for (int i = 0; i < N; ++i) {
        std::shared_lock<std::shared_mutex> lk(src->nodes[i]->lock);
        for (int nb : src->nodes[i]->links[0]) {
            if (nb >= 0 && nb < N) reverse_links[nb].push_back(i);
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < N; ++i) {
        std::vector<int> pool;
        pool.reserve(target_M * 4);
        {
            std::shared_lock lk(src->nodes[i]->lock);
            pool = src->nodes[i]->links[0];
        }
        pool.insert(pool.end(), reverse_links[i].begin(), reverse_links[i].end());
        std::sort(pool.begin(), pool.end());
        pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
        std::vector<Candidate> candidates;
        candidates.reserve(pool.size());
        const float* vec_i = src->getVec(i);
        for (int nb : pool) {
            if (nb == i || nb < 0 || nb >= N) continue;
            float d = l2sq_100d(vec_i, src->getVec(nb));
            candidates.push_back({nb, d});
        }
        std::sort(candidates.begin(), candidates.end());
        std::vector<int> result;
        result.reserve(target_M);
        if (!candidates.empty()) result.push_back(candidates[0].id);
        for (size_t c = 1; c < candidates.size() && result.size() < (size_t)target_M; ++c) {
            const auto& curr = candidates[c];
            bool keep = true;
            const float* vec_c = src->getVec(curr.id);
            for (int existing_nb : result) {
                float dist_between = l2sq_100d(vec_c, src->getVec(existing_nb));
                if (dist_between < prune_alpha * curr.dist) { keep = false; break; }
            }
            if (keep) result.push_back(curr.id);
        }
        std::unique_lock<std::shared_mutex> lk(src->nodes[i]->lock);
        src->nodes[i]->links[0] = std::move(result);
    }
}

static FlatHNSW* convert_to_flat(SimpleHNSW* src, bool debug) {
    // 调整后的参数：更保守的剪枝策略
    constexpr int TARGET_M_L0 = 28;        // 从 24 提升到 28
    constexpr int TARGET_M_UPPER = 20;     // 从 16 提升到 20
    constexpr float PRUNE_ALPHA = 1.25f;   // 从 1.0 提升到 1.25（松弛RNG）
    constexpr int MIN_EDGES = 3;           // 最小边数保证

    FlatHNSW* flat = new FlatHNSW(src->dim);
    int N = src->size();
    flat->num_nodes = N;
    flat->max_m = TARGET_M_L0;
    flat->max_m_upper = TARGET_M_UPPER;

    std::vector<int> old2new, new2old;
    bool do_reorder = !ABLATE_REORDER.load(std::memory_order_relaxed);
    if (do_reorder) {
        generate_dfs_reordering(src, old2new, new2old);
        flat->enter_point = (src->enter_point == -1) ? -1 : old2new[src->enter_point];
        flat->label_lookup = new2old;
    } else {
        flat->enter_point = src->enter_point;
        old2new.resize(N); new2old.resize(N);
        for (int i = 0; i < N; ++i) { old2new[i] = i; new2old[i] = i; }
    }

    flat->data.resize((size_t)N * flat->dim);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 4096)
#endif
    for (int nid = 0; nid < N; ++nid) {
        std::memcpy(flat->data.data() + (size_t)nid * flat->dim,
                    src->getVec(new2old[nid]), flat->dim * sizeof(float));
    }

    // 构建反向链接
    std::vector<std::vector<int>> reverse_links(N);
    for (int oid = 0; oid < N; ++oid) {
        std::shared_lock<std::shared_mutex> lk(src->nodes[oid]->lock);
        for (int nb : src->nodes[oid]->links[0]) if (nb >= 0 && nb < N) reverse_links[nb].push_back(oid);
    }

    flat->l0_offsets.resize(N + 1);
    std::vector<std::vector<int>> optimized_l0(N);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int nid = 0; nid < N; ++nid) {
        int oid = new2old[nid];
        const float* curr_vec = flat->data.data() + (size_t)nid * flat->dim;

        // Phase 1: 收集1-hop邻居（正向 + 反向）
        std::vector<int> l1_pool;
        l1_pool.reserve(128);
        {
            std::shared_lock<std::shared_mutex> lk(src->nodes[oid]->lock);
            l1_pool.insert(l1_pool.end(), src->nodes[oid]->links[0].begin(), src->nodes[oid]->links[0].end());
        }
        // 反向邻居是关键的"回溯路径"
        l1_pool.insert(l1_pool.end(), reverse_links[oid].begin(), reverse_links[oid].end());
        
        // 去重
        std::sort(l1_pool.begin(), l1_pool.end());
        l1_pool.erase(std::unique(l1_pool.begin(), l1_pool.end()), l1_pool.end());

        // Phase 2: 计算距离并排序1-hop
        std::vector<std::pair<float, int>> sorted_l1;
        sorted_l1.reserve(l1_pool.size());
        for (int nb_oid : l1_pool) {
            if (nb_oid == oid) continue;
            int nb_nid = old2new[nb_oid];
            if (nb_nid < 0) continue;
            float d = l2sq_100d(curr_vec, flat->data.data() + (size_t)nb_nid * flat->dim);
            sorted_l1.push_back({d, nb_nid});
        }
        std::sort(sorted_l1.begin(), sorted_l1.end());

        // Phase 3: 智能2-hop扩展 - 只对最近的5个邻居扩展
        std::vector<std::pair<float, int>> final_candidates;
        final_candidates.reserve(sorted_l1.size() + 50);
        
        // 先添加所有1-hop邻居
        for (const auto& p : sorted_l1) {
            final_candidates.push_back(p);
        }

        // 限制2-hop扩展：只对最近的5个邻居
        constexpr int TWO_HOP_EXPAND_LIMIT = 5;
        constexpr int TWO_HOP_PER_NEIGHBOR = 8;
        
        int expand_count = std::min((int)sorted_l1.size(), TWO_HOP_EXPAND_LIMIT);
        for (int i = 0; i < expand_count; ++i) {
            // sorted_l1[i].second 是 new_id，需要转回 old_id
            int neighbor_nid = sorted_l1[i].second;
            int neighbor_oid = new2old[neighbor_nid];
            float dist_to_neighbor = sorted_l1[i].first;
            
            std::shared_lock<std::shared_mutex> lk(src->nodes[neighbor_oid]->lock);
            const auto& nn_links = src->nodes[neighbor_oid]->links[0];
            
            int nn_count = 0;
            for (int nn_oid : nn_links) {
                if (nn_count++ >= TWO_HOP_PER_NEIGHBOR) break;
                if (nn_oid == oid) continue;
                int nn_nid = old2new[nn_oid];
                if (nn_nid < 0) continue;
                
                float d = l2sq_100d(curr_vec, flat->data.data() + (size_t)nn_nid * flat->dim);
                
                // 关键：只有当2-hop距离小于到中间节点的距离时才加入（构成更优三角边）
                if (d < dist_to_neighbor * 1.5f) {
                    final_candidates.push_back({d, nn_nid});
                }
            }
        }

        // Phase 4: 排序、去重、剪枝
        std::sort(final_candidates.begin(), final_candidates.end());
        
        // 去重（基于ID）
        std::vector<std::pair<float, int>> unique_candidates;
        unique_candidates.reserve(final_candidates.size());
        int last_id = -1;
        for (const auto& c : final_candidates) {
            if (c.second != last_id && c.second != nid) {
                unique_candidates.push_back(c);
                last_id = c.second;
            }
        }
        
        // 使用松弛RNG剪枝，带最小边数保证
        heuristic_prune(curr_vec, unique_candidates, optimized_l0[nid], 
                       TARGET_M_L0, flat->data.data(), flat->dim, 
                       PRUNE_ALPHA, MIN_EDGES);
    }

    uint64_t off = 0;
    for (const auto& v : optimized_l0) off += v.size();
    flat->l0_links.resize(off);
    off = 0;
    for (int i = 0; i < N; ++i) {
        flat->l0_offsets[i] = off;
        if (!optimized_l0[i].empty()) {
            std::memcpy(flat->l0_links.data() + off, optimized_l0[i].data(), optimized_l0[i].size() * sizeof(int));
            off += optimized_l0[i].size();
        }
    }
    flat->l0_offsets[N] = off;

    // ...existing code for upper layers...
    flat->max_level = 0;
    for (int i = 0; i < N; ++i) flat->max_level = std::max(flat->max_level, (int)src->nodes[i]->links.size() - 1);
    flat->max_level = std::min(flat->max_level, 100);
    flat->node_levels.resize(N);
    flat->upper_link_offsets.assign((size_t)N * (flat->max_level + 1), -1);

    for (int nid = 0; nid < N; ++nid) {
        int oid = new2old[nid];
        int lv = (int)src->nodes[oid]->links.size() - 1;
        flat->node_levels[nid] = lv;
        for (int l = 1; l <= lv && l <= flat->max_level; ++l) {
            const auto& src_links = src->nodes[oid]->links[l];
            std::vector<std::pair<float, int>> cand;
            const float* vec_u = flat->data.data() + (size_t)nid * flat->dim;
            for (int ob : src_links) {
                int nb = old2new[ob];
                if (nb < 0) continue;
                cand.push_back({l2sq_100d(vec_u, flat->data.data() + (size_t)nb * flat->dim), nb});
            }
            std::sort(cand.begin(), cand.end());
            std::vector<int> res;
            heuristic_prune(vec_u, cand, res, TARGET_M_UPPER, flat->data.data(), flat->dim, 1.15f, 2);
            int idx = (int)flat->upper_link_storage.size();
            flat->upper_link_offsets[(size_t)nid * (flat->max_level + 1) + l] = idx;
            flat->upper_link_storage.push_back((int)res.size());
            for (int nb : res) flat->upper_link_storage.push_back(nb);
        }
    }

    if (debug) std::cout << "[Convert] Relaxed RNG: M=" << TARGET_M_L0 << ", Alpha=" << PRUNE_ALPHA << std::endl;
    return flat;
}

// =========================================================
// Part 5: 主控类
// =========================================================
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    FlatHNSW* flat_index = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { delete hnsw; delete flat_index; }

    void build_from_memory(int d, const float* data, int n) {
        delete flat_index; flat_index = nullptr;
        delete hnsw; hnsw = nullptr;
        
        int M = g_HNSW_M.load(), ml = g_HNSW_MAX_LAYER.load(), efc = g_HNSW_EF_CONSTRUCTION.load();
        std::string cache_path = get_index_cache_path(n, d, M, ml, efc);
        
        #ifdef _WIN32
        _mkdir("cache");
        #else
        mkdir("cache", 0755);
        #endif
        
        point_ids.resize(n);
        for (int i = 0; i < n; ++i) point_ids[i] = i;
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Phase 1: 加载或构建原始图
        auto t0 = std::chrono::high_resolution_clock::now();
        SimpleHNSW* raw_hnsw = load_simple_index(cache_path, data, d, n, DEBUG_TIMING);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (raw_hnsw) {
            if (DEBUG_TIMING) std::cout << "[Phase1-Cache] " << std::chrono::duration<double,std::milli>(t1-t0).count() << " ms" << std::endl;
        } else {
            if (DEBUG_TIMING) std::cout << "[Phase1-Build] Starting..." << std::endl;

            auto build_start = std::chrono::high_resolution_clock::now();

            raw_hnsw = new SimpleHNSW(d, M, ml);
            raw_hnsw->data_flat.resize((size_t)n * d);
            std::memcpy(raw_hnsw->data_flat.data(), data, (size_t)n * d * sizeof(float));

            std::vector<int> levels(n);
            for (int i = 0; i < n; ++i) {
                levels[i] = std::min(raw_hnsw->randomLevel(), ml);
                raw_hnsw->nodes.push_back(new HNSWNode(levels[i], M));
            }
            if (n > 0) raw_hnsw->enter_point = 0;

            ThreadPool* pool = getThreadPool();
            std::atomic<int> processed(1);
            int chunk_size = 1000;

            // Enqueue work
            for (int i = 1; i < n; i += chunk_size) {
                int end = std::min(i + chunk_size, n);
                pool->enqueue([raw_hnsw, i, end, &levels, &processed]() {
                    for (int j = i; j < end; ++j) raw_hnsw->insertPointParallel(j, levels[j]);
                    processed.fetch_add(end - i);
                });
            }

            // New: progress reporter thread (only during building)
            std::thread progress_thread([&processed, n, build_start]() {
                int last_reported = 0;
                int step = std::max(50000, n / 10);
                while (processed.load(std::memory_order_acquire) < n) {
                    int curr = processed.load(std::memory_order_acquire);
                    if (curr - last_reported >= step) {
                        double pct = 100.0 * curr / n;
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed_ms = std::chrono::duration<double, std::milli>(now - build_start).count();
                        if (DEBUG_TIMING) {
                            std::cout << "[Progress] " << curr << "/" << n 
                                      << " (" << std::fixed << std::setprecision(1) << pct << "%) "
                                      << "Elapsed: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms"
                                      << std::endl;
                        }
                        last_reported = curr;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            });

            // Wait for build complete
            while (processed.load(std::memory_order_acquire) < n) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (progress_thread.joinable()) progress_thread.join();

            auto build_end = std::chrono::high_resolution_clock::now();
            if (DEBUG_TIMING) std::cout << "[Phase1-Build] " << std::chrono::duration<double,std::milli>(build_end-build_start).count() << " ms" << std::endl;

            // Save to cache
            auto save_start = std::chrono::high_resolution_clock::now();
            save_simple_index(raw_hnsw, cache_path, DEBUG_TIMING);
            auto save_end = std::chrono::high_resolution_clock::now();
            if (DEBUG_TIMING) std::cout << "[Phase1-Save] " << std::chrono::duration<double,std::milli>(save_end-save_start).count() << " ms" << std::endl;
        }

        // Phase 2: 后处理优化 (unchanged)
        if (ABLATE_CSR.load() || ABLATE_FLAT_INDEX.load()) {
            hnsw = raw_hnsw;
            if (DEBUG_TIMING) std::cout << "[Phase2] Ablation - using SimpleHNSW" << std::endl;
        } else {
            auto t2 = std::chrono::high_resolution_clock::now();
            flat_index = convert_to_flat(raw_hnsw, DEBUG_TIMING);
            auto t3 = std::chrono::high_resolution_clock::now();
            if (DEBUG_TIMING) std::cout << "[Phase2-Convert] " << std::chrono::duration<double,std::milli>(t3-t2).count() << " ms" << std::endl;
            delete raw_hnsw;
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double,std::milli>(total_end-total_start).count();
        g_last_build_ms.store(total_ms, std::memory_order_relaxed);
        if (DEBUG_TIMING) std::cout << "[Total] " << total_ms << " ms" << std::endl;
    }

    struct SearchDistanceCountingGuard {
        bool prev;
        SearchDistanceCountingGuard() : prev(ENABLE_RUNTIME_DIST_COUNTING.exchange(true)) {}
        ~SearchDistanceCountingGuard() { ENABLE_RUNTIME_DIST_COUNTING.store(prev); }
    };
    
    struct SearchDistanceStats {
        bool flushed = false;
        void flush() {
            if (flushed) return;
            flushed = true;
            uint64_t cnt = tl_dist_counter;
            g_total_dist_count.fetch_add(cnt, std::memory_order_relaxed);
            g_last_query_dist.store(cnt, std::memory_order_relaxed);
            g_total_query_count.fetch_add(1, std::memory_order_relaxed);
            tl_dist_counter = 0;
        }
        ~SearchDistanceStats() { flush(); }
    };

    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        SearchDistanceCountingGuard guard;
        SearchDistanceStats stats;
        if (hnsw) {
            int ep = hnsw->enter_point;
            if (ep < 0) return {};
            int curr = ep;
            for (int l = std::min((int)hnsw->nodes[ep]->links.size()-1, 4); l > 0; l--) 
                curr = hnsw->greedySearch<true>(curr, query.data(), l);
            auto top = hnsw->searchLayer<true>(query.data(), curr, 0, g_HNSW_EF_SEARCH.load());
            
            std::vector<std::pair<int, float>> out;
            for (int i = 0; i < std::min(k, (int)top.size()); ++i) 
                out.push_back({point_ids[top[i].second], top[i].first});
            return out;
        }
        if (!flat_index || flat_index->enter_point < 0) return {};
        
        int ep = flat_index->enter_point;
        int curr = ep;
        for (int l = std::min(flat_index->node_levels[ep], 4); l > 0; l = (l>1) ? l-2 : l-1)
            curr = flat_index->greedySearchUpper(curr, query.data(), l);
        
        auto top = flat_index->searchL0(query.data(), curr, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int, float>> out;
        for (int i = 0; i < std::min(k, (int)top.size()); ++i) {
            int iid = top[i].second;
            int oid = flat_index->label_lookup.empty() ? iid : flat_index->label_lookup[iid];
            out.push_back({point_ids[oid], top[i].first});
        }
        return out;
    }
};

// =========================================================
// Part 6: 对外接口
// =========================================================
static HnswSolutionParallel* g_impl = nullptr;

void build_hnsw(int d, const std::vector<float>& base) {
    delete g_impl;
    g_impl = new HnswSolutionParallel();
    g_impl->build_from_memory(d, base.data(), (int)base.size() / d);
}

std::vector<std::pair<int, float>> search_hnsw(const std::vector<float>& query, int k) {
    return g_impl ? g_impl->search(query, k) : std::vector<std::pair<int, float>>{};
}

Solution::Solution() : k_(10) {}
void Solution::build(int d, const std::vector<float>& base) { build_hnsw(d, base); }
void Solution::search(const std::vector<float>& query, int* result) {
    std::fill(result, result + k_, -1);
    auto res = search_hnsw(query, k_);
    for (size_t i = 0; i < res.size() && i < (size_t)k_; ++i) result[i] = res[i].first;
}

#include "extern.h"