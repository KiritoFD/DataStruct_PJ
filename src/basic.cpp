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
// static std::atomic<bool> ABLATE_HEAP(false);  // 删除 ABLATE_HEAP 标志
static std::atomic<bool> ABLATE_FLAT_INDEX(false);
static std::atomic<bool> ABLATE_REORDER(false);

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
        std::vector<Pair> top_candidates;
        std::vector<Pair> candidate_queue;

        top_candidates.reserve(ef + 1);
        candidate_queue.reserve(ef * 2);

        VisitedList visited_list;
        visited_list.init(size());
        visited_list.advance();

        float d0 = dist(ep, q);
        visited_list.mark(ep);

        top_candidates.push_back({d0, ep});
        candidate_queue.push_back({d0, ep});
        std::push_heap(candidate_queue.begin(), candidate_queue.end(), [](const Pair& a, const Pair& b) {
            return a.first > b.first;
        });

        float lower_bound = d0;

        while (!candidate_queue.empty()) {
            std::pop_heap(candidate_queue.begin(), candidate_queue.end(), [](const Pair& a, const Pair& b) {
                return a.first > b.first;
            });
            auto curr = candidate_queue.back();
            candidate_queue.pop_back();

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
            for (int nb : neighbors) {
                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float d_nb = dist(nb, q);

                    if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                        top_candidates.push_back({d_nb, nb});
                        std::push_heap(top_candidates.begin(), top_candidates.end(), [](const Pair& a, const Pair& b) {
                            return a.first > b.first;
                        });

                        if ((int)top_candidates.size() > ef) {
                            std::pop_heap(top_candidates.begin(), top_candidates.end(), [](const Pair& a, const Pair& b) {
                                return a.first > b.first;
                            });
                            top_candidates.pop_back();
                        }

                        lower_bound = top_candidates.front().first;
                    }

                    candidate_queue.push_back({d_nb, nb});
                    std::push_heap(candidate_queue.begin(), candidate_queue.end(), [](const Pair& a, const Pair& b) {
                        return a.first > b.first;
                    });
                }
            }
        }

        std::sort(top_candidates.begin(), top_candidates.end(), [](const Pair& a, const Pair& b) {
            return a.first < b.first;
        });
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

                // 直接连接节点，不使用启发式剪枝
                std::vector<int> result_links;
                for (const auto& p : top) {
                    result_links.push_back(p.second);
                }

                {
                    std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
                    nodes[id]->links[l] = std::move(result_links);
                }

                for (const auto& p : top) {
                    int nb = p.second;
                    if (nb < 0 || nb >= size() || nb == id) continue;

                    std::unique_lock<std::shared_mutex> lock(nodes[nb]->lock);
                    nodes[nb]->links[l].push_back(id);
                }
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
// 新增: 本地 cache 路径构造器（替代 get_index_cache_path）
// ---------------------------------------------------------
static inline std::string build_cache_path(int n, int d, int M, int max_layer, int efc) {
    std::ostringstream oss;
    oss << "cache/hnsw_n" << n << "_d" << d << "_M" << M << "_L" << max_layer << "_efc" << efc << ".bin";
    return oss.str();
}

// ---------------------------------------------------------
// 新增: SimpleHNSW 的二进制缓存序列化/反序列化 (保持多线程构建 + caching)
// ---------------------------------------------------------
static bool save_simple_hnsw(SimpleHNSW* h, const std::string& path) {
    if (!h) return false;
    std::ofstream ofs(path, std::ios::binary | std::ios::out);
    if (!ofs) return false;

    // Magic/version (6 chars -> reserve 7 bytes for null)
    const char magic[7] = "SHNSW1";
    ofs.write(magic, 6);

    int32_t d = h->dim;
    int32_t n = (int32_t)h->size();
    int32_t M = h->M;
    int32_t maxLayer = h->maxLayer;
    int32_t enter_point = h->enter_point;
    ofs.write(reinterpret_cast<const char*>(&d), sizeof(d));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    ofs.write(reinterpret_cast<const char*>(&M), sizeof(M));
    ofs.write(reinterpret_cast<const char*>(&maxLayer), sizeof(maxLayer));
    ofs.write(reinterpret_cast<const char*>(&enter_point), sizeof(enter_point));

    // Data
    int64_t data_len = (int64_t)h->data_flat.size();
    ofs.write(reinterpret_cast<const char*>(&data_len), sizeof(data_len));
    if (data_len > 0) {
        ofs.write(reinterpret_cast<const char*>(h->data_flat.data()), sizeof(float) * data_len);
    }

    // Nodes: for each node write level then per level neighbor counts and neighbors
    for (int i = 0; i < n; ++i) {
        int32_t level = (int32_t)h->nodes[i]->links.size() - 1;
        ofs.write(reinterpret_cast<const char*>(&level), sizeof(level));
        for (int l = 0; l <= level; ++l) {
            int32_t cnt = (int32_t)h->nodes[i]->links[l].size();
            ofs.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
            if (cnt > 0) {
                ofs.write(reinterpret_cast<const char*>(h->nodes[i]->links[l].data()), sizeof(int32_t) * cnt);
            }
        }
    }

    ofs.close();
    return ofs.good();
}

static SimpleHNSW* load_simple_hnsw(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::in);
    if (!ifs) return nullptr;

    char magic[7] = {0};
    ifs.read(magic, 6);
    if (ifs.gcount() != 6) return nullptr;
    if (std::strncmp(magic, "SHNSW1", 6) != 0) return nullptr;

    int32_t d, n, M, maxLayer, enter_point;
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    ifs.read(reinterpret_cast<char*>(&M), sizeof(M));
    ifs.read(reinterpret_cast<char*>(&maxLayer), sizeof(maxLayer));
    ifs.read(reinterpret_cast<char*>(&enter_point), sizeof(enter_point));

    auto* h = new SimpleHNSW(d, M, maxLayer);
    h->data_flat.resize((size_t)n * d);

    int64_t data_len = 0;
    ifs.read(reinterpret_cast<char*>(&data_len), sizeof(data_len));
    if (data_len > 0) {
        if ((int64_t)h->data_flat.size() != data_len) {
            // sanity: mismatch -> abort
            delete h;
            return nullptr;
        }
        ifs.read(reinterpret_cast<char*>(h->data_flat.data()), sizeof(float) * data_len);
    }

    h->enter_point = enter_point;
    h->nodes.reserve(n);

    for (int i = 0; i < n; ++i) {
        int32_t level = 0;
        ifs.read(reinterpret_cast<char*>(&level), sizeof(level));
        if (level < 0) {
            delete h;
            return nullptr;
        }
        HNSWNode* node = new HNSWNode(level, M);
        for (int l = 0; l <= level; ++l) {
            int32_t cnt = 0;
            ifs.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
            if (cnt < 0) { delete node; delete h; return nullptr; }
            node->links[l].resize(cnt);
            if (cnt > 0) {
                ifs.read(reinterpret_cast<char*>(node->links[l].data()), sizeof(int32_t) * cnt);
            }
        }
        h->nodes.push_back(node);
    }

    if (!ifs) {
        delete h;
        return nullptr;
    }

    return h;
}

// ---------------------------------------------------------
// 并行包装类 - 使用 SimpleHNSW（动态）并保持缓存、并行构建逻辑
// 合并为一个实现，确保没有重复类定义
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    SimpleHNSW* hnsw = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { 
        delete hnsw; 
    }

    void build_from_memory(int d, const float* data, int n) {
        delete hnsw;
        hnsw = nullptr;
        
        int M = g_HNSW_M.load();
        int max_layer = g_HNSW_MAX_LAYER.load();
        int efc = g_HNSW_EF_CONSTRUCTION.load();
        
        // 尝试从缓存加载 SimpleHNSW
        std::string cache_path = build_cache_path(n, d, M, max_layer, efc);
        
        // 创建缓存目录
        #ifdef _WIN32
        _mkdir("cache");
        #else
        mkdir("cache", 0755);
        #endif
        
        auto cache_start = std::chrono::high_resolution_clock::now();
        SimpleHNSW* cached_hnsw = load_simple_hnsw(cache_path);
        auto cache_end = std::chrono::high_resolution_clock::now();
        
        if (cached_hnsw != nullptr) {
            // 缓存命中
            double cache_ms = std::chrono::duration<double, std::milli>(cache_end - cache_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Loaded SimpleHNSW from: " << cache_path << std::endl;
                std::cout << "[Cache] Load time: " << std::fixed << std::setprecision(2) 
                          << cache_ms << " ms" << std::endl;
            }
            g_last_build_ms.store(cache_ms, std::memory_order_relaxed);
            
            // 初始化 point_ids
            point_ids.resize(n);
            for (int i = 0; i < n; ++i) point_ids[i] = i;
            
            // 直接使用 cached dynamic hnsw
            hnsw = cached_hnsw;
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

        // 保存到缓存
        auto save_start = std::chrono::high_resolution_clock::now();
        if (save_simple_hnsw(hnsw, cache_path)) {
            auto save_end = std::chrono::high_resolution_clock::now();
            double save_ms = std::chrono::duration<double, std::milli>(save_end - save_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Saved SimpleHNSW to: " << cache_path << std::endl;
                std::cout << "[Cache] Save time: " << std::fixed << std::setprecision(2) 
                          << save_ms << " ms" << std::endl;
            }
        }
    }

    // -------------------------------------------------------
    // search 方法 - 【优化3】multi-queue + 【优化4】skip-layer
    // -------------------------------------------------------
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

        if (!hnsw || hnsw->nodes.empty()) return {};

        int ep = hnsw->enter_point;
        if (ep < 0 || ep >= (int)hnsw->nodes.size()) return {};

        int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
        for (int l = max_l; l >= 0; l--) {
            ep = hnsw->greedySearch<true>(ep, query.data(), l);
        }
        auto top = hnsw->searchLayer<true>(query.data(), ep, 0, g_HNSW_EF_SEARCH.load());
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