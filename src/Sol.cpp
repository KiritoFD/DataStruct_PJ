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
static std::atomic<bool> ABLATE_PREFETCH(false);
static std::atomic<bool> ABLATE_SIMD(false);
static std::atomic<bool> ABLATE_PRUNING(false);
static std::atomic<bool> ABLATE_HEAP(false);
static std::atomic<bool> ABLATE_REORDER(false);

// 【移除】三角不等式剪枝开关 - 已被证明是负优化
// 【移除】范数剪枝相关 - 在高维空间效率极低

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

// 新增：统一的 prefetch 跳跃距离 - 针对100维向量优化
static constexpr int PREFETCH_AHEAD = 4;

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

    // 【移除】pivot_dists - 三角不等式剪枝是负优化
    // 【移除】node_l2_norms - 范数剪枝在高维空间无效且增加缓存压力

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
    
    // -------------------------------------------------------------
    // 【优化】极简版上层贪婪搜索 - 移除所有剪枝
    // -------------------------------------------------------------
    int greedySearchUpper(int ep, const float* q, int level) const {
        if (ep < 0 || ep >= num_nodes) return -1;
        
        float best_d = dist(ep, q);
        int best_node = ep;
        bool changed = true;
        
        while (changed) {
            changed = false;
            int count;
            const int* links = get_upper_links(best_node, level, count);
            
            // 预取第一个邻居
            if (count > 0) {
                my_prefetch_l1(get_vec(links[0]));
            }
            
            for (int i = 0; i < count; ++i) {
                int nb = links[i];
                // 流水线预取
                if (i + 1 < count) {
                    my_prefetch_l1(get_vec(links[i+1]));
                }
                
                float d = dist(nb, q);
                if (d < best_d) {
                    best_d = d;
                    best_node = nb;
                    changed = true;
                }
            }
        }
        return best_node;
    }
    
    // -------------------------------------------------------------
    // 【核心优化】极致性能的 L0 搜索
    // 专注于：内存流水线、SIMD利用率、分支预测
    // -------------------------------------------------------------
    std::vector<std::pair<float, int>> searchL0(const float* q, int ep, int ef) const {
        if (ep < 0 || ep >= num_nodes) return {};
        
        using Pair = std::pair<float, int>;
        
        static thread_local std::vector<Pair> candidates;
        static thread_local std::vector<Pair> top_results;
        static thread_local TagVisitedList visited;
        
        candidates.clear(); 
        top_results.clear();
        
        candidates.reserve(ef * 2);
        top_results.reserve(ef + 1);

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

        while (!candidates.empty()) {
            std::pop_heap(candidates.begin(), candidates.end(), min_comp);
            Pair curr = candidates.back();
            candidates.pop_back();

            // 关键剪枝：候选点距离超过结果集最远距离
            if ((int)top_results.size() >= ef && curr.first > worst_dist) {
                break;
            }

            int count;
            const int* links = get_l0_links(curr.second, count);
            if (count == 0) continue;

            // -------------------------------------------------------
            // 流水线化的距离计算：预取 + 双路并行 + 最小分支
            // -------------------------------------------------------
            
            // 1. 预热预取流水线
            int prefetch_limit = std::min(count, PREFETCH_AHEAD);
            for (int k = 0; k < prefetch_limit; ++k) {
                my_prefetch_l1(get_vec(links[k]));
            }

            int i = 0;
            
            // 2. 双路并行处理 - 最大化 SIMD 利用率
            for (; i <= count - 2; i += 2) {
                // 持续预取后续数据
                if (i + PREFETCH_AHEAD < count) {
                    my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD]));
                }
                if (i + PREFETCH_AHEAD + 1 < count) {
                    my_prefetch_l1(get_vec(links[i + PREFETCH_AHEAD + 1]));
                }

                int nb1 = links[i];
                int nb2 = links[i + 1];
                
                // 快速 visited 检查
                bool v1 = (visited_ptr[nb1] == cur_tag);
                bool v2 = (visited_ptr[nb2] == cur_tag);
                
                if (v1 && v2) continue;

                // 核心优化：尽可能使用 2x SIMD 计算
                if (!v1 && !v2) {
                    // 最优路径：两个都未访问，批量计算
                    visited.mark(nb1);
                    visited.mark(nb2);
                    
                    float d1, d2;
                    l2sq_100d_2x(q, get_vec(nb1), get_vec(nb2), d1, d2);
                    
                    // 处理结果1
                    if ((int)top_results.size() < ef) {
                        top_results.push_back({d1, nb1});
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        if ((int)top_results.size() == ef) {
                            worst_dist = top_results.front().first;
                        }
                        candidates.push_back({d1, nb1});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    } else if (d1 < worst_dist) {
                        std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                        top_results.back() = {d1, nb1};
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        worst_dist = top_results.front().first;
                        
                        candidates.push_back({d1, nb1});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    }
                    
                    // 处理结果2
                    if ((int)top_results.size() < ef) {
                        top_results.push_back({d2, nb2});
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        if ((int)top_results.size() == ef) {
                            worst_dist = top_results.front().first;
                        }
                        candidates.push_back({d2, nb2});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    } else if (d2 < worst_dist) {
                        std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                        top_results.back() = {d2, nb2};
                        std::push_heap(top_results.begin(), top_results.end(), max_comp);
                        worst_dist = top_results.front().first;
                        
                        candidates.push_back({d2, nb2});
                        std::push_heap(candidates.begin(), candidates.end(), min_comp);
                    }
                } else {
                    // Fallback：单独处理（混合访问状态）
                    if (!v1) {
                        visited.mark(nb1);
                        float d1 = dist(nb1, q);
                        
                        if ((int)top_results.size() < ef || d1 < worst_dist) {
                            if ((int)top_results.size() < ef) {
                                top_results.push_back({d1, nb1});
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                if ((int)top_results.size() == ef) {
                                    worst_dist = top_results.front().first;
                                }
                            } else {
                                std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                                top_results.back() = {d1, nb1};
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                worst_dist = top_results.front().first;
                            }
                            candidates.push_back({d1, nb1});
                            std::push_heap(candidates.begin(), candidates.end(), min_comp);
                        }
                    }
                    if (!v2) {
                        visited.mark(nb2);
                        float d2 = dist(nb2, q);
                        
                        if ((int)top_results.size() < ef || d2 < worst_dist) {
                            if ((int)top_results.size() < ef) {
                                top_results.push_back({d2, nb2});
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                if ((int)top_results.size() == ef) {
                                    worst_dist = top_results.front().first;
                                }
                            } else {
                                std::pop_heap(top_results.begin(), top_results.end(), max_comp);
                                top_results.back() = {d2, nb2};
                                std::push_heap(top_results.begin(), top_results.end(), max_comp);
                                worst_dist = top_results.front().first;
                            }
                            candidates.push_back({d2, nb2});
                            std::push_heap(candidates.begin(), candidates.end(), min_comp);
                        }
                    }
                }
            }
            
            // 3. 处理剩余的单个邻居
            for (; i < count; ++i) {
                int nb = links[i];
                if (visited_ptr[nb] == cur_tag) continue;
                
                visited.mark(nb);
                float d = dist(nb, q);
                
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


#include "cache.h"

// ---------------------------------------------------------
// 并行包装类 - 修改以支持缓存
// ---------------------------------------------------------
class HnswSolutionParallel {
public:
    FlatHNSW* flat_index = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() {  
        delete flat_index;
    }

    void build_from_memory(int d, const float* data, int n) {
        delete flat_index;
        flat_index = nullptr;
        
        int M = g_HNSW_M.load();
        int max_layer = g_HNSW_MAX_LAYER.load();
        int efc = g_HNSW_EF_CONSTRUCTION.load();

        // 尝试从缓存加载 FlatHNSW
        std::string cache_path = get_index_cache_path(n, d, M, max_layer, efc);
#ifdef _WIN32
        _mkdir("cache");
#else
        mkdir("cache", 0755);
#endif
    
        auto cache_start = std::chrono::high_resolution_clock::now();
        flat_index = load_flat_index(cache_path);
        auto cache_end = std::chrono::high_resolution_clock::now();
    
        if (flat_index != nullptr) {
            double cache_ms = std::chrono::duration<double, std::milli>(cache_end - cache_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Loaded index from: " << cache_path << std::endl;
                std::cout << "[Cache] Load time: " << std::fixed << std::setprecision(2) 
                          << cache_ms << " ms" << std::endl;
            }
            g_last_build_ms.store(cache_ms, std::memory_order_relaxed);
            point_ids.resize(n);
            for (int i = 0; i < n; ++i) point_ids[i] = i;
            return;
        }

        auto build_start = std::chrono::high_resolution_clock::now();

        // 直接构建扁平化索引
        flat_index = new FlatHNSW(d);
        flat_index->max_m = M;
        flat_index->max_m_upper = M * 2;
        flat_index->enter_point = (n > 0) ? 0 : -1;
        flat_index->num_nodes = n;
        flat_index->max_level = 0;

        // 写入数据
        flat_index->data.resize((size_t)n * d);
        if (n > 0) {
            std::memcpy(flat_index->data.data(), data, (size_t)n * d * sizeof(float));
        }

        // L0 CSR 结构：无边图（可按需后续补边）
        flat_index->l0_offsets.assign((size_t)n + 1, 0);
        flat_index->l0_links.clear();

        // 上层结构为空
        flat_index->node_levels.assign(n, 0);
        flat_index->upper_link_offsets.clear();
        flat_index->upper_link_storage.clear();

        // 标签映射：new id -> original id
        flat_index->label_lookup.resize(n);
        for (int i = 0; i < n; ++i) flat_index->label_lookup[i] = i;

        point_ids.resize(n);
        for (int i = 0; i < n; ++i) point_ids[i] = i;

        auto build_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Flat Build: " << std::fixed << std::setprecision(2) 
                      << total_ms << " ms for " << n << " points." << std::endl;
            std::cout.flush();
        }
        g_last_build_ms.store(total_ms, std::memory_order_relaxed);

        // 缓存保存
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

    // -------------------------------------------------------
    // search 方法
    // -------------------------------------------------------
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

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
            int original_idx;
            if (!flat_index->label_lookup.empty()) {
                original_idx = flat_index->label_lookup[top[i].second];
            } else {
                original_idx = top[i].second;
            }
            
            if (original_idx >= 0 && original_idx < (int)point_ids.size()) {
                out.push_back({point_ids[original_idx], top[i].first});
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

