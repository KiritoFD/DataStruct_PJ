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
static std::atomic<bool> ABLATE_ADAPTIVE_EP(false);  // 新增：自适应起点选择消融开关

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

// 新增：统一的 prefetch 跳跃距离 - 针对100维向量优化
static constexpr int PREFETCH_AHEAD = 7;

// [新增] 汉明距离阈值 (可调参数，建议范围 28-36)
static std::atomic<int> HAMMING_THRESHOLD{41};

// ---------------------------------------------------------
// 扁平化 HNSW 索引 (Unified Build & Search) - CSR 格式
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

    // [新增] 二进制指纹：每个向量 2 个 uint64_t (128 bits)
    std::vector<uint64_t> signatures;

    // =========================================================
    // [新增] 聚类起点选择相关成员
    // =========================================================
    std::vector<int> entry_candidates;           // 候选起点集（节点ID列表）
    AlignedFloatArray entry_candidates_data;     // 候选起点向量数据（紧凑存储）
    int num_entry_candidates = 0;                // 候选起点数量

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
        int offset = upper_link_offsets[(size_t)id * (max_level + 1) + level];
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
    
    // [新增] 计算向量的二进制指纹
    inline void compute_signature(int id, const float* vec) {
        uint64_t s0 = 0, s1 = 0;
        
        // 简单高效的二值化：基于原点划分 (vec[i] > 0 -> 1, else -> 0)
        // 前 64 维映射到 s0，后 36 维映射到 s1 的低位
        for (int i = 0; i < std::min(dim, 64); ++i) {
            if (vec[i] > 0.0f) s0 |= (1ULL << i);
        }
        for (int i = 64; i < dim && i < 128; ++i) {
            if (vec[i] > 0.0f) s1 |= (1ULL << (i - 64));
        }
        
        signatures[(size_t)id * 2] = s0;
        signatures[(size_t)id * 2 + 1] = s1;
    }
    
    // [新增] 为查询向量生成指纹（不需要存储）
    inline void compute_query_signature(const float* q, uint64_t& sig0, uint64_t& sig1) const {
        sig0 = sig1 = 0;
        for (int i = 0; i < std::min(dim, 64); ++i) {
            if (q[i] > 0.0f) sig0 |= (1ULL << i);
        }
        for (int i = 64; i < dim && i < 128; ++i) {
            if (q[i] > 0.0f) sig1 |= (1ULL << (i - 64));
        }
    }
    
    // [新增] 计算两个指纹的汉明距离
    inline int hamming_distance(uint64_t s0_a, uint64_t s1_a, uint64_t s0_b, uint64_t s1_b) const {
#ifdef __GNUC__
        // 使用内建函数，编译器会生成 POPCNT 指令
        return __builtin_popcountll(s0_a ^ s0_b) + __builtin_popcountll(s1_a ^ s1_b);
#elif defined(_MSC_VER)
        return (int)_mm_popcnt_u64(s0_a ^ s0_b) + (int)_mm_popcnt_u64(s1_a ^ s1_b);
#else
        // Fallback: 软件实现
        auto popcount = [](uint64_t x) {
            int c = 0;
            while (x) { c++; x &= x - 1; }
            return c;
        };
        return popcount(s0_a ^ s0_b) + popcount(s1_a ^ s1_b);
#endif
    }
    
    // =========================================================
    // [新增] K-means 聚类构建候选起点集
    // =========================================================
    void buildEntryCandidates(int K = 32, int max_iters = 20) {
        if (num_nodes < K) {
            // 数据量太少，直接使用默认入口点
            entry_candidates.clear();
            num_entry_candidates = 0;
            return;
        }

        // 1. K-means 聚类
        std::vector<AlignedFloatArray> centroids(K);
        for (int i = 0; i < K; ++i) {
            centroids[i].resize(dim);
        }
        
        // 初始化：随机选择 K 个点作为初始簇中心（K-means++简化版）
        std::mt19937 rng(42);
        std::vector<int> init_indices(num_nodes);
        for (int i = 0; i < num_nodes; ++i) init_indices[i] = i;
        std::shuffle(init_indices.begin(), init_indices.end(), rng);
        
        for (int i = 0; i < K; ++i) {
            const float* src = get_vec(init_indices[i]);
            std::memcpy(centroids[i].data(), src, dim * sizeof(float));
        }
        
        // 节点所属簇的分配
        std::vector<int> assignments(num_nodes, -1);
        std::vector<int> cluster_sizes(K);
        
        // 迭代优化
        for (int iter = 0; iter < max_iters; ++iter) {
            // E-step: 分配每个点到最近的簇中心
            std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
            
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_nodes; ++i) {
                const float* vec = get_vec(i);
                float best_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;
                
                for (int c = 0; c < K; ++c) {
                    float d = l2sq_100d(vec, centroids[c].data());
                    if (d < best_dist) {
                        best_dist = d;
                        best_cluster = c;
                    }
                }
                assignments[i] = best_cluster;
            }
            
            // 统计每个簇的大小
            for (int i = 0; i < num_nodes; ++i) {
                cluster_sizes[assignments[i]]++;
            }
            
            // M-step: 更新簇中心
            // 先清零
            for (int c = 0; c < K; ++c) {
                std::memset(centroids[c].data(), 0, dim * sizeof(float));
            }
            
            // 累加
            for (int i = 0; i < num_nodes; ++i) {
                int c = assignments[i];
                const float* vec = get_vec(i);
                float* cent = centroids[c].data();
                for (int d = 0; d < dim; ++d) {
                    cent[d] += vec[d];
                }
            }
            
            // 平均
            for (int c = 0; c < K; ++c) {
                if (cluster_sizes[c] > 0) {
                    float inv = 1.0f / cluster_sizes[c];
                    for (int d = 0; d < dim; ++d) {
                        centroids[c][d] *= inv;
                    }
                }
            }
        }
        
        // 2. 为每个簇中心找到数据库中最近的点
        entry_candidates.resize(K);
        
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < K; ++c) {
            const float* cent = centroids[c].data();
            float best_dist = std::numeric_limits<float>::max();
            int best_node = 0;
            
            for (int i = 0; i < num_nodes; ++i) {
                float d = l2sq_100d(get_vec(i), cent);
                if (d < best_dist) {
                    best_dist = d;
                    best_node = i;
                }
            }
            entry_candidates[c] = best_node;
        }
        
        // 去重（不同簇中心可能映射到同一个点）
        std::sort(entry_candidates.begin(), entry_candidates.end());
        entry_candidates.erase(std::unique(entry_candidates.begin(), entry_candidates.end()), 
                               entry_candidates.end());
        
        num_entry_candidates = (int)entry_candidates.size();
        
        // 3. 紧凑存储候选起点的向量数据（用于快速距离计算）
        entry_candidates_data.resize((size_t)num_entry_candidates * dim);
        for (int i = 0; i < num_entry_candidates; ++i) {
            std::memcpy(entry_candidates_data.data() + (size_t)i * dim,
                       get_vec(entry_candidates[i]),
                       dim * sizeof(float));
        }
        
        if (DEBUG_TIMING) {
            std::cout << "[Cluster] Built " << num_entry_candidates 
                      << " entry candidates from " << K << " clusters\n";
        }
    }
    
    // =========================================================
    // [新增] 自适应选择最佳起点
    // =========================================================
    int selectBestEntryPoint(const float* q) const {
        // 消融模式：禁用自适应起点选择，直接返回默认入口点
        if (ABLATE_ADAPTIVE_EP.load(std::memory_order_relaxed)) {
            return enter_point;
        }
        
        if (num_entry_candidates == 0) {
            // 没有候选起点，返回默认入口
            return enter_point;
        }
        
        float best_dist = std::numeric_limits<float>::max();
        int best_idx = 0;
        
        // 计算查询到所有候选起点的距离
        const float* cand_data = entry_candidates_data.data();
        
        for (int i = 0; i < num_entry_candidates; ++i) {
            float d = l2sq_100d(q, cand_data + (size_t)i * dim);
            if (d < best_dist) {
                best_dist = d;
                best_idx = i;
            }
        }
        
        return entry_candidates[best_idx];
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

        // [新增] 预计算查询向量的二进制指纹
        uint64_t q_sig0, q_sig1;
        compute_query_signature(q, q_sig0, q_sig1);
        
        // 获取阈值（一次性读取，避免多次原子操作）
        const int hamming_th = HAMMING_THRESHOLD.load(std::memory_order_relaxed);

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

            if ((int)top_results.size() >= ef && curr.first > worst_dist) {
                break;
            }

            int count;
            const int* links = get_l0_links(curr.second, count);
            if (count == 0) continue;

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

                // [新增] 二进制指纹过滤 - 极快的汉明距离预筛选
                bool pass1 = false, pass2 = false;
                
                if (!v1) {
                    uint64_t n_sig0 = signatures[(size_t)nb1 * 2];
                    uint64_t n_sig1 = signatures[(size_t)nb1 * 2 + 1];
                    int hd1 = hamming_distance(q_sig0, q_sig1, n_sig0, n_sig1);
                    pass1 = (hd1 <= hamming_th);
                }
                
                if (!v2) {
                    uint64_t n_sig0 = signatures[(size_t)nb2 * 2];
                    uint64_t n_sig1 = signatures[(size_t)nb2 * 2 + 1];
                    int hd2 = hamming_distance(q_sig0, q_sig1, n_sig0, n_sig1);
                    pass2 = (hd2 <= hamming_th);
                }
                
                // 跳过未通过指纹测试的向量
                if (!v1 && !pass1) v1 = true;  // 标记为已处理（跳过）
                if (!v2 && !pass2) v2 = true;

                if (v1 && v2) continue;

                // 核心优化：尽可能使用 2x SIMD 计算
                if (!v1 && !v2) {
                    // 最优路径：两个都未访问且通过指纹测试，批量计算
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
                
                // [新增] 指纹过滤
                uint64_t n_sig0 = signatures[(size_t)nb * 2];
                uint64_t n_sig1 = signatures[(size_t)nb * 2 + 1];
                int hd = hamming_distance(q_sig0, q_sig1, n_sig0, n_sig1);
                
                if (hd > hamming_th) continue;  // 跳过不相似的向量
                
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
// HNSWNode - 仅用于构建阶段的临时结构
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
// HnswGraphBuilder - 临时构建器（不保存数据副本）
// ---------------------------------------------------------
#include "GraphBuild.h"

// ---------------------------------------------------------
// 删除 SimpleHNSW 类定义
// ---------------------------------------------------------

#include "cache.h"

// ---------------------------------------------------------
// 并行包装类 - 简化为只管理 FlatHNSW
// ---------------------------------------------------------
#include "build.h"

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






#include "extern.h"// ---------------------------------------------------------// 设置参数接口// ---------------------------------------------------------