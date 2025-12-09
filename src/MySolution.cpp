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
    // 索引方式: signatures[id * 2], signatures[id * 2 + 1]
    std::vector<uint64_t> signatures;

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
class HnswGraphBuilder {
public:
    int dim;
    int M;
    int maxLayer;
    
    const float* data_ptr;  // 外部数据指针
    int data_count;
    
    std::vector<HNSWNode*> nodes;
    int enter_point;
    std::shared_mutex global_mutex;

    HnswGraphBuilder(int d, int m, int ml, const float* data, int n)
        : dim(d), M(m), maxLayer(ml), data_ptr(data), data_count(n), enter_point(-1) {}

    ~HnswGraphBuilder() { for (auto p : nodes) delete p; }

    inline int size() const { return (int)nodes.size(); }
    
    inline const float* getVec(int id) const {
        return data_ptr + (size_t)id * dim;
    }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    inline float dist(int id, const float* q) const {
        return l2sq_100d(getVec(id), q);
    }
    
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_100d(getVec(id_a), getVec(id_b));
    }

    int greedySearch(int ep, const float* q, int l) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return -1;
        
        float curd = dist(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            
            std::shared_lock<std::shared_mutex> lock_guard(nodes[ep]->lock);
            const auto& neighbors = nodes[ep]->links[l];
            
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

    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return {};
        
        using Pair = std::pair<float, int>;
        
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

            std::shared_lock<std::shared_mutex> lock_guard(nodes[curr.second]->lock);
            const auto& neighbors = nodes[curr.second]->links[l];
            
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

                    if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                        auto it = std::upper_bound(top_candidates.begin(), top_candidates.end(),
                            Pair{d_nb, nb}, [](const Pair& a, const Pair& b) { return a.first < b.first; });
                        top_candidates.insert(it, {d_nb, nb});

                        if ((int)top_candidates.size() > ef) {
                            top_candidates.pop_back();
                        }
                        lower_bound = top_candidates.back().first;
                    }

                    candidate_queue.push_back({d_nb, nb});
                    std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
                }
            }
        }

        return top_candidates;
    }

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
    }
    
    void tryAddReverseLink(int target_id, int new_neighbor_id, float dist_val, int level) {
        if (target_id == new_neighbor_id) return;
        if (target_id < 0 || target_id >= size()) return;
        if (new_neighbor_id < 0 || new_neighbor_id >= size()) return;
        
        int m_max = (level == 0) ? M * 2 : M;
        
        bool worth_trying = false;
        
        {
            std::shared_lock<std::shared_mutex> read_lock(nodes[target_id]->lock);
            
            if (level >= (int)nodes[target_id]->links.size()) {
                return;
            }
            
            const auto& links = nodes[target_id]->links[level];
            
            if ((int)links.size() < m_max) {
                worth_trying = true;
            } else {
                int worst_link_id = links.back();
                float worst_d = distNodes(target_id, worst_link_id);
                
                if (dist_val < worst_d * 1.0001f) {
                    worth_trying = true;
                }
            }
        }
        
        if (worth_trying) {
            std::vector<std::pair<float, int>> new_cand = {{dist_val, new_neighbor_id}};
            connectNodeHeuristic(target_id, new_cand, level);
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
                curr = greedySearch(curr, getVec(id), l);
            }

            for (int l = std::min(level, max_l); l >= 0; l--) {
                auto top = searchLayer(getVec(id), curr, l, g_HNSW_EF_CONSTRUCTION.load());
                if (!top.empty()) curr = top[0].second;
                
                connectNodeHeuristic(id, top, l);
                
                for (const auto& candidate : top) {
                    int neighbor_id = candidate.second;
                    float dist_val = candidate.first;
                    tryAddReverseLink(neighbor_id, id, dist_val, l);
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
// 删除 SimpleHNSW 类定义
// ---------------------------------------------------------

#include "cache.h"

// ---------------------------------------------------------
// 并行包装类 - 简化为只管理 FlatHNSW
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

        // 缓存未命中，使用临时构建器构建图
        auto build_start = std::chrono::high_resolution_clock::now();
        
        HnswGraphBuilder* builder = new HnswGraphBuilder(d, M, max_layer, data, n);
        
        builder->nodes.reserve(n);
        
        std::vector<int> levels(n);
        for (int i = 0; i < n; ++i) {
            levels[i] = std::min(builder->randomLevel(), max_layer);
            builder->nodes.push_back(new HNSWNode(levels[i], M));
        }
        
        if (n > 0) builder->enter_point = 0;

        ThreadPool* pool = getThreadPool();
        std::atomic<int> processed(1);
        int chunk_size = 1000;

        for (int i = 1; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            pool->enqueue([builder, i, end, &levels, &processed]() {
                for (int j = i; j < end; ++j) {
                    builder->insertPointParallel(j, levels[j]);
                }
                processed.fetch_add(end - i, std::memory_order_release);
            });
        }

        std::thread progress_thread([&processed, n, &build_start]() {
            int last_reported = 0;
            while (processed.load(std::memory_order_acquire) < n) {
                int curr = processed.load(std::memory_order_acquire);
                if (curr - last_reported >= std::max(50000, n / 100)) {
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

        // 转换为扁平化结构
        auto convert_start = std::chrono::high_resolution_clock::now();
        flat_index = convert_to_flat(builder);
        auto convert_end = std::chrono::high_resolution_clock::now();
        double convert_ms = std::chrono::duration<double, std::milli>(convert_end - convert_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Flat Conversion: " << std::fixed << std::setprecision(2) 
                      << convert_ms << " ms" << std::endl;
            std::cout.flush();
        }
        
        g_last_build_ms.store(total_ms + convert_ms, std::memory_order_relaxed);
        
        // 删除临时构建器
        delete builder;
        
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

        point_ids.resize(n);
        for (int i = 0; i < n; ++i) point_ids[i] = i;
    }

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

