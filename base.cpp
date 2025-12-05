#include "MySolution.h"
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <atomic>

// ---------------------------------------------------------
// Distance counting globals
// ---------------------------------------------------------
static std::atomic<uint64_t> g_total_dist_count{0};
static std::atomic<uint64_t> g_total_query_count{0};
static std::atomic<uint64_t> g_last_query_dist{0};
static thread_local uint64_t tl_dist_counter = 0;
static std::atomic<bool> ENABLE_RUNTIME_DIST_COUNTING{true};

// ---------------------------------------------------------
// 基础距离计算（无 SIMD，无计数）
// ---------------------------------------------------------
static inline float l2sq_basic(const float* a, const float* b, int dim) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) {
        ++tl_dist_counter;
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// ---------------------------------------------------------
// 基础 HNSW 节点
// ---------------------------------------------------------
struct BasicNode {
    std::vector<std::vector<int>> neighbors;  // neighbors[level] = list of neighbor ids
    
    BasicNode(int max_level) {
        neighbors.resize(max_level + 1);
    }
};

// ---------------------------------------------------------
// 基础 HNSW 索引（无任何优化）
// ---------------------------------------------------------
class BasicHNSW {
public:
    int dim;
    int M;           // 每层最大邻居数
    int M_max0;      // 第0层最大邻居数 (2*M)
    int max_level;   // 最大层数
    int ef_construction;
    int ef_search;
    
    std::vector<float> data;       // 所有向量数据 (n * dim)
    std::vector<BasicNode*> nodes;
    int enter_point;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
    BasicHNSW(int d, int m = 16, int ml = 16, int efc = 200, int efs = 200)
        : dim(d), M(m), M_max0(m * 2), max_level(ml), 
          ef_construction(efc), ef_search(efs),
          enter_point(-1), rng(std::random_device{}()), dist(0.0f, 1.0f) {}
    
    ~BasicHNSW() {
        for (auto p : nodes) delete p;
    }
    
    int size() const { return (int)nodes.size(); }
    
    const float* getVec(int id) const {
        return data.data() + (size_t)id * dim;
    }
    
    float distance(int id, const float* q) const {
        return l2sq_basic(getVec(id), q, dim);
    }
    
    float distance(int id_a, int id_b) const {
        return l2sq_basic(getVec(id_a), getVec(id_b), dim);
    }
    
    int randomLevel() {
        float r = dist(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }
    
    // 贪婪搜索：在指定层找到最近的节点
    int greedySearch(int ep, const float* q, int level) const {
        if (ep < 0) return -1;
        
        float cur_dist = distance(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            const auto& neighbors = nodes[ep]->neighbors[level];
            
            for (int nb : neighbors) {
                float d = distance(nb, q);
                if (d < cur_dist) {
                    cur_dist = d;
                    ep = nb;
                    changed = true;
                }
            }
        }
        return ep;
    }
    
    // 搜索层：返回最近的 ef 个候选
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int level, int ef) const {
        if (ep < 0) return {};
        
        std::unordered_set<int> visited;
        
        // min-heap for candidates (to expand)
        auto cmp_min = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_min)> candidates(cmp_min);
        
        // max-heap for results (to maintain top-ef)
        auto cmp_max = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp_max)> results(cmp_max);
        
        float d0 = distance(ep, q);
        visited.insert(ep);
        candidates.push({d0, ep});
        results.push({d0, ep});
        
        while (!candidates.empty()) {
            auto [cur_dist, cur_id] = candidates.top();
            candidates.pop();
            
            // 如果当前候选比结果集中最差的还差，停止
            if (results.size() >= (size_t)ef && cur_dist > results.top().first) {
                break;
            }
            
            const auto& neighbors = nodes[cur_id]->neighbors[level];
            
            for (int nb : neighbors) {
                if (visited.count(nb)) continue;
                visited.insert(nb);
                
                float d = distance(nb, q);
                
                if (results.size() < (size_t)ef || d < results.top().first) {
                    candidates.push({d, nb});
                    results.push({d, nb});
                    
                    if (results.size() > (size_t)ef) {
                        results.pop();
                    }
                }
            }
        }
        
        // 转换为 vector 并排序
        std::vector<std::pair<float, int>> result_vec;
        result_vec.reserve(results.size());
        while (!results.empty()) {
            result_vec.push_back(results.top());
            results.pop();
        }
        std::sort(result_vec.begin(), result_vec.end());
        return result_vec;
    }
    
    // 选择邻居（简单策略：取最近的 M 个）
    std::vector<int> selectNeighbors(const std::vector<std::pair<float, int>>& candidates, int max_neighbors) {
        std::vector<int> result;
        result.reserve(max_neighbors);
        for (size_t i = 0; i < candidates.size() && (int)result.size() < max_neighbors; ++i) {
            result.push_back(candidates[i].second);
        }
        return result;
    }
    
    // 插入单个点
    void insert(int id) {
        int level = std::min(randomLevel(), max_level);
        nodes[id] = new BasicNode(level);
        
        if (enter_point < 0) {
            enter_point = id;
            return;
        }
        
        int ep = enter_point;
        int max_l = (int)nodes[ep]->neighbors.size() - 1;
        
        // 从上层贪婪搜索到插入层
        for (int l = max_l; l > level; --l) {
            ep = greedySearch(ep, getVec(id), l);
        }
        
        // 在每层搜索并连接
        for (int l = std::min(level, max_l); l >= 0; --l) {
            auto candidates = searchLayer(getVec(id), ep, l, ef_construction);
            
            if (!candidates.empty()) {
                ep = candidates[0].second;
            }
            
            int m_max = (l == 0) ? M_max0 : M;
            auto neighbors = selectNeighbors(candidates, m_max);
            
            // 设置新节点的邻居
            nodes[id]->neighbors[l] = neighbors;
            
            // 反向连接
            for (int nb : neighbors) {
                auto& nb_neighbors = nodes[nb]->neighbors[l];
                nb_neighbors.push_back(id);
                
                // 如果超过限制，保留最近的
                if ((int)nb_neighbors.size() > m_max) {
                    std::vector<std::pair<float, int>> nb_candidates;
                    for (int x : nb_neighbors) {
                        nb_candidates.push_back({distance(nb, x), x});
                    }
                    std::sort(nb_candidates.begin(), nb_candidates.end());
                    nb_neighbors = selectNeighbors(nb_candidates, m_max);
                }
            }
        }
        
        // 更新入口点
        if (level > (int)nodes[enter_point]->neighbors.size() - 1) {
            enter_point = id;
        }
    }
    
    // 搜索 k 个最近邻
    std::vector<std::pair<int, float>> search(const float* q, int k) {
        if (enter_point < 0) return {};
        
        int ep = enter_point;
        int max_l = (int)nodes[ep]->neighbors.size() - 1;
        
        // 从上层贪婪搜索到第0层
        for (int l = max_l; l > 0; --l) {
            ep = greedySearch(ep, q, l);
        }
        
        // 在第0层搜索
        auto candidates = searchLayer(q, ep, 0, ef_search);
        
        std::vector<std::pair<int, float>> results;
        int cnt = std::min(k, (int)candidates.size());
        for (int i = 0; i < cnt; ++i) {
            results.push_back({candidates[i].second, candidates[i].first});
        }
        // accumulate distance counters
        uint64_t last = tl_dist_counter;
        tl_dist_counter = 0;
        g_last_query_dist.store(last, std::memory_order_relaxed);
        g_total_dist_count.fetch_add(last, std::memory_order_relaxed);
        g_total_query_count.fetch_add(1, std::memory_order_relaxed);
         return results;
     }
};

// ---------------------------------------------------------
// 全局实例
// ---------------------------------------------------------
static BasicHNSW* g_basic_hnsw = nullptr;

// ---------------------------------------------------------
// 对外接口
// ---------------------------------------------------------
void build_hnsw(int d, const std::vector<float>& base) {
    int n = (int)base.size() / d;
    
    delete g_basic_hnsw;
    g_basic_hnsw = new BasicHNSW(d, 16, 16, 200, 200);
    
    // 复制数据
    g_basic_hnsw->data = base;
    
    // 预分配节点
    g_basic_hnsw->nodes.resize(n, nullptr);
    
    // 逐个插入
    for (int i = 0; i < n; ++i) {
        g_basic_hnsw->insert(i);
    }
}

std::vector<std::pair<int, float>> search_hnsw(const std::vector<float>& query, int k) {
    if (!g_basic_hnsw) return {};
    return g_basic_hnsw->search(query.data(), k);
}

// ---------------------------------------------------------
// Solution 类实现
// ---------------------------------------------------------
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
// 缺失的接口函数（存根实现，供 test_hn_g.cpp 链接）
// ---------------------------------------------------------
extern "C" {

void set_hnsw_params(int m, int max_layer, int ef_construction, int ef_search, int build_threads) {
    (void)m; (void)max_layer; (void)ef_construction; (void)ef_search; (void)build_threads;
}

void set_hnsw_debug(int enable) {
    (void)enable;
}

void set_ablation_flags(int csr, int prefetch, int simd, int pruning, int heap) {
    (void)csr; (void)prefetch; (void)simd; (void)pruning; (void)heap;
}

double get_last_build_time_ms() {
    return 0.0;
}

int get_graph_num_nodes() {
    return g_basic_hnsw ? g_basic_hnsw->size() : 0;
}

int get_graph_max_level() {
    return g_basic_hnsw ? g_basic_hnsw->max_level : 0;
}

int get_graph_actual_max_layer() {
    if (!g_basic_hnsw || g_basic_hnsw->enter_point < 0) return 0;
    return (int)g_basic_hnsw->nodes[g_basic_hnsw->enter_point]->neighbors.size() - 1;
}

double get_graph_avg_degree_l0() {
    if (!g_basic_hnsw || g_basic_hnsw->size() == 0) return 0.0;
    long long total = 0;
    for (auto* node : g_basic_hnsw->nodes) {
        if (node && !node->neighbors.empty()) {
            total += node->neighbors[0].size();
        }
    }
    return (double)total / g_basic_hnsw->size();
}

double get_graph_avg_degree_upper() {
    if (!g_basic_hnsw || g_basic_hnsw->size() == 0) return 0.0;
    long long total = 0;
    int count = 0;
    for (auto* node : g_basic_hnsw->nodes) {
        if (node) {
            for (size_t l = 1; l < node->neighbors.size(); ++l) {
                total += node->neighbors[l].size();
                ++count;
            }
        }
    }
    return count > 0 ? (double)total / count : 0.0;
}

int get_graph_nodes_at_level(int level) {
    if (!g_basic_hnsw) return 0;
    int count = 0;
    for (auto* node : g_basic_hnsw->nodes) {
        if (node && (int)node->neighbors.size() > level) {
            ++count;
        }
    }
    return count;
}

void reset_dist_counters() {
    g_total_dist_count.store(0, std::memory_order_relaxed);
    g_total_query_count.store(0, std::memory_order_relaxed);
    g_last_query_dist.store(0, std::memory_order_relaxed);
    tl_dist_counter = 0;
}

uint64_t get_total_queries() {
    return g_total_query_count.load(std::memory_order_relaxed);
}

double get_avg_dists_per_query() {
    uint64_t q = g_total_query_count.load(std::memory_order_relaxed);
    return q ? static_cast<double>(g_total_dist_count.load(std::memory_order_relaxed)) / q : 0.0;
}

uint64_t get_last_query_dists() {
    return g_last_query_dist.load(std::memory_order_relaxed);
}

} // extern "C"
