// hnsw_solution_parallel_optimized.cpp
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
#include <shared_mutex> // C++17 读写锁，关键优化
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>

// ---------------------------------------------------------
// 全局线程池 (保持不变，用于并行构建)
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

// 硬件并发数
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

// ---------------------------------------------------------
// 配置参数
// ---------------------------------------------------------
constexpr int HNSW_M = 16;
constexpr int HNSW_MAX_LAYER = 16; // 稍微增加最大层数限制以适应大规模数据
constexpr int HNSW_EF_CONSTRUCTION = 200;
constexpr int HNSW_EF_SEARCH = 256;
static bool DEBUG_TIMING = true;

// ---------------------------------------------------------
// SIMD 距离计算 (保持 AVX2 优化)
// ---------------------------------------------------------
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
    // 使用非对齐存储，避免未满足 32 字节对齐时导致崩溃/栈损坏
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
    float s = 0.0f;
    const float* end = a + dim;
    // 简单的循环展开优化
    while (a < end) {
        float t = *a++ - *b++;
        s += t * t;
    }
    return s;
}
#endif

// ---------------------------------------------------------
// 访问标记优化 (Visited List)
// ---------------------------------------------------------
// 避免每次搜索都 malloc/memset
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
        if (curr_tag == 0) { // 溢出回绕
            std::fill(visited_tags.begin(), visited_tags.end(), 0);
            curr_tag = 1;
        }
    }

    bool isVisited(int id) const {
        if (id < 0 || (size_t)id >= visited_tags.size()) return false;
        return visited_tags[id] == curr_tag;
    }

    void mark(int id) {
        if (id < 0 || (size_t)id >= visited_tags.size()) return;
        visited_tags[id] = curr_tag;
    }
};

// ---------------------------------------------------------
// HNSW 节点与图结构
// ---------------------------------------------------------
struct HNSWNode {
    // 数据分离：向量数据通常是只读的，放在一起；
    // 连接关系需要加锁修改。
    std::vector<std::vector<int>> links;
    mutable std::shared_mutex lock; // 细粒度锁：每个节点一把锁
    
    // 初始化时预留空间，减少 vector 扩容带来的锁内耗时
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
    
    // 节点数据分开存储以优化缓存和锁
    std::vector<std::vector<float>> data_vecs; 
    std::vector<HNSWNode*> nodes; 
    
    // 入口点保护
    int enter_point; 
    std::shared_mutex global_mutex; // 保护 enter_point 和节点插入的整体结构

    SimpleHNSW(int d, int m = HNSW_M, int ml = HNSW_MAX_LAYER)
        : dim(d), M(m), maxLayer(ml), enter_point(-1) {}

    ~SimpleHNSW() {
        for (auto p : nodes) delete p;
    }

    int size() const { return (int)nodes.size(); }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        // -ln(r) * (1 / ln(M)) 实现
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    float dist(int id, const float* v) const {
        return l2sq_dense(data_vecs[id].data(), v, dim);
    }

    // 贪婪搜索 (Greedy Search)
    int greedySearch(int ep, const float* q, int l) const {
        if (ep < 0 || ep >= size()) return -1;
        float curd = dist(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            // 读锁：读取邻居列表
            std::shared_lock<std::shared_mutex> lock(nodes[ep]->lock);
            const auto& neighbors = nodes[ep]->links[l];
            
            for (int nb : neighbors) {
                if (nb < 0 || nb >= size()) continue; // 防止越界
                float nd = dist(nb, q);
                if (nd < curd) {
                    curd = nd;
                    ep = nb;
                    changed = true;
                    // 这里的逻辑可以优化：不需要立即跳出，可以遍历完当前节点所有邻居找最好的
                    // 但标准的 greedy 是找到一个更优就跳，或者遍历完找最优。
                    // 为了并发安全，我们不能一直持有锁，所以这里复制邻居列表或快速遍历是关键。
                    // 鉴于 M 很小，持有读锁遍历完是安全的。
                }
            }
        }
        return ep;
    }

    // 搜索层 (Search Layer)
    // 使用 thread_local 的 VisitedList 避免内存分配
    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (ep < 0) return {};
        
        using Pair = std::pair<float, int>;
        auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; }; // Min heap
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> candidates(cmp);
        std::priority_queue<Pair> top_results; // Max heap by default

        // 线程局部存储 VisitedList
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

            // 关键：加读锁获取邻居
            std::shared_lock<std::shared_mutex> lock(nodes[curid]->lock);
            const auto& neighbors = nodes[curid]->links[l];
            
            // 预取（可选，但在并发下可能不需要）
            // for(int nb : neighbors) __builtin_prefetch(data_vecs[nb].data());

            for (int nb : neighbors) {
                if (nb < 0 || nb >= size()) continue;
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

        std::vector<Pair> res;
        res.reserve(top_results.size());
        while (!top_results.empty()) {
            res.push_back(top_results.top());
            top_results.pop();
        }
        // 结果此时是反序的（距离大的在前），如果需要距离小的在前，需要反转
        std::reverse(res.begin(), res.end());
        return res;
    }

    // 连接节点 (加写锁)
    void connectNode(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        int m_max = (l == 0) ? M * 2 : M; // 第0层允许更多连接
        
        // 选出最近的 M 个
        std::vector<std::pair<float, int>> selected = candidates;
        if ((int)selected.size() > m_max) selected.resize(m_max);

        // 1. 将 selected 添加到 id 的连接表 (无需锁，因为 id 是新插入的，其他线程还不可见，或者由调用者保证)
        // 但为了通用性，我们还是加锁
        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            auto& links = nodes[id]->links[l];
            for (auto& p : selected) links.push_back(p.second);
        }

        // 2. 反向连接：把 id 添加到邻居的连接表
        for (auto& p : selected) {
            int nb = p.second;
            if (nb < 0 || nb >= size()) continue;
            std::unique_lock<std::shared_mutex> lock(nodes[nb]->lock);
            auto& links = nodes[nb]->links[l];
            
            bool exists = false;
            for(int x : links) if(x == id) { exists=true; break; }
            if(!exists) links.push_back(id);

            // 如果邻居连接数超限，需要修剪
            if ((int)links.size() > m_max) {
                // 重新计算距离并保留最近的 m_max 个
                // 注意：此时持有 nb 的写锁，不能调用需要 nb 锁的其他函数
                std::vector<std::pair<float, int>> nbr_dists;
                nbr_dists.reserve(links.size());
                for (int n_nb : links) {
                    nbr_dists.push_back({dist(n_nb, data_vecs[nb].data()), n_nb});
                }
                std::sort(nbr_dists.begin(), nbr_dists.end()); // 按距离排序
                
                links.clear();
                for(int i=0; i<m_max; ++i) links.push_back(nbr_dists[i].second);
            }
        }
    }

    // 并行插入的核心实现
    // 注意：addPoint 不直接接受 vector，而是接受索引，假定数据已预分配
    void insertPointParallel(int id, int level) {
        // 1. 保护性读取入口点
        int ep_curr;
        {
            std::shared_lock<std::shared_mutex> lock(global_mutex);
            ep_curr = enter_point;
        }

        if (ep_curr != -1) {
            // 2. 从顶层搜索到插入层
            // 这里需要获取 ep 节点的层数，注意多线程下节点可能被修改，但层数通常不变
            int max_l = (int)nodes[ep_curr]->links.size() - 1;
            int curr = ep_curr;
            
            for (int l = max_l; l > level; l--) {
                curr = greedySearch(curr, data_vecs[id].data(), l);
            }

            // 3. 在每层进行插入
            for (int l = std::min(level, max_l); l >= 0; l--) {
                // 搜索 candidates
                auto top = searchLayer(data_vecs[id].data(), curr, l, HNSW_EF_CONSTRUCTION);
                // 更新 curr 为下一层的最近点
                if (!top.empty()) curr = top[0].second;
                // 建立双向连接
                connectNode(id, top, l);
            }
        }

        // 4. 更新全局入口点
        // 如果新节点层级更高，它可能成为新的入口
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
    std::vector<int> point_ids; // 映射内部 ID 到外部 ID

    ~HnswSolutionParallel(){ delete hnsw; }

    void build_from_memory(int d, const std::vector<std::vector<float>>& data) {
        int n = (int)data.size();
        delete hnsw;
        hnsw = new SimpleHNSW(d, HNSW_M, HNSW_MAX_LAYER);
        
        // 1. 预分配所有内存
        // 这一步非常重要，避免多线程运行时 resize data_vecs 或 nodes 导致迭代器失效或锁竞争
        hnsw->data_vecs = data; // 拷贝数据
        hnsw->nodes.reserve(n);
        
        // 预生成每个节点的 Level，并构建 Node 对象
        // 这样可以提前分配好 links 的内存
        std::vector<int> levels(n);
        for(int i=0; i<n; ++i) {
            levels[i] = hnsw->randomLevel();
            if(levels[i] > HNSW_MAX_LAYER) levels[i] = HNSW_MAX_LAYER;
            hnsw->nodes.push_back(new HNSWNode(levels[i], HNSW_M));
        }
        
        point_ids.resize(n);
        for(int i=0; i<n; ++i) point_ids[i] = i;

        // 2. 并行插入
        // 第一个点必须串行插入以初始化 enter_point
        if (n > 0) {
            hnsw->enter_point = 0;
            // 第0个点已“逻辑”插入（它是入口，暂无邻居）
        }

        auto build_start = std::chrono::steady_clock::now();

        ThreadPool* pool = getThreadPool();
        std::atomic<int> processed(1); // 0号已处理

        // 将剩余点分块并行处理
        // HNSW 并行构建通常不需要严格顺序，但分块有助于利用缓存
        int start_idx = 1;
        int chunk_size = 1000; // 批次大小

        for (int i = start_idx; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            pool->enqueue([this, i, end, &levels, &processed, n]() {
                for (int j = i; j < end; ++j) {
                    hnsw->insertPointParallel(j, levels[j]);
                    
                    // 可选：打印进度
                    if (DEBUG_TIMING) {
                        int p = processed.fetch_add(1, std::memory_order_relaxed);
                        if (p % 10000 == 0) {
                            // std::cerr << "Built " << p << "/" << n << "\n";
                        }
                    }
                }
            });
        }
        
        // 等待队列清空（简单的 barrier 实现）
        // 注意：ThreadPool 的 wait() 并不是标准接口，这里简单通过析构或者智能指针控制
        // 由于原 ThreadPool 没有完善的 wait，我们利用析构或者额外的原子计数器等待
        // 简单修改：我们在外部等待所有任务完成。
        // 为简化代码，这里使用比较笨的方法：等待 processed 计数
        // 实际生产中应使用 std::future 或 ThreadPool 提供的 wait
        while (processed < n) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto build_end = std::chrono::steady_clock::now();
        if (DEBUG_TIMING) {
            double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
            std::cout << "[Timing] Parallel Build: " << total_ms << " ms for " << n << " points." << std::endl;
        }
    }

    std::vector<std::pair<int,float>> search(const std::vector<float>& query, int k) {
        if(!hnsw || hnsw->nodes.empty()) return {};
        
        // search 过程是只读的，本身不需要加全局锁
        // 但需要加节点的读锁（在 searchLayer 内部已处理）
        int ep = -1;
        {
            std::shared_lock<std::shared_mutex> lock(hnsw->global_mutex);
            ep = hnsw->enter_point;
        }
        if (ep < 0 || ep >= hnsw->size()) return {}; // 防御性返回
        
        int max_l = (int)hnsw->nodes[ep]->links.size() - 1;
        int curr = ep;
        
        for(int l = max_l; l > 0; l--) curr = hnsw->greedySearch(curr, query.data(), l);
        
        auto top = hnsw->searchLayer(query.data(), curr, 0, HNSW_EF_SEARCH);
        
        std::vector<std::pair<int,float>> out;
        if((int)top.size() > k) top.resize(k);
        out.reserve(top.size());
        for(const auto &p: top){
            out.push_back({point_ids[p.second], p.first});
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
    // 简单的并行化数据拷贝（可选优化）
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

class Solution {
    int k;
public:
    Solution(int d, int n, int kk); // 改为声明
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* result);
};

// Solution 类方法定义
void Solution::build(int d, const std::vector<float>& base) {
    build_hnsw(d, base);
}

void Solution::search(const std::vector<float>& query, int* result) {
    auto res = search_hnsw(query, k);
    for (size_t i = 0; i < res.size(); ++i) {
        result[i] = res[i].first;
    }
}

// 在类外添加构造函数定义以生成链接符号
Solution::Solution(int d, int n, int kk) : k(kk) {}