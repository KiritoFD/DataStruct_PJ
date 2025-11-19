#include "MySolution.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <numeric>
#include <limits>
#include <thread>
#include <mutex>
#include <utility>
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstring>
#include <memory>
#include <condition_variable>
#include <functional>
#include <future>

// --- 配置与常量 ---
constexpr int SEARCH_THREADS = 4;
const bool debug = false;

// --- 简易线程池定义 ---
class ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty()) return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// 全局线程池实例
static std::unique_ptr<ThreadPool> g_pool;

// 全局辅助：缓存每个点到其最终质心距离（可选，主要依赖 inverted_index 内的数据）
static std::vector<float> g_point_centroid_dist;

// --- 字符串解析辅助 ---
namespace {
    bool try_stod(const std::string& s, float& out) {
        try {
            size_t pos = 0;
            out = std::stod(s, &pos);
            return pos == s.size();
        } catch (...) {
            return false;
        }
    }
}

bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<float>& out_vec) {
    out_id.clear();
    out_vec.clear();
    std::istringstream iss(line);
    std::vector<std::string> toks;
    std::string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;

    float val = 0.0;
    bool allnum = true;
    for (const auto& s : toks) {
        if (!try_stod(s, val)) { allnum = false; break; }
    }
    if (allnum) {
        out_vec.reserve(toks.size());
        for (const auto& s : toks) out_vec.push_back(std::stod(s));
        return true;
    }

    if (toks.size() < 2) return false;
    out_id = toks[0];
    out_vec.reserve(toks.size() - 1);
    for (size_t i = 1; i < toks.size(); ++i) {
        if (!try_stod(toks[i], val)) return false;
        out_vec.push_back(std::stod(toks[i]));
    }
    return true;
}

// --- Solution 实现 ---

solution::solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob)
    : metric(metric_type),
      dim(0),
      num_threads(1),
      num_centroid(num_centroid),
      kmean_iter(kmean_iter),
      nprob(nprob),
      kd_root_(-1) {
    unsigned int hc = std::thread::hardware_concurrency();
    num_threads = static_cast<int>(hc > 0 ? hc : 1);
    if (debug) {
        std::cout << "[solution] hardware_concurrency=" << hc << ", using " << num_threads << " threads\n";
        std::cout << "[solution] metric=" << metric << ", num_centroid=" << num_centroid
                  << ", kmean_iter=" << kmean_iter << ", nprob=" << nprob << "\n";
    }
    
    // 初始化全局线程池 (Thread Safe)
    static std::once_flag pool_flag;
    std::call_once(pool_flag, []() {
        int t_cnt = std::max(1, SEARCH_THREADS);
        g_pool = std::make_unique<ThreadPool>(t_cnt);
    });
}

void solution::build(const std::string& base_file) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::ifstream fin(base_file);
    if (!fin) return;
    
    std::vector<std::vector<float>> vectors;
    std::string line;
    int local_dim = 0;
    while (std::getline(fin, line)) {
        std::string id;
        std::vector<float> vec;
        if (!parse_vector_line(line, id, vec)) continue;
        if (local_dim == 0) local_dim = static_cast<int>(vec.size());
        if (vec.size() != static_cast<size_t>(local_dim)) continue;
        vectors.push_back(std::move(vec));
    }

    if (vectors.empty()) {
        point_ids_.clear();
        point_data_.clear();
        centroid_data_.clear();
        inverted_index.clear();
        kd_nodes_.clear();
        kd_root_ = -1;
        dim = 0;
        return;
    }
    build_from_memory(local_dim, std::move(vectors));
}

void solution::build_from_memory(int d, std::vector<std::vector<float>> data) {
    dim = d;
    const size_t n = data.size();
    point_ids_.resize(n);
    point_data_.assign(n * static_cast<size_t>(dim), 0.0f);
    
    // 初始数据拷贝
    for (size_t i = 0; i < n; ++i) {
        point_ids_[i] = static_cast<int>(i);
        float* dst = point_ptr(static_cast<int>(i));
        for (int j = 0; j < dim; ++j) {
            dst[j] = static_cast<float>(data[i][j]);
        }
    }
    finalize_build();
}

void solution::finalize_build() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const int total = static_cast<int>(point_ids_.size());
    if (total <= 0 || dim == 0) return;

    // 1. 初始化质心
    centroid_data_.assign(static_cast<size_t>(num_centroid) * dim, 0.0f);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, total - 1);
    for (int i = 0; i < num_centroid; ++i) {
        std::memcpy(centroid_ptr(i), point_ptr(dist(rng)), sizeof(float) * dim);
    }

    // 2. K-Means 迭代
    std::vector<int> assignments(total, 0);
    for (int iter = 0; iter < kmean_iter; ++iter) {
        kmeans_assign_parallel(assignments);
        std::vector<float> new_centroids(static_cast<size_t>(num_centroid) * dim, 0.0f);
        kmeans_update_parallel(assignments, new_centroids);
        centroid_data_.swap(new_centroids);
    }

    // 3. 构建 KD-Tree
    kd_nodes_.clear();
    if (num_centroid > 0) {
        std::vector<int> ids(num_centroid);
        std::iota(ids.begin(), ids.end(), 0);
        kd_root_ = build_kdtree(ids, 0, num_centroid, 0);
    } else {
        kd_root_ = -1;
    }

    // 4. 构建倒排索引（带距离计算）
    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;
    std::vector<std::vector<std::vector<BucketItem>>> thread_results(
        threads_to_use, std::vector<std::vector<BucketItem>>(num_centroid));
    
    g_point_centroid_dist.assign(total, 0.0f);

    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);
    auto worker = [this, &thread_results, &assignments](int start, int end, int tid) {
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            float dist = compute_distance_simd(point_ptr(i), centroid_ptr(c));
            thread_results[tid][c].push_back({i, dist});
            g_point_centroid_dist[i] = dist;
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) workers.emplace_back(worker, start, end, t);
    }
    for (auto& th : workers) th.join();

    // 5. 合并倒排并排序
    inverted_index.clear();
    inverted_index.resize(num_centroid);
    for (int c = 0; c < num_centroid; ++c) {
        size_t total_bucket = 0;
        for (int t = 0; t < threads_to_use; ++t) total_bucket += thread_results[t][c].size();
        if (total_bucket == 0) continue;
        
        auto& dest = inverted_index[c];
        dest.reserve(total_bucket);
        for (int t = 0; t < threads_to_use; ++t) {
            auto& src = thread_results[t][c];
            dest.insert(dest.end(), std::make_move_iterator(src.begin()), std::make_move_iterator(src.end()));
            std::vector<BucketItem>().swap(src);
        }
        
        // --- 核心优化：桶内按距离排序 (Memory Layout Optimization) ---
        std::sort(dest.begin(), dest.end(),
                  [](const BucketItem& a, const BucketItem& b) { return a.dist_to_centroid < b.dist_to_centroid; });
    }

    // 6. 物理内存重排 (Data Packing)
    {
        std::vector<int> old2new(total, -1);
        std::vector<float> new_points(static_cast<size_t>(total) * dim);
        std::vector<int> new_ids(total);
        int write = 0;
        
        // 按桶顺序写入，由于桶内已排序，数据在物理上也是有序的
        for (int c = 0; c < num_centroid; ++c) {
            auto& bucket = inverted_index[c];
            for (auto& bi : bucket) {
                int old = bi.index;
                float* src = point_ptr(old);
                float* dst = new_points.data() + static_cast<size_t>(write) * dim;
                std::memcpy(dst, src, sizeof(float) * dim);
                
                old2new[old] = write;
                new_ids[write] = point_ids_[old];
                bi.index = write; // 更新为新索引
                write++;
            }
        }
        
        // 处理可能遗漏的点
        for (int old = 0; old < total; ++old) {
            if (old2new[old] == -1) {
                int write2 = write++;
                float* src = point_ptr(old);
                float* dst = new_points.data() + static_cast<size_t>(write2) * dim;
                std::memcpy(dst, src, sizeof(float) * dim);
                old2new[old] = write2;
                new_ids[write2] = point_ids_[old];
            }
        }
        point_data_.swap(new_points);
        point_ids_.swap(new_ids);
        
        // 同步 distance cache
        if (!g_point_centroid_dist.empty()) {
            std::vector<float> new_dist(total);
            for (int old = 0; old < total; ++old) {
                int nw = old2new[old];
                if(nw >= 0 && nw < total) new_dist[nw] = g_point_centroid_dist[old];
            }
            g_point_centroid_dist.swap(new_dist);
        }
    }

    if (debug) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[finalize_build] total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
}

// --- K-Means Helpers ---
void solution::kmeans_assign_parallel(std::vector<int>& assignments) {
    const int total = static_cast<int>(point_ids_.size());
    if (total == 0) return;
    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;

    std::vector<std::thread> threads;
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) {
            threads.emplace_back([this, &assignments, start, end]() {
                for (int i = start; i < end; ++i) {
                    _mm_prefetch(reinterpret_cast<const char*>(point_ptr(std::min(i + 1, end - 1))), _MM_HINT_T0);
                    assignments[i] = find_closest_centroid_linear(point_ptr(i));
                }
            });
        }
    }
    for (auto& th : threads) th.join();
}

void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<float>& new_centroids) {
    const int total = static_cast<int>(point_ids_.size());
    if (total == 0) return;
    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;

    std::vector<std::vector<float>> thread_sums(threads_to_use, std::vector<float>(static_cast<size_t>(num_centroid) * dim, 0.0f));
    std::vector<std::vector<int>> thread_counts(threads_to_use, std::vector<int>(num_centroid, 0));

    std::vector<std::thread> threads;
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) {
            threads.emplace_back([this, &assignments, &thread_sums, &thread_counts, start, end, t]() {
                auto& sums = thread_sums[t];
                auto& counts = thread_counts[t];
                for (int i = start; i < end; ++i) {
                    int c = assignments[i];
                    float* dst = sums.data() + static_cast<size_t>(c) * dim;
                    const float* src = point_ptr(i);
                    for (int d = 0; d < dim; ++d) dst[d] += src[d];
                    counts[c] += 1;
                }
            });
        }
    }
    for (auto& th : threads) th.join();

    std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
    for (int c = 0; c < num_centroid; ++c) {
        float* dst = new_centroids.data() + static_cast<size_t>(c) * dim;
        int count = 0;
        for (int t = 0; t < threads_to_use; ++t) {
            const float* src = thread_sums[t].data() + static_cast<size_t>(c) * dim;
            for (int d = 0; d < dim; ++d) dst[d] += src[d];
            count += thread_counts[t][c];
        }
        if (count > 0) {
            float inv = 1.0f / count;
            for (int d = 0; d < dim; ++d) dst[d] *= inv;
        } else {
            std::memcpy(dst, centroid_ptr(c), sizeof(float) * dim);
        }
    }
}

// --- 距离计算 Kernels (优化版) ---

int solution::find_closest_centroid_linear(const float* vec) const {
    if (centroid_data_.empty()) return 0;
    float best = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (int c = 0; c < num_centroid; ++c) {
        float dist = compute_distance_simd(vec, centroid_ptr(c));
        if (dist < best) {
            best = dist;
            best_idx = c;
        }
    }
    return best_idx;
}

int solution::find_closest_centroid(const std::vector<float>& vec) const {
    // 兼容接口，实际指向 SIMD 版本
    return find_closest_centroid_linear(vec.data());
}

float solution::compute_distance_simd(const float* a, const float* b) const {
#if defined(__AVX512F)
    if (dim >= 16) {
        __m512 sum512 = _mm512_setzero_ps();
        int i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            __m512 sq = _mm512_mul_ps(diff, diff);
            sum512 = _mm512_add_ps(sum512, sq);
        }
        alignas(64) float tmp512[16];
        _mm512_store_ps(tmp512, sum512);
        float total = 0.0f;
        for (int k = 0; k < 16; ++k) total += tmp512[k];
        for (; i <= dim - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 sq = _mm256_mul_ps(diff, diff);
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, sq);
            total += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        }
        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            total += diff * diff;
        }
        return total;
    }
#endif
    __m256 sumv = _mm256_setzero_ps();
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sumv = _mm256_add_ps(sumv, sq);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sumv);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    return total;
}

float solution::compute_distance_capped_simd(const float* a, const float* b, float cap) const {
    // 优化：移除 isinf 检查，假定 cap 合法
#if defined(__AVX512F)
    if (dim >= 16) {
        __m512 sum512 = _mm512_setzero_ps();
        float total = 0.0f;
        int i = 0;
        alignas(64) float tmp512[16];
        for (; i <= dim - 16; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            __m512 sq = _mm512_mul_ps(diff, diff);
            sum512 = _mm512_add_ps(sum512, sq);
            
            // 周期性检查 cap (每16维检查一次)
            _mm512_store_ps(tmp512, sum512);
            float current_block = 0.0f;
            for (int k = 0; k < 16; ++k) current_block += tmp512[k];
            // 注意：这里仅做近似检查，若要严格准确，需要累加到 total。
            // 为保持高性能和代码简洁，这里只在 AVX512 块结束后检查 total 可能更优，
            // 但为了及时退出，我们这里做一次累加判断。
            if (total + current_block >= cap) return total + current_block;
        }
        
        _mm512_store_ps(tmp512, sum512);
        for (int k = 0; k < 16; ++k) total += tmp512[k];
        
        for (; i <= dim - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            __m256 sq = _mm256_mul_ps(diff, diff);
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, sq);
            for (int k = 0; k < 8; ++k) {
                total += tmp[k];
                if (total >= cap) return total;
            }
        }
        for (; i < dim; ++i) {
            float d = a[i] - b[i];
            total += d * d;
            if (total >= cap) return total;
        }
        return total;
    }
#endif
    // AVX2 Path
    float total = 0.0f;
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff);
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sq);
        for (int k = 0; k < 8; ++k) {
            total += tmp[k];
            if (total >= cap) return total;
        }
    }
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
        if (total >= cap) return total;
    }
    return total;
}

// --- KD-Tree Helpers ---

int solution::build_kdtree(std::vector<int>& indices, int begin, int end, int depth) {
    if (begin >= end) return -1;
    int axis = depth % dim;
    int mid = (begin + end) / 2;
    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end,
                     [this, axis](int lhs, int rhs) {
                         return centroid_ptr(lhs)[axis] < centroid_ptr(rhs)[axis];
                     });
    int centroid_index = indices[mid];
    KDNode node{axis, centroid_index, -1, -1, centroid_ptr(centroid_index)[axis]};
    int node_id = static_cast<int>(kd_nodes_.size());
    kd_nodes_.push_back(node);
    kd_nodes_[node_id].left = build_kdtree(indices, begin, mid, depth + 1);
    kd_nodes_[node_id].right = build_kdtree(indices, mid + 1, end, depth + 1);
    return node_id;
}

void solution::search_kdtree(const float* query, int node_idx, int nprobe,
                             std::priority_queue<std::pair<float, int>>& best) const {
    if (node_idx < 0) return;
    const KDNode& node = kd_nodes_[node_idx];
    float dist = compute_distance_simd(query, centroid_ptr(node.centroid_index));
    if (static_cast<int>(best.size()) < nprobe) {
        best.emplace(dist, node.centroid_index);
    } else if (dist < best.top().first) {
        best.pop();
        best.emplace(dist, node.centroid_index);
    }

    float diff = query[node.axis] - node.split_value;
    int near = diff <= 0.0f ? node.left : node.right;
    int far = diff <= 0.0f ? node.right : node.left;

    search_kdtree(query, near, nprobe, best);
    float worst = best.empty() ? std::numeric_limits<float>::max() : best.top().first;
    if (static_cast<int>(best.size()) < nprobe || diff * diff < worst) {
        search_kdtree(query, far, nprobe, best);
    }
}

std::vector<std::pair<int, float>> solution::find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const {
    if (centroid_data_.empty() || nprobe <= 0) return {};
    nprobe = std::min(nprobe, num_centroid);
    std::priority_queue<std::pair<float, int>> best;
    if (kd_root_ >= 0) {
        search_kdtree(query.data(), kd_root_, nprobe, best);
    } else {
        // Fallback linear
        for (int c = 0; c < num_centroid; ++c) {
            float dist = compute_distance_simd(query.data(), centroid_ptr(c));
            if (static_cast<int>(best.size()) < nprobe) {
                best.emplace(dist, c);
            } else if (dist < best.top().first) {
                best.pop();
                best.emplace(dist, c);
            }
        }
    }
    std::vector<std::pair<int, float>> result;
    result.reserve(best.size());
    while (!best.empty()) {
        result.push_back({best.top().second, best.top().first});
        best.pop();
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    if (result.size() > static_cast<size_t>(nprobe)) result.resize(nprobe);
    return result;
}

// --- Search 核心入口 (高度优化) ---

std::vector<std::pair<int, float>> solution::search(const std::vector<float>& query, int k) {
    if (point_ids_.empty() || inverted_index.empty() || k <= 0) return {};

    // 1. 粗排
    auto close_centroids = find_closest_centroids_simd(query, std::min(nprob, num_centroid));
    if (close_centroids.empty()) return {};

    std::vector<float> centroid_dists(close_centroids.size());
    for (size_t i = 0; i < close_centroids.size(); ++i) centroid_dists[i] = close_centroids[i].second;

    int threads_to_use = std::max(1, SEARCH_THREADS);
    int total_centroids = static_cast<int>(close_centroids.size());
    int chunk_size = (total_centroids + threads_to_use - 1) / threads_to_use;

    std::vector<std::future<std::vector<std::pair<float, int>>>> futures;
    futures.reserve(threads_to_use);

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total_centroids);
        if (start >= end) continue;

        futures.push_back(g_pool->enqueue([this, start, end, k, &query, &close_centroids, &centroid_dists]() {
            std::vector<std::pair<float, int>> local_top;
            local_top.reserve(k + 1);
            float current_limit = std::numeric_limits<float>::max();
            
            // 预计算 Prefetch 偏移 (j+8)
            const int prefetch_stride = 8;
            const size_t prefetch_offset_bytes = static_cast<size_t>(prefetch_stride) * dim * sizeof(float);

            for (int idx = start; idx < end; ++idx) {
                int c_id = close_centroids[idx].first;
                float d_qc = centroid_dists[idx];
                
                const auto& bucket = inverted_index[c_id];
                if (bucket.empty()) continue;

                // --- 优化：二分查找跳过头部 (利用单调性) ---
                float min_dist_pc = d_qc - current_limit;
                auto it_start = bucket.begin();
                if (min_dist_pc > 0) {
                    it_start = std::lower_bound(bucket.begin(), bucket.end(), min_dist_pc,
                        [](const BucketItem& item, float val) {
                            return item.dist_to_centroid < val;
                        });
                }

                size_t start_idx = std::distance(bucket.begin(), it_start);
                const size_t bucket_size = bucket.size();
                if (start_idx >= bucket_size) continue;

                // --- 优化：指针直接算术，避免重复 point_ptr 计算 ---
                const float* vec_ptr = point_ptr(bucket[start_idx].index);
                float max_dist_pc = d_qc + current_limit;

                for (size_t j = start_idx; j < bucket_size; ++j) {
                    // Prefetch
                    if (j + prefetch_stride < bucket_size) {
                         _mm_prefetch(reinterpret_cast<const char*>(vec_ptr) + prefetch_offset_bytes, _MM_HINT_T0);
                    }

                    float dist_pc = bucket[j].dist_to_centroid;
                    
                    // --- 优化：提前退出 (Monotonic Break) ---
                    if (dist_pc > max_dist_pc) break;

                    // 标量过滤
                    if (std::fabs(d_qc - dist_pc) >= current_limit) {
                        vec_ptr += dim;
                        continue;
                    }

                    // 精确计算
                    float exact = compute_distance_capped_simd(query.data(), vec_ptr, current_limit);
                    vec_ptr += dim;

                    // 堆维护
                    if (exact < current_limit) {
                        local_top.emplace_back(exact, bucket[j].index);
                        std::push_heap(local_top.begin(), local_top.end());
                        
                        if (local_top.size() > static_cast<size_t>(k)) {
                            std::pop_heap(local_top.begin(), local_top.end());
                            local_top.pop_back();
                            current_limit = local_top.front().first;
                            max_dist_pc = d_qc + current_limit; // 收紧 Break 条件
                        } else if (local_top.size() == static_cast<size_t>(k)) {
                             current_limit = local_top.front().first;
                             max_dist_pc = d_qc + current_limit;
                        }
                    }
                }
            }
            return local_top;
        }));
    }

    std::vector<std::pair<float, int>> all_candidates;
    for (auto& f : futures) {
        auto res = f.get();
        all_candidates.insert(all_candidates.end(), res.begin(), res.end());
    }
    if (all_candidates.empty()) return {};

    if (static_cast<int>(all_candidates.size()) > k) {
        std::partial_sort(all_candidates.begin(), all_candidates.begin() + k, all_candidates.end());
        all_candidates.resize(k);
    } else {
        std::sort(all_candidates.begin(), all_candidates.end());
    }

    std::vector<std::pair<int, float>> final_result;
    final_result.reserve(all_candidates.size());
    for (auto& cand : all_candidates) {
        final_result.push_back({point_ids_[cand.second], cand.first});
    }
    return final_result;
}

// --- 外部接口封装 ---

static solution* g_impl = nullptr;

Solution::Solution(int num_centroid, int kmean_iter, int nprob) 
    : num_centroid_(num_centroid), kmean_iter_(kmean_iter), nprob_(nprob) {}

void Solution::build(int d, const std::vector<float>& base) {
    if (d <= 0) return;
    int n = static_cast<int>(base.size()) / d;
    if (n <= 0) return;

    std::vector<std::vector<float>> data;
    data.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::vector<float> vec;
        vec.reserve(d);
        for (int j = 0; j < d; ++j) {
            vec.push_back(static_cast<float>(base[i * d + j]));
        }
        data.push_back(std::move(vec));
    }

    delete g_impl;
    g_impl = new solution("l2", num_centroid_, kmean_iter_, nprob_);
    g_impl->build_from_memory(d, std::move(data));
}

void Solution::search(const std::vector<float>& query, int* res) {
    if (!g_impl) {
        for (int i = 0; i < 10; ++i) res[i] = -1;
        return;
    }
    auto ans = g_impl->search(query, 10);
    int idx = 0;
    for (; idx < static_cast<int>(ans.size()) && idx < 10; ++idx) {
        res[idx] = ans[idx].first;
    }
    for (; idx < 10; ++idx) {
        res[idx] = -1;
    }
}