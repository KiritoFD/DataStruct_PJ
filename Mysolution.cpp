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
constexpr int SEARCH_THREADS = 8;
const bool debug = false;
constexpr bool USE_QUANTIZATION = true;
constexpr bool USE_RESIDUALS = true;
constexpr int PREFETCH_DIST_VEC = 16;
constexpr int PREFETCH_DIST_SCALAR = 64;

// --- 量化数据结构实现 ---
void QuantizedData::quantize(const std::vector<float>& data, int n, int d) {
    codes.resize(n * d);
    scales.resize(d);
    mins.resize(d);
    
    // 统计每维 min/max
    for (int j = 0; j < d; ++j) {
        float minv = std::numeric_limits<float>::max();
        float maxv = std::numeric_limits<float>::lowest();
        for (int i = 0; i < n; ++i) {
            float val = data[i * d + j];
            minv = std::min(minv, val);
            maxv = std::max(maxv, val);
        }
        mins[j] = minv;
        float range = maxv - minv;
        scales[j] = (range > 1e-8f) ? (range / 255.0f) : 1.0f;
    }
    
    // 编码
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            float val = data[i * d + j];
            int code = static_cast<int>((val - mins[j]) / scales[j]);
            codes[i * d + j] = static_cast<uint8_t>(std::clamp(code, 0, 255));
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float QuantizedData::compute_distance_dequant_avx2(const float* query, int idx, int dim) const {
    const uint8_t* code_ptr = codes.data() + idx * dim;
    __m256 sum = _mm256_setzero_ps();
    
    int i = 0;
    for (; i <= dim - 8; i += 8) {
        __m128i codes_i8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code_ptr + i));
        __m256i codes_i32 = _mm256_cvtepu8_epi32(codes_i8);
        __m256 codes_f32 = _mm256_cvtepi32_ps(codes_i32);
        
        __m256 scale = _mm256_loadu_ps(scales.data() + i);
        __m256 minv = _mm256_loadu_ps(mins.data() + i);
        __m256 decoded = _mm256_fmadd_ps(codes_f32, scale, minv);
        
        __m256 q = _mm256_loadu_ps(query + i);
        __m256 diff = _mm256_sub_ps(q, decoded);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sum);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    for (; i < dim; ++i) {
        float decoded = codes.data()[idx * dim + i] * scales[i] + mins[i];
        float diff = query[i] - decoded;
        total += diff * diff;
    }
    return total;
}

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

    // 1. 初始化质心（不变）
    centroid_data_.assign(static_cast<size_t>(num_centroid) * dim, 0.0f);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, total - 1);
    for (int i = 0; i < num_centroid; ++i) {
        std::memcpy(centroid_ptr(i), point_ptr(dist(rng)), sizeof(float) * dim);
    }

    // 2. K-Means（不变）
    std::vector<int> assignments(total, 0);
    for (int iter = 0; iter < kmean_iter; ++iter) {
        kmeans_assign_parallel(assignments);
        std::vector<float> new_centroids(static_cast<size_t>(num_centroid) * dim, 0.0f);
        kmeans_update_parallel(assignments, new_centroids);
        centroid_data_.swap(new_centroids);
    }

    // 3. 构建倒排（SoA + 残差）
    std::vector<std::vector<int>> temp_buckets(num_centroid);
    std::vector<std::vector<float>> temp_dists(num_centroid);
    std::vector<std::vector<float>> temp_residuals; // 若使用残差
    
    if (USE_RESIDUALS) {
        temp_residuals.resize(num_centroid);
    }

    for (int i = 0; i < total; ++i) {
        int c = assignments[i];
        float dist = compute_distance_simd(point_ptr(i), centroid_ptr(c));
        temp_buckets[c].push_back(i);
        temp_dists[c].push_back(dist);
        
        if (USE_RESIDUALS) {
            // 计算残差向量：point - centroid
            std::vector<float> residual(dim);
            const float* pt = point_ptr(i);
            const float* ct = centroid_ptr(c);
            for (int d = 0; d < dim; ++d) {
                residual[d] = pt[d] - ct[d];
            }
            temp_residuals[c].insert(temp_residuals[c].end(), residual.begin(), residual.end());
        }
    }

    // 4. 按距离排序并重排
    std::vector<int> new_order;
    new_order.reserve(total);
    std::vector<int> new_ids;
    new_ids.reserve(total);
    std::vector<float> new_data_flat;
    new_data_flat.reserve(total * dim);

    compact_inverted_index.clear();
    compact_inverted_index.resize(num_centroid);

    for (int c = 0; c < num_centroid; ++c) {
        auto& bucket_indices = temp_buckets[c];
        auto& bucket_dists = temp_dists[c];
        if (bucket_indices.empty()) continue;

        // 排序
        std::vector<size_t> order(bucket_indices.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return bucket_dists[a] < bucket_dists[b];
        });

        CompactBucket& cb = compact_inverted_index[c];
        cb.start_offset = static_cast<uint32_t>(new_order.size());
        cb.count = static_cast<uint32_t>(bucket_indices.size());
        cb.sorted_dists.reserve(cb.count);
        cb.original_ids.reserve(cb.count);

        for (size_t idx : order) {
            int old_i = bucket_indices[idx];
            cb.sorted_dists.push_back(bucket_dists[idx]);
            cb.original_ids.push_back(point_ids_[old_i]);
            new_order.push_back(old_i);

            if (USE_RESIDUALS) {
                // 从 temp_residuals[c] 中提取残差
                const float* res_ptr = temp_residuals[c].data() + idx * dim;
                new_data_flat.insert(new_data_flat.end(), res_ptr, res_ptr + dim);
            } else {
                const float* pt = point_ptr(old_i);
                new_data_flat.insert(new_data_flat.end(), pt, pt + dim);
            }
            new_ids.push_back(point_ids_[old_i]);
        }
    }

    // 5. 量化（可选）
    if (USE_QUANTIZATION) {
        quantized_data_.quantize(new_data_flat, total, dim);
        point_data_.clear(); // 释放原始浮点数据
    } else {
        point_data_ = std::move(new_data_flat);
    }
    point_ids_ = std::move(new_ids);

    // 6. 移除 KD-Tree（小规模质心用线性扫描更快）
    kd_root_ = -1;
    kd_nodes_.clear();

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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float solution::compute_distance_simd(const float* a, const float* b) const {
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

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float solution::compute_distance_capped_simd(const float* a, const float* b, float cap) const {
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

    // 小规模质心：直接暴力 SIMD 扫描（比 KD-Tree 快）
    std::vector<std::pair<float, int>> dists;
    dists.reserve(num_centroid);
    for (int c = 0; c < num_centroid; ++c) {
        float d = compute_distance_simd(query.data(), centroid_ptr(c));
        dists.emplace_back(d, c);
    }

    std::partial_sort(dists.begin(), dists.begin() + nprobe, dists.end());
    dists.resize(nprobe);

    std::vector<std::pair<int, float>> result;
    result.reserve(nprobe);
    for (auto& p : dists) result.push_back({p.second, p.first});
    return result;
}

// --- Search 核心入口 (高度优化) ---

std::vector<std::pair<int, float>> solution::search(const std::vector<float>& query, int k) {
    if (point_ids_.empty() || compact_inverted_index.empty() || k <= 0) return {};

    auto close_centroids = find_closest_centroids_simd(query, std::min(nprob, num_centroid));
    if (close_centroids.empty()) return {};

    // 计算查询点的残差（若使用残差编码）
    std::vector<std::vector<float>> query_residuals;
    if (USE_RESIDUALS) {
        query_residuals.resize(close_centroids.size());
        for (size_t i = 0; i < close_centroids.size(); ++i) {
            int c_id = close_centroids[i].first;
            const float* ct = centroid_ptr(c_id);
            query_residuals[i].resize(dim);
            for (int d = 0; d < dim; ++d) {
                query_residuals[i][d] = query[d] - ct[d];
            }
        }
    }

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

        futures.push_back(g_pool->enqueue([this, start, end, k, &query, &query_residuals, &close_centroids, &centroid_dists]() {
            std::vector<std::pair<float, int>> local_top;
            local_top.reserve(k + 1);
            float current_limit = std::numeric_limits<float>::max();

            for (int idx = start; idx < end; ++idx) {
                int c_id = close_centroids[idx].first;
                float d_qc = centroid_dists[idx];
                const CompactBucket& bucket = compact_inverted_index[c_id];
                if (bucket.count == 0) continue;

                const float* query_ptr = USE_RESIDUALS ? query_residuals[idx].data() : query.data();

                // --- 预取标量数组（距离） ---
                const float* dist_array = bucket.sorted_dists.data();
                for (uint32_t prefetch_i = 0; prefetch_i < bucket.count; prefetch_i += PREFETCH_DIST_SCALAR) {
                    _mm_prefetch(reinterpret_cast<const char*>(dist_array + prefetch_i), _MM_HINT_T0);
                }

                // --- 二分跳过头部 ---
                float min_dist_pc = d_qc - current_limit;
                uint32_t start_j = 0;
                if (min_dist_pc > 0) {
                    auto it = std::lower_bound(bucket.sorted_dists.begin(), bucket.sorted_dists.end(), min_dist_pc);
                    start_j = static_cast<uint32_t>(std::distance(bucket.sorted_dists.begin(), it));
                }

                float max_dist_pc = d_qc + current_limit;

                for (uint32_t j = start_j; j < bucket.count; ++j) {
                    // --- 激进预取向量数据 ---
                    if (j + PREFETCH_DIST_VEC < bucket.count) {
                        uint32_t prefetch_idx = bucket.start_offset + j + PREFETCH_DIST_VEC;
                        if (USE_QUANTIZATION) {
                            _mm_prefetch(reinterpret_cast<const char*>(quantized_data_.codes.data() + prefetch_idx * dim), _MM_HINT_T0);
                        } else {
                            _mm_prefetch(reinterpret_cast<const char*>(point_data_.data() + prefetch_idx * dim), _MM_HINT_T0);
                        }
                    }

                    float dist_pc = bucket.sorted_dists[j];
                    if (dist_pc > max_dist_pc) break;
                    if (std::fabs(d_qc - dist_pc) >= current_limit) continue;

                    // --- 计算精确距离 ---
                    float exact;
                    uint32_t global_idx = bucket.start_offset + j;
                    if (USE_QUANTIZATION) {
                        exact = quantized_data_.compute_distance_dequant_avx2(query_ptr, global_idx, dim);
                    } else {
                        const float* vec_ptr = point_data_.data() + global_idx * dim;
                        exact = compute_distance_capped_simd(query_ptr, vec_ptr, current_limit);
                    }

                    // --- 堆维护 ---
                    if (exact < current_limit) {
                        local_top.emplace_back(exact, global_idx);
                        std::push_heap(local_top.begin(), local_top.end());
                        if (local_top.size() > static_cast<size_t>(k)) {
                            std::pop_heap(local_top.begin(), local_top.end());
                            local_top.pop_back();
                        }
                        if (local_top.size() == static_cast<size_t>(k)) {
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
        // 从 SoA 结构中恢复原始 ID
        uint32_t global_idx = cand.second;
        // 找到对应的桶与桶内偏移
        int bucket_id = -1;
        uint32_t local_offset = 0;
        for (int c = 0; c < num_centroid; ++c) {
            const auto& bucket = compact_inverted_index[c];
            if (global_idx >= bucket.start_offset && global_idx < bucket.start_offset + bucket.count) {
                bucket_id = c;
                local_offset = global_idx - bucket.start_offset;
                break;
            }
        }
        if (bucket_id >= 0) {
            int orig_id = compact_inverted_index[bucket_id].original_ids[local_offset];
            final_result.push_back({orig_id, cand.first});
        }
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