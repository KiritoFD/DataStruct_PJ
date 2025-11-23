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

constexpr int SEARCH_THREADS =16;
const bool debug = true;

// --- 简易线程池 ---
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
    auto enqueue(F&& f, Args&&... args) {
        using return_type = typename std::invoke_result_t<F, Args...>;
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

static std::unique_ptr<ThreadPool> g_pool;
static std::vector<float> g_point_centroid_dist;

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

// --- Solution 实现 ---

solution::solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob)
    : metric(metric_type), dim(0), num_threads(1), num_centroid(num_centroid),
      kmean_iter(kmean_iter), nprob(nprob), kd_root_(-1) {
    unsigned int hc = std::thread::hardware_concurrency();
    num_threads = static_cast<int>(hc > 0 ? hc : 1);
    if (debug) {
        std::cout << "[solution] threads=" << num_threads << ", centroid=" << num_centroid
                  << ", iter=" << kmean_iter << ", nprob=" << nprob << "\n";
    }
    
    static std::once_flag pool_flag;
    std::call_once(pool_flag, []() {
        g_pool = std::make_unique<ThreadPool>(std::max(1, SEARCH_THREADS));
    });
}

void solution::build_from_memory(int d, std::vector<std::vector<float>> data) {
    dim = d;
    const size_t n = data.size();
    point_ids_.resize(n);
    point_data_.assign(n * static_cast<size_t>(dim), 0.0f);
    
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

    // 4. 构建倒排索引
    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;
    std::vector<std::vector<std::vector<BucketItem>>> thread_results(
        threads_to_use, std::vector<std::vector<BucketItem>>(num_centroid));
    
    g_point_centroid_dist.assign(total, 0.0f);

    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) {
            workers.emplace_back([this, &thread_results, &assignments, start, end, t]() {
                for (int i = start; i < end; ++i) {
                    int c = assignments[i];
                    float d = compute_distance_simd(point_ptr(i), centroid_ptr(c));
                    thread_results[t][c].push_back({i, d});
                    g_point_centroid_dist[i] = d;
                }
            });
        }
    }
    for (auto& th : workers) th.join();

    // 5. 合并倒排索引并排序
    inverted_index.clear();
    inverted_index.resize(num_centroid);
    for (int c = 0; c < num_centroid; ++c) {
        size_t total_bucket = 0;
        for (int t = 0; t < threads_to_use; ++t) {
            total_bucket += thread_results[t][c].size();
        }
        if (total_bucket == 0) continue;
        
        auto& dest = inverted_index[c].items;
        dest.reserve(total_bucket);
        for (int t = 0; t < threads_to_use; ++t) {
            auto& src = thread_results[t][c];
            dest.insert(dest.end(), std::make_move_iterator(src.begin()), 
                       std::make_move_iterator(src.end()));
        }
        
        std::sort(dest.begin(), dest.end(),
                  [](const BucketItem& a, const BucketItem& b) { 
                      return a.dist_to_centroid < b.dist_to_centroid; 
                  });
        
        inverted_index[c].max_radius = dest.empty() ? 0.0f : dest.back().dist_to_centroid;
        inverted_index[c].max_radius_sq8 = static_cast<int>(inverted_index[c].max_radius * global_scale_ * global_scale_ * dim);
    }

    // 6. 物理内存重排
    {
        std::vector<int> old2new(total, -1);
        std::vector<float> new_points(static_cast<size_t>(total) * dim);
        std::vector<int> new_ids(total);
        int write = 0;
        
        for (int c = 0; c < num_centroid; ++c) {
            auto& bucket = inverted_index[c].items;
            for (auto& bi : bucket) {
                int old = bi.index;
                float* src = point_ptr(old);
                float* dst = new_points.data() + static_cast<size_t>(write) * dim;
                std::memcpy(dst, src, sizeof(float) * dim);
                
                old2new[old] = write;
                new_ids[write] = point_ids_[old];
                bi.index = write;
                write++;
            }
        }
        
        for (int old = 0; old < total; ++old) {
            if (old2new[old] == -1) {
                int nw = write++;
                float* src = point_ptr(old);
                float* dst = new_points.data() + static_cast<size_t>(nw) * dim;
                std::memcpy(dst, src, sizeof(float) * dim);
                old2new[old] = nw;
                new_ids[nw] = point_ids_[old];
            }
        }
        point_data_.swap(new_points);
        point_ids_.swap(new_ids);
        
        if (!g_point_centroid_dist.empty()) {
            std::vector<float> new_dist(total);
            for (int old = 0; old < total; ++old) {
                int nw = old2new[old];
                if (nw >= 0 && nw < total) new_dist[nw] = g_point_centroid_dist[old];
            }
            g_point_centroid_dist.swap(new_dist);
        }
    }

    // 7. SQ8 Quantization
    {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (float v : point_data_) {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        global_min_ = min_val;
        global_scale_ = 255.0f / (max_val - min_val + 1e-9f);
        
        padded_dim = (dim + 31) / 32 * 32; // Align to 32 bytes
        quantized_point_data_.resize(static_cast<size_t>(total) * padded_dim, 0);
        
        int threads_to_use = std::min(num_threads, std::max(1, total));
        int chunk_size = (total + threads_to_use - 1) / threads_to_use;
        std::vector<std::thread> q_workers;
        for(int t=0; t<threads_to_use; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, total);
            if(start < end) {
                q_workers.emplace_back([this, start, end]() {
                    for(int i=start; i<end; ++i) {
                        const float* src = point_ptr(i);
                        uint8_t* dst = const_cast<uint8_t*>(quantized_ptr(i));
                        for(int j=0; j<dim; ++j) {
                            float val = (src[j] - global_min_) * global_scale_;
                            dst[j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
                        }
                    }
                });
            }
        }
        for(auto& w : q_workers) w.join();
    }

    if (debug) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[finalize_build] " << ms << " ms\n";
    }
}

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

    std::vector<std::vector<float>> thread_sums(threads_to_use, 
        std::vector<float>(static_cast<size_t>(num_centroid) * dim, 0.0f));
    std::vector<std::vector<int>> thread_counts(threads_to_use, 
        std::vector<int>(num_centroid, 0));

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
                    counts[c]++;
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

float solution::compute_distance_simd(const float* a, const float* b) const {
#if defined(__AVX512F)
    if (dim >= 16) {
        __m512 sum512 = _mm512_setzero_ps();
        int i = 0;
        for (; i <= dim - 16; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(diff, diff));
        }
        float total = _mm512_reduce_add_ps(sum512);
        for (; i <= dim - 8; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, _mm256_mul_ps(diff, diff));
            for (int k = 0; k < 8; ++k) total += tmp[k];
        }
        for (; i < dim; ++i) {
            float d = a[i] - b[i];
            total += d * d;
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
        sumv = _mm256_add_ps(sumv, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sumv);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

int solution::compute_distance_sq8(const uint8_t* a, const uint8_t* b, int limit) const {
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < padded_dim; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
        
        __m128i va_lo = _mm256_castsi256_si128(va);
        __m128i va_hi = _mm256_extracti128_si256(va, 1);
        __m128i vb_lo = _mm256_castsi256_si128(vb);
        __m128i vb_hi = _mm256_extracti128_si256(vb, 1);

        __m256i va_16_lo = _mm256_cvtepu8_epi16(va_lo);
        __m256i vb_16_lo = _mm256_cvtepu8_epi16(vb_lo);
        __m256i diff_lo = _mm256_sub_epi16(va_16_lo, vb_16_lo);
        __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        sum = _mm256_add_epi32(sum, sq_lo);

        __m256i va_16_hi = _mm256_cvtepu8_epi16(va_hi);
        __m256i vb_16_hi = _mm256_cvtepu8_epi16(vb_hi);
        __m256i diff_hi = _mm256_sub_epi16(va_16_hi, vb_16_hi);
        __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);
        sum = _mm256_add_epi32(sum, sq_hi);

        // Early break check
        __m128i sum128_partial = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        __m128i tmp1_partial = _mm_hadd_epi32(sum128_partial, sum128_partial);
        __m128i tmp2_partial = _mm_hadd_epi32(tmp1_partial, tmp1_partial);
        int partial = _mm_cvtsi128_si32(tmp2_partial);
        if (partial > limit) return std::numeric_limits<int>::max();
    }
    
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    __m128i tmp1 = _mm_hadd_epi32(sum128, sum128);
    __m128i tmp2 = _mm_hadd_epi32(tmp1, tmp1);
    return _mm_cvtsi128_si32(tmp2);
}

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

std::vector<std::pair<int, float>> solution::search(const std::vector<float>& query, int k) {
    auto t_search_start = std::chrono::high_resolution_clock::now();
    
    if (point_ids_.empty() || inverted_index.empty() || k <= 0) return {};

    // Phase 1: 粗排 - 找最近的质心
    auto t_coarse_start = std::chrono::high_resolution_clock::now();
    auto close_centroids = find_closest_centroids_simd(query, std::min(nprob, num_centroid));
    auto t_coarse_end = std::chrono::high_resolution_clock::now();
    
    if (close_centroids.empty()) return {};

    int total_centroids = static_cast<int>(close_centroids.size());

    // Quantize Query
    std::vector<uint8_t> q_quant(padded_dim, 0);
    for(int j=0; j<dim; ++j) {
        float val = (query[j] - global_min_) * global_scale_;
        q_quant[j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
    }

    std::vector<int> centroid_sq8_dists(total_centroids);
    for(int idx=0; idx<total_centroids; ++idx){
        int c_id = close_centroids[idx].first;
        std::vector<uint8_t> c_quant(padded_dim, 0);
        for(int j=0; j<dim; ++j){
            float val = (centroid_ptr(c_id)[j] - global_min_) * global_scale_;
            c_quant[j] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, val)));
        }
        centroid_sq8_dists[idx] = compute_distance_sq8(q_quant.data(), c_quant.data());
    }

    unsigned int hw_threads = std::thread::hardware_concurrency();
    int threads_to_use = std::max(1u, std::min(hw_threads, static_cast<unsigned int>(total_centroids)));
    int buckets_per_thread = (total_centroids + threads_to_use - 1) / threads_to_use;

    // Use a larger K for refinement (SQ8 -> Float)
    int k_refine = std::max(k * 20, 200);

    struct ThreadResult {
        std::vector<std::pair<int, int>> local_candidates; // <sq8_dist, index>
        int checked = 0;
    };
    std::vector<ThreadResult> results(threads_to_use);

    // Phase 2: Fine Search (SQ8)
    auto t_fine_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> workers;
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * buckets_per_thread;
        int end = std::min(start + buckets_per_thread, total_centroids);
        if (start >= end) continue;

        workers.emplace_back([this, start, end, k_refine, &q_quant, &close_centroids, &results, t, &centroid_sq8_dists]() {
            auto& res = results[t];
            std::vector<std::pair<int, int>> heap;
            heap.reserve(k_refine + 1);
            int limit = std::numeric_limits<int>::max();
            bool has_k = false;

            for (int idx = start; idx < end; ++idx) {
                int c_id = close_centroids[idx].first;
                const auto& bucket = inverted_index[c_id];
                
                // Bucket-level early prune
                if (centroid_sq8_dists[idx] - bucket.max_radius_sq8 > limit) continue;
                
                for (const auto& item : bucket.items) {
                    res.checked++;
                    int dist_sq8 = compute_distance_sq8(q_quant.data(), quantized_ptr(item.index), limit);

                    if (!has_k) {
                        heap.emplace_back(dist_sq8, item.index);
                        if (static_cast<int>(heap.size()) == k_refine) {
                            std::make_heap(heap.begin(), heap.end());
                            limit = heap.front().first;
                            has_k = true;
                        }
                    } else if (dist_sq8 < limit) {
                        std::pop_heap(heap.begin(), heap.end());
                        heap.back() = {dist_sq8, item.index};
                        std::push_heap(heap.begin(), heap.end());
                        limit = heap.front().first;
                    }
                }
            }
            res.local_candidates = std::move(heap);
        });
    }
    
    for (auto& w : workers) w.join();
    auto t_fine_end = std::chrono::high_resolution_clock::now();

    // Phase 3: Merge & Refine
    auto t_merge_start = std::chrono::high_resolution_clock::now();
    std::vector<int> candidate_indices;
    candidate_indices.reserve(threads_to_use * k_refine);
    int total_checked = 0;
    for (const auto& res : results) {
        total_checked += res.checked;
        for (const auto& p : res.local_candidates) {
            candidate_indices.push_back(p.second);
        }
    }
    
    // Refine: Compute float distance for candidates
    std::vector<std::pair<float, int>> refined_results;
    refined_results.reserve(candidate_indices.size());
    
    for (int idx : candidate_indices) {
        float d = compute_distance_simd(query.data(), point_ptr(idx));
        refined_results.push_back({d, idx});
    }
    auto t_merge_end = std::chrono::high_resolution_clock::now();

    if (refined_results.empty()) return {};

    // Phase 4: 最终排序
    auto t_sort_start = std::chrono::high_resolution_clock::now();
    if (static_cast<int>(refined_results.size()) > k) {
        std::partial_sort(refined_results.begin(), refined_results.begin() + k, refined_results.end());
        refined_results.resize(k);
    } else {
        std::sort(refined_results.begin(), refined_results.end());
    }
    auto t_sort_end = std::chrono::high_resolution_clock::now();

    // Phase 5: ID 映射
    auto t_map_start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, float>> final;
    final.reserve(refined_results.size());
    for (const auto& c : refined_results) {
        final.push_back({point_ids_[c.second], c.first});
    }
    auto t_map_end = std::chrono::high_resolution_clock::now();
    
    auto t_search_end = std::chrono::high_resolution_clock::now();

    if (debug) {
        auto us_coarse = std::chrono::duration_cast<std::chrono::microseconds>(t_coarse_end - t_coarse_start).count();
        auto us_fine = std::chrono::duration_cast<std::chrono::microseconds>(t_fine_end - t_fine_start).count();
        auto us_merge = std::chrono::duration_cast<std::chrono::microseconds>(t_merge_end - t_merge_start).count();
        auto us_sort = std::chrono::duration_cast<std::chrono::microseconds>(t_sort_end - t_sort_start).count();
        auto us_map = std::chrono::duration_cast<std::chrono::microseconds>(t_map_end - t_map_start).count();
        auto us_total = std::chrono::duration_cast<std::chrono::microseconds>(t_search_end - t_search_start).count();
        
        std::cout << "[Timing] coarse=" << us_coarse << "us fine=" << us_fine << "us merge=" << us_merge 
                  << "us sort=" << us_sort << "us map=" << us_map << "us TOTAL=" << us_total << "us\n";
        std::cout << "[Stats] checked=" << total_checked << " refined=" << candidate_indices.size() << "\n";
    }

    return final;
}

// 外部接口
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