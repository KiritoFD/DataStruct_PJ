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

const bool debug = true;

// 新增：提供 try_stod 与 parse_vector_line 的定义
namespace {
	bool try_stod(const std::string& s, double& out) {
		try {
			size_t pos = 0;
			out = std::stod(s, &pos);
			return pos == s.size();
		} catch (...) {
			return false;
		}
	}
}

bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec) {
	out_id.clear();
	out_vec.clear();
	std::istringstream iss(line);
	std::vector<std::string> toks;
	std::string t;
	while (iss >> t) toks.push_back(t);
	if (toks.empty()) return false;

	double val = 0.0;
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




solution::solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob)
    : metric(metric_type),
      dim(0),
      num_threads(1),
      num_centroid(num_centroid),
      kmean_iter(kmean_iter),
      nprob(nprob){
    unsigned int hc = std::thread::hardware_concurrency();
    num_threads = static_cast<int>(hc > 0 ? hc : 1);
    if (debug) {
        std::cout << "[solution] hardware_concurrency=" << hc << ", using " << num_threads << " threads\n";
        std::cout << "[solution] metric=" << metric << ", num_centroid=" << num_centroid << ", kmean_iter=" << kmean_iter << ", nprob=" << nprob << "\n";
    }
}

void solution::build(const std::string& base_file) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::ifstream fin(base_file);
    if (!fin) {
        return;
    }
    std::vector<std::vector<double>> vectors;
    std::string line;
    int local_dim = 0;
    while (std::getline(fin, line)) {
        std::string id;
        std::vector<double> vec;
        if (!parse_vector_line(line, id, vec)) continue;
        if (local_dim == 0) local_dim = static_cast<int>(vec.size());
        if (vec.size() != static_cast<size_t>(local_dim)) {
            continue;
        }
        vectors.push_back(std::move(vec));
    }

    if (vectors.empty()) {
        database.clear();
        centroids.clear();
        inverted_index.clear();
        dim = 0;
        return;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[build] Data loading time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
    build_from_memory(local_dim, std::move(vectors));
}

void solution::build_from_memory(int d, std::vector<std::vector<double>> data) {
    auto t0 = std::chrono::high_resolution_clock::now();
    dim = d;
    database.clear();
    database.reserve(data.size());
    // 将 double -> float 存储以节省内存并配合 SIMD
    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<float> vec_f;
        vec_f.reserve(d);
        for (int j = 0; j < d; ++j) vec_f.push_back(static_cast<float>(data[i][j]));
        database.push_back({static_cast<int>(i), std::move(vec_f)});
    }
    // 初始化 float 型质心容器（后续 K-means 会用到）
    centroids.clear();
    centroids.resize(num_centroid, std::vector<float>(dim, 0.0f));
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[build_from_memory] Data conversion (double->float) time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
    finalize_build();
}

void solution::finalize_build() {
    auto t0 = std::chrono::high_resolution_clock::now();
    if (database.empty() || dim == 0) {
        centroids.clear();
        inverted_index.clear();
        return;
    }
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(database.size()) - 1);

    // 随机初始化质心（从数据库拷贝 float 向量）
    centroids.clear();
    centroids.reserve(num_centroid);
    for (int i = 0; i < num_centroid; ++i) {
        centroids.push_back(database[dist(rng)].second); // second == vec (float)
    }

    std::vector<int> assignments(database.size());
    for (int iter = 0; iter < kmean_iter; ++iter) {
        auto t_assign0 = std::chrono::high_resolution_clock::now();
        kmeans_assign_parallel(assignments);
        auto t_assign1 = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::cout << "[finalize_build] kmeans_assign_parallel iter " << iter
                      << " time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t_assign1 - t_assign0).count()
                      << " ms\n";
        }
        // new_centroids 使用 float
        std::vector<std::vector<float>> new_centroids(num_centroid, std::vector<float>(dim, 0.0f));
        auto t_update0 = std::chrono::high_resolution_clock::now();
        // 需要对应重载或模板支持 kmeans_update_parallel 使用 float centroids
        kmeans_update_parallel(assignments, new_centroids);
        auto t_update1 = std::chrono::high_resolution_clock::now();
        if (debug) {
            std::cout << "[finalize_build] kmeans_update_parallel iter " << iter
                      << " time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t_update1 - t_update0).count()
                      << " ms\n";
        }
        for (int i = 0; i < num_centroid; ++i) {
            centroids[i] = std::move(new_centroids[i]);
        }
    }

    // 构建倒排索引并预计算到质心的距离（使用线程局部结果避免锁）
    int threads_to_use = std::min(num_threads, static_cast<int>(database.size()));
    if (threads_to_use <= 0) threads_to_use = 1;
    int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;

    std::vector<std::vector<std::vector<BucketItem>>> thread_results(threads_to_use,
                                                                     std::vector<std::vector<BucketItem>>(num_centroid));
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);
    auto build_worker = [this, &thread_results](int start, int end, int thread_id) {
        for (int i = start; i < end; ++i) {
            // database[i].second = vec (float)
            int c = find_closest_centroid(database[i].second);
            float dist = compute_distance_simd(database[i].second, centroids[c]);
            thread_results[thread_id][c].push_back({database[i].first, dist});
        }
    };

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(database.size()));
        if (start < end) workers.emplace_back(build_worker, start, end, t);
    }
    for (auto& th : workers) th.join();

    // 合并到 inverted_index 并预分配
    inverted_index.clear();
    try { inverted_index.reserve(static_cast<size_t>(num_centroid)); } catch (...) {}
    for (int c = 0; c < num_centroid; ++c) {
        size_t total = 0;
        for (int t = 0; t < threads_to_use; ++t) total += thread_results[t][c].size();
        if (total == 0) continue;
        std::vector<BucketItem> bucket;
        bucket.reserve(total);
        for (int t = 0; t < threads_to_use; ++t) {
            auto& src = thread_results[t][c];
            for (auto& it : src) bucket.push_back(std::move(it));
            std::vector<BucketItem>().swap(src);
        }
        inverted_index[c] = std::move(bucket);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[finalize_build] Total finalize_build time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
}

void solution::kmeans_assign_parallel(std::vector<int>& assignments) {
    auto t0 = std::chrono::high_resolution_clock::now();
	// 更稳健地计算实际使用的线程数（不超过 database.size()）
	int threads_to_use = std::min(num_threads, static_cast<int>(database.size()));
	if (threads_to_use <= 0) threads_to_use = 1;
	int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;
	std::vector<std::thread> threads;
	threads.reserve(threads_to_use);

	// 注意：database 存储为 pair<int, vector<float>>，使用 .second
	auto worker = [this, &assignments](int start, int end) {
		for (int i = start; i < end; ++i) {
			assignments[i] = find_closest_centroid(database[i].second);
		}
	};

	for (int t = 0; t < threads_to_use; ++t) {
		int start = t * chunk_size;
		int end = std::min(start + chunk_size, static_cast<int>(database.size()));
		if (start < end) threads.emplace_back(worker, start, end);
	}
	for (auto& th : threads) th.join();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[kmeans_assign_parallel] time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
}

// 将 kmeans_update_parallel 改为使用 float new_centroids（与 finalize_build 中调用一致）
void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<float>>& new_centroids) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int threads_to_use = std::min(num_threads, static_cast<int>(database.size()));
    if (threads_to_use <= 0) threads_to_use = 1;
    int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;

    // 使用 float 存储线程局部和计数
    std::vector<std::vector<std::vector<float>>> thread_sums(threads_to_use);
    std::vector<std::vector<int>> thread_counts(threads_to_use);
    
    for (int t = 0; t < threads_to_use; ++t) {
        thread_sums[t].assign(num_centroid, std::vector<float>(dim, 0.0f));
        thread_counts[t].assign(num_centroid, 0);
    }
    
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    
    auto worker = [this, &assignments, &thread_sums, &thread_counts](int start, int end, int thread_id) {
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            // database[i].second 是 vector<float>
            for (int d = 0; d < dim; ++d) {
                thread_sums[thread_id][c][d] += database[i].second[d];
            }
            thread_counts[thread_id][c]++;
        }
    };
    
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(database.size()));
        if (start < end) threads.emplace_back(worker, start, end, t);
    }
    for (auto& th : threads) th.join();
    
    // 合并线程结果（只合并实际使用的线程数）
    for (int c = 0; c < num_centroid; ++c) {
        for (int d = 0; d < dim; ++d) {
            new_centroids[c][d] = 0.0f;
        }
        int total_count = 0;
        
        for (int t = 0; t < threads_to_use; ++t) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] += thread_sums[t][c][d];
            }
            total_count += thread_counts[t][c];
        }
        
        if (total_count > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] /= static_cast<float>(total_count);
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[kmeans_update_parallel] time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
}

// 新增：对 float 向量的质心查找（与 centroids 的类型一致）
int solution::find_closest_centroid(const std::vector<float>& vec) const {
    float min_dist = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (int i = 0; i < (int)centroids.size(); ++i) {
        float d = compute_distance_simd(vec, centroids[i]);
        if (d < min_dist) {
            min_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

double solution::compute_distance(const std::vector<double>& a, const std::vector<double>& b) const {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return (sum);
   
}

// 替换为基于 float 的距离计算（SIMD + fallback）
float solution::compute_distance_simd(const std::vector<float>& a, const std::vector<float>& b) const {
    if (dim < 8) return compute_distance_fallback(a, b); // 降级到普通计算
    const float* pa = a.data();
    const float* pb = b.data();

    __m256 sumv = _mm256_setzero_ps();
    int i = 0;

    // 8元素并行处理
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(pa + i);
        __m256 vb = _mm256_loadu_ps(pb + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 sq = _mm256_mul_ps(diff, diff); // 使用乘法再加法，避免 FMA 要求
        sumv = _mm256_add_ps(sumv, sq);
    }

    // 横向累加 __m256 到标量
    float tmp[8];
    _mm256_storeu_ps(tmp, sumv);
    float total = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    // 处理剩余元素
    for (; i < dim; ++i) {
        float diff = pa[i] - pb[i];
        total += diff * diff;
    }
    return total;
}
 
float solution::compute_distance_fallback(const std::vector<float>& a, const std::vector<float>& b) const {
     float sum = 0.0f;
     for (int i = 0; i < dim; ++i) {
         float diff = a[i] - b[i];
         sum += diff * diff;
     }
     return sum;
}

std::vector<std::pair<int, double>> solution::find_closest_centroids(const std::vector<double>& query, int nprobe) const {
    std::vector<std::pair<double, int>> distances;
    distances.reserve(centroids.size());

    // centroids 存储为 float；对每个质心按 double 计算距离，避免类型不匹配
    for (int i = 0; i < (int)centroids.size(); ++i) {
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = query[d] - static_cast<double>(centroids[i][d]);
            sum += diff * diff;
        }
        distances.emplace_back(sum, i);
    }

    if (nprobe >= (int)distances.size()) {
        std::sort(distances.begin(), distances.end());
    } else {
        std::partial_sort(distances.begin(), distances.begin() + nprobe, distances.end());
        distances.resize(nprobe);
    }

    std::vector<std::pair<int, double>> result;
    result.reserve(distances.size());
    for (auto& p : distances) {
        result.push_back({p.second, p.first});
    }
    return result;
}

// 新的基于 float 的 SIMD 质心搜索
std::vector<std::pair<int, float>> solution::find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const {
    std::vector<std::pair<float, int>> distances;
    distances.reserve(centroids.size());
    for (int i = 0; i < (int)centroids.size(); ++i) {
        float dist = compute_distance_simd(query, centroids[i]);
        distances.push_back({dist, i});
    }
    if (nprobe >= (int)distances.size()) {
        std::sort(distances.begin(), distances.end());
    } else {
        std::partial_sort(distances.begin(), distances.begin() + nprobe, distances.end());
        distances.resize(nprobe);
    }
    std::vector<std::pair<int, float>> result;
    result.reserve(distances.size());
    for (auto& p : distances) result.push_back({p.second, p.first});
    return result;
}

// 新的 search（float）实现：SIMD + 移除激进剪枝，保持召回率
std::vector<std::pair<int, float>> solution::search(const std::vector<float>& query, int k) {
    // 第一阶段：找到最近的质心（SIMD）
    auto close_centroids = find_closest_centroids_simd(query, nprob);

    // 并行计算所有候选距离
    std::vector<std::vector<std::pair<float, int>>> thread_candidates(num_threads);
    int threads_to_use = std::min<int>(num_threads, std::max<int>(1, static_cast<int>(close_centroids.size())));
    int chunk_size = (static_cast<int>(close_centroids.size()) + threads_to_use - 1) / threads_to_use;

    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    auto worker = [this, &query, &close_centroids, &thread_candidates](int start, int end, int thread_id) {
        for (int i = start; i < end && i < (int)close_centroids.size(); ++i) {
            int c_id = close_centroids[i].first;
            auto it = inverted_index.find(c_id);
            if (it == inverted_index.end()) continue;
            
            const auto& bucket = it->second;
            for (const auto& item : bucket) {
                // 直接计算精确距离，不进行预剪枝以保证召回率
                float dist = compute_distance_simd(query, database[item.id].second);
                thread_candidates[thread_id].push_back({dist, item.id});
            }
        }
    };

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)close_centroids.size());
        if (start < end) threads.emplace_back(worker, start, end, t);
    }
    for (auto& th : threads) th.join();

    // 合并所有候选
    std::vector<std::pair<float, int>> all_candidates;
    for (const auto& tc : thread_candidates) {
        all_candidates.insert(all_candidates.end(), tc.begin(), tc.end());
    }

    // 对所有候选进行top-k排序
    if (k >= all_candidates.size()) {
        std::sort(all_candidates.begin(), all_candidates.end());
    } else {
        std::partial_sort(all_candidates.begin(), all_candidates.begin() + k, all_candidates.end());
        all_candidates.resize(k);
    }

    std::vector<std::pair<int, float>> final_result;
    final_result.reserve(all_candidates.size());
    for (auto& p : all_candidates) {
        final_result.push_back({p.second, p.first});
    }
    return final_result;
}

// 新增全局内部实现指针（保持索引状态）
static solution* g_impl = nullptr;

Solution::Solution(int num_centroid, int kmean_iter, int nprob) 
    : num_centroid_(num_centroid), kmean_iter_(kmean_iter), nprob_(nprob) {}

void Solution::build(int d, const std::vector<float>& base) {
    if (d <= 0) return;

    int n = static_cast<int>(base.size()) / d;
    if (n <= 0) return;

    std::vector<std::vector<double>> data;
    data.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::vector<double> vec;
        vec.reserve(d);
        for (int j = 0; j < d; ++j) {
            vec.push_back(static_cast<double>(base[i * d + j]));
        }
        data.push_back(std::move(vec));
    }

    delete g_impl;
    g_impl = new solution("l2", num_centroid_, kmean_iter_, nprob_);
    g_impl->build_from_memory(d, std::move(data));
}

void Solution::search(const std::vector<float>& query, int* res) {
	// 若还未构建索引，返回 -1 填充
	if (!g_impl) {
		for (int i = 0; i < 10; ++i) res[i] = -1;
		return;
	}

	// 直接调用浮点版本的 search（query已经是float vector）
	auto ans = g_impl->search(query, 10);

	// 将前 10 个 id 填入 res，不足处填 -1
	int idx = 0;
	for (; idx < static_cast<int>(ans.size()) && idx < 10; ++idx) {
		res[idx] = ans[idx].first;
	}
	for (; idx < 10; ++idx) {
		res[idx] = -1;
	}
}