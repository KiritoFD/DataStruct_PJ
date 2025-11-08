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
    for (size_t i = 0; i < data.size(); ++i) {
        database.push_back({static_cast<int>(i), std::move(data[i])});
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[build_from_memory] Data conversion time: "
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

    centroids.clear();
    centroids.reserve(num_centroid);
    for (int i = 0; i < num_centroid; ++i) {
        centroids.push_back(database[dist(rng)].vec);
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
        std::vector<std::vector<double>> new_centroids(num_centroid, std::vector<double>(dim, 0.0));
        auto t_update0 = std::chrono::high_resolution_clock::now();
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

    inverted_index.clear();
    for (const auto& dp : database) {
        int c = find_closest_centroid(dp.vec);
        inverted_index[c].push_back(dp.id);
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
	int threads_to_use = std::min<int>(num_threads, std::max<size_t>(1, database.size()));
	int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;
	std::vector<std::thread> threads;
	threads.reserve(threads_to_use);

	auto worker = [this, &assignments](int start, int end) {
		for (int i = start; i < end; ++i) {
			assignments[i] = find_closest_centroid(database[i].vec);
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

void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<double>>& new_centroids) {
    auto t0 = std::chrono::high_resolution_clock::now();
    // 使用线程局部变量避免锁竞争
    std::vector<std::vector<std::vector<double>>> thread_sums(num_threads);
    std::vector<std::vector<int>> thread_counts(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        thread_sums[t].assign(num_centroid, std::vector<double>(dim, 0.0));
        thread_counts[t].assign(num_centroid, 0);
    }
    
    int threads_to_use = std::min<int>(num_threads, std::max<size_t>(1, database.size()));
    int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;
    
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    
    auto worker = [this, &assignments, &thread_sums, &thread_counts](int start, int end, int thread_id) {
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            for (int d = 0; d < dim; ++d) {
                thread_sums[thread_id][c][d] += database[i].vec[d];
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
    
    // 合并线程结果
    for (int c = 0; c < num_centroid; ++c) {
        for (int d = 0; d < dim; ++d) {
            new_centroids[c][d] = 0.0;
        }
        int total_count = 0;
        
        for (int t = 0; t < num_threads; ++t) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] += thread_sums[t][c][d];
            }
            total_count += thread_counts[t][c];
        }
        
        if (total_count > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] /= total_count;
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
int solution::find_closest_centroid(const std::vector<double>& vec) const {
    double min_dist = std::numeric_limits<double>::max();
    int best_idx = 0;
    for (int i = 0; i < (int)centroids.size(); ++i) {
        double d = compute_distance(vec, centroids[i]);
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

std::vector<std::pair<int, double>> solution::find_closest_centroids(const std::vector<double>& query, int nprobe) const {
    std::vector<std::pair<double, int>> distances;
    distances.reserve(centroids.size());
    
    for (int i = 0; i < centroids.size(); ++i) {
        distances.push_back({compute_distance(query, centroids[i]), i});
    }
    
    // 只对前nprobe个元素进行部分排序
    if (nprobe >= distances.size()) {
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

std::vector<std::pair<int, double>> solution::search(const std::vector<double>& query, int k) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto close_centroids = find_closest_centroids(query, nprob); // 修正 nprobe -> nprob
    auto t1 = std::chrono::high_resolution_clock::now();
    // 使用线程局部变量收集候选
    std::vector<std::vector<std::pair<int, double>>> thread_candidates(num_threads);
    int threads_to_use = std::min<int>(num_threads, std::max<int>(1, static_cast<int>(close_centroids.size())));
    int chunk_size = (static_cast<int>(close_centroids.size()) + threads_to_use - 1) / threads_to_use;
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    auto worker = [this, &query, &close_centroids, &thread_candidates](int start, int end, int thread_id) { // 捕获 close_centroids
        for (int i = start; i < end && i < close_centroids.size(); ++i) {
            int c_id = close_centroids[i].first;
            auto it = inverted_index.find(c_id);
            if (it != inverted_index.end()) {
                for (int vec_id : it->second) {
                    double dist = compute_distance(query, database[vec_id].vec);
                    thread_candidates[thread_id].push_back({vec_id, dist});
                }
            }
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(close_centroids.size()));
        if (start < end) threads.emplace_back(worker, start, end, t);
    }
    for (auto& th : threads) th.join();
    auto t2 = std::chrono::high_resolution_clock::now();
    // 合并所有候选
    std::vector<std::pair<int, double>> all_candidates;
    for (auto& tc : thread_candidates) {
        all_candidates.insert(all_candidates.end(), tc.begin(), tc.end());
    }
    // 使用部分排序找top-k
    if (k >= all_candidates.size()) {
        std::sort(all_candidates.begin(), all_candidates.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    } else {
        std::partial_sort(all_candidates.begin(), all_candidates.begin() + k, all_candidates.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        all_candidates.resize(k);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    return all_candidates;
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

	// 转换查询向量为 double，并调用现有搜索接口（返回 id,dist 列表）
	std::vector<double> q(query.begin(), query.end());
	auto ans = g_impl->search(q, 10);

	// 将前 10 个 id 填入 res，不足处填 -1
	int idx = 0;
	for (; idx < (int)ans.size() && idx < 10; ++idx) {
		res[idx] = ans[idx].first;
	}
	for (; idx < 10; ++idx) res[idx] = -1;
}
