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

solution::solution(const std::string& metric_type) : metric(metric_type), dim(0) {
	unsigned int hc = std::thread::hardware_concurrency();
	num_threads = static_cast<int>(hc > 0 ? hc : 1);
	std::cout << "[solution] hardware_concurrency = " << hc
	          << ", using " << num_threads << " threads\n";
}

void solution::build(const std::string& base_file) {
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

    build_from_memory(local_dim, std::move(vectors));
}

void solution::build_from_memory(int d, std::vector<std::vector<double>> data) {
    dim = d;
    database.clear();
    database.reserve(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        database.push_back({static_cast<int>(i), std::move(data[i])});
    }
    // 移除 storage.initialize(dim, data);
    finalize_build();
}

void solution::finalize_build() {
    if (database.empty() || dim == 0) {
        std::cout << "[solution] finalize_build skipped (empty dataset)\n";
        centroids.clear();
        inverted_index.clear();
        return;
    }
    std::cout << "[solution] finalize_build: vectors=" << database.size()
              << ", dim=" << dim << ", centroids=" << NUM_CENTROID << '\n';
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(database.size()) - 1);

    centroids.clear();
    centroids.reserve(NUM_CENTROID);
    for (int i = 0; i < NUM_CENTROID; ++i) {
        centroids.push_back(database[dist(rng)].vec);
    }

    std::vector<int> assignments(database.size());
    for (int iter = 0; iter < KMEAN_ITER; ++iter) {
        kmeans_assign_parallel(assignments);
        std::vector<std::vector<double>> new_centroids(NUM_CENTROID, std::vector<double>(dim, 0.0));
        kmeans_update_parallel(assignments, new_centroids);
        for (int i = 0; i < NUM_CENTROID; ++i) {
            centroids[i] = std::move(new_centroids[i]);
        }
        std::cout << "[solution] kmeans iter " << (iter + 1)
                  << "/" << KMEAN_ITER << " done\n";
    }

    inverted_index.clear(); // 替换 resize
    for (const auto& dp : database) {
        int c = find_closest_centroid(dp.vec);
        inverted_index[c].push_back(dp.id);
    }
    std::cout << "[solution] inverted index built with "
              << inverted_index.size() << " buckets\n";
}

void solution::kmeans_assign_parallel(std::vector<int>& assignments) {
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
}

void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<double>>& new_centroids) {
    std::vector<std::mutex> mutexes(NUM_CENTROID);
    std::vector<int> counts(NUM_CENTROID, 0);

    int threads_to_use = std::min<int>(num_threads, std::max<size_t>(1, database.size()));
    int chunk_size = (static_cast<int>(database.size()) + threads_to_use - 1) / threads_to_use;
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    auto worker = [this, &assignments, &new_centroids, &mutexes, &counts](int start, int end) {
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            {
                std::lock_guard<std::mutex> lock(mutexes[c]);
                for (int d = 0; d < dim; ++d) {
                    new_centroids[c][d] += database[i].vec[d];
                }
                counts[c]++;
            }
        }
    };

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(database.size()));
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();

    for (int i = 0; i < NUM_CENTROID; ++i) {
        if (counts[i] > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[i][d] /= counts[i];
            }
        }
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

std::vector<std::pair<int, double>> solution::find_closest_centroids(const std::vector<double>& query, int nprobe) const {
    std::priority_queue<std::pair<double, int>> pq;
    for (int i = 0; i < (int)centroids.size(); ++i) {
        double d = compute_distance(query, centroids[i]);
        pq.push({d, i});
        if ((int)pq.size() > nprobe) {
            pq.pop();
        }
    }
    std::vector<std::pair<int, double>> result;
    while (!pq.empty()) {
        result.push_back({pq.top().second, pq.top().first});
        pq.pop();
    }
    return result;
}

std::vector<std::pair<int, double>> solution::search(const std::vector<double>& query, int k) {
    auto close_centroids = find_closest_centroids(query, NPROB);

    std::vector<std::pair<int, double>> all_candidates;
    std::mutex candidates_mutex;

    int threads_to_use = std::min<int>(num_threads, std::max<int>(1, static_cast<int>(close_centroids.size())));
    int chunk_size = (static_cast<int>(close_centroids.size()) + threads_to_use - 1) / threads_to_use;
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    auto worker = [this, &query, &close_centroids, &all_candidates, &candidates_mutex](int start, int end) {
        std::vector<std::pair<int, double>> local_candidates;
        for (int i = start; i < end && i < (int)close_centroids.size(); ++i) {
            int c_id = close_centroids[i].first;
            auto it = inverted_index.find(c_id);
            if (it != inverted_index.end()) {
                for (int vec_id : it->second) {
                    double dist = compute_distance(query, database[vec_id].vec);
                    local_candidates.push_back({vec_id, dist});
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(candidates_mutex);
            all_candidates.insert(all_candidates.end(), local_candidates.begin(), local_candidates.end());
        }
    };

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(close_centroids.size()));
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();

    // 找 top-k
    std::priority_queue<std::pair<double, int>> topk;
    for (const auto& [vec_id, dist] : all_candidates) {
        if ((int)topk.size() < k) {
            topk.push({dist, vec_id});
        } else if (dist < topk.top().first) {
            topk.pop();
            topk.push({dist, vec_id});
        }
    }

    std::vector<std::pair<int, double>> result;
    while (!topk.empty()) {
        result.push_back({topk.top().second, topk.top().first});
        topk.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// 新增全局内部实现指针（保持索引状态）
static solution* g_impl = nullptr;
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
    g_impl = new solution("l2");
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
