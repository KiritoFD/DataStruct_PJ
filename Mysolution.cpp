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
#include <atomic>
#include <chrono>

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
              << ", dim=" << dim << ", centroids=" << NUM_CENTROIDS << '\n';
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(database.size()) - 1);

    centroids.clear();
    centroids.reserve(NUM_CENTROIDS);
    for (int i = 0; i < NUM_CENTROIDS; ++i) {
        centroids.push_back(database[dist(rng)].vec);
    }

    std::vector<int> assignments(database.size());
    for (int iter = 0; iter < KMEANS_ITER; ++iter) {
        kmeans_assign_parallel(assignments);
        std::vector<std::vector<double>> new_centroids(NUM_CENTROIDS, std::vector<double>(dim, 0.0));
        kmeans_update_parallel(assignments, new_centroids);
        for (int i = 0; i < NUM_CENTROIDS; ++i) {
            centroids[i] = std::move(new_centroids[i]);
        }
        std::cout << "[solution] kmeans iter " << (iter + 1)
                  << "/" << KMEANS_ITER << " done\n";
    }

    // Replaced: sequential inverted_index build -> parallel with watchdog and timeout
    {
        std::cout << "[solution] building inverted index (parallel) ..." << std::endl;
        auto idx_start = std::chrono::high_resolution_clock::now();

        int dbsize = static_cast<int>(database.size());
        if (dbsize == 0) {
            std::cout << "[solution] empty database, skipped inverted index build\n";
        } else {
            // Limit threads to avoid oversubscription; choose up to 16 or num_threads
            int threads_to_use = std::min(num_threads, std::min(16, dbsize));
            int chunk = (dbsize + threads_to_use - 1) / threads_to_use;

            // per-thread per-centroid local buckets to avoid locking during writes
            std::vector<std::vector<std::vector<int>>> local_buckets(
                threads_to_use, std::vector<std::vector<int>>(NUM_CENTROIDS)
            );

            std::atomic<int> processed{0};
            std::atomic<bool> stop_flag{false};
            const int WATCHDOG_INTERVAL_SEC = 5;
            const int MAX_WAIT_SEC = 300; // timeout to avoid indefinite hang
            std::vector<std::thread> threads;
            threads.reserve(threads_to_use);

            // worker threads
            for (int t = 0; t < threads_to_use; ++t) {
                int start = t * chunk;
                int end = std::min(start + chunk, dbsize);
                if (start >= end) continue;
                threads.emplace_back([this, &local_buckets, t, start, end, &processed, dbsize, &stop_flag]() {
                    std::cout << "[solution] worker " << t << " start range [" << start << "," << end << ")\n";
                    int local_count = 0;
                    for (int i = start; i < end; ++i) {
                        if (stop_flag.load(std::memory_order_relaxed)) break;
                        int c = find_closest_centroid(database[i].vec);
                        local_buckets[t][c].push_back(database[i].id);
                        ++local_count;
                        int cur = ++processed;
                        // coarse-grained progress logging
                        if ((cur % 500000) == 0) {
                            std::cout << "[solution] inverted_index progress: " << cur << "/" << dbsize << std::endl;
                        }
                        // occasionally yield to allow watchdog/other threads to run
                        if ((local_count & 0x7FF) == 0) std::this_thread::yield();
                    }
                    std::cout << "[solution] worker " << t << " finished (processed " << local_count << ")\n";
                });
            }

            // watchdog thread monitors progress and enforces timeout
            std::atomic<bool> watchdog_stopped{false};
            // capture WATCHDOG_INTERVAL_SEC and MAX_WAIT_SEC by value as they are local consts
            std::thread watchdog([&processed, dbsize, &stop_flag, &watchdog_stopped, idx_start, WATCHDOG_INTERVAL_SEC, MAX_WAIT_SEC]() {
                int last = 0;
                int elapsed = 0;
                while (true) {
                    std::this_thread::sleep_for(std::chrono::seconds(WATCHDOG_INTERVAL_SEC));
                    int cur = processed.load();
                    auto now = std::chrono::high_resolution_clock::now();
                    elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - idx_start).count());
                    std::cout << "[solution][watchdog] progress " << cur << "/" << dbsize
                              << ", elapsed=" << elapsed << "s\n";
                    if (cur >= dbsize) break; // completed
                    if (elapsed > MAX_WAIT_SEC) {
                        std::cout << "[solution][watchdog][WARN] inverted_index build exceeded " << MAX_WAIT_SEC
                                  << "s, requesting stop\n";
                        stop_flag.store(true);
                        break;
                    }
                    if (cur == last && cur > 0 && elapsed > WATCHDOG_INTERVAL_SEC * 6) {
                        // no progress for a while; warn
                        std::cout << "[solution][watchdog][WARN] no progress detected for " 
                                  << (elapsed) << "s, continuing to monitor\n";
                    }
                    last = cur;
                }
                watchdog_stopped.store(true);
            });

            // join workers
            for (auto &th : threads) {
                if (th.joinable()) th.join();
            }

            // ensure watchdog stops
            if (!watchdog_stopped.load()) {
                // if build finished early, watchdog will exit; otherwise give it a chance then join
                stop_flag.store(true);
            }
            if (watchdog.joinable()) watchdog.join();

            // merge local buckets into global inverted_index (may be partial if stopped)
            inverted_index.clear();
            for (int c = 0; c < NUM_CENTROIDS; ++c) {
                size_t total_size = 0;
                for (int t = 0; t < threads_to_use; ++t) total_size += local_buckets[t][c].size();
                if (total_size == 0) continue;
                std::vector<int> merged;
                merged.reserve(total_size);
                for (int t = 0; t < threads_to_use; ++t) {
                    auto &v = local_buckets[t][c];
                    if (!v.empty()) merged.insert(merged.end(), v.begin(), v.end());
                }
                inverted_index[c] = std::move(merged);
            }
        }

        auto idx_end = std::chrono::high_resolution_clock::now();
        auto idx_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(idx_end - idx_start).count();
        // count non-empty buckets
        size_t non_empty = 0;
        size_t total_ids = 0;
        for (const auto &p : inverted_index) {
            if (!p.second.empty()) ++non_empty;
            total_ids += p.second.size();
        }
        std::cout << "[solution] inverted index built: non-empty buckets=" << non_empty
                  << ", total entries=" << total_ids
                  << ", time=" << (idx_time_ms / 1000.0) << "s\n";
    }
}

void solution::kmeans_assign_parallel(std::vector<int>& assignments) {
	// safer int conversions for thread counts
	int dbsize = static_cast<int>(database.size());
	int threads_to_use = std::min(num_threads, std::max(1, dbsize));
	int chunk_size = (dbsize + threads_to_use - 1) / threads_to_use;
	std::vector<std::thread> threads;
	threads.reserve(threads_to_use);

	const auto warn_threshold = std::chrono::seconds(30);
	auto worker = [this, &assignments, dbsize, &warn_threshold](int start, int end) {
		auto tstart = std::chrono::high_resolution_clock::now();
		int counter = 0;
		for (int i = start; i < end; ++i) {
			assignments[i] = find_closest_centroid(database[i].vec);
			++counter;
			if ((counter & 0x3FF) == 0) { // every 1024 items check elapsed
				auto now = std::chrono::high_resolution_clock::now();
				if (now - tstart > warn_threshold) {
					std::cout << "[solution][WARN] kmeans_assign worker processing items[" << start
							  << "," << end << ") has been running > " << warn_threshold.count() << "s\n";
					tstart = now; // throttle further logs
				}
			}
		}
	};

	for (int t = 0; t < threads_to_use; ++t) {
		int start = t * chunk_size;
		int end = std::min(start + chunk_size, dbsize);
		if (start < end) threads.emplace_back(worker, start, end);
	}
	for (auto& th : threads) th.join();
}

void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<double>>& new_centroids) {
    // 线程安全的累加
    std::vector<std::mutex> mutexes(NUM_CENTROIDS);
    std::vector<int> counts(NUM_CENTROIDS, 0);

    int dbsize = static_cast<int>(database.size());
    int threads_to_use = std::min(num_threads, std::max(1, dbsize));
    int chunk_size = (dbsize + threads_to_use - 1) / threads_to_use;
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);

    const auto warn_threshold = std::chrono::seconds(30);
    auto worker = [this, &assignments, &new_centroids, &mutexes, &counts, dbsize, &warn_threshold](int start, int end) {
        auto tstart = std::chrono::high_resolution_clock::now();
        int local_counter = 0;
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            {
                std::lock_guard<std::mutex> lock(mutexes[c]);
                for (int d = 0; d < dim; ++d) {
                    new_centroids[c][d] += database[i].vec[d];
                }
                counts[c]++;
            }
            ++local_counter;
            if ((local_counter & 0x3FF) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                if (now - tstart > warn_threshold) {
                    std::cout << "[solution][WARN] kmeans_update worker processed " << local_counter
                              << " items in range [" << start << "," << end << ") > " << warn_threshold.count() << "s\n";
                    tstart = now;
                }
            }
        }
    };

    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, dbsize);
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();

    // 归一化中心
    for (int i = 0; i < NUM_CENTROIDS; ++i) {
        if (counts[i] > 0) {
            for (int d = 0; d < dim; ++d) {
                new_centroids[i][d] /= counts[i];
            }
        }
    }

    // consistency check: counts sum should equal dbsize
    long long sum_counts = 0;
    for (int c : counts) sum_counts += c;
    if (sum_counts != dbsize) {
        std::cout << "[solution][WARN] kmeans_update counts sum (" << sum_counts
                  << ") != database size (" << dbsize << ")\n";
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
    if (metric == "l2") {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } else if (metric == "cosine") {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0.0 || nb == 0.0) return 1.0;
        return 1.0 - dot / (std::sqrt(na) * std::sqrt(nb));
    }
    return std::numeric_limits<double>::infinity();
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
	auto close_centroids = find_closest_centroids(query, NPROBE);

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
