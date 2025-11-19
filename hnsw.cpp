#include "MySolution.h"
#include "CommonUtils.h"
#include "DistanceUtils.h"
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
#include <unordered_set>
#include <functional>

// --- replace previous HNSWGraph with improved lightweight implementation ---
// filepath: g:\Github\DataStruct_PJ\MySolution.cpp
struct HNSWGraph {
	// 存储指向原始向量的指针和对应的全局索引
	std::vector<const float*> vecs;
	std::vector<int> ids;
	std::vector<std::vector<int>> neighbors; // neighbor indices inside this vecs
	int M = 16;

	void add_point(const float* vec, int global_id) {
		vecs.push_back(vec);
		ids.push_back(global_id);
		neighbors.emplace_back();
	}

	bool empty() const { return vecs.empty(); }
	size_t size() const { return vecs.size(); }

	// build: create M-NN candidates then diversification + mutual links + trim
	// dist_fn(a,b) 返回 L2 distance between pointers a and b
	void build(const std::function<float(const float*, const float*)>& dist_fn, int M_, float diversification_coef = 0.7f) {
		M = M_;
		const int n = static_cast<int>(vecs.size());
		if (n <= 0) return;

		// 对每个点 i，计算与所有 j<i 的距离并选前 K (这里 K=M*2 以保证有更多候选做 diversification)
		int K = std::min(n, M * 2);
		std::vector<std::vector<std::pair<float,int>>> cand(n);
		for (int i = 0; i < n; ++i) {
			cand[i].reserve(std::max(1, std::min(i, K)));
			for (int j = 0; j < i; ++j) {
				float d = dist_fn(vecs[i], vecs[j]);
				cand[i].emplace_back(d, j);
			}
			if (!cand[i].empty()) {
				std::sort(cand[i].begin(), cand[i].end(), [](const auto& a, const auto& b){ return a.first < b.first; });
				if ((int)cand[i].size() > K) cand[i].resize(K);
			}
		}

		// diversification: 从最近到远挑选，保证候选彼此不太相似（使用 dist between candidate and selected）
		for (int i = 0; i < n; ++i) {
			if (cand[i].empty()) continue;
			std::vector<int> selected;
			for (auto &p : cand[i]) {
				int cid = p.second;
				float dij = p.first;
				bool ok = true;
				for (int s : selected) {
					float dcs = dist_fn(vecs[cid], vecs[s]);
					// 如果 candidate 与已有 selected 太近（相对于 candidate->i 的比例），则跳过
					if (dcs <= dij * diversification_coef) {
						ok = false;
						break;
					}
				}
				if (ok) {
					selected.push_back(cid);
					if ((int)selected.size() >= M) break;
				}
			}
			// 如果 diversification 选不到足够数，补齐最近的
			if ((int)selected.size() < M) {
				for (auto &p : cand[i]) {
					int cid = p.second;
					if (std::find(selected.begin(), selected.end(), cid) == selected.end()) {
						selected.push_back(cid);
						if ((int)selected.size() >= M) break;
					}
				}
			}
			// 添加双向边
			for (int nb : selected) {
				neighbors[i].push_back(nb);
				neighbors[nb].push_back(i);
			}
		}

		// 最后修剪每个节点的邻居到 <= M（按距离从近到远）
		for (int u = 0; u < n; ++u) {
			auto &nbrs = neighbors[u];
			if ((int)nbrs.size() <= M) continue;
			std::vector<std::pair<float,int>> tmp;
			tmp.reserve(nbrs.size());
			for (int v : nbrs) tmp.emplace_back(dist_fn(vecs[u], vecs[v]), v);
			std::sort(tmp.begin(), tmp.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
			nbrs.clear();
			int keep = std::min(M, (int)tmp.size());
			for (int i = 0; i < keep; ++i) nbrs.push_back(tmp[i].second);
		}
	}

	// search: 多入口（entry_count），位图 visited，best-first (min-heap) 探索
	std::vector<std::pair<int,float>> search(const std::function<float(const float*, const float*)>& dist_fn,
	                                        const float* q, int ef, int entry_count = 4) const {
		std::vector<std::pair<int,float>> res;
		const int n = static_cast<int>(vecs.size());
		if (n == 0) return res;

		// 选择 deterministic 多入口：0, n/4, n/2, 3n/4（如果 n<entry_count 则全部使用）
		std::vector<int> entries;
		if (n <= entry_count) {
			for (int i = 0; i < n; ++i) entries.push_back(i);
		} else {
			entries.push_back(0);
			entries.push_back(n / 4);
			entries.push_back(n / 2);
			entries.push_back((3 * n) / 4);
		}
		// visited bitset
		std::vector<uint8_t> visited(n, 0);
		int visited_count = 0;

		// min-heap for candidates (distance, node)
		struct MinComp { bool operator()(const std::pair<float,int>& a, const std::pair<float,int>& b) const { return a.first > b.first; } };
		std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, MinComp> cand;

		// max-heap to keep best ef results so far (distance, node)
		auto cmp_max = [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first < b.first; };
		std::priority_queue<std::pair<float,int>, std::vector<std::pair<float,int>>, decltype(cmp_max)> topk(cmp_max);

		// push entries
		for (int e : entries) {
			if (!visited[e]) {
				float de = dist_fn(q, vecs[e]);
				cand.emplace(de, e);
				visited[e] = 1;
				++visited_count;
				topk.emplace(de, e);
				if ((int)topk.size() > ef) topk.pop();
			}
		}

		// best-first exploration until visited_count >= ef or cand empty
		while (!cand.empty() && visited_count < ef) {
			auto cur = cand.top(); cand.pop();
			int u = cur.second;
			// 拉取邻居
			for (int v : neighbors[u]) {
				if (visited[v]) continue;
				visited[v] = 1;
				++visited_count;
				float dv = dist_fn(q, vecs[v]);
				cand.emplace(dv, v);
				if ((int)topk.size() < ef) {
					topk.emplace(dv, v);
				} else if (dv < topk.top().first) {
					topk.pop();
					topk.emplace(dv, v);
				}
			}
		}

		// extract topk -> convert to (global_id, distance)
		while (!topk.empty()) {
			auto p = topk.top(); topk.pop();
			res.emplace_back(ids[p.second], p.first);
		}
		std::sort(res.begin(), res.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
		return res;
	}
};

// 全局（针对当前 cpp 内使用的）HNSW Graph 列表，与 inverted_index 对应（size = num_centroid）
static std::vector<HNSWGraph> g_hnsw_graphs;

// 新增全局实现指针（保持索引状态）
static solution* g_impl = nullptr;

const bool debug = false;

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
        if (!common::parse_vector_line(line, id, vec)) continue;
        if (local_dim == 0) local_dim = static_cast<int>(vec.size());
        if (vec.size() != static_cast<size_t>(local_dim)) {
            continue;
        }
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
    if (debug) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[build_from_memory] SoA conversion time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
    finalize_build();
}

void solution::finalize_build() {
    auto t0 = std::chrono::high_resolution_clock::now();
    const int total = static_cast<int>(point_ids_.size());
    if (total <= 0 || dim == 0) {
        centroid_data_.clear();
        inverted_index.clear();
        kd_nodes_.clear();
        kd_root_ = -1;
        return;
    }

    centroid_data_.assign(static_cast<size_t>(num_centroid) * dim, 0.0f);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, total - 1);
    for (int i = 0; i < num_centroid; ++i) {
        std::memcpy(centroid_ptr(i), point_ptr(dist(rng)), sizeof(float) * dim);
    }

    std::vector<int> assignments(total, 0);
    for (int iter = 0; iter < kmean_iter; ++iter) {
        auto t_assign0 = std::chrono::high_resolution_clock::now();
        kmeans_assign_parallel(assignments);
        auto t_assign1 = std::chrono::high_resolution_clock::now();

        std::vector<float> new_centroids(static_cast<size_t>(num_centroid) * dim, 0.0f);
        auto t_update0 = std::chrono::high_resolution_clock::now();
        kmeans_update_parallel(assignments, new_centroids);
        auto t_update1 = std::chrono::high_resolution_clock::now();

        centroid_data_.swap(new_centroids);
        if (debug) {
            std::cout << "[finalize_build] iter " << iter
                      << " assign=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_assign1 - t_assign0).count()
                      << " ms, update="
                      << std::chrono::duration_cast<std::chrono::milliseconds>(t_update1 - t_update0).count()
                      << " ms\n";
        }
    }

    kd_nodes_.clear();
    if (num_centroid > 0) {
        std::vector<int> ids(num_centroid);
        std::iota(ids.begin(), ids.end(), 0);
        kd_root_ = build_kdtree(ids, 0, num_centroid, 0);
    } else {
        kd_root_ = -1;
    }

    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;
    std::vector<std::vector<std::vector<BucketItem>>> thread_results(
        threads_to_use, std::vector<std::vector<BucketItem>>(num_centroid));
    std::vector<std::thread> workers;
    workers.reserve(threads_to_use);
    auto worker = [this, &thread_results](int start, int end, int tid) {
        for (int i = start; i < end; ++i) {
            const float* vec = point_ptr(i);
            int c = find_closest_centroid_linear(vec);
            float dist = compute_distance_simd(vec, centroid_ptr(c));
            thread_results[tid][c].push_back({i, dist});
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) workers.emplace_back(worker, start, end, t);
    }
    for (auto& th : workers) th.join();

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
    }

    // ----------------------
    // 新增：为每个 bucket 构建 HNSW（并行）
    // ----------------------
    const int HNSW_M = 16;
    const int HNSW_EF_BUILD = 100;
    g_hnsw_graphs.clear();
    g_hnsw_graphs.resize(num_centroid);

    // 距离函数（避免访问 solution 私有方法）
    auto dist_fn_build = [this](const float* a, const float* b) {
        return compute_distance_simd(a, b);
    };

    // 并行构建每个 bucket 的 HNSW
    std::vector<std::thread> h_workers;
    h_workers.reserve(threads_to_use);
    auto h_worker = [this, &HNSW_M, &dist_fn_build](int start, int end) {
        for (int c = start; c < end; ++c) {
            auto& bucket = inverted_index[c];
            if (bucket.empty()) continue;
            HNSWGraph g;
            // 添加点（保留全局 id）
            for (size_t i = 0; i < bucket.size(); ++i) {
                int global_idx = bucket[i].index;
                g.add_point(point_ptr(global_idx), global_idx);
            }
            // 构建本地 HNSW（使用 dist_fn_build）
            g.build(dist_fn_build, HNSW_M);
            g_hnsw_graphs[c] = std::move(g);
        }
    };
    int h_chunk = (num_centroid + threads_to_use - 1) / threads_to_use;
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * h_chunk;
        int end = std::min(start + h_chunk, num_centroid);
        if (start < end) h_workers.emplace_back(h_worker, start, end);
    }
    for (auto& th : h_workers) th.join();

    if (debug) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "[finalize_build] total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }
}

void solution::kmeans_assign_parallel(std::vector<int>& assignments) {
    const int total = static_cast<int>(point_ids_.size());
    if (total == 0) return;
    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;

    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    auto worker = [this, &assignments](int start, int end) {
        for (int i = start; i < end; ++i) {
            _mm_prefetch(reinterpret_cast<const char*>(point_ptr(std::min(i + 1, end - 1))), _MM_HINT_T0);
            assignments[i] = find_closest_centroid_linear(point_ptr(i));
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();
}

void solution::kmeans_update_parallel(const std::vector<int>& assignments, std::vector<float>& new_centroids) {
    const int total = static_cast<int>(point_ids_.size());
    if (total == 0) return;

    int threads_to_use = std::min(num_threads, std::max(1, total));
    int chunk_size = (total + threads_to_use - 1) / threads_to_use;

    std::vector<std::vector<float>> thread_sums(
        threads_to_use, std::vector<float>(static_cast<size_t>(num_centroid) * dim, 0.0f));
    std::vector<std::vector<int>> thread_counts(threads_to_use, std::vector<int>(num_centroid, 0));

    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    auto worker = [this, &assignments, &thread_sums, &thread_counts](int start, int end, int tid) {
        auto& sums = thread_sums[tid];
        auto& counts = thread_counts[tid];
        for (int i = start; i < end; ++i) {
            int c = assignments[i];
            float* dst = sums.data() + static_cast<size_t>(c) * dim;
            const float* src = point_ptr(i);
            for (int d = 0; d < dim; ++d) dst[d] += src[d];
            counts[c] += 1;
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, total);
        if (start < end) threads.emplace_back(worker, start, end, t);
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

int solution::find_closest_centroid(const std::vector<double>& vec) const {
    if (centroid_data_.empty()) return 0;
    float best = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (int c = 0; c < num_centroid; ++c) {
        const float* ctr = centroid_ptr(c);
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = static_cast<float>(vec[d]) - ctr[d];
            acc += diff * diff;
        }
        if (acc < best) {
            best = acc;
            best_idx = c;
        }
    }
    return best_idx;
}

float solution::compute_distance_simd(const float* a, const float* b) const {
    return common::compute_distance_simd(dim, a, b);
}

float solution::compute_distance_fallback(const float* a, const float* b) const {
    return common::compute_distance_fallback(dim, a, b);
}

std::vector<std::pair<int, double>> solution::find_closest_centroids(const std::vector<double>& query, int nprobe) const {
    if (centroid_data_.empty()) return {};
    std::vector<std::pair<double, int>> distances;
    distances.reserve(num_centroid);
    for (int c = 0; c < num_centroid; ++c) {
        const float* ctr = centroid_ptr(c);
        double sum = 0.0;
        for (int d = 0; d < dim; ++d) {
            double diff = query[d] - static_cast<double>(ctr[d]);
            sum += diff * diff;
        }
        distances.emplace_back(sum, c);
    }
    if (nprobe >= num_centroid) {
        std::sort(distances.begin(), distances.end());
    } else {
        std::partial_sort(distances.begin(), distances.begin() + nprobe, distances.end());
        distances.resize(nprobe);
    }
    std::vector<std::pair<int, double>> result;
    result.reserve(distances.size());
    for (auto& p : distances) result.push_back({p.second, p.first});
    return result;
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
    if (point_ids_.empty() || inverted_index.empty() || k <= 0) return {};
    auto close_centroids = find_closest_centroids_simd(query, std::min(nprob, num_centroid));
    if (close_centroids.empty()) return {};

    std::vector<float> centroid_dists(close_centroids.size());
    for (size_t i = 0; i < close_centroids.size(); ++i) centroid_dists[i] = close_centroids[i].second;

    int threads_to_use = std::min<int>(num_threads, std::max<size_t>(1, close_centroids.size()));
    int chunk_size = (static_cast<int>(close_centroids.size()) + threads_to_use - 1) / threads_to_use;

    std::vector<std::vector<std::pair<float, int>>> thread_candidates(threads_to_use);
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
	// 使用 HNSW 的 ef_search 参数
	const int ef_search = 80;

	// 距离函数供 HNSW/search 使用
	auto dist_fn_search = [this](const float* a, const float* b) {
		return compute_distance_simd(a, b);
	};

    auto worker = [this, &query, &close_centroids, &centroid_dists, &thread_candidates, k, ef_search, &dist_fn_search](int start, int end, int tid) {
        std::priority_queue<std::pair<float, int>> local;
        for (int idx = start; idx < end && idx < static_cast<int>(close_centroids.size()); ++idx) {
            int c_id = close_centroids[idx].first;
            float cq = centroid_dists[idx];

            // 仅使用已构建的 HNSW；若没有 HNSW 则跳过（移除了线性扫描回退）
            if (!(c_id >= 0 && c_id < static_cast<int>(g_hnsw_graphs.size()) && !g_hnsw_graphs[c_id].empty())) {
                continue;
            }

            auto local_res = g_hnsw_graphs[c_id].search(dist_fn_search, query.data(), ef_search, 4);
            for (const auto& pr : local_res) {
                float exact = pr.second;
                int global_idx = pr.first;
                if (static_cast<int>(local.size()) < k) {
                    local.emplace(exact, global_idx);
                } else if (exact < local.top().first) {
                    local.pop();
                    local.emplace(exact, global_idx);
                }
            }
        }
        auto& out = thread_candidates[tid];
        while (!local.empty()) {
            out.push_back(local.top());
            local.pop();
        }
    };
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, static_cast<int>(close_centroids.size()));
        if (start < end) threads.emplace_back(worker, start, end, t);
    }
    for (auto& th : threads) th.join();

    std::vector<std::pair<float, int>> all_candidates;
    for (auto& tc : thread_candidates) {
        all_candidates.insert(all_candidates.end(), tc.begin(), tc.end());
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

solution::solution(const std::string& metric, int num_centroid_val, int kmean_iter_val, int nprob_val)
    : num_centroid(num_centroid_val),
      kmean_iter(kmean_iter_val),
      nprob(nprob_val),
      num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency()))),
      dim(0),
      kd_root_(-1) {
    (void)metric;
    point_ids_.clear();
    point_data_.clear();
    centroid_data_.clear();
    inverted_index.clear();
    kd_nodes_.clear();
}