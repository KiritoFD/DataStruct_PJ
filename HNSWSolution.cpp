#include "HNSWSolution.h"
#include <random>
#include <queue>
#include <cmath>
#include <limits>
#include <algorithm>

namespace {
inline float l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}
}
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

HNSWSolution::HNSWSolution(int M, int efConstruction, int efSearch)
    : dim(0), M(M), efConstruction(efConstruction), efSearch(efSearch), maxLayer(0) {}

void HNSWSolution::build(int d, const std::vector<float>& base) {
    dim = d;
    nodes.clear();
    enterpoint.clear();
    int n = static_cast<int>(base.size()) / d;
    if (n == 0) return;

    // HNSW参数
    float ml = 1.0f / std::log(1.0f * M);
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> urd(0.0f, 1.0f);

    // 构建节点
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::vector<float> vec(base.begin() + i * d, base.begin() + (i + 1) * d);
        nodes.push_back({i, std::move(vec), {}});
    }

    // 分配层数
    std::vector<int> nodeLevel(n);
    maxLayer = 0;
    for (int i = 0; i < n; ++i) {
        int level = (int)(-std::log(urd(rng)) * ml);
        nodeLevel[i] = level;
        if (level > maxLayer) maxLayer = level;
    }
    for (auto& node : nodes) node.neighbors.resize(maxLayer + 1);
    enterpoint.resize(maxLayer + 1, -1);

    // 构建图
    for (int i = 0; i < n; ++i) {
        int level = nodeLevel[i];
        if (enterpoint[level] == -1) {
            enterpoint[level] = i;
            continue;
        }
        int ep = enterpoint[level];
        for (int l = maxLayer; l >= 0; --l) {
            if (l > level) continue;
            // 搜索邻居
            std::priority_queue<std::pair<float, int>> top_candidates;
            top_candidates.emplace(l2_distance(nodes[i].vec, nodes[ep].vec), ep);
            std::vector<bool> visited(n, false);
            visited[ep] = true;
            while (!top_candidates.empty()) {
                int curr = top_candidates.top().second;
                float curr_dist = top_candidates.top().first;
                top_candidates.pop();
                for (int nb : nodes[curr].neighbors[l]) {
                    if (!visited[nb]) {
                        visited[nb] = true;
                        float dist = l2_distance(nodes[i].vec, nodes[nb].vec);
                        if ((int)top_candidates.size() < M || dist < curr_dist) {
                            top_candidates.emplace(dist, nb);
                        }
                    }
                }
            }
            // 选M个最近邻
            std::vector<std::pair<float, int>> candidates;
            while (!top_candidates.empty()) {
                candidates.push_back(top_candidates.top());
                top_candidates.pop();
            }
            std::sort(candidates.begin(), candidates.end());
            for (int k = 0; k < std::min(M, (int)candidates.size()); ++k) {
                int nb = candidates[k].second;
                nodes[i].neighbors[l].push_back(nb);
                nodes[nb].neighbors[l].push_back(i);
            }
        }
    }
}

void HNSWSolution::search(const std::vector<float>& query, int* res) {
    if (nodes.empty() || dim == 0) {
        for (int i = 0; i < 10; ++i) res[i] = -1;
        return;
    }
    int ep = -1;
    for (int l = maxLayer; l >= 0; --l) {
        if (enterpoint[l] != -1) {
            ep = enterpoint[l];
            break;
        }
    }
    if (ep == -1) {
        for (int i = 0; i < 10; ++i) res[i] = -1;
        return;
    }

    // 层间导航
    int curr = ep;
    for (int l = maxLayer; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            float best_dist = l2_distance(query, nodes[curr].vec);
            for (int nb : nodes[curr].neighbors[l]) {
                float d = l2_distance(query, nodes[nb].vec);
                if (d < best_dist) {
                    curr = nb;
                    best_dist = d;
                    changed = true;
                }
            }
        }
    }

    // 最底层efSearch近邻
    struct HeapItem {
        float dist;
        int id;
        bool operator<(const HeapItem& o) const { return dist > o.dist; }
    };
    std::priority_queue<HeapItem> heap;
    std::vector<bool> visited(nodes.size(), false);
    heap.push({l2_distance(query, nodes[curr].vec), curr});
    visited[curr] = true;
    std::vector<int> candidates = {curr};

    while (!heap.empty() && (int)candidates.size() < efSearch) {
        int v = heap.top().id;
        heap.pop();
        for (int nb : nodes[v].neighbors[0]) {
            if (!visited[nb]) {
                visited[nb] = true;
                heap.push({l2_distance(query, nodes[nb].vec), nb});
                candidates.push_back(nb);
            }
        }
    }

    // 取前10个最近
    std::vector<std::pair<float, int>> dists;
    for (int id : candidates) {
        dists.emplace_back(l2_distance(query, nodes[id].vec), id);
    }
    std::partial_sort(dists.begin(), dists.begin() + std::min(10, (int)dists.size()), dists.end());
    for (int i = 0; i < 10; ++i) {
        res[i] = (i < (int)dists.size()) ? dists[i].second : -1;
    }
}
