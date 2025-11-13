#include "MySolution.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <sstream>
#include <string>
#include <cstring> // for memcpy

const bool debug = true; // 确保调试输出开启

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

// 全局指针，用于存储索引状态
static class HNSWIndex {
public:
    int dim; // 向量维度
    std::vector<float*> data; // 存储所有向量的指针
    std::vector<int> entry_points; // 入口点 (顶层节点)
    std::vector<std::vector<std::vector<int>>> graph; // 多层图结构: graph[level][node_id] = neighbors
    std::vector<int> node_levels; // 每个节点的层级
    std::mt19937 rng; // 随机数生成器

    // HNSW 参数
    int M = 16;         // 每个节点的最大连接数
    int efConstruction = 200; // 构建时的候选集大小
    int efSearch = 50;      // 搜索时的候选集大小

    // 插入节点计数器
    int insert_count = 0;

    HNSWIndex() : rng(42) {}

    // 计算两个向量之间的 L2 距离的平方
    float compute_distance(const float* a, const float* b) const {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // 为新节点随机分配层级
    int get_random_level() {
        int level = 0;
        while (level < 100 && (rng() % 1000) < 100) { // 简化的几何分布
            level++;
        }
        return level;
    }

    // 在指定层级查找最近的邻居（启发式搜索）
    std::vector<std::pair<float, int>> find_nearest_neighbors(int level, const float* vec, int k) {
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> pq;
        std::unordered_set<int> visited;

        // 从入口点开始搜索
        for (int ep : entry_points) {
            if (ep < static_cast<int>(data.size()) && node_levels[ep] >= level) {
                float dist = compute_distance(vec, data[ep]);
                pq.emplace(dist, ep);
                visited.insert(ep);
            }
        }

        while (!pq.empty() && visited.size() < efConstruction) {
            auto [dist, node_id] = pq.top();
            pq.pop();

            // 遍历该节点的所有邻居
            for (int neighbor : graph[level][node_id]) {
                if (visited.find(neighbor) == visited.end()) {
                    float neighbor_dist = compute_distance(vec, data[neighbor]);
                    pq.emplace(neighbor_dist, neighbor);
                    visited.insert(neighbor);
                }
            }
        }

        // 收集最近的 k 个邻居
        std::vector<std::pair<float, int>> neighbors;
        while (!pq.empty() && neighbors.size() < k) {
            auto [dist, node_id] = pq.top();
            pq.pop();
            neighbors.emplace_back(dist, node_id);
        }

        return neighbors;
    }

    // 插入新节点（真正的 HNSW 实现）
    void insert_node(const float* vec) {
        int node_id = static_cast<int>(data.size());
        // 分配内存并复制向量
        float* new_vec = new float[dim];
        std::memcpy(new_vec, vec, dim * sizeof(float));
        data.push_back(new_vec);
        int level = get_random_level();
        node_levels.push_back(level);

        // 扩展图结构
        while (static_cast<int>(graph.size()) <= level) {
            graph.push_back(std::vector<std::vector<int>>(data.size()));
        }
        for (int l = 0; l <= level; ++l) {
            if (static_cast<int>(graph[l].size()) <= node_id) {
                graph[l].resize(node_id + 1);
            }
        }

        // 从顶层到底层，逐层建立连接
        for (int l = level; l >= 0; --l) {
            if (l == level) {
                // 顶层，只连接到入口点
                if (entry_points.empty()) {
                    entry_points.push_back(node_id);
                } else {
                    auto neighbors = find_nearest_neighbors(l, vec, M);
                    for (auto& [dist, neighbor_id] : neighbors) {
                        graph[l][node_id].push_back(neighbor_id);
                        graph[l][neighbor_id].push_back(node_id); // 双向连接
                    }
                }
            } else {
                // 低层，连接到该层的最近邻居
                auto neighbors = find_nearest_neighbors(l, vec, M);
                for (auto& [dist, neighbor_id] : neighbors) {
                    graph[l][node_id].push_back(neighbor_id);
                    graph[l][neighbor_id].push_back(node_id); // 双向连接
                }
            }
        }

        // 更新入口点
        if (level > 0) {
            entry_points.push_back(node_id);
        }

        // 每 10,000 条打印一次进度，并附带时间
        insert_count++;
        if (debug && insert_count % 10000 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            std::cout << "[HNSWIndex] Inserted " << insert_count << " vectors at " << duration << " seconds\n";
        }
    }

    // 搜索 Top-K 最近邻
    std::vector<std::pair<int, float>> search(const float* query, int k) {
        if (data.empty()) return {};

        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> pq;
        std::unordered_set<int> visited;
        std::vector<std::pair<int, float>> results;

        // 从顶层入口点开始搜索
        int start_level = static_cast<int>(graph.size()) - 1;
        for (int ep : entry_points) {
            if (ep < static_cast<int>(data.size()) && node_levels[ep] >= start_level) {
                float dist = compute_distance(query, data[ep]);
                pq.emplace(dist, ep);
                visited.insert(ep);
            }
        }

        // 逐层向下搜索
        for (int l = start_level; l >= 0; --l) {
            std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> local_pq;
            std::unordered_set<int> local_visited;

            // 初始化局部搜索
            while (!pq.empty()) {
                auto [dist, node_id] = pq.top();
                pq.pop();
                local_pq.emplace(dist, node_id);
                local_visited.insert(node_id);
            }

            // 在当前层进行搜索
            while (!local_pq.empty() && local_visited.size() < efSearch) {
                auto [dist, node_id] = local_pq.top();
                local_pq.pop();

                // 遍历该节点的所有邻居
                for (int neighbor : graph[l][node_id]) {
                    if (local_visited.find(neighbor) == local_visited.end()) {
                        float neighbor_dist = compute_distance(query, data[neighbor]);
                        local_pq.emplace(neighbor_dist, neighbor);
                        local_visited.insert(neighbor);
                    }
                }
            }

            // 准备下一层搜索
            pq = std::move(local_pq);
            visited = std::move(local_visited);
        }

        // 收集结果
        while (!pq.empty() && results.size() < k) {
            auto [dist, node_id] = pq.top();
            pq.pop();
            results.emplace_back(node_id, dist);
        }

        return results;
    }

    ~HNSWIndex() {
        // 释放所有向量内存
        for (float* vec : data) {
            delete[] vec;
        }
    }

} *g_hnsw_index = nullptr;

// Solution 类的实现
Solution::Solution(int num_centroid, int kmean_iter, int nprob) {
    // 这里可以忽略参数，因为我们不使用它们
    // 或者用它们来设置 HNSW 的参数
    if (g_hnsw_index == nullptr) {
        g_hnsw_index = new HNSWIndex();
        // 根据参数调整 HNSW 参数
        g_hnsw_index->M = std::max(1, num_centroid / 10); // 示例：用 num_centroid 控制 M
        g_hnsw_index->efConstruction = std::max(100, kmean_iter);
        g_hnsw_index->efSearch = std::max(10, nprob);
    }
}

void Solution::build(int d, const std::vector<float>& base) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (d <= 0 || base.empty()) return;

    // 清理旧索引
    delete g_hnsw_index;
    g_hnsw_index = new HNSWIndex();

    // 设置维度
    g_hnsw_index->dim = d;

    // 转换数据
    int n = static_cast<int>(base.size()) / d;
    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[Solution::build] Data conversion time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }

    // 插入所有节点
    for (int i = 0; i < n; ++i) {
        const float* vec = &base[i * d];
        g_hnsw_index->insert_node(vec);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[Solution::build] Total build time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count()
                  << " ms\n";
    }
}

void Solution::search(const std::vector<float>& query, int* res) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 若还未构建索引，返回 -1 填充
    if (!g_hnsw_index || g_hnsw_index->data.empty()) {
        for (int i = 0; i < 10; ++i) res[i] = -1;
        return;
    }

    // 转换查询向量
    std::vector<float> q;
    q.reserve(g_hnsw_index->dim);
    for (float val : query) {
        q.push_back(val);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[Solution::search] Vector conversion time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms\n";
    }

    // 执行搜索
    auto ans = g_hnsw_index->search(q.data(), 10);

    auto t2 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[Solution::search] Search time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
                  << " ms\n";
    }

    // 填充结果
    int idx = 0;
    for (; idx < (int)ans.size() && idx < 10; ++idx) {
        res[idx] = ans[idx].first;
    }
    for (; idx < 10; ++idx) {
        res[idx] = -1;
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    if (debug) {
        std::cout << "[Solution::search] Result filling time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " ms\n";
        std::cout << "[Solution::search] Total time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count()
                  << " ms\n";
    }
}