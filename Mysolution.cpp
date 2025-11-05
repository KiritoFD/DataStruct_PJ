#include "Mysolution.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <iostream>
#include <random>
#include <numeric>
#include <limits>

bool try_stod(const std::string &s, double &out) {
    try {
        size_t pos;
        out = stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parse_vector_line(const std::string &line, std::string &out_id, std::vector<double> &out_vec) {
    out_id.clear();
    out_vec.clear();
    std::istringstream iss(line);
    std::vector<std::string> toks;
    std::string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;
    double val;
    bool allnum = true;
    for (auto &s : toks) {
        if (!try_stod(s, val)) { allnum = false; break; }
    }
    if (allnum) {
        out_id.clear();
        out_vec.reserve(toks.size());
        for (auto &s : toks) out_vec.push_back(stod(s));
        return true;
    }
    if (toks.size() < 2) return false;
    out_id = toks[0];
    out_vec.reserve(toks.size()-1);
    for (size_t i = 1; i < toks.size(); ++i) {
        if (!try_stod(toks[i], val)) return false;
        out_vec.push_back(stod(toks[i]));
    }
    return true;
}

solution::solution(const std::string& metric_type) : metric(metric_type), dim(0) {}

void solution::build(const std::string& base_file) {
    std::ifstream fin(base_file);
    if (!fin) {
        std::cerr << "Cannot open " << base_file << std::endl;
        return;
    }

    // 加载所有数据
    std::cout << "Loading vectors..." << std::endl;
    std::string line;
    while (std::getline(fin, line)) {
        std::string id;
        std::vector<double> vec;
        if (parse_vector_line(line, id, vec)) {
            if (dim == 0) dim = vec.size();
            database.push_back({(int)database.size(), vec});
        }
    }
    std::cout << "Loaded " << database.size() << " vectors, dim=" << dim << std::endl;

    // K-Means 聚类
    std::cout << "Running K-Means..." << std::endl;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, database.size() - 1);

    centroids.clear();
    centroids.reserve(NUM_CENTROIDS);
    for (int i = 0; i < NUM_CENTROIDS; ++i) {
        centroids.push_back(database[dist(rng)].vec);
    }

    // K-Means 迭代
    for (int iter = 0; iter < KMEANS_ITER; ++iter) {
        std::vector<std::vector<double>> new_centroids(NUM_CENTROIDS, std::vector<double>(dim, 0.0));
        std::vector<int> counts(NUM_CENTROIDS, 0);

        // 分配点到最近的中心
        for (const auto& dp : database) {
            int c = find_closest_centroid(dp.vec);
            for (int d = 0; d < dim; ++d) {
                new_centroids[c][d] += dp.vec[d];
            }
            counts[c]++;
        }

        // 更新中心
        for (int i = 0; i < NUM_CENTROIDS; ++i) {
            if (counts[i] > 0) {
                for (int d = 0; d < dim; ++d) {
                    centroids[i][d] = new_centroids[i][d] / counts[i];
                }
            }
        }

        if ((iter + 1) % 2 == 0) {
            std::cout << "  Iteration " << (iter + 1) << "/" << KMEANS_ITER << " done" << std::endl;
        }
    }

    // 构建倒排索引
    std::cout << "Building inverted index..." << std::endl;
    inverted_index.clear();
    for (const auto& dp : database) {
        int c = find_closest_centroid(dp.vec);
        inverted_index[c].push_back(dp.id);
    }

    std::cout << "Index built successfully" << std::endl;
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
    std::priority_queue<std::pair<double, int>> pq;  // max-heap
    
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
    // 找到 NPROBE 个最近的聚类中心
    auto close_centroids = find_closest_centroids(query, NPROBE);

    // 收集候选向量
    std::unordered_map<int, bool> candidate_ids;
    for (const auto& [c_id, _] : close_centroids) {
        auto it = inverted_index.find(c_id);
        if (it != inverted_index.end()) {
            for (int vec_id : it->second) {
                candidate_ids[vec_id] = true;
            }
        }
    }

    // 在候选集中计算距离并找 top-k
    std::priority_queue<std::pair<double, int>> topk;  // max-heap
    for (const auto& [vec_id, _] : candidate_ids) {
        double dist = compute_distance(query, database[vec_id].vec);
        if ((int)topk.size() < k) {
            topk.push({dist, vec_id});
        } else if (dist < topk.top().first) {
            topk.pop();
            topk.push({dist, vec_id});
        }
    }

    // 整理结果
    std::vector<std::pair<int, double>> result;
    while (!topk.empty()) {
        result.push_back({topk.top().second, topk.top().first});
        topk.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
}
