#include "Mysolution.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <utility>
#include <sys/stat.h>

/// 新增：声明 MySolution.cpp 中导出的 C 接口函数（用于距离计数统计）
extern "C" {
    void reset_dist_counters();
    uint64_t get_total_queries();
    double get_avg_dists_per_query();
    uint64_t get_last_query_dists();
    double get_last_build_time_ms();
}

const std::string dataset = "sift";

// --- 本地辅助函数：解析向量行 ---
static bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<float>& out_vec) {
    out_id.clear();
    out_vec.clear();
    std::istringstream iss(line);
    std::vector<std::string> toks;
    std::string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;

    auto try_stod = [](const std::string& s, float& out) -> bool {
        try {
            size_t pos = 0;
            out = std::stod(s, &pos);
            return pos == s.size();
        } catch (...) {
            return false;
        }
    };

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

// 简单的JSON解析器
struct SimpleJSON {
    static std::unordered_map<int, std::vector<std::pair<int, float>>> parse_gt(const std::string& json_file) {
        std::unordered_map<int, std::vector<std::pair<int, float>>> gt;
        std::ifstream ifs(json_file);
        if (!ifs) return gt;

        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        
        // 找到 "results" 数组
        size_t results_pos = content.find("\"results\":");
        if (results_pos == std::string::npos) return gt;
        
        size_t array_start = content.find('[', results_pos);
        size_t array_end = content.rfind(']');
        if (array_start == std::string::npos || array_end == std::string::npos) return gt;

        std::string results_str = content.substr(array_start + 1, array_end - array_start - 1);
        
        // 按 "query_index" 分割
        size_t pos = 0;
        while ((pos = results_str.find("\"query_index\"", pos)) != std::string::npos) {
            size_t colon = results_str.find(':', pos);
            size_t comma = results_str.find(',', colon);
            std::string idx_str = results_str.substr(colon + 1, comma - colon - 1);
            idx_str.erase(0, idx_str.find_first_not_of(" \t\n\r"));
            int query_idx = std::stoi(idx_str);

            std::vector<std::pair<int, float>> neighbors;
            
            size_t neighbors_pos = results_str.find("\"neighbors\"", pos);
            size_t brackets_start = results_str.find('[', neighbors_pos);
            size_t brackets_end = results_str.find(']', brackets_start);
            std::string neighbors_str = results_str.substr(brackets_start + 1, brackets_end - brackets_start - 1);

            size_t idx_pos = 0;
            while ((idx_pos = neighbors_str.find("\"index\"", idx_pos)) != std::string::npos) {
                size_t idx_colon = neighbors_str.find(':', idx_pos);
                size_t idx_comma = neighbors_str.find(',', idx_colon);
                std::string neighbor_id_str = neighbors_str.substr(idx_colon + 1, idx_comma - idx_colon - 1);
                neighbor_id_str.erase(0, neighbor_id_str.find_first_not_of(" \t\n\r"));
                int neighbor_id = std::stoi(neighbor_id_str);

                size_t dist_pos = neighbors_str.find("\"distance\"", idx_comma);
                size_t dist_colon = neighbors_str.find(':', dist_pos);
                size_t dist_brace = neighbors_str.find('}', dist_colon);
                std::string dist_str = neighbors_str.substr(dist_colon + 1, dist_brace - dist_colon - 1);
                dist_str.erase(0, dist_str.find_first_not_of(" \t\n\r"));
                float distance = std::stod(dist_str);

                neighbors.emplace_back(neighbor_id, distance);
                idx_pos = dist_brace + 1;
            }

            if (!neighbors.empty()) {
                gt[query_idx] = std::move(neighbors);
            }
            pos = brackets_end + 1;
        }

        return gt;
    }
};

// 计算召回率
float compute_recall(const std::vector<std::pair<int, float>>& result, 
                     const std::vector<std::pair<int, float>>& ground_truth,
                     int k) {
    if (ground_truth.empty()) return 0.0;
    
    std::unordered_set<int> result_set;
    for (int i = 0; i < std::min(k, (int)result.size()); ++i) {
        result_set.insert(result[i].first);
    }

    int matches = 0;
    for (int i = 0; i < std::min(k, (int)ground_truth.size()); ++i) {
        if (result_set.count(ground_truth[i].first)) {
            ++matches;
        }
    }

    return matches / (float)std::min(k, (int)ground_truth.size());
}

// 新增：从 base 文件加载为一维 float 向量，并返回维度 d
static std::vector<float> load_base_flat(const std::string& base_file, int& out_d,
                                         std::vector<std::pair<int, std::vector<float>>>* out_queries = nullptr) {
    std::vector<float> base_flat;
    out_d = 0;
    std::ifstream ifs(base_file);
    if (!ifs) {
        std::cerr << "Cannot open file: " << base_file << std::endl;
        return base_flat;
    }
    std::string line;
    bool first = true;
    int idx = 0;
    while (std::getline(ifs, line)) {
        std::string id;
        std::vector<float> vec;
        if (!parse_vector_line(line, id, vec)) {
            ++idx;
            continue;
        }
        if (first) {
            out_d = static_cast<int>(vec.size());
            first = false;
        }
        if (vec.size() != static_cast<size_t>(out_d)) {
            ++idx;
            continue;
        }
        for (float v : vec) base_flat.push_back(static_cast<float>(v));
        if (out_queries) out_queries->emplace_back(idx, std::move(vec));
        ++idx;
    }
    return base_flat;
}

int main() {
	const std::string base_file = "data_o/" + dataset + "/base.txt";
	const std::string gt_file = "data_o/" + dataset + "/test.json";
	const int K = 10;

	int d = 0;
	std::vector<std::pair<int, std::vector<float>>> queries;
	auto base_flat = load_base_flat(base_file, d, &queries);
	if (d <= 0 || base_flat.empty() || queries.empty()) {
		std::cerr << "Empty base or invalid dimension." << std::endl;
		return 1;
	}

	auto ground_truth = SimpleJSON::parse_gt(gt_file);
	if (ground_truth.empty()) {
		std::cerr << "Failed to load ground truth." << std::endl;
		return 1;
	}

	std::cout << "Building index..." << std::endl;
	Solution sol;
	sol.build(d, base_flat);
	std::cout << "Index built." << std::endl;

	// 在开始查询前重置计数器（确保统计仅包含当前 run）
	reset_dist_counters();

	float total_recall = 0.0f;
	int query_count = 0;
	auto search_start = std::chrono::steady_clock::now();
	int res_arr[10];

	for (const auto& [qid, vec] : queries) {
		auto it = ground_truth.find(qid);
		if (it == ground_truth.end()) continue;

		sol.search(vec, res_arr);

		std::vector<std::pair<int, float>> result;
		for (int i = 0; i < K; ++i) {
			if (res_arr[i] >= 0) result.emplace_back(res_arr[i], 0.0f);
		}

		total_recall += compute_recall(result, it->second, K);
		++query_count;

		if (query_count % 50 == 0) {
			std::cout << "Processed " << query_count << " queries\r" << std::flush;
		}
	}

	auto search_end = std::chrono::steady_clock::now();
	auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

	std::cout << "\n=== Results ===" << std::endl;
	std::cout << "Total queries: " << query_count << std::endl;
	if (query_count > 0) {
		std::cout << "Average recall@" << K << ": " << std::fixed << std::setprecision(4)
				  << (total_recall / query_count) << std::endl;
		std::cout << "Average query time: " << std::fixed << std::setprecision(2)
				  << (search_ms / static_cast<float>(query_count)) << " ms" << std::endl;
	}

	// 新增：打印距离运算统计（来自 MySolution.cpp 的全局计数）
	uint64_t total_queries_reported = get_total_queries();
	double avg_dists = get_avg_dists_per_query();
	uint64_t last_query_dists = get_last_query_dists();
	double last_build_ms = get_last_build_time_ms();

	std::cout << "\n=== Distance Statistics ===" << std::endl;
	std::cout << "Total queries reported by index: " << total_queries_reported << std::endl;
	std::cout << "Average distance ops per query: " << std::fixed << std::setprecision(2) << avg_dists << std::endl;
	std::cout << "Last query distance ops: " << last_query_dists << std::endl;
	std::cout << "Last build time (ms): " << std::fixed << std::setprecision(2) << last_build_ms << " ms" << std::endl;

	return 0;
}