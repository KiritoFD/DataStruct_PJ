#include "Mysolution.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <cctype>

// 简单的JSON解析器
struct SimpleJSON {
    static std::unordered_map<int, std::vector<std::pair<int, double>>> parse_gt(const std::string& json_file) {
        std::unordered_map<int, std::vector<std::pair<int, double>>> gt;
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

            std::vector<std::pair<int, double>> neighbors;
            
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
                double distance = std::stod(dist_str);

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

// 从文件加载查询向量
std::vector<std::pair<int, std::vector<double>>> load_queries(const std::string& base_file) {
    std::vector<std::pair<int, std::vector<double>>> queries;
    std::ifstream ifs(base_file);
    if (!ifs) {
        std::cerr << "Cannot open file: " << base_file << std::endl;
        return queries;
    }

    std::string line;
    int idx = 0;
    while (std::getline(ifs, line)) {
        std::string id;
        std::vector<double> vec;
        if (parse_vector_line(line, id, vec)) {
            queries.emplace_back(idx, std::move(vec));
        }
        ++idx;
    }
    return queries;
}

// 计算召回率
double compute_recall(const std::vector<std::pair<int, double>>& result, 
                     const std::vector<std::pair<int, double>>& ground_truth,
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

    return matches / (double)std::min(k, (int)ground_truth.size());
}

int main() {
    // 配置参数
    const std::string base_file = "data_o/sift/base.txt";
    const std::string gt_file = "data_o/sift/base.json";
    const int K = 5;  // top-k
    
    // 初始化解决方案
    solution sol("l2");
    
    // 构建索引并计时
    auto build_start = std::chrono::high_resolution_clock::now();
    std::cout << "Building index..." << std::endl;
    sol.build(base_file);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start).count();
    std::cout << "Index built in " << build_time << " seconds" << std::endl;

    // 加载查询和ground truth
    std::cout << "Loading queries and ground truth..." << std::endl;
    auto queries = load_queries(base_file);
    auto ground_truth = SimpleJSON::parse_gt(gt_file);
    std::cout << "Loaded " << ground_truth.size() << " ground truth entries" << std::endl;

    // 执行查询并评估
    std::cout << "Running queries..." << std::endl;
    double total_recall = 0.0;
    int query_count = 0;
    auto search_start = std::chrono::high_resolution_clock::now();

    for (const auto& gt_pair : ground_truth) {
        int query_idx = gt_pair.first;
        // 找到对应的查询向量
        auto it = std::find_if(queries.begin(), queries.end(),
            [query_idx](const auto& q) { return q.first == query_idx; });
        
        if (it != queries.end() && !it->second.empty()) {
            // 执行查询
            auto result = sol.search(it->second, K);
            // 计算召回率
            double recall = compute_recall(result, gt_pair.second, K);
            total_recall += recall;
            ++query_count;

            // 打印进度
            if (query_count % 50 == 0) {
                std::cout << "Processed " << query_count << " queries\r" << std::flush;
            }
        }
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

    // 打印结果
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total queries: " << query_count << std::endl;
    if (query_count > 0) {
        std::cout << "Average recall@" << K << ": " << std::fixed << std::setprecision(4) 
                  << (total_recall / query_count) << std::endl;
        std::cout << "Average query time: " << std::fixed << std::setprecision(2) 
                  << (search_time / (double)query_count) << " ms" << std::endl;
    }
    std::cout << "Index build time: " << build_time << " seconds" << std::endl;

    return 0;
}
