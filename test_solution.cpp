#include "Mysolution.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <vector>
#include <utility>
#include <sys/stat.h>
const std::string dataset = "sift";
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

// 从文件加载查询向量 —— 改为由 load_base_flat 内联返回，删去独立函数
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

// 新增：从 base 文件加载为一维 float 向量，并返回维度 d
static std::vector<float> load_base_flat(const std::string& base_file, int& out_d,
                                         std::vector<std::pair<int, std::vector<double>>>* out_queries = nullptr) {
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
        std::vector<double> vec;
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
        for (double v : vec) base_flat.push_back(static_cast<float>(v));
        if (out_queries) out_queries->emplace_back(idx, std::move(vec));
        ++idx;
    }
    return base_flat;
}

// 新增：二进制缓存加速
static bool file_exists(const std::string& fname) {
    struct stat st;
    return stat(fname.c_str(), &st) == 0;
}

static bool save_base_bin(const std::string& bin_file, int d, const std::vector<float>& base_flat, const std::vector<std::pair<int, std::vector<double>>>& queries) {
    std::ofstream ofs(bin_file, std::ios::binary);
    if (!ofs) return false;
    int n = (int)base_flat.size() / d;
    ofs.write((char*)&d, sizeof(int));
    ofs.write((char*)&n, sizeof(int));
    ofs.write((char*)base_flat.data(), sizeof(float) * base_flat.size());
    int nq = (int)queries.size();
    ofs.write((char*)&nq, sizeof(int));
    for (const auto& q : queries) {
        int idx = q.first;
        ofs.write((char*)&idx, sizeof(int));
        int qdim = (int)q.second.size();
        ofs.write((char*)&qdim, sizeof(int));
        for (double v : q.second) {
            float fv = (float)v;
            ofs.write((char*)&fv, sizeof(float));
        }
    }
    return true;
}

static bool load_base_bin(const std::string& bin_file, int& out_d, std::vector<float>& base_flat, std::vector<std::pair<int, std::vector<double>>>& queries) {
    std::ifstream ifs(bin_file, std::ios::binary);
    if (!ifs) return false;
    int d = 0, n = 0;
    ifs.read((char*)&d, sizeof(int));
    ifs.read((char*)&n, sizeof(int));
    if (d <= 0 || n <= 0) return false;
    base_flat.resize(n * d);
    ifs.read((char*)base_flat.data(), sizeof(float) * n * d);
    int nq = 0;
    ifs.read((char*)&nq, sizeof(int));
    queries.clear();
    for (int i = 0; i < nq; ++i) {
        int idx = 0, qdim = 0;
        ifs.read((char*)&idx, sizeof(int));
        ifs.read((char*)&qdim, sizeof(int));
        std::vector<double> qv(qdim);
        for (int j = 0; j < qdim; ++j) {
            float fv;
            ifs.read((char*)&fv, sizeof(float));
            qv[j] = fv;
        }
        queries.emplace_back(idx, std::move(qv));
    }
    out_d = d;
    return true;
}

static std::vector<float> load_base_flat_cached(const std::string& base_file, int& out_d,
                                         std::vector<std::pair<int, std::vector<double>>>* out_queries = nullptr) {
    std::string bin_file = base_file + ".bin";
    std::vector<float> base_flat;
    std::vector<std::pair<int, std::vector<double>>> queries;
    if (file_exists(bin_file)) {
        if (load_base_bin(bin_file, out_d, base_flat, queries)) {
            if (out_queries) *out_queries = queries;
            std::cout << "[cache] loaded base vectors from " << bin_file << std::endl;
            return base_flat;
        }
    }
    base_flat = load_base_flat(base_file, out_d, &queries);
    if (out_d > 0 && !base_flat.empty() && !queries.empty()) {
        save_base_bin(bin_file, out_d, base_flat, queries);
        std::cout << "[cache] saved base vectors to " << bin_file << std::endl;
    }
    if (out_queries) *out_queries = queries;
    return base_flat;
}

// Add forward declaration for parse_vector_line
bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec);

// 替换 main：使用 Solution 接口进行构建与查询
int main() {
    // 配置参数
    
    const std::string base_file = std::string("data_o/") + dataset + "/base.txt";
    const std::string gt_file = std::string("data_o/") + dataset + "/test.json";
    const int K = 10;  // top-k

    // 加载底库为一维 float 向量
    int d = 0;
    std::vector<std::pair<int, std::vector<double>>> queries;
    auto base_flat = load_base_flat_cached(base_file, d, &queries);
    if (d <= 0 || base_flat.empty() || queries.empty()) {
        std::cerr << "Empty base or invalid dimension." << std::endl;
        return 1;
    }

    // 初始化并构建索引（使用题目接口 Solution）
    Solution sol;
    auto build_start = std::chrono::high_resolution_clock::now();
    std::cout << "Building index..." << std::endl;
    sol.build(d, base_flat);
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::seconds>(build_end - build_start).count();
    std::cout << "Index built in " << build_time << " seconds" << std::endl;

    // 加载查询和ground truth（复用原有函数）
    std::cout << "Loading queries and ground truth..." << std::endl;
    auto ground_truth = SimpleJSON::parse_gt(gt_file);
    std::cout << "Loaded " << ground_truth.size() << " ground truth entries" << std::endl;

    // 执行查询并评估
    std::cout << "Running queries..." << std::endl;
    double total_recall = 0.0;
    int query_count = 0;
    auto search_start = std::chrono::high_resolution_clock::now();

    // 直接使用上面获得的 queries，移除重复文件读取
    for (const auto& gt_pair : ground_truth) {
        int query_idx = gt_pair.first;
        auto it = std::find_if(queries.begin(), queries.end(),
            [query_idx](const auto& q) { return q.first == query_idx; });

        if (it != queries.end() && !it->second.empty()) {
            // 转换为 float 并调用 Solution::search
            std::vector<float> qf;
            qf.reserve(it->second.size());
            for (double v : it->second) qf.push_back(static_cast<float>(v));
            int res_arr[10];
            sol.search(qf, res_arr);

            // 将返回的 id 转为 result 格式（distance 不影响 recall）
            std::vector<std::pair<int, double>> result;
            for (int i = 0; i < K && i < 10; ++i) {
                if (res_arr[i] >= 0) result.emplace_back(res_arr[i], 0.0);
            }

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
