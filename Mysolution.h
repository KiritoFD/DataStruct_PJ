#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>

// 将解析函数对外声明，供其它翻译单元使用（例如 test_solution.cpp）
bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec);

struct BucketItem {
    int id;
    float dist_to_centroid;
};

class solution {
public:
    solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob);
    void build(const std::string& base_file);
    void build_from_memory(int d, std::vector<std::vector<double>> data);
    void finalize_build();

    // 搜索：浮点向量接口（与 wrapper 对应）
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k);

    // 保留旧的 double 接口（若还有使用）
    std::vector<std::pair<int, double>> find_closest_centroids(const std::vector<double>& query, int nprobe) const;

private:
    // metric & 超参数（从 .h 中定义或构造时传入）
    std::string metric;
    int dim;
    int num_threads;
    int num_centroid;
    int kmean_iter;
    int nprob;

    // 改为 float 存储以配合 SIMD 和节省内存
    std::vector<std::pair<int, std::vector<float>>> database;
    std::vector<std::vector<float>> centroids;

    // 倒排索引：每个桶保存预计算到质心的距离
    std::unordered_map<int, std::vector<BucketItem>> inverted_index;

    // K-means 并行函数（assign 使用数据库的 float 向量）
    void kmeans_assign_parallel(std::vector<int>& assignments);
    // 注意：此处声明匹配实现，new_centroids 使用 float
    void kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<float>>& new_centroids);

    // 质心查找（float 版本）
    int find_closest_centroid(const std::vector<float>& vec) const;
    // 保留旧 double 版本声明（若仍需）
    int find_closest_centroid(const std::vector<double>& vec) const;

    // 距离计算：SIMD 与回退实现
    float compute_distance_simd(const std::vector<float>& a, const std::vector<float>& b) const;
    float compute_distance_fallback(const std::vector<float>& a, const std::vector<float>& b) const;
    double compute_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // SIMD 版质心搜索（float）
    std::vector<std::pair<int, float>> find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const;


};

class Solution {
public:
    Solution(int num_centroid = 9600, int kmean_iter = 1, int nprob = 1024);
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
private:
    int num_centroid_;
    int kmean_iter_;
    int nprob_;
};

#endif // MYSOLUTION_H
