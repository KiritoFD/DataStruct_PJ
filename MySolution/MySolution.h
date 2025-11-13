#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>

// 将解析函数对外声明，供其它翻译单元使用（例如 test_solution.cpp）
bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec);

struct BucketItem {
    int index;
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
    std::vector<int> point_ids_;
    std::vector<float> point_data_;
    std::vector<float> centroid_data_;

    struct KDNode {
        int axis;
        int centroid_index;
        int left;
        int right;
        float split_value;
    };
    std::vector<KDNode> kd_nodes_;
    int kd_root_;

    std::vector<std::vector<BucketItem>> inverted_index;

    inline const float* point_ptr(int idx) const { return point_data_.data() + static_cast<size_t>(idx) * dim; }
    inline float* point_ptr(int idx) { return point_data_.data() + static_cast<size_t>(idx) * dim; }
    inline const float* centroid_ptr(int idx) const { return centroid_data_.data() + static_cast<size_t>(idx) * dim; }
    inline float* centroid_ptr(int idx) { return centroid_data_.data() + static_cast<size_t>(idx) * dim; }

    // K-means 并行函数（assign 使用数据库的 float 向量）
    void kmeans_assign_parallel(std::vector<int>& assignments);
    // 注意：此处声明匹配实现，new_centroids 使用 float
    void kmeans_update_parallel(const std::vector<int>& assignments, std::vector<float>& new_centroids);

    // 质心查找（float 版本）
    int find_closest_centroid_linear(const float* vec) const;
    // 保留旧 double 版本声明（若仍需）
    int find_closest_centroid(const std::vector<double>& vec) const;

    // 距离计算：SIMD 与回退实现
    float compute_distance_simd(const float* a, const float* b) const;
    float compute_distance_fallback(const float* a, const float* b) const;
    double compute_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // SIMD 版质心搜索（float）
    std::vector<std::pair<int, float>> find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const;
    int build_kdtree(std::vector<int>& indices, int begin, int end, int depth);
    void search_kdtree(const float* query, int node_idx, int nprobe,
                       std::priority_queue<std::pair<float, int>>& best) const;
};

class Solution {
public:
    Solution(int num_centroid = 4096, int kmean_iter = 1, int nprob = 112);
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
private:
    int num_centroid_;
    int kmean_iter_;
    int nprob_;
};

#endif // MYSOLUTION_H
