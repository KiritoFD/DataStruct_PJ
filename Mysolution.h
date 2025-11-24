#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <limits>


// --- 量化数据结构（从 cpp 移入头文件）---
struct QuantizedData {
    int dim_padded; // Padded dimension for SIMD alignment
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<float> mins;
    
    void quantize(const std::vector<float>& data, int n, int d);
    float compute_distance_dequant_avx2(const float* query, int idx, int dim) const;
};

// --- SoA 倒排索引结构（从 cpp 移入头文件）---
struct CompactBucket {
    uint32_t start_offset;
    uint32_t count;
    std::vector<float> sorted_dists;
    std::vector<int> original_ids;
};

struct BucketItem {
    int index;
    float dist_to_centroid;
};

class solution {
public:
    solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob);
    void build(const std::string& base_file);
    void build_from_memory(int d, std::vector<std::vector<float>> data);
    void finalize_build();

    // 搜索：浮点向量接口（与 wrapper 对应）
    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k);
    std::vector<std::pair<int, float>> find_closest_centroids(const std::vector<float>& query, int nprobe) const;
    // 新增：批量查询（并行按查询级别）
    void search_batch(const std::vector<std::vector<float>>& queries, int k,
                      std::vector<std::vector<std::pair<int,float>>>& results);

    // 开关 profiling
    static void set_profile(bool on);

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
    std::vector<std::vector<BucketItem>> inverted_index; // 旧版（已废弃）
    
    // --- 新增 ---
    std::vector<CompactBucket> compact_inverted_index;
    QuantizedData quantized_data_;
    
    struct KDNode {
        int axis;
        int centroid_index;
        int left;
        int right;
        float split_value;
    };
    std::vector<KDNode> kd_nodes_;
    int kd_root_;

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
    // 保留旧 float 版本声明（若仍需）
    int find_closest_centroid(const std::vector<float>& vec) const;

    // 距离计算：SIMD 与回退实现
    float compute_distance_simd(const float* a, const float* b) const;
    float compute_distance_fallback(const float* a, const float* b) const;
    float compute_distance(const std::vector<float>& a, const std::vector<float>& b) const;
    // 新增：带上界的距离（超过 cap 提前退出）
    float compute_distance_capped_simd(const float* a, const float* b, float cap) const;

    // SIMD 版质心搜索（float）
    std::vector<std::pair<int, float>> find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const;
    int build_kdtree(std::vector<int>& indices, int begin, int end, int depth);
    void search_kdtree(const float* query, int node_idx, int nprobe,
                       std::priority_queue<std::pair<float, int>>& best) const;
};

class Solution {
public:
    Solution(int num_centroid = 4045, int kmean_iter = 16, int nprob = 324);
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
    // 包装批量查询
    void search_batch(const std::vector<std::vector<float>>& queries, int k,
                      std::vector<std::vector<int>>& out_ids);
private:
    int num_centroid_;
    int kmean_iter_;
    int nprob_;
};

#endif // MYSOLUTION_H
