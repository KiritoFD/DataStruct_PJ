#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <queue>
#include <limits>

struct BucketItem {
    int index;
    float dist_to_centroid;
};

struct Bucket {
    std::vector<BucketItem> items;
    float max_radius = 0.0f;
};

class solution {
public:
    solution(const std::string& metric_type, int num_centroid, int kmean_iter, int nprob);
    void build_from_memory(int d, std::vector<std::vector<float>> data);
    void finalize_build();

    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k);

private:
    std::string metric;
    int dim;
    int num_threads;
    int num_centroid;
    int kmean_iter;
    int nprob;

    std::vector<int> point_ids_;
    std::vector<float> point_data_;
    std::vector<float> centroid_data_;

    // SQ8 Optimization
    std::vector<uint8_t> quantized_point_data_;
    float global_min_ = 0.0f;
    float global_scale_ = 1.0f;
    int padded_dim = 0;

    struct KDNode {
        int axis;
        int centroid_index;
        int left;
        int right;
        float split_value;
    };
    std::vector<KDNode> kd_nodes_;
    int kd_root_;

    std::vector<Bucket> inverted_index;

    inline const float* point_ptr(int idx) const { return point_data_.data() + static_cast<size_t>(idx) * dim; }
    inline float* point_ptr(int idx) { return point_data_.data() + static_cast<size_t>(idx) * dim; }
    inline const float* centroid_ptr(int idx) const { return centroid_data_.data() + static_cast<size_t>(idx) * dim; }
    inline float* centroid_ptr(int idx) { return centroid_data_.data() + static_cast<size_t>(idx) * dim; }

    inline const uint8_t* quantized_ptr(int idx) const { return quantized_point_data_.data() + static_cast<size_t>(idx) * padded_dim; }

    void kmeans_assign_parallel(std::vector<int>& assignments);
    void kmeans_update_parallel(const std::vector<int>& assignments, std::vector<float>& new_centroids);

    int find_closest_centroid_linear(const float* vec) const;
    float compute_distance_simd(const float* a, const float* b) const;
    int compute_distance_sq8(const uint8_t* a, const uint8_t* b) const;

    std::vector<std::pair<int, float>> find_closest_centroids_simd(const std::vector<float>& query, int nprobe) const;
    int build_kdtree(std::vector<int>& indices, int begin, int end, int depth);
    void search_kdtree(const float* query, int node_idx, int nprobe,
                       std::priority_queue<std::pair<float, int>>& best) const;
};

class Solution {
public:
    Solution(int num_centroid = 5422, int kmean_iter = 16, int nprob = 1024);
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
private:
    int num_centroid_;
    int kmean_iter_;
    int nprob_;
};

#endif // MYSOLUTION_H
