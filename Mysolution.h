#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <shared_mutex>

bool parse_vector_line(const std::string &line, std::string &out_id, std::vector<double> &out_vec);

class solution {
  public:
    solution(const std::string& metric = "l2");
    void build(const std::string& base_file);
    std::vector<std::pair<int, double>> search(const std::vector<double>& query, int k);
    void build_from_memory(int d, std::vector<std::vector<double>> data);

  private:
    struct DataPoint { int id; std::vector<double> vec; };
#ifndef NUM_CENTROID
#define NUM_CENTROID 1024
#endif
#ifndef KMEAN_ITER
#define KMEAN_ITER 6
#endif
#ifndef NPROB
#define NPROB 256
#endif
    std::unordered_map<int, std::vector<int>> inverted_index;
    std::vector<DataPoint> database;
    std::vector<std::vector<double>> centroids;
    std::string metric;
    int dim;
    int num_threads;
    int find_closest_centroid(const std::vector<double>& vec) const;
    int find_closest_centroid_cached(int vec_idx) const;
    double compute_distance(const std::vector<double>& a, const std::vector<double>& b) const;
    double compute_distance_simd(const double* a, const double* b, double norm_a, double norm_b) const;
    std::vector<std::pair<int, double>> find_closest_centroids(const std::vector<double>& query, int nprobe) const;
    void kmeans_assign_parallel(std::vector<int>& assignments);
    void kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<double>>& new_centroids);
    void finalize_build();
    
    // 预计算的聚类中心到所有聚类中心的距离矩阵
    std::vector<std::vector<double>> centroid_distance_matrix;
};

class Solution {
  public:
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* res);
};

#endif