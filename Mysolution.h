#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>

bool parse_vector_line(const std::string &line, std::string &out_id, std::vector<double> &out_vec);

class solution {
public:
    solution(const std::string& metric = "l2");
    void build(const std::string& base_file);
    std::vector<std::pair<int, double>> search(const std::vector<double>& query, int k);

private:
    struct DataPoint {
        int id;
        std::vector<double> vec;
    };

    static const int NUM_CENTROIDS = 256;
    static const int KMEANS_ITER = 10;
    static const int NPROBE = 25;
    static const int NUM_THREADS = 8;

    std::vector<DataPoint> database;
    std::vector<std::vector<double>> centroids;
    std::unordered_map<int, std::vector<int>> inverted_index;
    
    std::string metric;
    int dim;
    
    int find_closest_centroid(const std::vector<double>& vec) const;
    double compute_distance(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<std::pair<int, double>> find_closest_centroids(const std::vector<double>& query, int nprobe) const;
    
    // 多线程辅助函数
    void kmeans_assign_parallel(std::vector<int>& assignments);
    void kmeans_update_parallel(const std::vector<int>& assignments, std::vector<std::vector<double>>& new_centroids);
};

#endif // MYSOLUTION_H
