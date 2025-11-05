#ifndef MYSOLUTION_H
#define MYSOLUTION_H

#include <string>
#include <vector>
#include <unordered_map>

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

    static const int NUM_CENTROIDS = 256;  // K-Means聚类中心数
    static const int KMEANS_ITER = 10;     // K-Means迭代次数
    static const int NPROBE = 20;          // 查询时搜索的中心数量

    std::vector<DataPoint> database;
    std::vector<std::vector<double>> centroids;  // 聚类中心
    std::unordered_map<int, std::vector<int>> inverted_index;  // 倒排索引
    
    std::string metric;
    int dim;
    
    int find_closest_centroid(const std::vector<double>& vec) const;
    double compute_distance(const std::vector<double>& a, const std::vector<double>& b) const;
    std::vector<std::pair<int, double>> find_closest_centroids(const std::vector<double>& query, int nprobe) const;
};

#endif // MYSOLUTION_H
