#pragma once
#include <vector>

class HNSWSolution {
public:
    HNSWSolution(int M = 16, int efConstruction = 200, int efSearch = 50);

    // 构建索引，d为维度，base为数据（按行主序排布的float向量）
    void build(int d, const std::vector<float>& base);

    // 查询，query为float向量，res为结果id数组（长度10）
    void search(const std::vector<float>& query, int* res);

private:
    struct Node {
        int id;
        std::vector<float> vec;
        std::vector<std::vector<int>> neighbors; // 多层邻居
    };

    int dim;
    int M;
    int efConstruction;
    int efSearch;
    int maxLayer;
    std::vector<Node> nodes;
    std::vector<int> enterpoint; // 每层入口点
    // ...可扩展更多成员...
};
