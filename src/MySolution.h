#pragma once
#include <vector>
#include <cstdint>
#include <cstring>

// 默认参数常量
constexpr int HNSW_DEFAULT_M = 80;
constexpr int HNSW_DEFAULT_MAX_LAYER = 11;
constexpr int HNSW_DEFAULT_EF_CONSTRUCTION = 1301;
constexpr int HNSW_DEFAULT_EF_SEARCH = 318;


// 使用宏守卫避免与 visited_list.h 重复定义
#ifndef VISITED_LIST_DEFINED
#define VISITED_LIST_DEFINED

class VisitedList {
public:
    std::vector<uint8_t> arr;
    uint8_t cur_tag = 1;
    int cap = 0;

    void init(int n) {
        if (n > cap) { arr.assign(n, 0); cap = n; cur_tag = 1; }
    }
    void advance() {
        if (++cur_tag == 0) { std::memset(arr.data(), 0, cap); cur_tag = 1; }
    }
    bool isVisited(int id) const { return arr[id] == cur_tag; }
    void mark(int id) { arr[id] = cur_tag; }
};

class TagVisitedList {
public:
    std::vector<uint16_t> arr;
    uint16_t cur_tag = 1;
    int cap = 0;

    void init(int n) {
        if (n > cap) { arr.assign(n, 0); cap = n; cur_tag = 1; }
    }
    void advance() {
        if (++cur_tag == 0) { std::memset(arr.data(), 0, cap * sizeof(uint16_t)); cur_tag = 1; }
    }
    bool isVisited(int id) const { return arr[id] == cur_tag; }
    void mark(int id) { arr[id] = cur_tag; }
    const uint16_t* data() const { return arr.data(); }
    uint16_t currentTag() const { return cur_tag; }
};

#endif // VISITED_LIST_DEFINED

class Solution {
public:
    Solution();
    void build(int d, const std::vector<float>& base);
    void search(const std::vector<float>& query, int* result);
    void setK(int k) { k_ = k; }
private:
    int k_;
};

void build_hnsw(int d, const std::vector<float>& base);
std::vector<std::pair<int, float>> search_hnsw(const std::vector<float>& query, int k);

