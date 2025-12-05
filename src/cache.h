#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>

// 需要 SimpleHNSW 和 HNSWNode 的完整定义，此头文件必须在它们定义之后 include

static inline std::string get_index_cache_path(int n, int d, int M, int max_layer, int efc) {
    std::ostringstream oss;
    oss << "cache/hnsw_raw_" << n << "_" << d << "_" << M << "_" << max_layer << "_" << efc << ".bin";
    return oss.str();
}

static inline bool save_simple_index(SimpleHNSW* hnsw, const std::string& path, bool debug = false) {
    if (!hnsw) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    
    uint32_t magic = 0x48535752, version = 1;
    out.write((char*)&magic, sizeof(magic));
    out.write((char*)&version, sizeof(version));
    
    int num_nodes = hnsw->size();
    out.write((char*)&hnsw->dim, sizeof(int));
    out.write((char*)&hnsw->M, sizeof(int));
    out.write((char*)&hnsw->maxLayer, sizeof(int));
    out.write((char*)&hnsw->enter_point, sizeof(int));
    out.write((char*)&num_nodes, sizeof(int));
    
    for (int i = 0; i < num_nodes; ++i) {
        int level = (int)hnsw->nodes[i]->links.size() - 1;
        out.write((char*)&level, sizeof(level));
        for (int l = 0; l <= level; ++l) {
            int cnt = (int)hnsw->nodes[i]->links[l].size();
            out.write((char*)&cnt, sizeof(cnt));
            if (cnt > 0) out.write((char*)hnsw->nodes[i]->links[l].data(), cnt * sizeof(int));
        }
    }
    if (debug) std::cout << "[Cache] Saved: " << num_nodes << " nodes to " << path << std::endl;
    return true;
}

static inline SimpleHNSW* load_simple_index(const std::string& path, const float* data, int expected_dim, int expected_n, bool debug = false) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return nullptr;
    
    uint32_t magic, version;
    in.read((char*)&magic, sizeof(magic));
    if (magic != 0x48535752) return nullptr;
    in.read((char*)&version, sizeof(version));
    if (version != 1) return nullptr;
    
    int dim, M, maxLayer, enter_point, num_nodes;
    in.read((char*)&dim, sizeof(int));
    in.read((char*)&M, sizeof(int));
    in.read((char*)&maxLayer, sizeof(int));
    in.read((char*)&enter_point, sizeof(int));
    in.read((char*)&num_nodes, sizeof(int));
    
    if (!in || num_nodes <= 0 || num_nodes != expected_n || dim != expected_dim) return nullptr;
    
    SimpleHNSW* hnsw = new SimpleHNSW(dim, M, maxLayer);
    hnsw->enter_point = enter_point;
    hnsw->data_flat.resize((size_t)num_nodes * dim);
    std::memcpy(hnsw->data_flat.data(), data, (size_t)num_nodes * dim * sizeof(float));
    
    hnsw->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        int level;
        in.read((char*)&level, sizeof(level));
        if (!in || level < 0 || level > 100) { delete hnsw; return nullptr; }
        
        HNSWNode* node = new HNSWNode(level, M);
        for (int l = 0; l <= level; ++l) {
            int cnt;
            in.read((char*)&cnt, sizeof(cnt));
            if (!in || cnt < 0) { delete node; delete hnsw; return nullptr; }
            if (cnt > 0) {
                node->links[l].resize(cnt);
                in.read((char*)node->links[l].data(), cnt * sizeof(int));
            }
        }
        hnsw->nodes.push_back(node);
    }
    if (debug) std::cout << "[Cache] Loaded: " << num_nodes << " nodes from " << path << std::endl;
    return hnsw;
}