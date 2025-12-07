#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

// Forward declare FlatHNSW and SimpleHNSW so cache helpers can be declared without needing full type here
struct FlatHNSW;
class SimpleHNSW;

// Note: AlignedFloatArray is defined in Sol.cpp before this header is included

// 【修改】版本4：移除无效的剪枝数组（pivot_dists, node_l2_norms）
static const uint32_t INDEX_VERSION = 4u;


// 保存 FlatHNSW 到文件
static bool save_flat_index(FlatHNSW* flat, const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    
    // 写入魔数和版本
    uint32_t magic = 0x48535746; // 'HSWF'
    uint32_t version = INDEX_VERSION;
    ofs.write((char*)&magic, sizeof(magic));
    ofs.write((char*)&version, sizeof(version));
    
    // 写入基本参数
    ofs.write((char*)&flat->dim, sizeof(flat->dim));
    ofs.write((char*)&flat->max_m, sizeof(flat->max_m));
    ofs.write((char*)&flat->max_m_upper, sizeof(flat->max_m_upper));
    ofs.write((char*)&flat->enter_point, sizeof(flat->enter_point));
    ofs.write((char*)&flat->num_nodes, sizeof(flat->num_nodes));
    ofs.write((char*)&flat->max_level, sizeof(flat->max_level));
    
    // 写入数据向量（使用对齐数组）
    uint64_t data_size = flat->data.size();
    ofs.write((char*)&data_size, sizeof(data_size));
    if (data_size > 0) {
        ofs.write((char*)flat->data.data(), data_size * sizeof(float));
    }
    
    // 写入 L0 CSR 结构
    uint64_t l0_offsets_size = flat->l0_offsets.size();
    ofs.write((char*)&l0_offsets_size, sizeof(l0_offsets_size));
    ofs.write((char*)flat->l0_offsets.data(), l0_offsets_size * sizeof(uint64_t));
    
    uint64_t l0_links_size = flat->l0_links.size();
    ofs.write((char*)&l0_links_size, sizeof(l0_links_size));
    ofs.write((char*)flat->l0_links.data(), l0_links_size * sizeof(int));
    
    // 写入上层结构
    uint64_t node_levels_size = flat->node_levels.size();
    ofs.write((char*)&node_levels_size, sizeof(node_levels_size));
    ofs.write((char*)flat->node_levels.data(), node_levels_size * sizeof(int));
    
    uint64_t upper_offsets_size = flat->upper_link_offsets.size();
    ofs.write((char*)&upper_offsets_size, sizeof(upper_offsets_size));
    ofs.write((char*)flat->upper_link_offsets.data(), upper_offsets_size * sizeof(int));
    
    uint64_t upper_storage_size = flat->upper_link_storage.size();
    ofs.write((char*)&upper_storage_size, sizeof(upper_storage_size));
    ofs.write((char*)flat->upper_link_storage.data(), upper_storage_size * sizeof(int));
    
    // 写入 label_lookup
    uint64_t label_lookup_size = flat->label_lookup.size();
    ofs.write((char*)&label_lookup_size, sizeof(label_lookup_size));
    if (label_lookup_size > 0) {
        ofs.write((char*)flat->label_lookup.data(), label_lookup_size * sizeof(int));
    }
    
    // 【移除】不再保存 node_l2_norms 和 pivot_dists（版本4）
    
    return ofs.good();
}

// 从文件加载 FlatHNSW
static FlatHNSW* load_flat_index(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return nullptr;
    
    // 验证魔数和版本
    uint32_t magic, version;
    ifs.read((char*)&magic, sizeof(magic));
    ifs.read((char*)&version, sizeof(version));
    
    if (magic != 0x48535746) return nullptr;
    if (version < 1 || version > INDEX_VERSION) return nullptr;  // 支持版本1-4
    
    int dim;
    ifs.read((char*)&dim, sizeof(dim));
    
    FlatHNSW* flat = new FlatHNSW(dim);
    
    ifs.read((char*)&flat->max_m, sizeof(flat->max_m));
    ifs.read((char*)&flat->max_m_upper, sizeof(flat->max_m_upper));
    ifs.read((char*)&flat->enter_point, sizeof(flat->enter_point));
    ifs.read((char*)&flat->num_nodes, sizeof(flat->num_nodes));
    ifs.read((char*)&flat->max_level, sizeof(flat->max_level));
    
    // 读取数据向量（使用对齐数组）
    uint64_t data_size;
    ifs.read((char*)&data_size, sizeof(data_size));
    flat->data.resize(data_size);
    if (data_size > 0) {
        ifs.read((char*)flat->data.data(), data_size * sizeof(float));
    }
    
    // 读取 L0 CSR 结构
    uint64_t l0_offsets_size;
    ifs.read((char*)&l0_offsets_size, sizeof(l0_offsets_size));
    flat->l0_offsets.resize(l0_offsets_size);
    ifs.read((char*)flat->l0_offsets.data(), l0_offsets_size * sizeof(uint64_t));
    
    uint64_t l0_links_size;
    ifs.read((char*)&l0_links_size, sizeof(l0_links_size));
    flat->l0_links.resize(l0_links_size);
    ifs.read((char*)flat->l0_links.data(), l0_links_size * sizeof(int));
    
    // 读取上层结构
    uint64_t node_levels_size;
    ifs.read((char*)&node_levels_size, sizeof(node_levels_size));
    flat->node_levels.resize(node_levels_size);
    ifs.read((char*)flat->node_levels.data(), node_levels_size * sizeof(int));
    
    uint64_t upper_offsets_size;
    ifs.read((char*)&upper_offsets_size, sizeof(upper_offsets_size));
    flat->upper_link_offsets.resize(upper_offsets_size);
    ifs.read((char*)flat->upper_link_offsets.data(), upper_offsets_size * sizeof(int));
    
    uint64_t upper_storage_size;
    ifs.read((char*)&upper_storage_size, sizeof(upper_storage_size));
    flat->upper_link_storage.resize(upper_storage_size);
    ifs.read((char*)flat->upper_link_storage.data(), upper_storage_size * sizeof(int));
    
    // 读取 label_lookup（版本2+）
    if (version >= 2) {
        uint64_t label_lookup_size;
        ifs.read((char*)&label_lookup_size, sizeof(label_lookup_size));
        if (label_lookup_size > 0) {
            flat->label_lookup.resize(label_lookup_size);
            ifs.read((char*)flat->label_lookup.data(), label_lookup_size * sizeof(int));
        }
    }
    
    // 【兼容性处理】版本3的文件包含 L2 范数数据，需要跳过
    if (version == 3) {
        uint64_t norms_size;
        ifs.read((char*)&norms_size, sizeof(norms_size));
        if (norms_size > 0) {
            // 跳过这部分数据
            ifs.seekg(norms_size * sizeof(float), std::ios::cur);
        }
    }
    
    // 版本4及以后不存储剪枝数组
    
    if (!ifs.good()) {
        delete flat;
        return nullptr;
    }
    
    return flat;
}



// 生成索引缓存文件名（基于参数的哈希）
static std::string get_index_cache_path(int n, int d, int M, int max_layer, int efc) {
    // 简单哈希：使用参数组合生成唯一文件名
    std::stringstream ss;
    ss << "cache/hnsw_n" << n << "_d" << d 
       << "_M" << M << "_L" << max_layer << "_efc" << efc << ".idx";
    return ss.str();
}