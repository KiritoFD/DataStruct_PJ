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

static void generate_dfs_reordering(SimpleHNSW* src, std::vector<int>& old_to_new, std::vector<int>& new_to_old) {
    int N = src->size();
    // 初始化映射表：old_to_new 初始化为 -1 表示未访问
    old_to_new.assign(N, -1);
    new_to_old.resize(N);
    
    int new_id_counter = 0;
    std::vector<int> stack;
    // 预分配内存以避免频繁扩容，N 为节点总数
    stack.reserve(N);
    
    // 优先从入口点开始遍历，保证搜索起点的局部性
    if (src->enter_point != -1 && src->enter_point < N) {
        stack.push_back(src->enter_point);
    }
    
    // 用于处理非连通图或初始点的扫描指针
    int scan_idx = 0;
    
    while (new_id_counter < N) {
        // 如果栈空了，说明当前连通分量遍历完毕，或者刚开始
        if (stack.empty()) {
            // 寻找下一个未访问的节点
            while (scan_idx < N && old_to_new[scan_idx] != -1) {
                scan_idx++;
            }
            if (scan_idx < N) {
                stack.push_back(scan_idx);
            } else {
                // 所有节点都处理完毕
                break;
            }
        }
        
        // 弹出栈顶元素
        int u = stack.back();
        stack.pop_back();
        
        // 如果已访问过，跳过
        if (old_to_new[u] != -1) continue;
        
        // 建立映射关系：旧ID -> 新ID (连续递增)
        old_to_new[u] = new_id_counter;
        new_to_old[new_id_counter] = u;
        new_id_counter++;
        
        // 将邻居入栈
        // 使用 Layer 0 的连接，因为这层最密集，对缓存影响最大
        if (u >= 0 && u < (int)src->nodes.size()) {
            const auto& links = src->nodes[u]->links[0];
            // 倒序遍历邻居并入栈，这样出栈顺序（即访问顺序）就是正序的
            // 这有助于保持与原始构建顺序或启发式选边顺序的一致性
            for (auto it = links.rbegin(); it != links.rend(); ++it) {
                int v = *it;
                // 只将合法且未访问的邻居入栈
                if (v >= 0 && v < N && old_to_new[v] == -1) {
                    stack.push_back(v);
                }
            }
        }
    }
}
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

static FlatHNSW* convert_to_flat(SimpleHNSW* src) {
    FlatHNSW* flat = new FlatHNSW(src->dim);

    int N = src->size();
    int M = src->M;
    flat->num_nodes = N;
    flat->max_m = M * 2;
    flat->max_m_upper = M;

    if (DEBUG_TIMING) {
        std::cout << "[FlatConvert] Starting conversion for " << N << " nodes, dim=" << src->dim << std::endl;
        std::cout.flush();
    }

    // 判断是否启用重排优化
    bool do_reorder = !ABLATE_REORDER.load(std::memory_order_relaxed);

    std::vector<int> old_to_new, new_to_old;

    if (do_reorder) {
        // 1. 生成 ID 映射（使用 DFS 重排）
        generate_dfs_reordering(src, old_to_new, new_to_old);

        // 更新入口点
        flat->enter_point = (src->enter_point == -1) ? -1 : old_to_new[src->enter_point];

        // 计算 max_level
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) flat->max_level = level;
        }

        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder enabled, enter_point=" << flat->enter_point
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }

        flat->label_lookup = new_to_old;
    } else {
        flat->enter_point = src->enter_point;
        flat->max_level = 0;
        for (int i = 0; i < N; ++i) {
            int level = (int)src->nodes[i]->links.size() - 1;
            if (level > flat->max_level) flat->max_level = level;
        }

        if (DEBUG_TIMING) {
            std::cout << "[FlatConvert] Reorder disabled, enter_point=" << flat->enter_point
                      << ", max_level=" << flat->max_level << std::endl;
            std::cout.flush();
        }

        old_to_new.resize(N);
        new_to_old.resize(N);
        for (int i = 0; i < N; ++i) { old_to_new[i] = i; new_to_old[i] = i; }
    }

    // 2. 重排/复制向量数据
    flat->data.resize((size_t)N * flat->dim);
    if (flat->data.data() == nullptr) {
        std::cerr << "[FlatConvert] ERROR: Failed to allocate data array!" << std::endl;
        delete flat;
        return nullptr;
    }

    for (int new_id = 0; new_id < N; ++new_id) {
        int old_id = new_to_old[new_id];
        const float* src_vec = src->getVec(old_id);
        float* dst_vec = flat->data.data() + (size_t)new_id * flat->dim;
        std::memcpy(dst_vec, src_vec, flat->dim * sizeof(float));
    }
    
    // 【移除】不再预计算 L2 范数和 pivot 距离
    // 这些剪枝策略在高维空间反而降低性能

    // 3. 构建 L0 CSR
    flat->l0_offsets.resize(N + 1);
    uint64_t total_l0_links = 0;
    for (int i = 0; i < N; ++i) total_l0_links += src->nodes[i]->links[0].size();
    flat->l0_links.resize(total_l0_links);
    uint64_t current_offset = 0;

    std::vector<std::pair<float,int>> temp_neighbors;
    temp_neighbors.reserve(M * 2);

    for (int new_id = 0; new_id < N; ++new_id) {
        flat->l0_offsets[new_id] = current_offset;
        int old_id = new_to_old[new_id];
        const auto& src_links = src->nodes[old_id]->links[0];
        temp_neighbors.clear();
        const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;

        for (int old_nb : src_links) {
            if (old_nb < 0 || old_nb >= N) continue;
            int new_nb = old_to_new[old_nb];
            if (new_nb < 0 || new_nb >= N) continue;
            const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
            float d = l2sq_100d(vec_u, vec_v);
            temp_neighbors.emplace_back(d, new_nb);
        }
        std::sort(temp_neighbors.begin(), temp_neighbors.end());
        for (const auto &p : temp_neighbors) {
            flat->l0_links[current_offset++] = p.second;
        }
    }
    flat->l0_offsets[N] = current_offset;

    // 4. 构建上层结构
    flat->node_levels.resize(N);
    if (flat->max_level < 0) flat->max_level = 0;
    if (flat->max_level > 100) flat->max_level = 100;
    flat->upper_link_offsets.assign((size_t)N * (flat->max_level + 1), -1);
    flat->upper_link_storage.reserve(N * M);

    for (int new_id = 0; new_id < N; ++new_id) {
        int old_id = new_to_old[new_id];
        int level = (int)src->nodes[old_id]->links.size() - 1;
        flat->node_levels[new_id] = level;
        for (int l = 1; l <= level && l <= flat->max_level; ++l) {
            const auto& src_links = src->nodes[old_id]->links[l];
            int storage_idx = (int)flat->upper_link_storage.size();
            size_t offset_idx = (size_t)new_id * (flat->max_level + 1) + l;
            if (offset_idx < flat->upper_link_offsets.size()) flat->upper_link_offsets[offset_idx] = storage_idx;
            flat->upper_link_storage.push_back((int)src_links.size());

            temp_neighbors.clear();
            const float* vec_u = flat->data.data() + (size_t)new_id * flat->dim;
            for (int old_nb : src_links) {
                if (old_nb < 0 || old_nb >= N) continue;
                int new_nb = old_to_new[old_nb];
                if (new_nb < 0 || new_nb >= N) continue;
                const float* vec_v = flat->data.data() + (size_t)new_nb * flat->dim;
                float d = l2sq_100d(vec_u, vec_v);
                temp_neighbors.emplace_back(d, new_nb);
            }
            std::sort(temp_neighbors.begin(), temp_neighbors.end());
            for (const auto &p : temp_neighbors) flat->upper_link_storage.push_back(p.second);
        }
    }

    if (DEBUG_TIMING) {
        std::cout << "[FlatHNSW-CSR] Converted " << N << " nodes (optimized for cache, no pruning overhead)." << std::endl;
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