extern "C" {

void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads) {
    if (M > 0) g_HNSW_M.store(M);
    if (max_layer > 0) g_HNSW_MAX_LAYER.store(max_layer);
    if (ef_construction > 0) g_HNSW_EF_CONSTRUCTION.store(ef_construction);
    if (ef_search > 0) g_HNSW_EF_SEARCH.store(ef_search);

    if (build_threads > 0) {
        int old = HNSW_BUILD_THREADS.load();
        if (build_threads != old) {
            std::lock_guard<std::mutex> lock(g_pool_mutex);
            HNSW_BUILD_THREADS.store(build_threads);
            if (g_thread_pool) {
                delete g_thread_pool;
                g_thread_pool = new ThreadPool(build_threads);
            }
        }
    }
}

void set_hnsw_debug(int dbg) { DEBUG_TIMING = (dbg != 0); }

// Set ablation flags at runtime to toggle features for experiments
void set_ablation_flags(int csr, int prefetch, int simd, int pruning, int heap) {
    ABLATE_CSR.store(csr != 0);
    ABLATE_PREFETCH.store(prefetch != 0);
    ABLATE_SIMD.store(simd != 0);
    ABLATE_PRUNING.store(pruning != 0);
    ABLATE_HEAP.store(heap != 0);
}

void get_ablation_flags(int* csr, int* prefetch, int* simd, int* pruning, int* heap) {
    if (csr) *csr = ABLATE_CSR.load() ? 1 : 0;
    if (prefetch) *prefetch = ABLATE_PREFETCH.load() ? 1 : 0;
    if (simd) *simd = ABLATE_SIMD.load() ? 1 : 0;
    if (pruning) *pruning = ABLATE_PRUNING.load() ? 1 : 0;
    if (heap) *heap = ABLATE_HEAP.load() ? 1 : 0;
}

// Convenience setters
void set_ablate_csr(int on) { ABLATE_CSR.store(on != 0); }
void set_ablate_prefetch(int on) { ABLATE_PREFETCH.store(on != 0); }
void set_ablate_simd(int on) { ABLATE_SIMD.store(on != 0); }
void set_ablate_pruning(int on) { ABLATE_PRUNING.store(on != 0); }
void set_ablate_heap(int on) { ABLATE_HEAP.store(on != 0); }
void set_ablate_flat_index(int on) { ABLATE_FLAT_INDEX.store(on != 0); }  // 新增
void set_ablate_reorder(bool v) {
    ABLATE_REORDER.store(v, std::memory_order_relaxed);
}

bool get_ablate_reorder() {
    return ABLATE_REORDER.load(std::memory_order_relaxed);
}

// New: enable/disable runtime distance counting
void set_enable_dist_counting(int on) {
    ENABLE_RUNTIME_DIST_COUNTING.store(on != 0, std::memory_order_relaxed);
}

uint64_t get_total_queries() { return g_total_query_count.load(std::memory_order_relaxed); }
double get_avg_dists_per_query() {
    uint64_t q = g_total_query_count.load(std::memory_order_relaxed);
    if (q == 0) return 0.0;
    return double(g_total_dist_count.load(std::memory_order_relaxed)) / double(q);
}
uint64_t get_last_query_dists() { return g_last_query_dist.load(std::memory_order_relaxed); }
void reset_dist_counters() {
    g_total_dist_count.store(0, std::memory_order_relaxed);
    g_total_query_count.store(0, std::memory_order_relaxed);
    g_last_query_dist.store(0, std::memory_order_relaxed);
}
double get_last_build_time_ms() { return g_last_build_ms.load(std::memory_order_relaxed); }

// 图质量统计函数
int get_graph_max_level() {
    if (!g_impl || !g_impl->flat_index) return 0;
    return g_impl->flat_index->max_level;
}

int get_graph_num_nodes() {
    if (!g_impl || !g_impl->flat_index) return 0;
    return g_impl->flat_index->num_nodes;
}

double get_graph_avg_degree_l0() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0) return 0.0;
    
    uint64_t total_degree = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        int count;
        idx->get_l0_links(i, count);
        total_degree += count;
    }
    return double(total_degree) / double(idx->num_nodes);
}

int get_graph_actual_max_layer() {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int max_lv = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        if (idx->node_levels[i] > max_lv) {
            max_lv = idx->node_levels[i];
        }
    }
    return max_lv;
}

// 获取各层级节点数量分布 (返回层级 l 的节点数)
int get_graph_nodes_at_level(int level) {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int count = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        if (idx->node_levels[i] >= level) {
            ++count;
        }
    }
    return count;
}

// 获取上层平均度数
double get_graph_avg_degree_upper() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0 || idx->max_level == 0) return 0.0;
    
    uint64_t total_degree = 0;
    int total_upper_nodes = 0;
    
    for (int i = 0; i < idx->num_nodes; ++i) {
        for (int l = 1; l <= idx->node_levels[i]; ++l) {
            int count;
            idx->get_upper_links(i, l, count);
            total_degree += count;
            ++total_upper_nodes;
        }
    }
    
    return total_upper_nodes > 0 ? double(total_degree) / double(total_upper_nodes) : 0.0;
}



inline void set_post_optimization(bool enable) {
    ENABLE_POST_OPTIMIZATION.store(enable, std::memory_order_relaxed);
}

inline void set_post_opt_m(int m) {
    POST_OPT_M.store(m, std::memory_order_relaxed);
}

inline void set_pruning_alpha(float alpha) {
    PRUNING_ALPHA.store(alpha, std::memory_order_relaxed);
}

// 便捷函数：设置优化配置组合
inline void configure_for_low_ndc(int target_m = 32, float alpha = 1.0f) {
    set_post_optimization(true);
    set_post_opt_m(target_m);
    set_pruning_alpha(alpha);
}

inline void configure_for_high_recall(int target_m = 55, float alpha = 1.2f) {
    set_post_optimization(false);
    set_post_opt_m(target_m);
    set_pruning_alpha(alpha);
}

} // extern "C"