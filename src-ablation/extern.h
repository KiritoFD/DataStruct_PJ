extern "C" {

void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads) {
    if (M > 0) g_HNSW_M.store(M);
    if (max_layer > 0) g_HNSW_MAX_LAYER.store(max_layer);
    if (ef_construction > 0) g_HNSW_EF_CONSTRUCTION.store(ef_construction);
    if (ef_search > 0) g_HNSW_EF_SEARCH.store(ef_search);
    if (build_threads > 0) {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        HNSW_BUILD_THREADS.store(build_threads);
        if (g_thread_pool) { delete g_thread_pool; g_thread_pool = new ThreadPool(build_threads); }
    }
}

void set_hnsw_debug(int dbg) { DEBUG_TIMING = (dbg != 0); }

// 消融标志设置
void set_ablation_flags(int prefetch, int simd, int pruning, int heap, int reorder, int adaptive_ep) {
    ABLATE_PREFETCH.store(prefetch != 0);
    ABLATE_SIMD.store(simd != 0);
    ABLATE_PRUNING.store(pruning != 0);
    ABLATE_HEAP.store(heap != 0);
    ABLATE_REORDER.store(reorder != 0);
    ABLATE_ADAPTIVE_EP.store(adaptive_ep != 0);
}

void get_ablation_flags(int* prefetch, int* simd, int* pruning, int* heap, int* reorder, int* adaptive_ep) {
    if (prefetch) *prefetch = ABLATE_PREFETCH.load() ? 1 : 0;
    if (simd) *simd = ABLATE_SIMD.load() ? 1 : 0;
    if (pruning) *pruning = ABLATE_PRUNING.load() ? 1 : 0;
    if (heap) *heap = ABLATE_HEAP.load() ? 1 : 0;
    if (reorder) *reorder = ABLATE_REORDER.load() ? 1 : 0;
    if (adaptive_ep) *adaptive_ep = ABLATE_ADAPTIVE_EP.load() ? 1 : 0;
}

void set_ablate_prefetch(int on) { ABLATE_PREFETCH.store(on != 0); }
void set_ablate_simd(int on) { ABLATE_SIMD.store(on != 0); }
void set_ablate_pruning(int on) { ABLATE_PRUNING.store(on != 0); }
void set_ablate_heap(int on) { ABLATE_HEAP.store(on != 0); }
void set_ablate_reorder(int on) { ABLATE_REORDER.store(on != 0); }
void set_ablate_adaptive_ep(int on) { ABLATE_ADAPTIVE_EP.store(on != 0); }

bool get_ablate_reorder() { return ABLATE_REORDER.load(); }
bool get_ablate_adaptive_ep() { return ABLATE_ADAPTIVE_EP.load(); }

// 自适应起点参数
void set_adaptive_ep_k(int k) { ADAPTIVE_EP_K.store(k); }
void set_kmeans_iterations(int iter) { KMEANS_ITERATIONS.store(iter); }
void set_adaptive_ep_num_probes(int num) { ADAPTIVE_EP_NUM_PROBES.store(num); }
int get_adaptive_ep_num_probes() { return ADAPTIVE_EP_NUM_PROBES.load(); }

// 距离统计
uint64_t get_total_queries() { return g_total_query_count.load(); }
double get_avg_dists_per_query() {
    uint64_t q = g_total_query_count.load();
    return q == 0 ? 0.0 : (double)g_total_dist_count.load() / q;
}
uint64_t get_last_query_dists() { return g_last_query_dist.load(); }
void reset_dist_counters() {
    g_total_dist_count.store(0); g_total_query_count.store(0); g_last_query_dist.store(0);
}
double get_last_build_time_ms() { return g_last_build_ms.load(); }

// 图统计
int get_graph_max_level() { return g_impl && g_impl->flat_index ? g_impl->flat_index->max_level : 0; }
int get_graph_num_nodes() { return g_impl && g_impl->flat_index ? g_impl->flat_index->num_nodes : 0; }

double get_graph_avg_degree_l0() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0) return 0.0;
    uint64_t total = 0;
    for (int i = 0; i < idx->num_nodes; ++i) { int c; idx->get_l0_links(i, c); total += c; }
    return (double)total / idx->num_nodes;
}

int get_graph_actual_max_layer() {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int mx = 0;
    for (int i = 0; i < idx->num_nodes; ++i) if (idx->node_levels[i] > mx) mx = idx->node_levels[i];
    return mx;
}

int get_graph_nodes_at_level(int level) {
    if (!g_impl || !g_impl->flat_index) return 0;
    auto* idx = g_impl->flat_index;
    int cnt = 0;
    for (int i = 0; i < idx->num_nodes; ++i) if (idx->node_levels[i] >= level) ++cnt;
    return cnt;
}

double get_graph_avg_degree_upper() {
    if (!g_impl || !g_impl->flat_index) return 0.0;
    auto* idx = g_impl->flat_index;
    if (idx->num_nodes == 0 || idx->max_level == 0) return 0.0;
    uint64_t total = 0; int cnt = 0;
    for (int i = 0; i < idx->num_nodes; ++i) {
        for (int l = 1; l <= idx->node_levels[i]; ++l) {
            int c; idx->get_upper_links(i, l, c); total += c; ++cnt;
        }
    }
    return cnt > 0 ? (double)total / cnt : 0.0;
}

} // extern "C"