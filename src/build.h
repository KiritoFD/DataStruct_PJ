class HnswSolutionParallel {
public:
    FlatHNSW* flat_index = nullptr;
    std::vector<int> point_ids;

    ~HnswSolutionParallel() { 
        delete flat_index;
    }

    void build_from_memory(int d, const float* data, int n) {
        delete flat_index;
        flat_index = nullptr;
        
        int M = g_HNSW_M.load();
        int max_layer = g_HNSW_MAX_LAYER.load();
        int efc = g_HNSW_EF_CONSTRUCTION.load();

        // 尝试从缓存加载 FlatHNSW
        std::string cache_path = get_index_cache_path(n, d, M, max_layer, efc);
        #ifdef _WIN32
        _mkdir("cache");
        #else
        mkdir("cache", 0755);
        #endif
        
        auto cache_start = std::chrono::high_resolution_clock::now();
        flat_index = load_flat_index(cache_path);
        auto cache_end = std::chrono::high_resolution_clock::now();
        
        if (flat_index != nullptr) {
            double cache_ms = std::chrono::duration<double, std::milli>(cache_end - cache_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Loaded index from: " << cache_path << std::endl;
                std::cout << "[Cache] Load time: " << std::fixed << std::setprecision(2) 
                          << cache_ms << " ms" << std::endl;
            }
            g_last_build_ms.store(cache_ms, std::memory_order_relaxed);
            point_ids.resize(n);
            for (int i = 0; i < n; ++i) point_ids[i] = i;
            return;
        }

        // 缓存未命中，使用临时构建器构建图
        auto build_start = std::chrono::high_resolution_clock::now();
        
        HnswGraphBuilder* builder = new HnswGraphBuilder(d, M, max_layer, data, n);
        
        builder->nodes.reserve(n);
        
        std::vector<int> levels(n);
        for (int i = 0; i < n; ++i) {
            levels[i] = std::min(builder->randomLevel(), max_layer);
            builder->nodes.push_back(new HNSWNode(levels[i], M));
        }
        
        if (n > 0) builder->enter_point = 0;

        ThreadPool* pool = getThreadPool();
        std::atomic<int> processed(1);
        int chunk_size = 1000;

        for (int i = 1; i < n; i += chunk_size) {
            int end = std::min(i + chunk_size, n);
            pool->enqueue([builder, i, end, &levels, &processed]() {
                for (int j = i; j < end; ++j) {
                    builder->insertPointParallel(j, levels[j]);
                }
                processed.fetch_add(end - i, std::memory_order_release);
            });
        }

        std::thread progress_thread([&processed, n, &build_start]() {
            int last_reported = 0;
            while (processed.load(std::memory_order_acquire) < n) {
                int curr = processed.load(std::memory_order_acquire);
                if (curr - last_reported >= std::max(50000, n / 100)) {
                    double pct = 100.0 * curr / n;
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_ms = std::chrono::duration<double, std::milli>(now - build_start).count();
                    if (DEBUG_TIMING) {
                        std::cout << "[Progress] " << curr << "/" << n 
                                  << " (" << std::fixed << std::setprecision(1) << pct << "%) "
                                  << "Time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms" << std::endl;
                        std::cout.flush();
                    }
                    last_reported = curr;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        });

        while (processed.load(std::memory_order_acquire) < n) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        progress_thread.join();

        auto build_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Parallel Build: " << std::fixed << std::setprecision(2) 
                      << total_ms << " ms for " << n << " points." << std::endl;
            std::cout.flush();
        }

        // 转换为扁平化结构
        auto convert_start = std::chrono::high_resolution_clock::now();
        flat_index = convert_to_flat(builder);
        auto convert_end = std::chrono::high_resolution_clock::now();
        double convert_ms = std::chrono::duration<double, std::milli>(convert_end - convert_start).count();
        
        if (DEBUG_TIMING) {
            std::cout << "[Timing] Flat Conversion: " << std::fixed << std::setprecision(2) 
                      << convert_ms << " ms" << std::endl;
            std::cout.flush();
        }
        
        g_last_build_ms.store(total_ms + convert_ms, std::memory_order_relaxed);
        
        // 删除临时构建器
        delete builder;
        
        // 保存到缓存
        auto save_start = std::chrono::high_resolution_clock::now();
        if (save_flat_index(flat_index, cache_path)) {
            auto save_end = std::chrono::high_resolution_clock::now();
            double save_ms = std::chrono::duration<double, std::milli>(save_end - save_start).count();
            if (DEBUG_TIMING) {
                std::cout << "[Cache] Saved index to: " << cache_path << std::endl;
                std::cout << "[Cache] Save time: " << std::fixed << std::setprecision(2) 
                          << save_ms << " ms" << std::endl;
            }
        }

        point_ids.resize(n);
        for (int i = 0; i < n; ++i) point_ids[i] = i;
    }

    std::vector<std::pair<int, float>> search(const std::vector<float>& query, int k) {
        tl_dist_counter = 0;

        if (!flat_index || flat_index->enter_point < 0) return {};

        int ep = flat_index->enter_point;
        int max_l = flat_index->node_levels[ep];
        int curr = ep;

        int l = std::min(max_l, 4);
        while (l > 0) {
            curr = flat_index->greedySearchUpper(curr, query.data(), l);
            l = (l > 1) ? (l - 2) : (l - 1);
        }
    
        auto top = flat_index->searchL0(query.data(), curr, g_HNSW_EF_SEARCH.load());
        
        std::vector<std::pair<int, float>> out;
        int cnt = std::min(k, (int)top.size());
        out.reserve(cnt);
        for (int i = 0; i < cnt; ++i) {
            int original_idx;
            if (!flat_index->label_lookup.empty()) {
                original_idx = flat_index->label_lookup[top[i].second];
            } else {
                original_idx = top[i].second;
            }
            
            if (original_idx >= 0 && original_idx < (int)point_ids.size()) {
                out.push_back({point_ids[original_idx], top[i].first});
            }
        }

        uint64_t last = tl_dist_counter;
        tl_dist_counter = 0;
        g_last_query_dist.store(last, std::memory_order_relaxed);
        g_total_dist_count.fetch_add(last, std::memory_order_relaxed);
        g_total_query_count.fetch_add(1, std::memory_order_relaxed);

        return out;
    }
};