class HnswGraphBuilder {
public:
    int dim;
    int M;
    int maxLayer;
    
    const float* data_ptr;  // 外部数据指针
    int data_count;
    
    std::vector<HNSWNode*> nodes;
    int enter_point;
    std::shared_mutex global_mutex;

    HnswGraphBuilder(int d, int m, int ml, const float* data, int n)
        : dim(d), M(m), maxLayer(ml), data_ptr(data), data_count(n), enter_point(-1) {}

    ~HnswGraphBuilder() { for (auto p : nodes) delete p; }

    inline int size() const { return (int)nodes.size(); }
    
    inline const float* getVec(int id) const {
        return data_ptr + (size_t)id * dim;
    }

    int randomLevel() {
        static thread_local std::minstd_rand rng((unsigned)std::random_device{}());
        static thread_local std::uniform_real_distribution<float> ud(0.f, 1.f);
        float r = ud(rng);
        return (int)(-std::log(r) * (1.0 / std::log((float)M)));
    }

    inline float dist(int id, const float* q) const {
        return l2sq_100d(getVec(id), q);
    }
    
    inline float distNodes(int id_a, int id_b) const {
        return l2sq_100d(getVec(id_a), getVec(id_b));
    }

    int greedySearch(int ep, const float* q, int l) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return -1;
        
        float curd = dist(ep, q);
        bool changed = true;
        
        while (changed) {
            changed = false;
            
            std::shared_lock<std::shared_mutex> lock_guard(nodes[ep]->lock);
            const auto& neighbors = nodes[ep]->links[l];
            
            const int nsize = (int)neighbors.size();
            
            if (nsize > 0) {
                my_prefetch_l1(getVec(neighbors[0]));
            }
            
            int best_nb = -1;
            float best_d = curd;
            
            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    my_prefetch_l1(getVec(neighbors[i+1]));
                }
                float nd = dist(nb, q);
                if (nd < best_d) {
                    best_d = nd;
                    best_nb = nb;
                }
            }
            
            if (best_nb >= 0) {
                curd = best_d;
                ep = best_nb;
                changed = true;
            }
        }
        return ep;
    }

    std::vector<std::pair<float, int>> searchLayer(const float* q, int ep, int l, int ef) const {
        if (__builtin_expect(ep < 0 || ep >= size(), 0)) return {};
        
        using Pair = std::pair<float, int>;
        
        static thread_local std::vector<Pair> top_candidates;
        static thread_local std::vector<Pair> candidate_queue;
        static thread_local VisitedList visited_list;
        
        top_candidates.clear();
        candidate_queue.clear();
        top_candidates.reserve(ef + 1);
        candidate_queue.reserve(ef * 2);
        
        visited_list.init(size());
        visited_list.advance();

        // 最小堆比较器
        auto greater_comp = [](const Pair& a, const Pair& b) { return a.first > b.first; };

        float d0 = dist(ep, q);
        visited_list.mark(ep);
        
        top_candidates.push_back({d0, ep});
        candidate_queue.push_back({d0, ep});
        std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);

        float lower_bound = d0;

        while (!candidate_queue.empty()) {
            std::pop_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
            auto curr = candidate_queue.back();
            candidate_queue.pop_back();

            // 关键剪枝：当前最近候选已超过结果集最远距离
            if (curr.first > lower_bound && (int)top_candidates.size() >= ef) {
                break;
            }

            std::shared_lock<std::shared_mutex> lock_guard(nodes[curr.second]->lock);
            const auto& neighbors = nodes[curr.second]->links[l];
            
            const int nsize = (int)neighbors.size();

            if (nsize > 0) {
                my_prefetch_l1(getVec(neighbors[0]));
            }

            for (int i = 0; i < nsize; ++i) {
                int nb = neighbors[i];
                if (i + 1 < nsize) {
                    my_prefetch_l1(getVec(neighbors[i+1]));
                }

                if (!visited_list.isVisited(nb)) {
                    visited_list.mark(nb);
                    float d_nb = dist(nb, q);

                    if ((int)top_candidates.size() < ef || d_nb < lower_bound) {
                        auto it = std::upper_bound(top_candidates.begin(), top_candidates.end(),
                            Pair{d_nb, nb}, [](const Pair& a, const Pair& b) { return a.first < b.first; });
                        top_candidates.insert(it, {d_nb, nb});

                        if ((int)top_candidates.size() > ef) {
                            top_candidates.pop_back();
                        }
                        lower_bound = top_candidates.back().first;
                    }

                    candidate_queue.push_back({d_nb, nb});
                    std::push_heap(candidate_queue.begin(), candidate_queue.end(), greater_comp);
                }
            }
        }

        return top_candidates;
    }

    void connectNodeHeuristic(int id, const std::vector<std::pair<float, int>>& candidates, int l) {
        if (id < 0 || id >= size()) return;
        int m_max = (l == 0) ? M * 2 : M;

        std::vector<std::pair<float, int>> all_candidates;
        all_candidates.reserve(candidates.size() + m_max);
        
        for (const auto& p : candidates) {
            all_candidates.push_back(p);
        }

        {
            std::shared_lock<std::shared_mutex> lock(nodes[id]->lock);
            const auto& old_links = nodes[id]->links[l];
            for (int old_nb : old_links) {
                if (old_nb >= 0 && old_nb < size()) {
                    all_candidates.push_back({distNodes(id, old_nb), old_nb});
                }
            }
        }

        std::sort(all_candidates.begin(), all_candidates.end());
        all_candidates.erase(
            std::unique(all_candidates.begin(), all_candidates.end(),
                [](const auto& a, const auto& b) { return a.second == b.second; }),
            all_candidates.end()
        );

        std::vector<int> result_links;
        result_links.reserve(m_max);

        if (ABLATE_PRUNING.load(std::memory_order_relaxed)) {
            int taken = 0;
            for (const auto& cand : all_candidates) {
                if (taken >= m_max) break;
                if (cand.second == id) continue;
                result_links.push_back(cand.second);
                ++taken;
            }
        } else {
            for (const auto& cand : all_candidates) {
                if ((int)result_links.size() >= m_max) break;

                float d_cand_to_curr = cand.first;
                int cand_id = cand.second;
                
                if (cand_id == id) continue;

                bool keep = true;
                for (int selected_nbr : result_links) {
                    float d_cand_to_selected = distNodes(cand_id, selected_nbr);
                    if (d_cand_to_selected < d_cand_to_curr) {
                        keep = false;
                        break;
                    }
                }

                if (keep) {
                    result_links.push_back(cand_id);
                }
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(nodes[id]->lock);
            nodes[id]->links[l] = std::move(result_links);
        }
    }
    
    void tryAddReverseLink(int target_id, int new_neighbor_id, float dist_val, int level) {
        if (target_id == new_neighbor_id) return;
        if (target_id < 0 || target_id >= size()) return;
        if (new_neighbor_id < 0 || new_neighbor_id >= size()) return;
        
        int m_max = (level == 0) ? M * 2 : M;
        
        bool worth_trying = false;
        
        {
            std::shared_lock<std::shared_mutex> read_lock(nodes[target_id]->lock);
            
            if (level >= (int)nodes[target_id]->links.size()) {
                return;
            }
            
            const auto& links = nodes[target_id]->links[level];
            
            if ((int)links.size() < m_max) {
                worth_trying = true;
            } else {
                int worst_link_id = links.back();
                float worst_d = distNodes(target_id, worst_link_id);
                
                if (dist_val < worst_d * 1.0001f) {
                    worth_trying = true;
                }
            }
        }
        
        if (worth_trying) {
            std::vector<std::pair<float, int>> new_cand = {{dist_val, new_neighbor_id}};
            connectNodeHeuristic(target_id, new_cand, level);
        }
    }

    void insertPointParallel(int id, int level) {
        int ep_curr;
        {
            std::shared_lock<std::shared_mutex> lock(global_mutex);
            ep_curr = enter_point;
        }

        if (ep_curr != -1) {
            int max_l = (int)nodes[ep_curr]->links.size() - 1;
            int curr = ep_curr;
            
            for (int l = max_l; l > level; l--) {
                curr = greedySearch(curr, getVec(id), l);
            }

            for (int l = std::min(level, max_l); l >= 0; l--) {
                auto top = searchLayer(getVec(id), curr, l, g_HNSW_EF_CONSTRUCTION.load());
                if (!top.empty()) curr = top[0].second;
                
                connectNodeHeuristic(id, top, l);
                
                for (const auto& candidate : top) {
                    int neighbor_id = candidate.second;
                    float dist_val = candidate.first;
                    tryAddReverseLink(neighbor_id, id, dist_val, l);
                }
            }
        }

        {
            std::unique_lock<std::shared_mutex> lock(global_mutex);
            if (enter_point == -1 || level > (int)nodes[enter_point]->links.size() - 1) {
                enter_point = id;
            }
        }
    }
};