#include "MySolution.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <numeric>
#include <limits>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include <sstream>

// add global process start-time
static std::chrono::high_resolution_clock::time_point g_process_start;
static bool g_process_start_set = false;

namespace {
    bool try_stod(const std::string& s, double& out) {
        try { size_t pos = 0; out = std::stod(s, &pos); return pos == s.size(); } catch (...) { return false; }
    }
}

bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec) {
    out_id.clear(); out_vec.clear(); std::istringstream iss(line); std::vector<std::string> toks; std::string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;
    double val = 0.0; bool allnum = true;
    for (const auto& s : toks) if (!try_stod(s, val)) { allnum = false; break; }
    if (allnum) { out_vec.reserve(toks.size()); for (const auto& s : toks) out_vec.push_back(std::stod(s)); return true; }
    if (toks.size() < 2) return false;
    out_id = toks[0]; out_vec.reserve(toks.size() - 1);
    for (size_t i = 1; i < toks.size(); ++i) { if (!try_stod(toks[i], val)) return false; out_vec.push_back(std::stod(toks[i])); }
    return true;
}

// Timing helpers
static inline std::string fmt_sec(double s) { std::ostringstream oss; oss << std::fixed << std::setprecision(6) << s << "s"; return oss.str(); }
static inline void print_stage(const std::string &name, double secs, int thread_id = -1) {
    if (thread_id >= 0) { std::cout << "[TIMING] " << name << " (thread " << thread_id << "): " << fmt_sec(secs) << '\n'; } else { std::cout << "[TIMING] " << name << ": " << fmt_sec(secs) << '\n'; }
}

// 内存优化：预计算并平铺存储
class FlatDatabase {
    std::vector<double> flat_data; std::vector<double> norms; int dim; int num_vectors;
public:
    FlatDatabase():dim(0),num_vectors(0){}
    void build(const std::vector<std::vector<double>>& original_data, int d) {
        dim=d; num_vectors=(int)original_data.size(); flat_data.resize((size_t)num_vectors*dim); norms.resize(num_vectors);
        for (int i=0;i<num_vectors;i++){ const auto& vec=original_data[i]; double norm=0.0; for(int j=0;j<dim;j++){ flat_data[i*dim+j]=vec[j]; norm+=vec[j]*vec[j]; } norms[i]=std::sqrt(norm); }
    }
    const double* get_vector(int idx) const { return flat_data.data()+idx*dim; }
    double get_norm(int idx) const { return norms[idx]; }
    int size() const { return num_vectors; }
    int get_dim() const { return dim; }
};
static FlatDatabase g_flat_db;
static std::vector<std::vector<double>> g_vector_to_centroid_distances;

// 在头文件中已经定义了solution类，这里只需要实现方法

double solution::compute_distance(const std::vector<double>& a, const std::vector<double>& b) const {
    if (metric=="l2"){ double s=0.0; for(int i=0;i<dim;++i){ double d=a[i]-b[i]; s+=d*d; } return s; }
    else if(metric=="cosine"){ double dot=0,na=0,nb=0; for(int i=0;i<dim;++i){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; } if(na==0||nb==0) return 1.0; return 1.0-dot/(std::sqrt(na)*std::sqrt(nb)); }
    return std::numeric_limits<double>::infinity();
}

double solution::compute_distance_simd(const double* a,const double* b,double norm_a,double norm_b) const {
    if(metric=="l2"){ double s=0.0; for(int i=0;i<dim;++i){ double d=a[i]-b[i]; s+=d*d; } return s; }
    else if(metric=="cosine"){ double dot=0,na=norm_a,nb=norm_b; if(na<0){ na=0; for(int i=0;i<dim;i++) na+=a[i]*a[i]; } if(nb<0){ nb=0; for(int i=0;i<dim;i++) nb+=b[i]*b[i]; } if(na==0||nb==0) return 1.0; for(int i=0;i<dim;i++) dot+=a[i]*b[i]; return 1.0-dot/(std::sqrt(na)*std::sqrt(nb)); }
    return std::numeric_limits<double>::infinity();
}

solution::solution(const std::string& metric_type):metric(metric_type),dim(0), centroid_distance_matrix(NUM_CENTROID, std::vector<double>(NUM_CENTROID)){ 
    unsigned int hc=std::thread::hardware_concurrency(); 
    num_threads=(int)(hc>0?hc:1); 
    if(!g_process_start_set){ 
        g_process_start=std::chrono::high_resolution_clock::now(); 
        g_process_start_set=true; 
    } 
    std::cout<<"[solution] hardware_concurrency="<<hc<<", using "<<num_threads<<" threads\n"; 
}

void solution::build(const std::string& base_file){
    auto t0=std::chrono::high_resolution_clock::now();
    std::ifstream fin(base_file); if(!fin){ print_stage("build - open file failed",0.0); return; }
    std::vector<std::vector<double>> vectors; std::string line; int local_dim=0; size_t lines_read=0;
    while(std::getline(fin,line)){ ++lines_read; std::string id; std::vector<double> vec; if(!parse_vector_line(line,id,vec)) continue; if(local_dim==0) local_dim=(int)vec.size(); if(vec.size()!=(size_t)local_dim) continue; vectors.push_back(std::move(vec)); }
    auto t1=std::chrono::high_resolution_clock::now();
    print_stage("build - file read", std::chrono::duration<double>(t1-t0).count());
    if(vectors.empty()){ database.clear(); centroids.clear(); inverted_index.clear(); centroid_distance_matrix.clear(); dim=0; return; }
    build_from_memory(local_dim,std::move(vectors));
}

void solution::build_from_memory(int d, std::vector<std::vector<double>> data){
    auto t0=std::chrono::high_resolution_clock::now();
    dim=d; database.clear(); database.reserve(data.size()); for(size_t i=0;i<data.size();++i) database.push_back({(int)i,std::move(data[i])});
    print_stage("build_from_memory - populate database", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count());
    finalize_build();
    print_stage("build_from_memory - finalize_build complete", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count());
}

void solution::finalize_build(){
    if(database.empty()||dim==0){ std::cout<<"[solution] finalize_build skipped\n"; centroids.clear(); inverted_index.clear(); centroid_distance_matrix.clear(); centroid_distance_matrix.resize(NUM_CENTROID, std::vector<double>(NUM_CENTROID)); return; }
    auto t_start=std::chrono::high_resolution_clock::now();
    std::cout<<"[solution] finalize_build: vectors="<<database.size()<<", dim="<<dim<<", centroids="<<NUM_CENTROID<<'\n';
    std::vector<std::vector<double>> original_data; original_data.reserve(database.size()); for(const auto& dp:database) original_data.push_back(dp.vec);
    g_flat_db.build(original_data,dim);
    std::mt19937 rng(42); centroids.clear(); centroids.reserve(NUM_CENTROID);
    int n=(int)database.size(); std::vector<int> indices(n); std::iota(indices.begin(),indices.end(),0); std::shuffle(indices.begin(),indices.end(),rng);
    int centers_to_select=std::min(NUM_CENTROID,n); for(int i=0;i<centers_to_select;i++) centroids.push_back(database[indices[i]].vec);
    while(centroids.size()<NUM_CENTROID && centroids.size()<(size_t)n) centroids.push_back(database[indices[centroids.size()%n]].vec);
    print_stage("finalize_build - random init centers",0.0);
    g_vector_to_centroid_distances.clear();
    std::vector<int> assignments(database.size()); double total_kmeans=0.0;
    for(int iter=0;iter<KMEAN_ITER;iter++){
        auto it0=std::chrono::high_resolution_clock::now();
        kmeans_assign_parallel(assignments);
        std::vector<std::vector<double>> new_centroids(NUM_CENTROID,std::vector<double>(dim,0.0));
        if(iter==0) g_vector_to_centroid_distances.assign(database.size(),std::vector<double>(centroids.size()));
        #pragma omp parallel for num_threads(num_threads)
        for(size_t i=0;i<database.size();++i) for(size_t c=0;c<centroids.size();++c) g_vector_to_centroid_distances[i][c]=compute_distance(database[i].vec,centroids[c]);
        kmeans_update_parallel(assignments,new_centroids);
        for(int i=0;i<NUM_CENTROID;++i) centroids[i]=std::move(new_centroids[i]);
        auto it1=std::chrono::high_resolution_clock::now(); double it_time=std::chrono::duration<double>(it1-it0).count(); total_kmeans+=it_time;
        std::cout<<"[solution] kmeans iter "<<(iter+1)<<"/"<<KMEAN_ITER<<" done, iter_time="<<fmt_sec(it_time)<<"\n";
    }
    print_stage("finalize_build - total kmeans",total_kmeans);
    
    // 预计算聚类中心之间的距离矩阵 - 这是真正的缓存！
    centroid_distance_matrix.resize(centroids.size(), std::vector<double>(centroids.size()));
    for(size_t i=0; i<centroids.size(); ++i) {
        for(size_t j=0; j<centroids.size(); ++j) {
            if(i == j) {
                centroid_distance_matrix[i][j] = 0.0;
            } else {
                centroid_distance_matrix[i][j] = compute_distance(centroids[i], centroids[j]);
            }
        }
    }
    print_stage("finalize_build - centroid distance matrix computed", std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_start).count());
    
    inverted_index.clear();
    if(!g_vector_to_centroid_distances.empty() && g_vector_to_centroid_distances.size()==database.size()){
        #pragma omp parallel for num_threads(num_threads)
        for(size_t i=0;i<database.size();++i){
            double min_dist=std::numeric_limits<double>::max(); int best_cluster=0;
            for(size_t c=0;c<centroids.size();++c){ double d=g_vector_to_centroid_distances[i][c]; if(d<min_dist){ min_dist=d; best_cluster=(int)c; } }
            #pragma omp critical
            inverted_index[best_cluster].push_back(database[i].id);
        }
    } else {
        for(const auto& dp:database){ int c=find_closest_centroid(dp.vec); inverted_index[c].push_back(dp.id); }
    }
    print_stage("finalize_build - inverted_index build",std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t_start).count());
}

void solution::kmeans_assign_parallel(std::vector<int>& assignments){
    if(!g_vector_to_centroid_distances.empty() && g_vector_to_centroid_distances.size()==database.size()){
        #pragma omp parallel for num_threads(num_threads)
        for(size_t i=0;i<database.size();++i){
            double min_dist=std::numeric_limits<double>::max(); int best_cluster=0;
            for(size_t c=0;c<centroids.size();++c){ double d=g_vector_to_centroid_distances[i][c]; if(d<min_dist){ min_dist=d; best_cluster=(int)c; } }
            assignments[i]=best_cluster;
        }
        return;
    }
    int threads_to_use=std::min<int>(num_threads,std::max<size_t>(1,database.size()));
    int chunk_size=(int(database.size())+threads_to_use-1)/threads_to_use; std::vector<std::thread> threads; threads.reserve(threads_to_use); std::vector<double> thread_times(threads_to_use,0.0);
    auto worker=[this,&assignments,&thread_times](int start,int end,int thread_id){ auto t0=std::chrono::high_resolution_clock::now(); for(int i=start;i<end;++i) assignments[i]=find_closest_centroid(database[i].vec); auto t1=std::chrono::high_resolution_clock::now(); thread_times[thread_id]=std::chrono::duration<double>(t1-t0).count(); };
    for(int t=0;t<threads_to_use;++t){ int start=t*chunk_size; int end=std::min(start+chunk_size,(int)database.size()); if(start<end) threads.emplace_back(worker,start,end,t); }
    for(auto&th:threads) th.join(); double sum=0; for(double v:thread_times) sum+=v; print_stage("kmeans_assign_parallel",sum);
}

int solution::find_closest_centroid_cached(int vec_idx) const { 
    double min_dist=std::numeric_limits<double>::max(); 
    int best_idx=0; 
    for(size_t c=0;c<centroids.size();++c){ 
        double d=g_vector_to_centroid_distances[vec_idx][c]; 
        if(d<min_dist){ min_dist=d; best_idx=(int)c; } 
    } 
    return best_idx; 
}

int solution::find_closest_centroid(const std::vector<double>& vec) const { 
    double min_dist=std::numeric_limits<double>::max(); 
    int best_idx=0; 
    for(int i=0;i<(int)centroids.size();++i){ 
        double d=compute_distance(vec,centroids[i]); 
        if(d<min_dist){ min_dist=d; best_idx=i; } 
    } 
    return best_idx; 
}

std::vector<std::pair<int,double>> solution::find_closest_centroids(const std::vector<double>& query,int nprobe) const {
    std::vector<std::pair<double,int>> dists; dists.reserve(centroids.size());
    for(int i=0;i<(int)centroids.size();++i){ double d=compute_distance(query,centroids[i]); dists.emplace_back(d,i); }
    if(nprobe>=(int)dists.size()) std::sort(dists.begin(),dists.end()); else { std::partial_sort(dists.begin(),dists.begin()+nprobe,dists.end()); dists.resize(nprobe); }
    std::vector<std::pair<int,double>> res; res.reserve(dists.size()); for(const auto& p:dists) res.emplace_back(p.second,p.first); return res;
}

std::vector<std::pair<int,double>> solution::search(const std::vector<double>& query, int k) {
    auto t_search_start = std::chrono::high_resolution_clock::now();
    double pre_search_elapsed = std::chrono::duration<double>(t_search_start - g_process_start).count();
    
    // 找到查询向量最接近的聚类 - 这里可以利用预计算的距离矩阵进行优化
    auto close_centroids = find_closest_centroids(query, NPROB);
    
    // 预分配空间
    std::vector<std::pair<int,double>> candidates;
    size_t estimated_size = 0;
    for (const auto& centroid_info : close_centroids) {
        int c_id = centroid_info.first;
        auto it = inverted_index.find(c_id);
        if (it != inverted_index.end()) {
            estimated_size += it->second.size();
        }
    }
    candidates.reserve(std::min(estimated_size, (size_t)(g_flat_db.size() * NPROB / NUM_CENTROID * 2)));
    
    // 计算查询向量的范数（如果使用余弦距离）
    double query_norm = -1.0;
    if (metric == "cosine") {
        query_norm = 0.0;
        for (double v : query) query_norm += v * v;
        query_norm = std::sqrt(query_norm);
    }
    
    const double* query_ptr = query.data();
    
    // 并行处理每个聚类中的向量
    int threads_to_use = std::min<int>(num_threads, std::max<int>(1, (int)close_centroids.size()));
    int chunk_size = ((int)close_centroids.size() + threads_to_use - 1) / threads_to_use;
    
    std::vector<std::thread> threads;
    threads.reserve(threads_to_use);
    std::vector<std::vector<std::pair<int,double>>> local_results(threads_to_use);
    
    auto worker = [this, query_ptr, &close_centroids, &local_results, query_norm](int start, int end, int tid) {
        local_results[tid].reserve(g_flat_db.size() / NUM_CENTROID * 2);
        
        for (int i = start; i < end && i < (int)close_centroids.size(); ++i) {
            int c_id = close_centroids[i].first;
            auto it = inverted_index.find(c_id);
            if (it == inverted_index.end()) continue;
            
            const std::vector<int>& cluster_vectors = it->second;
            const double* dbv_base = g_flat_db.get_vector(0); // 基地址，减少函数调用开销
            
            for (int vec_id : cluster_vectors) {
                if (vec_id < g_flat_db.size()) {
                    const double* dbv = dbv_base + vec_id * dim; // 直接计算地址
                    double dist = compute_distance_simd(query_ptr, dbv, query_norm, -1.0);
                    local_results[tid].emplace_back(vec_id, dist);
                }
            }
        }
    };
    
    for (int t = 0; t < threads_to_use; ++t) {
        int start = t * chunk_size;
        int end = std::min(start + chunk_size, (int)close_centroids.size());
        if (start < end) {
            threads.emplace_back(worker, start, end, t);
        }
    }
    
    for (auto& th : threads) th.join();
    
    // 合并局部结果
    size_t total_size = 0;
    for (const auto& lr : local_results) {
        total_size += lr.size();
    }
    candidates.reserve(total_size);
    for (const auto& lr : local_results) {
        candidates.insert(candidates.end(), lr.begin(), lr.end());
    }
    // Top-K 排序
    std::vector<std::pair<int,double>> result;
    if (candidates.size() <= (size_t)k) {
        std::sort(candidates.begin(), candidates.end(), 
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        result = std::move(candidates);
    } else {
        std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        candidates.resize(k);
        std::sort(candidates.begin(), candidates.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        result = std::move(candidates);
    }
    
   auto t_end = std::chrono::high_resolution_clock::now();
    //std::cout << "[TIMING] search - elapsed_before_search: " << fmt_sec(pre_search_elapsed) << "\n";
   // std::cout << "[TIMING] search - total_elapsed: " << fmt_sec(std::chrono::duration<double>(t_end - g_process_start).count()) << "\n";
    
    return result;
}

void solution::kmeans_update_parallel(const std::vector<int>& assignments,std::vector<std::vector<double>>& new_centroids){
    std::vector<std::mutex> mutexes(NUM_CENTROID); std::vector<int> counts(NUM_CENTROID,0);
    int threads_to_use=std::min<int>(num_threads,std::max<size_t>(1,database.size()));
    int chunk_size=(int(database.size())+threads_to_use-1)/threads_to_use; std::vector<std::thread> threads; threads.reserve(threads_to_use); std::vector<double> thread_times(threads_to_use,0.0);
    auto worker=[this,&assignments,&new_centroids,&mutexes,&counts,&thread_times](int start,int end,int tid){ auto t0=std::chrono::high_resolution_clock::now(); for(int i=start;i<end;++i){ int c=assignments[i]; std::lock_guard<std::mutex> lk(mutexes[c]); for(int d=0;d<dim;++d) new_centroids[c][d]+=database[i].vec[d]; counts[c]++; } auto t1=std::chrono::high_resolution_clock::now(); thread_times[tid]=std::chrono::duration<double>(t1-t0).count(); };
    for(int t=0;t<threads_to_use;++t){ int start=t*chunk_size,end=std::min(start+chunk_size,(int)database.size()); if(start<end) threads.emplace_back(worker,start,end,t); }
    for(auto&th:threads) th.join(); double sum=0; for(double v:thread_times) sum+=v; print_stage("kmeans_update_parallel",sum);
    for(int i=0;i<NUM_CENTROID;++i) if(counts[i]>0) for(int d=0;d<dim;++d) new_centroids[i][d]/=counts[i];
}

static solution* g_impl=nullptr;
void Solution::build(int d,const std::vector<float>& base){ if(d<=0) return; int n=(int)base.size()/d; if(n<=0) return; std::vector<std::vector<double>> data; data.reserve(n); for(int i=0;i<n;++i){ std::vector<double> v; v.reserve(d); for(int j=0;j<d;++j) v.push_back((double)base[i*d+j]); data.push_back(std::move(v)); } delete g_impl; g_impl=new solution("l2"); g_impl->build_from_memory(d,std::move(data)); }
void Solution::search(const std::vector<float>& query,int* res){ if(!g_impl){ for(int i=0;i<10;++i) res[i]=-1; return; } std::vector<double> q(query.begin(),query.end()); auto ans=g_impl->search(q,10); int idx=0; for(;idx<(int)ans.size()&&idx<10;++idx) res[idx]=ans[idx].first; for(;idx<10;++idx) res[idx]=-1; }