#include "MySolution.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include <cstdint>
#include <ctime>
#include <sys/stat.h>

// 导出函数声明
extern "C" {
    uint64_t get_total_queries();
    double get_avg_dists_per_query();
    uint64_t get_last_query_dists();
    double get_last_build_time_ms();
    void reset_dist_counters();
    void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads);
    void set_hnsw_debug(int dbg);
    
    // 图质量统计
    int get_graph_max_level();
    int get_graph_num_nodes();
    double get_graph_avg_degree_l0();
    int get_graph_actual_max_layer();
    int get_graph_nodes_at_level(int level);
    double get_graph_avg_degree_upper();
}

// 新增：尝试从二进制缓存加载 base；格式：magic(8) + uint32_t d + uint64_t n_vec + floats...
static bool load_base_flat_from_bin(const std::string &bin_path, std::vector<float> &flat, int &d_out) {
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) return false;
    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() != 8) return false;
    if (std::string(magic, 7) != "BINFMT1") return false;
    uint32_t d;
    uint64_t n;
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!ifs) return false;
    flat.resize(static_cast<size_t>(n) * d);
    ifs.read(reinterpret_cast<char*>(flat.data()), flat.size() * sizeof(float));
    if (!ifs) return false;
    d_out = static_cast<int>(d);
    return true;
}

static bool save_base_flat_to_bin(const std::string &bin_path, const std::vector<float> &flat, int d) {
    std::ofstream ofs(bin_path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    char magic[8] = "BINFMT1";
    ofs.write(magic, 8);
    uint32_t du = static_cast<uint32_t>(d);
    uint64_t n = flat.empty() ? 0 : (flat.size() / d);
    ofs.write(reinterpret_cast<const char*>(&du), sizeof(du));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (!flat.empty()) ofs.write(reinterpret_cast<const char*>(flat.data()), flat.size() * sizeof(float));
    return !!ofs;
}

// 载入 base.txt：优先二进制缓存
static std::vector<float> load_base_flat(const std::string &path, int &d_out) {
     std::string bin = path + ".bin";
     std::vector<float> flat;
     if (load_base_flat_from_bin(bin, flat, d_out)) {
         std::cerr << "Loaded base from binary cache: " << bin << std::endl;
         return flat;
     }

     // 文本加载（原逻辑）
     std::ifstream in(path);
     d_out = 0;
     std::string line;
     while (std::getline(in, line)) {
         if (line.empty()) continue;
         std::istringstream ss(line);
         float v;
         std::vector<float> tmp;
         while (ss >> v) tmp.push_back(v);
         if (tmp.empty()) continue;
         if (d_out == 0) d_out = (int)tmp.size();
         if ((int)tmp.size() != d_out) continue;
         flat.insert(flat.end(), tmp.begin(), tmp.end());
     }

     // 尝试写入二进制缓存（忽略失败）
     if (!flat.empty() && d_out > 0) {
         if (save_base_flat_to_bin(bin, flat, d_out)) {
             std::cerr << "Saved base binary cache: " << bin << std::endl;
         }
     }
     return flat;
}

// 新增：二进制加载/保存 queries
static bool load_queries_from_bin(const std::string &bin_path, std::vector<std::vector<float>> &qs, int expected_d) {
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) return false;
    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() != 8) return false;
    if (std::string(magic,7) != "BINFMT1") return false;
    uint32_t d;
    uint64_t n;
    ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!ifs) return false;
    if (expected_d != 0 && expected_d != static_cast<int>(d)) return false;
    qs.resize(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        qs[i].resize(d);
        ifs.read(reinterpret_cast<char*>(qs[i].data()), d * sizeof(float));
        if (!ifs) return false;
    }
    return true;
}

static bool save_queries_to_bin(const std::string &bin_path, const std::vector<std::vector<float>> &qs, int d) {
    std::ofstream ofs(bin_path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    char magic[8] = "BINFMT1";
    ofs.write(magic, 8);
    uint32_t du = static_cast<uint32_t>(d);
    uint64_t n = qs.size();
    ofs.write(reinterpret_cast<const char*>(&du), sizeof(du));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto &v : qs) {
        ofs.write(reinterpret_cast<const char*>(v.data()), d * sizeof(float));
    }
    return !!ofs;
}

// 载入 query.txt：优先二进制缓存
static std::vector<std::vector<float>> load_queries(const std::string &path, int expected_d) {
    std::string bin = path + ".bin";
    std::vector<std::vector<float>> qs;
    if (load_queries_from_bin(bin, qs, expected_d)) {
        std::cerr << "Loaded queries from binary cache: " << bin << std::endl;
        return qs;
    }

    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        float v;
        std::vector<float> tmp;
        while (ss >> v) tmp.push_back(v);
        if ((int)tmp.size() != expected_d) continue;
        qs.push_back(std::move(tmp));
    }

    if (!qs.empty() && expected_d > 0) {
        if (save_queries_to_bin(bin, qs, expected_d)) {
            std::cerr << "Saved queries binary cache: " << bin << std::endl;
        }
    }
    return qs;
}

// 载入 truth.txt：优先二进制缓存
static bool load_truth_from_bin(const std::string &bin_path, std::vector<std::vector<int>> &gt) {
    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs) return false;
    char magic[8];
    ifs.read(magic, 8);
    if (ifs.gcount() != 8) return false;
    if (std::string(magic,7) != "BINFMT1") return false;
    uint64_t nq;
    ifs.read(reinterpret_cast<char*>(&nq), sizeof(nq));
    if (!ifs) return false;
    gt.resize(static_cast<size_t>(nq));
    for (uint64_t i = 0; i < nq; ++i) {
        uint32_t len;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!ifs) return false;
        gt[i].resize(len);
        ifs.read(reinterpret_cast<char*>(gt[i].data()), len * sizeof(int32_t));
        if (!ifs) return false;
    }
    return true;
}

static bool save_truth_to_bin(const std::string &bin_path, const std::vector<std::vector<int>> &gt) {
    std::ofstream ofs(bin_path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    char magic[8] = "BINFMT1";
    ofs.write(magic, 8);
    uint64_t nq = gt.size();
    ofs.write(reinterpret_cast<const char*>(&nq), sizeof(nq));
    for (const auto &row : gt) {
        uint32_t len = static_cast<uint32_t>(row.size());
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        if (len) ofs.write(reinterpret_cast<const char*>(row.data()), len * sizeof(int32_t));
    }
    return !!ofs;
}

static std::vector<std::vector<int>> load_truth(const std::string &path) {
    std::string bin = path + ".bin";
    std::vector<std::vector<int>> gt;
    if (load_truth_from_bin(bin, gt)) {
        std::cerr << "Loaded truth from binary cache: " << bin << std::endl;
        return gt;
    }

    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) { gt.emplace_back(); continue; }
        std::istringstream ss(line);
        int id;
        std::vector<int> row;
        while (ss >> id) row.push_back(id);
        gt.push_back(std::move(row));
    }

    if (!gt.empty()) {
        if (save_truth_to_bin(bin, gt)) {
            std::cerr << "Saved truth binary cache: " << bin << std::endl;
        }
    }
    return gt;
}

static float recall_at_k(const int* res, const std::vector<int> &gt, int k) {
    if (gt.empty()) return 0.0f;
    // 使用位图加速小规模集合查找
    std::unordered_set<int> s;
    s.reserve(k);
    for (int i = 0; i < (int)gt.size() && i < k; ++i) s.insert(gt[i]);
    int hit = 0;
    for (int i = 0; i < k; ++i) {
        if (res[i] >= 0 && s.count(res[i])) ++hit;
    }
    return float(hit) / float(k);
}

// 简易命令行解析（与 test_hn.cpp 对齐）
struct CmdOptsG {
    int M = -1, max_layer = -1, efc = -1, efs = -1, threads = -1, debug = -1;
    std::string base_file, query_file, truth_file;
    int K = -1;
    // ablation flags
    int ablate_csr = -1;
    int ablate_prefetch = -1;
    int ablate_simd = -1;
    int ablate_pruning = -1;
    int ablate_heap = -1;
};
static CmdOptsG parse_args_g(int argc, char** argv) {
    CmdOptsG o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i+1 < argc) o.M = std::stoi(argv[++i]);
        else if (a == "--max_layer" && i+1 < argc) o.max_layer = std::stoi(argv[++i]);
        else if (a == "--efc" && i+1 < argc) o.efc = std::stoi(argv[++i]);
        else if (a == "--efs" && i+1 < argc) o.efs = std::stoi(argv[++i]);
        else if (a == "--threads" && i+1 < argc) o.threads = std::stoi(argv[++i]);
        else if (a == "--debug" && i+1 < argc) o.debug = std::stoi(argv[++i]);
        else if (a == "--base" && i+1 < argc) o.base_file = argv[++i];
        else if (a == "--query" && i+1 < argc) o.query_file = argv[++i];
        else if (a == "--truth" && i+1 < argc) o.truth_file = argv[++i];
        else if (a == "--k" && i+1 < argc) o.K = std::stoi(argv[++i]);
        else if (a == "--ablate_csr" && i+1 < argc) o.ablate_csr = std::stoi(argv[++i]);
        else if (a == "--ablate_prefetch" && i+1 < argc) o.ablate_prefetch = std::stoi(argv[++i]);
        else if (a == "--ablate_simd" && i+1 < argc) o.ablate_simd = std::stoi(argv[++i]);
        else if (a == "--ablate_pruning" && i+1 < argc) o.ablate_pruning = std::stoi(argv[++i]);
        else if (a == "--ablate_heap" && i+1 < argc) o.ablate_heap = std::stoi(argv[++i]);
    }
    return o;
}

// 新增：日志输出辅助类
class Logger {
private:
    std::ofstream log_file;
    std::string log_path;

public:
    Logger(const std::string& log_dir) {
        // 创建日志目录
        mkdir(log_dir.c_str(), 0755);
        
        // 生成带时间戳的日志文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << log_dir << "/run_"
           << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S")
           << "_" << std::setfill('0') << std::setw(3) << ms.count()
           << ".log";
        log_path = ss.str();
        
        log_file.open(log_path, std::ios::app);
    }

    ~Logger() {
        if (log_file.is_open()) log_file.close();
    }

    void write(const std::string& msg) {
        if (log_file.is_open()) {
            log_file << msg;
            log_file.flush();
        }
        std::cout << msg;
    }

    void writeline(const std::string& msg) {
        write(msg + "\n");
    }

    template<typename T>
    Logger& operator<<(const T& val) {
        std::stringstream ss;
        ss << val;
        write(ss.str());
        return *this;
    }

    const std::string& get_path() const { return log_path; }
};

int main(int argc, char** argv) {
    auto opts = parse_args_g(argc, argv);
    const std::string base_file_default = "data_o/glove/base.txt";
    const std::string query_file_default = "data_o/glove/query.txt";
    const std::string truth_file_default = "data_o/glove/truth.txt";
    const std::string base_file = opts.base_file.empty() ? base_file_default : opts.base_file;
    const std::string query_file = opts.query_file.empty() ? query_file_default : opts.query_file;
    const std::string truth_file = opts.truth_file.empty() ? truth_file_default : opts.truth_file;
    const int K = (opts.K > 0) ? opts.K : 10;

    // 创建日志记录器
    Logger logger("Log");
    
    // 记录启动时间和配置
    {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        logger.writeline(std::string("========== HNSW Search Test ==========\n"));
        logger.writeline(std::string("Timestamp: ") + std::ctime(&time_t));
    }

    logger.writeline("Command line arguments:");
    for (int i = 0; i < argc; ++i) {
        logger.writeline(std::string("  argv[") + std::to_string(i) + "]: " + argv[i]);
    }

    logger.writeline("\nConfiguration:");
    logger.writeline(std::string("  Base file: ") + base_file);
    logger.writeline(std::string("  Query file: ") + query_file);
    logger.writeline(std::string("  Truth file: ") + truth_file);
    logger.writeline(std::string("  K: ") + std::to_string(K));
    logger.writeline(std::string("  M: ") + std::to_string(opts.M));
    logger.writeline(std::string("  Max layer: ") + std::to_string(opts.max_layer));
    logger.writeline(std::string("  EFC: ") + std::to_string(opts.efc));
    logger.writeline(std::string("  EFS: ") + std::to_string(opts.efs));
    logger.writeline(std::string("  Threads: ") + std::to_string(opts.threads));
    logger.writeline(std::string("  Debug: ") + std::to_string(opts.debug));
    logger.writeline("");

    if (opts.M>0 || opts.max_layer>0 || opts.efc>0 || opts.efs>0 || opts.threads>0) {
        set_hnsw_params(opts.M, opts.max_layer, opts.efc, opts.efs, opts.threads);
        logger.writeline("HNSW parameters set.\n");
    }
    if (opts.debug >= 0) set_hnsw_debug(opts.debug);

    // apply ablation flags when provided; default to 0 otherwise
    int csr = (opts.ablate_csr >= 0) ? opts.ablate_csr : 0;
    int prefetch = (opts.ablate_prefetch >= 0) ? opts.ablate_prefetch : 0;
    int simd = (opts.ablate_simd >= 0) ? opts.ablate_simd : 0;
    int pruning = (opts.ablate_pruning >= 0) ? opts.ablate_pruning : 0;
    int heap = (opts.ablate_heap >= 0) ? opts.ablate_heap : 0;
    set_ablation_flags(csr, prefetch, simd, pruning, heap);
    logger.writeline((std::string("Ablation flags: csr=") + std::to_string(csr)
                      + ", prefetch=" + std::to_string(prefetch)
                      + ", simd=" + std::to_string(simd)
                      + ", pruning=" + std::to_string(pruning)
                      + ", heap=" + std::to_string(heap) + "\n"));

    // 加载数据
    logger.writeline("Loading base vectors...");
    int d = 0;
    auto load_base_t0 = std::chrono::steady_clock::now();
    auto base_flat = load_base_flat(base_file, d);
    auto load_base_t1 = std::chrono::steady_clock::now();
    double load_base_ms = std::chrono::duration<double, std::milli>(load_base_t1 - load_base_t0).count();

    if (d <= 0 || base_flat.empty()) {
        logger.writeline("ERROR: Failed to load base or invalid dimension.\n");
        return 1;
    }
    logger.writeline(std::string("✓ Base loaded in ") + std::to_string(load_base_ms) + " ms\n");

    logger.writeline("Loading query vectors...");
    auto load_query_t0 = std::chrono::steady_clock::now();
    auto queries = load_queries(query_file, d);
    auto load_query_t1 = std::chrono::steady_clock::now();
    double load_query_ms = std::chrono::duration<double, std::milli>(load_query_t1 - load_query_t0).count();

    if (queries.empty()) {
        logger.writeline("ERROR: Failed to load queries.\n");
        return 1;
    }
    logger.writeline(std::string("✓ Queries loaded in ") + std::to_string(load_query_ms) + " ms\n");

    logger.writeline("Loading ground truth...");
    auto load_truth_t0 = std::chrono::steady_clock::now();
    auto truths = load_truth(truth_file);
    auto load_truth_t1 = std::chrono::steady_clock::now();
    double load_truth_ms = std::chrono::duration<double, std::milli>(load_truth_t1 - load_truth_t0).count();

    if (truths.empty()) {
        logger.writeline("WARNING: Failed to load ground truth (recall will be 0).\n");
    } else {
        logger.writeline(std::string("✓ Truth loaded in ") + std::to_string(load_truth_ms) + " ms\n");
    }

    logger.writeline(std::string("Data Summary:\n  Dimension d=") + std::to_string(d) 
                     + ", Base vectors=" + std::to_string(base_flat.size() / d)
                     + ", Queries=" + std::to_string(queries.size()) + "\n\n");

    // 构建索引
    logger.writeline("Building HNSW index...");
    Solution sol;
    auto build_t0 = std::chrono::steady_clock::now();
    sol.build(d, base_flat);
    auto build_t1 = std::chrono::steady_clock::now();
    double build_ms_local = std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();
    double build_ms_internal = get_last_build_time_ms();
    
    logger.writeline(std::string("✓ Index built.\n  build_time(local)=") + std::to_string(build_ms_local)
                     + " ms\n  build_time(internal)=" + std::to_string(build_ms_internal) + " ms\n");
    
    // 输出图质量统计
    logger.writeline("\n========== GRAPH QUALITY ==========\n");
    int num_nodes = get_graph_num_nodes();
    int max_level = get_graph_max_level();
    int actual_max_layer = get_graph_actual_max_layer();
    double avg_degree_l0 = get_graph_avg_degree_l0();
    double avg_degree_upper = get_graph_avg_degree_upper();
    
    logger.writeline(std::string("  num_nodes: ") + std::to_string(num_nodes) + "\n");
    logger.writeline(std::string("  max_level (entry point): ") + std::to_string(max_level) + "\n");
    logger.writeline(std::string("  actual_max_layer: ") + std::to_string(actual_max_layer) + "\n");
    logger.writeline(std::string("  avg_degree_l0: ") + std::to_string(avg_degree_l0) + "\n");
    logger.writeline(std::string("  avg_degree_upper: ") + std::to_string(avg_degree_upper) + "\n");
    
    // 输出层级分布
    logger.writeline("\n  Layer Distribution:\n");
    for (int l = 0; l <= actual_max_layer && l <= 10; ++l) {
        int nodes_at_l = get_graph_nodes_at_level(l);
        double pct = num_nodes > 0 ? 100.0 * nodes_at_l / num_nodes : 0.0;
        std::stringstream ss;
        ss << "    L" << l << ": " << nodes_at_l << " nodes (" 
           << std::fixed << std::setprecision(2) << pct << "%)\n";
        logger.writeline(ss.str());
    }
    logger.writeline("\n");

    // 重置距离计数器
    reset_dist_counters();

    // 执行查询
    logger.writeline("Executing queries...\n");
    const size_t nq = queries.size();
    double total_recall = 0.0;
    int res[10];
    
    auto t0 = std::chrono::steady_clock::now();
    
    for (size_t qi = 0; qi < nq; ++qi) {
        sol.search(queries[qi], res);
        const std::vector<int> &gt = (qi < truths.size() ? truths[qi] : std::vector<int>{});
        total_recall += recall_at_k(res, gt, K);
        
        // 每 1000 次查询输出一次进度
        if ((qi + 1) % 1000 == 0) {
            logger.writeline(std::string("  Processed ") + std::to_string(qi + 1) + " queries\n");
        }
    }
    
    auto t1 = std::chrono::steady_clock::now();
    
    double total_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_recall = nq ? (total_recall / nq) : 0.0;
    double avg_query_ms = nq ? (total_time_ms / nq) : 0.0;

    logger.writeline("\n========== RESULTS ==========\n");
    logger.writeline(std::string("Total queries: ") + std::to_string(nq) + "\n");
    
    // 修复：使用 stringstream 格式化浮点数
    {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(6) 
           << "Average recall@" << K << ": " << avg_recall << "\n";
        logger.writeline(ss.str());
    }
    
    logger.writeline(std::string("Average query time: ") + std::to_string(avg_query_ms) + " ms\n");
    logger.writeline(std::string("Total query time: ") + std::to_string(total_time_ms) + " ms\n");

    // 打印距离统计（优化版：更详细）
    uint64_t total_queries_reported = get_total_queries();
    double avg_dists = get_avg_dists_per_query();
    uint64_t last_query_dists = get_last_query_dists();
    
    logger.writeline("\n========== DISTANCE STATISTICS ==========\n");
    logger.writeline(std::string("Total queries reported by index: ") + std::to_string(total_queries_reported) + "\n");
    logger.writeline(std::string("Total distance computations: ") + std::to_string((uint64_t)(avg_dists * total_queries_reported)) + "\n");
    logger.writeline(std::string("Average distance ops per query: ") + std::to_string(avg_dists) + "\n");
    logger.writeline(std::string("Last query distance ops: ") + std::to_string(last_query_dists) + "\n");

    // 性能摘要
    logger.writeline("\n========== PERFORMANCE SUMMARY ==========\n");
    logger.writeline(std::string("Data loading time: ") + std::to_string(load_base_ms + load_query_ms + load_truth_ms) + " ms\n");
    logger.writeline(std::string("Index build time: ") + std::to_string(build_ms_internal) + " ms\n");
    logger.writeline(std::string("Query execution time: ") + std::to_string(total_time_ms) + " ms\n");
    double total_elapsed = load_base_ms + load_query_ms + load_truth_ms + build_ms_local + total_time_ms;
    logger.writeline(std::string("Total elapsed time: ") + std::to_string(total_elapsed) + " ms\n");
    
    // 新增：距离计算效率指标
    logger.writeline("\n========== EFFICIENCY METRICS ==========\n");
    if (nq > 0 && total_time_ms > 0) {
        double queries_per_sec = (nq * 1000.0) / total_time_ms;
        logger.writeline(std::string("Queries per second: ") + std::to_string(queries_per_sec) + "\n");
        
        if (avg_dists > 0) {
            double dist_ops_per_second = (avg_dists * nq * 1000.0) / total_time_ms;
            logger.writeline(std::string("Distance ops per second: ") + std::to_string(dist_ops_per_second) + "\n");
        }
    }

    logger.writeline("\n========== END OF LOG ==========\n");
    logger.writeline(std::string("Log saved to: ") + logger.get_path() + "\n");

    return 0;
}