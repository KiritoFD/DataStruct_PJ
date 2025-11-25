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

// 导出函数声明
extern "C" {
    uint64_t get_total_queries();
    double get_avg_dists_per_query();
    uint64_t get_last_query_dists();
    double get_last_build_time_ms();
    void reset_dist_counters();
    void set_hnsw_params(int M, int max_layer, int ef_construction, int ef_search, int build_threads);
    void set_hnsw_debug(int dbg);
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

static float recall_at_k(const std::vector<int> &res, const std::vector<int> &gt, int k) {
	if (gt.empty()) return 0.0f;
	std::unordered_set<int> s;
	for (int i = 0; i < (int)gt.size() && i < k; ++i) s.insert(gt[i]);
	int hit = 0;
	for (int i = 0; i < k; ++i) {
		if (res[i] >= 0 && s.find(res[i]) != s.end()) ++hit;
	}
	return float(hit) / float(k);
}

// 简易命令行解析（与 test_hn.cpp 对齐）
struct CmdOptsG {
    int M = -1, max_layer = -1, efc = -1, efs = -1, threads = -1, debug = -1;
    std::string base_file, query_file, truth_file;
    int K = -1;
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
    }
    return o;
}

int main(int argc, char** argv) {
    auto opts = parse_args_g(argc, argv);
    const std::string base_file_default = "data_o/glove/base.txt";
    const std::string query_file_default = "data_o/glove/query.txt";
    const std::string truth_file_default = "data_o/glove/truth.txt";
    const std::string base_file = opts.base_file.empty() ? base_file_default : opts.base_file;
    const std::string query_file = opts.query_file.empty() ? query_file_default : opts.query_file;
    const std::string truth_file = opts.truth_file.empty() ? truth_file_default : opts.truth_file;
    const int K = (opts.K > 0) ? opts.K : 10;

    if (opts.M>0 || opts.max_layer>0 || opts.efc>0 || opts.efs>0 || opts.threads>0) {
        set_hnsw_params(opts.M, opts.max_layer, opts.efc, opts.efs, opts.threads);
    }
    if (opts.debug >= 0) set_hnsw_debug(opts.debug);
    std::cout << "Config: base=" << base_file << ", query=" << query_file << ", truth=" << truth_file << ", K=" << K << std::endl;

	int d = 0;
	auto base_flat = load_base_flat(base_file, d);
 	if (d <= 0 || base_flat.empty()) {
 		std::cerr << "加载 base 失败或维度无效: " << base_file << std::endl;
 		return 1;
 	}
 
	auto queries = load_queries(query_file, d);
 	if (queries.empty()) {
 		std::cerr << "加载 query 失败或没有合法查询: " << query_file << std::endl;
 		return 1;
 	}
 
	auto truths = load_truth(truth_file);
 	if (truths.empty()) {
 		std::cerr << "加载 truth 失败或为空: " << truth_file << std::endl;
 		// 仍可继续但召回为 0
 	}
 
 	std::cout << "向量维度 d=" << d << ", 底库向量数=" << (base_flat.size() / d)
 			  << ", 查询数=" << queries.size() << std::endl;
 
 	// 构建索引
 	Solution sol;
     auto build_t0 = std::chrono::steady_clock::now();
 	sol.build(d, base_flat);
     auto build_t1 = std::chrono::steady_clock::now();
     double build_ms_local = std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();
     double build_ms_internal = get_last_build_time_ms();
     std::cout << "Index built. build_time(local)=" << std::fixed << std::setprecision(2) << build_ms_local
               << " ms, build_time(internal)=" << std::fixed << std::setprecision(2) << build_ms_internal << " ms" << std::endl;
    
	// 查询并统计
	long long total_ms = 0;
	double total_recall = 0.0;
	int cnt = 0;
	int res[10];
	auto t0 = std::chrono::steady_clock::now();
	for (size_t qi = 0; qi < queries.size(); ++qi) {
		const auto &q = queries[qi];
		// 调用接口
		auto qt0 = std::chrono::steady_clock::now();
		sol.search(q, res);
		auto qt1 = std::chrono::steady_clock::now();
		total_ms += std::chrono::duration_cast<std::chrono::microseconds>(qt1 - qt0).count();
		std::vector<int> out(res, res + K);
		const std::vector<int> &gt = (qi < truths.size() ? truths[qi] : std::vector<int>{});
		total_recall += recall_at_k(out, gt, K);
		++cnt;
	}
	auto t1 = std::chrono::steady_clock::now();
	double avg_recall = cnt ? (total_recall / cnt) : 0.0;
	double avg_query_ms = cnt ? (total_ms / 1000.0 / cnt) : 0.0;
	double total_time_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "queries=" << cnt << ", avg recall@" << K << "=" << avg_recall
			  << ", avg_query_time=" << avg_query_ms << " ms"
			  << ", total_time=" << total_time_ms << " ms" << std::endl;
    // 打印距离统计
    std::cout << "Distance stats: total_queries = " << get_total_queries()
              << ", avg_dists_per_query = " << std::fixed << std::setprecision(2) << get_avg_dists_per_query()
              << ", last_query_dists = " << get_last_query_dists() << std::endl;
 	return 0;
 }
