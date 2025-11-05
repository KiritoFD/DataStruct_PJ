// compute_ground_truth.cpp
// 多线程生产者-消费者暴力 top-k 最近邻 (流式、可限制扫描条数)
// 支持：reservoir 抽样 query，--num_queries, --k, --metric (l2|cosine), --threads, --batch_size, --max_base, --out
// 输出：简单 JSON 文件（无外部依赖）

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

struct VecItem {
    int index;
    string id; // empty if none
    vector<double> v;
};

bool try_stod(const string &s, double &out) {
    try {
        size_t pos;
        out = stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parse_vector_line(const string &line, string &out_id, vector<double> &out_vec) {
    out_id.clear();
    out_vec.clear();
    istringstream iss(line);
    vector<string> toks;
    string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;
    double val;
    // try treat ALL tokens as numbers
    bool allnum = true;
    for (auto &s : toks) {
        if (!try_stod(s, val)) { allnum = false; break; }
    }
    if (allnum) {
        out_id.clear();
        out_vec.reserve(toks.size());
        for (auto &s : toks) out_vec.push_back(stod(s));
        return true;
    }
    // otherwise first token is id
    if (toks.size() < 2) return false;
    out_id = toks[0];
    out_vec.reserve(toks.size()-1);
    for (size_t i = 1; i < toks.size(); ++i) {
        if (!try_stod(toks[i], val)) return false;
        out_vec.push_back(stod(toks[i]));
    }
    return true;
}

double l2_distance(const vector<double> &a, const vector<double> &b) {
    double s = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i]; s += d*d;
    }
    return sqrt(s);
}

double cosine_distance(const vector<double> &a, const vector<double> &b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    if (na == 0.0 || nb == 0.0) return 1.0;
    double sim = dot / (sqrt(na)*sqrt(nb));
    return 1.0 - sim;
}

// Reservoir sampling queries from file
vector<VecItem> reservoir_sample(const string &path, int num, unsigned seed) {
    vector<VecItem> reservoir;
    reservoir.reserve(num);
    ifstream ifs(path);
    if (!ifs) return reservoir;
    string line;
    std::mt19937 rng(seed);
    int idx = 0;
    while (std::getline(ifs, line)) {
        string id; vector<double> vec;
        if (!parse_vector_line(line, id, vec)) { ++idx; continue; }
        VecItem it{idx, id, std::move(vec)};
        if ((int)reservoir.size() < num) {
            reservoir.push_back(std::move(it));
        } else {
            uniform_int_distribution<int> dist(0, idx);
            int j = dist(rng);
            if (j < num) reservoir[j] = std::move(it);
        }
        ++idx;
    }
    return reservoir;
}

// Thread-safe queue of batches
template<typename T>
class ThreadQueue {
public:
    void push(T &&v) {
        {
            unique_lock<mutex> lk(m_);
            q_.push(std::move(v));
        }
        cv_.notify_one();
    }
    bool pop(T &out) {
        unique_lock<mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty() || finished_; });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop();
        return true;
    }
    void set_finished() {
        {
            unique_lock<mutex> lk(m_);
            finished_ = true;
        }
        cv_.notify_all();
    }
private:
    queue<T> q_;
    mutex m_;
    condition_variable cv_;
    bool finished_ = false;
};

struct QueryState {
    int qidx;
    string id;
    vector<double> vec;
    // max-heap of size up to k; store pair(distance, base_index). largest distance on top
    priority_queue<pair<double,int>> heap;
    mutex m;
};

void worker_func(ThreadQueue<vector<VecItem>> &tq, vector<unique_ptr<QueryState>> &queries, int k, const string &metric, atomic<long long> &processed) {
    vector<VecItem> batch;
    while (tq.pop(batch)) {
        for (auto &b : batch) {
            for (auto &qptr : queries) {
                auto &q = *qptr;
                double dist = numeric_limits<double>::infinity();
                if (metric == "l2") dist = l2_distance(q.vec, b.v);
                else dist = cosine_distance(q.vec, b.v);
                // update q.heap
                {
                    lock_guard<mutex> lk(q.m);
                    if ((int)q.heap.size() < k) q.heap.emplace(dist, b.index);
                    else if (q.heap.top().first > dist) {
                        q.heap.pop(); q.heap.emplace(dist, b.index);
                    }
                }
            }
            ++processed;
        }
    }
}

void write_json_output(const string &out_path, const string &dataset, const vector<unique_ptr<QueryState>> &queries, const string &metric, int k) {
    ofstream ofs(out_path);
    if (!ofs) {
        cerr << "Cannot open output file: " << out_path << "\n";
        return;
    }
    ofs << "{\n";
    ofs << "  \"dataset\": \"" << dataset << "\",\n";
    ofs << "  \"metric\": \"" << metric << "\",\n";
    ofs << "  \"k\": " << k << ",\n";
    ofs << "  \"results\": [\n";
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto &q = *queries[i];
        // extract neighbors sorted ascending by distance
        vector<pair<double,int>> list;
        {
            // copy heap
            auto tmp = q.heap;
            while (!tmp.empty()) { list.push_back(tmp.top()); tmp.pop(); }
        }
        sort(list.begin(), list.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
        ofs << "    {\n";
        ofs << "      \"query_index\": " << q.qidx << ",\n";
        ofs << "      \"query_id\": \"" << q.id << "\",\n";
        ofs << "      \"neighbors\": [\n";
        for (size_t j = 0; j < list.size(); ++j) {
            ofs << "        { \"index\": " << list[j].second << ", \"distance\": " << list[j].first << " }";
            if (j+1 < list.size()) ofs << ",";
            ofs << "\n";
        }
        ofs << "      ]\n";
        ofs << "    }";
        if (i+1 < queries.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}

int main(int argc, char **argv) {
    // simple arg parsing
    string dataset = "";
    int num_queries = 100;
    int k = 5;
    string metric = "l2";
    int threads = 32;
    int batch_size = 1000;
    int max_base = -1;
    unsigned seed = 1;
    string out = "";

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--dataset" && i+1 < argc) { dataset = argv[++i]; }
        else if (a == "--num_queries" && i+1 < argc) num_queries = stoi(argv[++i]);
        else if (a == "--k" && i+1 < argc) k = stoi(argv[++i]);
        else if (a == "--metric" && i+1 < argc) metric = argv[++i];
        else if (a == "--threads" && i+1 < argc) threads = stoi(argv[++i]);
        else if (a == "--batch_size" && i+1 < argc) batch_size = stoi(argv[++i]);
        else if (a == "--max_base" && i+1 < argc) max_base = stoi(argv[++i]);
        else if (a == "--seed" && i+1 < argc) seed = stoi(argv[++i]);
        else if (a == "--out" && i+1 < argc) out = argv[++i];
        else if (a == "--help") {
            cout << "Usage: compute_ground_truth.exe --dataset PATH [--num_queries N] [--k K] [--metric l2|cosine] [--threads T] [--batch_size B] [--max_base N] [--out out.json]\n";
            return 0;
        }
    }
    if (dataset.empty()) {
        cerr << "Please provide --dataset path\n";
        return 1;
    }

    cout << "Reservoir sampling " << num_queries << " queries from " << dataset << "...\n";
    auto queries_items = reservoir_sample(dataset, num_queries, seed);
    if (queries_items.empty()) { cerr << "No queries sampled (file empty or parse failed)\n"; return 1; }

    vector<unique_ptr<QueryState>> queries;
    for (size_t i = 0; i < queries_items.size(); ++i) {
        auto qs = std::make_unique<QueryState>();
        qs->qidx = queries_items[i].index;
        qs->id = std::move(queries_items[i].id);
        qs->vec = std::move(queries_items[i].v);
        queries.push_back(std::move(qs));
    }

    ThreadQueue<vector<VecItem>> tq;
    atomic<long long> processed{0};

    // start workers
    vector<thread> workers;
    for (int i = 0; i < threads; ++i)
        workers.emplace_back(worker_func, std::ref(tq), std::ref(queries), k, metric, std::ref(processed));

    // producer: read dataset in batches and push into queue
    ifstream ifs(dataset);
    if (!ifs) { cerr << "Cannot open dataset: " << dataset << "\n"; tq.set_finished(); for (auto &t: workers) t.join(); return 1; }
    string line; vector<VecItem> batch; batch.reserve(batch_size);
    int idx = 0;
    while (std::getline(ifs, line)) {
        if (max_base >= 0 && idx >= max_base) break;
        string id; vector<double> vec;
        if (!parse_vector_line(line, id, vec)) { ++idx; continue; }
        batch.push_back(VecItem{idx, id, std::move(vec)});
        ++idx;
        if ((int)batch.size() >= batch_size) { tq.push(std::move(batch)); batch.clear(); batch.reserve(batch_size); }
    }
    if (!batch.empty()) { tq.push(std::move(batch)); }
    tq.set_finished();

    // simple progress display
    while (processed < idx) {
        cout << "Processed base vectors (approx): " << processed << "\r" << flush;
        this_thread::sleep_for(chrono::milliseconds(200));
    }

    // join workers
    for (auto &t : workers) if (t.joinable()) t.join();
    cout << "\nFinished.\n";

    // write output
    string outpath = out.empty() ? dataset.substr(0, dataset.find_last_of(".")) + ".json" : out;
    write_json_output(outpath, dataset, queries, metric, k);
    cout << "Wrote output to " << outpath << "\n";

    return 0;
}
