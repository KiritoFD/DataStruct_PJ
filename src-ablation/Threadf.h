#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>

class ThreadPool {
public:
    ThreadPool(int n) : stop(false) {
        for (int i = 0; i < n; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(qmtx);
                        cv.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(qmtx); stop = true; }
        cv.notify_all();
        for (auto &w : workers) w.join();
    }
    template<class F> void enqueue(F&& f) {
        { std::unique_lock<std::mutex> lock(qmtx); tasks.emplace(std::forward<F>(f)); }
        cv.notify_one();
    }
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex qmtx;
    std::condition_variable cv;
    bool stop;
};

// 默认参数（如果未定义）
#ifndef HNSW_DEFAULT_M
#define HNSW_DEFAULT_M 16
#endif

#ifndef HNSW_DEFAULT_MAX_LAYER
#define HNSW_DEFAULT_MAX_LAYER 16
#endif

#ifndef HNSW_DEFAULT_EF_CONSTRUCTION
#define HNSW_DEFAULT_EF_CONSTRUCTION 200
#endif

#ifndef HNSW_DEFAULT_EF_SEARCH
#define HNSW_DEFAULT_EF_SEARCH 200
#endif