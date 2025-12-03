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