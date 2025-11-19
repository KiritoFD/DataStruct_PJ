#pragma once
#include <string>
#include <vector>
#include <sstream>

namespace common {

inline bool try_stod(const std::string& s, double& out) {
    try {
        size_t pos = 0;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

inline bool parse_vector_line(const std::string& line, std::string& out_id, std::vector<double>& out_vec) {
    out_id.clear();
    out_vec.clear();
    std::istringstream iss(line);
    std::vector<std::string> toks;
    std::string t;
    while (iss >> t) toks.push_back(t);
    if (toks.empty()) return false;

    double val = 0.0;
    bool allnum = true;
    for (const auto& s : toks) {
        if (!try_stod(s, val)) { allnum = false; break; }
    }
    if (allnum) {
        out_vec.reserve(toks.size());
        for (const auto& s : toks) out_vec.push_back(std::stod(s));
        return true;
    }

    if (toks.size() < 2) return false;
    out_id = toks[0];
    out_vec.reserve(toks.size() - 1);
    for (size_t i = 1; i < toks.size(); ++i) {
        if (!try_stod(toks[i], val)) return false;
        out_vec.push_back(std::stod(toks[i]));
    }
    return true;
}

} // namespace common
