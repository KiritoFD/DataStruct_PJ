#!/usr/bin/env python3
import csv
import json
import math
import os
import subprocess
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# 配置
PARAM_BOUNDS = {
    "NUM_CENTROIDS": (32, 8192),
    "NPROBE": (4, 2048),
}
FIXED_KMEANS_ITER = 4

CACHE_DIR = Path("tune")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = CACHE_DIR / "history.csv"
BEST_JSON = CACHE_DIR / "best_config.json"

CXX = os.environ.get("CXX", "g++")
CXXFLAGS = "-O3 -std=c++17 -pthread -march=native"
TIMEOUT_SEC = 1800

def stamp(params):
    # KMEANS_ITER固定为4
    return f"c{params['NUM_CENTROIDS']}_i{FIXED_KMEANS_ITER}_p{params['NPROBE']}"

class Cache:
    def __init__(self):
        self.data = {}
        if HISTORY_CSV.exists():
            with HISTORY_CSV.open() as f:
                for row in csv.DictReader(f):
                    try:
                        p = {k: int(row[k]) for k in PARAM_BOUNDS}
                        p['KMEANS_ITER'] = FIXED_KMEANS_ITER
                        self.data[stamp(p)] = {
                            'params': p,
                            'recall': float(row['RECALL']) if row['RECALL'] != 'NA' else None,
                            'avg_time': float(row['AVG_QUERY_TIME_ms']) if row['AVG_QUERY_TIME_ms'] != 'NA' else None,
                            'build_time': float(row['INDEX_BUILD_TIME_s']) if row.get('INDEX_BUILD_TIME_s', 'NA') != 'NA' else None,
                            'status': row['STATUS']
                        }
                    except: pass

    def get(self, params):
        params = dict(params)
        params['KMEANS_ITER'] = FIXED_KMEANS_ITER
        key = stamp(params)
        if key in self.data:
            print(f"[Cache] 命中: {key} -> {self.data[key]}")
        else:
            print(f"[Cache] 未命中: {key}")
        return self.data.get(key)

    def add(self, params, recall, avg_time, build_time, status):
        params = dict(params)
        params['KMEANS_ITER'] = FIXED_KMEANS_ITER
        key = stamp(params)
        print(f"[Cache] 写入: {key} recall={recall} avg_time={avg_time} build_time={build_time} status={status}")
        self.data[key] = {'params': params, 'recall': recall, 'avg_time': avg_time, 'build_time': build_time, 'status': status}
        header = not HISTORY_CSV.exists()
        with HISTORY_CSV.open('a') as f:
            w = csv.writer(f)
            if header:
                w.writerow(['STAMP', 'NUM_CENTROIDS', 'KMEANS_ITER', 'NPROBE', 'STATUS', 'ELAPSED_s', 'RECALL', 'AVG_QUERY_TIME_ms', 'INDEX_BUILD_TIME_s'])
            w.writerow([key, params['NUM_CENTROIDS'], FIXED_KMEANS_ITER, params['NPROBE'], status, 'NA', recall or 'NA', avg_time or 'NA', build_time or 'NA'])

def run_experiment(params, timeout):
    params = dict(params)
    params['KMEANS_ITER'] = FIXED_KMEANS_ITER

    stamp_str = stamp(params)
    logfile = CACHE_DIR / f"run_{stamp_str}.log"
    buildlog = CACHE_DIR / f"run_{stamp_str}.build.log"
    
    # 编译 - 使用与bash脚本相同的宏定义
    compile_cmd = [
        CXX, *CXXFLAGS.split(), 
        f"-DNUM_CENTROID={params['NUM_CENTROIDS']}",
        f"-DKMEAN_ITER={params['KMEANS_ITER']}", 
        f"-DNPROB={params['NPROBE']}",
        "MySolution.cpp", "test_solution.cpp", "-o", "test"
    ]
    
    print(f"[run_experiment] 编译命令: {' '.join(compile_cmd)}")
    print(f"[run_experiment] 编译日志: {buildlog}")
    with open(buildlog, 'w') as f:
        compile_result = subprocess.run(compile_cmd, stdout=f, stderr=subprocess.STDOUT)
    
    if compile_result.returncode != 0:
        print(f"[run_experiment] 编译失败: {stamp_str}, 返回码: {compile_result.returncode}")
        return 'COMPILE_FAIL', None, None, None
    
    # 运行 - 使用timeout
    print(f"[run_experiment] 运行 (超时 {timeout}s)... 日志: {logfile}")
    start_time = time.time()
    
    try:
        with open(logfile, 'w') as f:
            run_result = subprocess.run(
                ["timeout", f"{timeout}s", "./test"], 
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout + 10  # 额外缓冲
            )
        
        elapsed = time.time() - start_time
        print(f"[run_experiment] 运行完成, 返回码: {run_result.returncode}, 用时: {elapsed:.2f}s")
        
        if run_result.returncode == 124 or run_result.returncode == 137:
            print(f"[run_experiment] 超时: {stamp_str}")
            return 'TIMEOUT', None, None, None
        elif run_result.returncode != 0:
            print(f"[run_experiment] 运行错误: {stamp_str}, 返回码: {run_result.returncode}")
            return f'ERROR_{run_result.returncode}', None, None, None
            
    except subprocess.TimeoutExpired:
        print(f"[run_experiment] Python超时: {stamp_str}")
        return 'TIMEOUT', None, None, None
    
    # 解析结果 - 使用与bash脚本相同的解析逻辑
    recall = avg_time = build_time = None
    print(f"[run_experiment] 解析日志: {logfile}")
    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            if "Average recall@" in line:
                try:
                    recall = float(line.split(":")[-1].strip())
                    print(f"[run_experiment] 解析到 recall: {recall}")
                except Exception as e:
                    print(f"[run_experiment] recall解析异常: {e}")
            elif "Average query time" in line:
                try:
                    avg_time = float(line.split(":")[-1].split()[0])
                    print(f"[run_experiment] 解析到 avg_time: {avg_time}")
                except Exception as e:
                    print(f"[run_experiment] avg_time解析异常: {e}")
            elif "Index build time" in line:
                try:
                    build_time = float(line.split(":")[-1].split()[0])
                    print(f"[run_experiment] 解析到 build_time: {build_time}")
                except Exception as e:
                    print(f"[run_experiment] build_time解析异常: {e}")
    
    return 'OK', recall, avg_time, build_time

def objective(recall, avg_time, target):
    if recall is None or avg_time is None:
        return float('inf')
    if recall >= target:
        # 召回率达标，权重很小
        return avg_time + 10 * (target - recall)
    elif recall >= 0.975:
        # 0.975~0.98之间，惩罚急剧增大（指数形式）
        penalty = avg_time + 50 * math.exp(100 * (target - recall))
        return penalty
    else:
        # 0.975以下直接截断
        return float('inf')

class SPSA:
    def __init__(self, target_recall):
        self.target = target_recall
        self.a, self.c = 0.8, 0.1

    def optimize(self, cache, init_params, max_iters, timeout):
        # 只优化NUM_CENTROIDS和NPROBE
        current = self.vec(init_params)
        best_vec, best_obj = current, float('inf')
        best_result = None

        for k in range(1, max_iters + 1):
            a_k, c_k = self.a/k**0.6, self.c/k**0.1
            delta = np.random.choice([-1, 1], len(PARAM_BOUNDS))
            print(f"[SPSA] 迭代{k} a_k={a_k:.4f} c_k={c_k:.4f} delta={delta}")

            plus_vec = current + c_k * delta
            minus_vec = current - c_k * delta
            print(f"[SPSA] plus_vec={plus_vec}, minus_vec={minus_vec}")
            plus_res = self.eval(cache, plus_vec, timeout)
            minus_res = self.eval(cache, minus_vec, timeout)
            print(f"[SPSA] plus_res={plus_res}, minus_res={minus_res}")

            if plus_res['obj'] < float('inf') and minus_res['obj'] < float('inf'):
                grad = (plus_res['obj'] - minus_res['obj']) / (2 * c_k * delta)
                print(f"[SPSA] grad={grad}")
                candidate = self.project(current - a_k * grad)
                print(f"[SPSA] candidate_vec={candidate}, candidate_params={self.params(candidate)}")
                candidate_res = self.eval(cache, candidate, timeout)
                print(f"[SPSA] candidate_res={candidate_res}")

                if candidate_res['obj'] < best_obj:
                    current, best_vec, best_obj = candidate, candidate, candidate_res['obj']
                    best_result = candidate_res
                    print(f"[SPSA] 迭代 {k}: 新最佳参数 {candidate_res['params']} 召回率={candidate_res['recall']} 时间={candidate_res['avg_time']}ms obj={candidate_res['obj']}")

            if k > 10 and a_k < 0.01:
                print("[SPSA] 收敛，提前终止")
                break

        print(f"[SPSA] 最佳结果: {best_result}")
        return best_result or self.eval(cache, best_vec, timeout)

    def vec(self, params):
        # 只处理NUM_CENTROIDS和NPROBE
        v = np.array([math.log(params[k]) for k in PARAM_BOUNDS])
        print(f"[SPSA] params->vec: {params} -> {v}")
        return v

    def params(self, vec):
        p = {}
        for i, k in enumerate(PARAM_BOUNDS):
            lo, hi = PARAM_BOUNDS[k]
            p[k] = int(round(np.clip(math.exp(vec[i]), lo, hi)))
        p['KMEANS_ITER'] = FIXED_KMEANS_ITER
        print(f"[SPSA] vec->params: {vec} -> {p}")
        return p

    def project(self, vec):
        projected = []
        for i, k in enumerate(PARAM_BOUNDS):
            lo, hi = math.log(PARAM_BOUNDS[k][0]), math.log(PARAM_BOUNDS[k][1])
            projected.append(np.clip(vec[i], lo, hi))
        print(f"[SPSA] 投影: {vec} -> {projected}")
        return np.array(projected)

    def eval(self, cache, vec, timeout):
        params = self.params(vec)
        print(f"[SPSA] 评估参数: {params}")
        cached = cache.get(params)
        if cached:
            cached['obj'] = objective(cached['recall'], cached['avg_time'], self.target)
            print(f"[SPSA] 使用缓存 obj={cached['obj']}")
            return cached

        status, recall, avg_time, build_time = run_experiment(params, timeout)
        print(f"[SPSA] 实验结果: status={status}, recall={recall}, avg_time={avg_time}, build_time={build_time}")
        cache.add(params, recall, avg_time, build_time, status)
        result = {'params': params, 'recall': recall, 'avg_time': avg_time, 'build_time': build_time, 'status': status}
        result['obj'] = objective(recall, avg_time, self.target)
        print(f"[SPSA] 评估完成 obj={result['obj']}")
        return result

def main():
    target_recall = 0.98
    init_params = {'NUM_CENTROIDS': 4096, 'NPROBE': 1024, 'KMEANS_ITER': FIXED_KMEANS_ITER}

    cache = Cache()
    optimizer = SPSA(target_recall)

    print(f"优化目标: 召回率 > {target_recall}, 最小化查询时间")
    print(f"初始参数: {init_params}")
    print(f"超时设置: {TIMEOUT_SEC}秒")
    print()

    best = optimizer.optimize(cache, init_params, max_iters=25, timeout=TIMEOUT_SEC)

    print(f"\n=== 最终结果 ===")
    print(f"最佳参数: {best['params']}")
    print(f"召回率: {best['recall']:.4f}")
    print(f"平均查询时间: {best['avg_time']:.2f}ms")
    if best['build_time']:
        print(f"索引构建时间: {best['build_time']:.2f}s")

    with open(BEST_JSON, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"配置已保存至: {BEST_JSON}")

if __name__ == "__main__":
    main()