#!/usr/bin/env python3
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 配置
PARAM_BOUNDS = {
    "NUM_CENTROIDS": (64,3000),
    "NPROBE": (128,1024),
    "KMEAN_ITER": (4, 8),
}

CACHE_DIR = Path("tune")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUM_CSV = CACHE_DIR / "sum.csv"
BEST_JSON = CACHE_DIR / "best_config.json"

CXX = os.environ.get("CXX", "g++")
CXXFLAGS = "-O3 -std=c++17 -pthread -march=native"
TIMEOUT_SEC = 1800

def stamp(params):
    return f"c{params['NUM_CENTROIDS']}_i{params['KMEAN_ITER']}_p{params['NPROBE']}"

class Cache:
    def __init__(self):
        self.data = {}
        if SUM_CSV.exists():
            with SUM_CSV.open() as f:
                for row in csv.DictReader(f):
                    try:
                        p = {k: int(row[k]) for k in PARAM_BOUNDS}
                        self.data[stamp(p)] = {
                            'params': p,
                            'recall': float(row['RECALL']) if row['RECALL'] != 'NA' else None,
                            'avg_time': float(row['AVG_QUERY_TIME_ms']) if row['AVG_QUERY_TIME_ms'] != 'NA' else None,
                            'build_time': float(row['INDEX_BUILD_TIME_s']) if row.get('INDEX_BUILD_TIME_s', 'NA') != 'NA' else None,
                            'status': row['STATUS']
                        }
                    except: 
                        pass

    def get(self, params):
        return self.data.get(stamp(params))

    def add(self, params, recall, avg_time, build_time, status):
        key = stamp(params)
        self.data[key] = {
            'params': params, 
            'recall': recall, 
            'avg_time': avg_time, 
            'build_time': build_time, 
            'status': status
        }
        
        header = not SUM_CSV.exists()
        with SUM_CSV.open('a') as f:
            w = csv.writer(f)
            if header:
                w.writerow([
                    'STAMP', 'NUM_CENTROIDS', 'KMEAN_ITER', 'NPROBE', 
                    'STATUS', 'ELAPSED_s', 'RECALL', 'AVG_QUERY_TIME_ms', 'INDEX_BUILD_TIME_s'
                ])
            w.writerow([
                key, params['NUM_CENTROIDS'], params['KMEAN_ITER'], params['NPROBE'],
                status, 'NA', recall or 'NA', avg_time or 'NA', build_time or 'NA'
            ])

def run_experiment(params, timeout):
    """运行实验并返回结果"""
    stamp_str = stamp(params)
    logfile = CACHE_DIR / f"run_{stamp_str}.log"
    
    print(f"运行实验: {stamp_str}")
    start_time = time.time()
    
    try:
        with open(logfile, 'w') as f:
            run_result = subprocess.run(
                ["timeout", f"{timeout}s", "./test", 
                 "--num-centroids", str(params['NUM_CENTROIDS']), 
                 "--kmean-iter", str(params['KMEAN_ITER']), 
                 "--nprob", str(params['NPROBE'])], 
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout + 10
            )
        
        elapsed = time.time() - start_time
        
        if run_result.returncode == 124 or run_result.returncode == 137:
            return 'TIMEOUT', None, None, None
        elif run_result.returncode != 0:
            return f'ERROR_{run_result.returncode}', None, None, None
            
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', None, None, None
    
    # 解析结果
    recall = avg_time = build_time = None
    
    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            if "Average recall@" in line:
                try:
                    recall = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Average query time" in line:
                try:
                    avg_time = float(line.split(":")[-1].split()[0])
                except:
                    pass
            elif "Index build time" in line:
                try:
                    build_time = float(line.split(":")[-1].split()[0])
                except:
                    pass
    
    return 'OK', recall, avg_time, build_time

def evaluate_params(params, cache, timeout, target_recall):
    """评估参数配置"""
    cached = cache.get(params)
    if cached:
        return cached
    
    status, recall, avg_time, build_time = run_experiment(params, timeout)
    cache.add(params, recall, avg_time, build_time, status)
    
    result = {
        'params': params, 
        'recall': recall, 
        'avg_time': avg_time, 
        'build_time': build_time, 
        'status': status
    }
    
    # 计算得分：召回率达标时得分=1/查询时间，否则为负分
    if status == 'OK' and recall is not None and avg_time is not None:
        if recall >= target_recall:
            result['score'] = (20-avg_time)*7+(recall-target_recall)*100  # 查询时间越小，得分越高
        else:
            # 召回率不足，根据差距给负分
            result['score'] = avg_time+10**((target_recall-recall)*100) # 差距越大惩罚越大
    else:
        result['score'] = -float('inf')
    
    return result

class AdaptiveSearch:
    def __init__(self, target_recall):
        self.target_recall = target_recall
        # 初始搜索步长（相对比例）
        self.step_sizes = {
            'NUM_CENTROIDS': 0.05,  # 5% 变化
            'NPROBE': 0.15,      # 15% 变化
            'KMEAN_ITER': 0.05     # 5% 变化（范围小）
        }
        self.min_step_sizes = {
            'NUM_CENTROIDS': 0.01,  # 最小1%变化
            'NPROBE': 0.02,         # 最小2%变化
            'KMEAN_ITER': 0.01      # 最小1%变化
        }
        self.max_step_sizes = {
            'NUM_CENTROIDS': 0.3,   # 最大30%变化
            'NPROBE': 0.4,          # 最大40%变化
            'KMEAN_ITER': 0.2       # 最大20%变化
        }
        
        # 跟踪历史改进
        self.improvement_history = []
        self.consecutive_failures = 0
        
    def get_neighbors(self, params, last_improvement=None):
        """根据历史表现动态生成邻居"""
        neighbors = []
        
        # 基础方向：如果上次有改进，继续类似方向
        base_directions = [1, -1]  # 增加和减少
        
        # 如果有明显的改进方向，加强该方向
        if last_improvement and len(self.improvement_history) > 2:
            recent_trend = sum(self.improvement_history[-3:]) / 3
            if abs(recent_trend) > 0.1:  # 有明显趋势
                if recent_trend > 0:
                    base_directions = [1, 1, -1]  # 加强正向
                else:
                    base_directions = [-1, -1, 1]  # 加强负向
        
        for param_name in ['NUM_CENTROIDS', 'NPROBE', 'KMEAN_ITER']:
            current_val = params[param_name]
            step_size = self.step_sizes[param_name]
            
            # 根据连续失败次数调整步长
            if self.consecutive_failures > 3:
                step_size = max(step_size * 0.8, self.min_step_sizes[param_name])
            elif self.consecutive_failures == 0 and len(self.improvement_history) > 0:
                # 连续成功，适当增大步长
                step_size = min(step_size * 1.1, self.max_step_sizes[param_name])
            
            for direction in base_directions:
                # 计算新值（相对变化）
                if param_name == 'KMEAN_ITER':
                    # KMEAN_ITER 使用绝对变化
                    new_val = current_val + direction * max(1, int(current_val * step_size))
                else:
                    # 其他参数使用相对变化
                    new_val = int(current_val * (1 + direction * step_size))
                
                # 确保在边界内
                new_val = max(PARAM_BOUNDS[param_name][0], 
                            min(PARAM_BOUNDS[param_name][1], new_val))
                
                if new_val != current_val:
                    neighbor = params.copy()
                    neighbor[param_name] = new_val
                    neighbors.append(neighbor)
        
        # 添加一些随机组合的邻居（探索新方向）
        if len(neighbors) < 8:  # 如果邻居太少，添加一些组合
            for _ in range(3):
                neighbor = params.copy()
                for param_name in ['NUM_CENTROIDS', 'NPROBE']:
                    direction = 1 if np.random.random() > 0.5 else -1
                    step_size = self.step_sizes[param_name] * (0.5 + np.random.random())
                    current_val = neighbor[param_name]
                    new_val = int(current_val * (1 + direction * step_size))
                    new_val = max(PARAM_BOUNDS[param_name][0], 
                                min(PARAM_BOUNDS[param_name][1], new_val))
                    neighbor[param_name] = new_val
                if stamp(neighbor) != stamp(params):
                    neighbors.append(neighbor)
        
        # 去重
        seen = set()
        unique_neighbors = []
        for neighbor in neighbors:
            key = stamp(neighbor)
            if key not in seen:
                seen.add(key)
                unique_neighbors.append(neighbor)
        
        return unique_neighbors
    
    def update_step_sizes(self, improvement_ratio):
        """根据改进情况更新步长"""
        self.improvement_history.append(improvement_ratio)
        
        if improvement_ratio > 0.01:  # 有明显改进
            self.consecutive_failures = 0
            # 成功时稍微增大步长（但不超过最大值）
            for param in self.step_sizes:
                self.step_sizes[param] = min(
                    self.step_sizes[param] * 1.05, 
                    self.max_step_sizes[param]
                )
        else:
            self.consecutive_failures += 1
            # 失败时减小步长（但不小于最小值）
            for param in self.step_sizes:
                self.step_sizes[param] = max(
                    self.step_sizes[param] * 0.9,
                    self.min_step_sizes[param]
                )
        
        # 保持历史长度
        if len(self.improvement_history) > 10:
            self.improvement_history.pop(0)

def adaptive_greedy_search(cache, init_params, target_recall, max_iterations, timeout):
    """自适应贪心搜索算法"""
    current_params = init_params.copy()
    current_result = evaluate_params(current_params, cache, timeout, target_recall)
    best_result = current_result
    
    search_engine = AdaptiveSearch(target_recall)
    
    print(f"初始配置: {current_params}")
    print(f"初始结果: 召回率={current_result['recall']}, 查询时间={current_result['avg_time']}ms")
    print(f"初始步长: {search_engine.step_sizes}")
    
    last_score = current_result.get('score', -float('inf'))
    
    for iteration in range(max_iterations):
        print(f"\n--- 迭代 {iteration + 1} ---")
        print(f"当前步长: {search_engine.step_sizes}")
        print(f"连续失败: {search_engine.consecutive_failures}")
        
        # 生成并评估邻居
        neighbors = search_engine.get_neighbors(current_params)
        print(f"评估 {len(neighbors)} 个邻居 (步长: {search_engine.step_sizes})...")
        
        best_neighbor = None
        best_score = last_score
        
        neighbor_results = []
        for neighbor in neighbors:
            result = evaluate_params(neighbor, cache, timeout, target_recall)
            score = result.get('score', -float('inf'))
            neighbor_results.append((result, score))
            
            improvement = score - last_score
            status = "↑" if improvement > 0.001 else "↓" if improvement < -0.001 else "→"
            
            print(f"  {status} 邻居 {neighbor}: 召回率={result['recall']:.4f}, 时间={result['avg_time']:.2f}ms, 得分={score:.6f}")
            
            if score > best_score + 1e-6:  # 避免浮点误差
                best_score = score
                best_neighbor = result
        
        # 更新步长
        if best_neighbor:
            improvement_ratio = (best_score - last_score) / (abs(last_score) + 1e-6)
            search_engine.update_step_sizes(improvement_ratio)
        
        # 如果没有更好的邻居，尝试更激进的搜索
        if best_neighbor is None and search_engine.consecutive_failures < 5:
            print("未找到更好邻居，尝试扩大搜索范围...")
            # 临时增大步长
            original_steps = search_engine.step_sizes.copy()
            for param in search_engine.step_sizes:
                search_engine.step_sizes[param] = min(
                    search_engine.step_sizes[param] * 1.5,
                    search_engine.max_step_sizes[param]
                )
            
            # 重新生成邻居
            expanded_neighbors = search_engine.get_neighbors(current_params)
            for neighbor in expanded_neighbors:
                if any(stamp(neighbor) == stamp(n['params']) for n, _ in neighbor_results):
                    continue  # 跳过已评估的
                    
                result = evaluate_params(neighbor, cache, timeout, target_recall)
                score = result.get('score', -float('inf'))
                
                improvement = score - last_score
                print(f"  *扩展* 邻居 {neighbor}: 召回率={result['recall']:.4f}, 时间={result['avg_time']:.2f}ms, 得分={score:.6f}")
                
                if score > best_score + 1e-6:
                    best_score = score
                    best_neighbor = result
            
            # 恢复步长
            search_engine.step_sizes = original_steps
        
        # 如果还是没有改进，停止搜索
        if best_neighbor is None:
            print("无法找到更好的配置，停止搜索")
            break
        
        # 移动到最佳邻居
        current_params = best_neighbor['params'].copy()
        current_result = best_neighbor
        last_score = best_score
        
        # 更新全局最佳
        if best_score > best_result.get('score', -float('inf')):
            best_result = current_result
            print(f"🎯 新的最佳配置: {best_result['params']}")
            print(f"   召回率={best_result['recall']:.4f}, 查询时间={best_result['avg_time']:.2f}ms")
        
        print(f"当前最佳: {best_result['params']} (得分={best_result.get('score', 'N/A'):.6f})")
        
        # 收敛检查
        if search_engine.consecutive_failures >= 5 and all(
            sz <= min_sz * 1.1 for sz, min_sz in 
            zip(search_engine.step_sizes.values(), search_engine.min_step_sizes.values())
        ):
            print("步长已收敛到最小值，停止搜索")
            break
    
    return best_result

def main():
    target_recall = 0.98
    init_params = {'NUM_CENTROIDS': 947, 'KMEAN_ITER': 6, 'NPROBE': 277}
    
    cache = Cache()
    
    print(f"优化目标: 召回率 >= {target_recall}, 最小化查询时间")
    print(f"初始参数: {init_params}")
    print(f"超时设置: {TIMEOUT_SEC}秒")
    print()
    
    best = adaptive_greedy_search(cache, init_params, target_recall, max_iterations=200, timeout=TIMEOUT_SEC)
    
    print(f"\n=== 最终结果 ===")
    print(f"最佳参数: {best['params']}")
    print(f"召回率: {best['recall']:.4f}")
    print(f"平均查询时间: {best['avg_time']:.2f}ms")
    if best['build_time']:
        print(f"索引构建时间: {best['build_time']:.2f}s")
    
    # 保存结果
    with open(BEST_JSON, 'w') as f:
        json.dump({
            'params': best['params'],
            'recall': best['recall'],
            'avg_query_time_ms': best['avg_time'],
            'index_build_time_s': best['build_time'],
            'status': best['status']
        }, f, indent=2)
    
    print(f"配置已保存至: {BEST_JSON}")

if __name__ == "__main__":
    import numpy as np
    main()