import optuna
import subprocess
import re
import csv
import os
import math
from typing import Tuple, Dict
from datetime import datetime

# --- 配置 ---
BIN_PATH = "./hn"           # HNSW 可执行文件的路径
MIN_RECALL = 0.985          # 最小召回率约束 (目标: 98.5%)
NUM_RUNS = 1                # 每次参数组合运行的次数 (建议 > 3 以取平均值，但为了快速测试可设为 1)
FIXED_K = 10                # 固定的 Top-K 搜索结果数
N_TRIALS = 50000            # Optuna 最大试验次数

RESULT_CSV = "optuna_search_hn.csv"
PRECISION_DIGITS = 6        # 统一使用 6 位小数精度

# --- 奖励/惩罚常量 (用于最大化目标) ---
SUCCESS_FACTOR = 1000.0     # 奖励缩放系数，用于放大 1/Time (主导因素)
RECALL_BONUS_SCALE = 1000.0 # 溢出召回率的奖励系数 (次要因素)

LARGE_PENALTY = 100000.0    # 巨大的负数，确保失败的奖励远低于成功的奖励
RECALL_SCALE_NEG = 100.0    # 召回率差距的惩罚缩放


# --- 全局缓存字典 (key: m, efc, efs, k) ---
# k 必须是 key 的一部分，因为 K 值变化会影响 Recall 和 Time
CACHE: Dict[Tuple[int, int, int, int], Tuple[float, float]] = {}


def load_cache_data():
    """从 CSV 文件加载历史测试结果到内存缓存中，Key 结构为 (m, efc, efs, k)。"""
    global CACHE
    if not os.path.exists(RESULT_CSV):
        return

    with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get('status') == 'COMPLETE':
                    m = int(row.get('m') or -1)
                    efc = int(row.get('efc') or -1)
                    efs = int(row.get('efs') or -1)
                    # 关键：将 K (I) 加入 Key
                    k = int(row.get('K') or FIXED_K) 
                    key = (m, efc, efs, k)

                    recall = round(float(row['avg_recall']), PRECISION_DIGITS)
                    time_ms = round(float(row['avg_time_ms']), PRECISION_DIGITS)
                    CACHE[key] = (recall, time_ms)
            except (ValueError, KeyError):
                continue
    print(f"Loaded {len(CACHE)} unique results from {RESULT_CSV} cache.")


def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """
    Optuna 回调函数：在每次试验完成后，将结果记录到 CSV 文件中。
    """
    if trial.state.is_finished():
        data = {
            'timestamp': datetime.now().isoformat(),
            'trial_id': trial.number,
            'm': trial.params.get('m'),
            'efc': trial.params.get('efc'),
            'efs': trial.params.get('efs'),
            'K': FIXED_K, # 使用全局 K 值
            'avg_recall': trial.user_attrs.get('avg_recall'),
            'avg_time_ms': trial.user_attrs.get('avg_time_ms'),
            'reward_score': trial.value if trial.value is not None else -float('inf'),
            'status': trial.state.name,
        }
        
        fieldnames = list(data.keys())
        file_exists = os.path.exists(RESULT_CSV)
        
        try:
            with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
        except IOError as e:
            print(f"Error writing to CSV file: {e}")


def run_test(k: int, m: int, efc: int, efs: int) -> Tuple[float, float]:
    """
    运行外部测试程序并返回平均召回率和平均查询时间。
    """
    key = (m, efc, efs, k)
    if key in CACHE:
        recall, time_ms = CACHE[key]
        print(f"  [CACHE HIT] m={m}, efc={efc}, efs={efs}, k={k} -> Recall={recall:.4f}, Time={time_ms:.{PRECISION_DIGITS}f}ms")
        return recall, time_ms
        
    total_recall = 0.0
    total_time_ms = 0.0
    BASE_FAIL_METRIC = 100000.0 # 失败时的默认高时间
    
    # 构造 CLI 命令
    cmd = [BIN_PATH, "--k", str(k), "--m", str(m), "--efc", str(efc), "--efs", str(efs)]

    for run_num in range(1, NUM_RUNS + 1):
        try:
            # 运行子进程，设置超时
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
            
            if result.returncode != 0:
                print(f"  Run {run_num}: Test failed (Code {result.returncode}). Output: {result.stdout.strip()[:100]}...")
                return 0.0, BASE_FAIL_METRIC
            
            # 正则表达式解析 Recall 和 Time
            recall_match = re.search(r"Average recall@.*: *([0-9]*\.?[0-9]+)", result.stdout)
            time_match = re.search(r"Average query time.*: *([0-9]*\.?[0-9]+) *ms", result.stdout)

            if recall_match and time_match:
                recall = float(recall_match.group(1))
                time_ms = float(time_match.group(1))
                total_recall += recall
                total_time_ms += time_ms
            else:
                print(f"  Run {run_num}: Could not parse output.")
                return 0.0, BASE_FAIL_METRIC
                
        except subprocess.TimeoutExpired:
            print(f"  Run {run_num}: Timeout.")
            return 0.0, BASE_FAIL_METRIC
        except FileNotFoundError:
            print(f"Error: Executable '{BIN_PATH}' not found.")
            exit(1)

    avg_recall = total_recall / NUM_RUNS
    avg_time_ms = total_time_ms / NUM_RUNS
    
    # 四舍五入和更新缓存
    rounded_recall = round(avg_recall, PRECISION_DIGITS)
    rounded_time_ms = round(avg_time_ms, PRECISION_DIGITS)

    print(f"  Avg Result: m={m}, efc={efc}, efs={efs} -> Recall={rounded_recall:.4f}, Time={rounded_time_ms:.{PRECISION_DIGITS}f}ms (NEW RUN)")
    
    CACHE[key] = (rounded_recall, rounded_time_ms)
    
    return rounded_recall, rounded_time_ms


def objective(trial: optuna.trial.Trial) -> float:
    # 1. 定义 HNSW 参数搜索范围 (基于合理范围设置)
    m = trial.suggest_int("m", 12, 36)      # 常见范围 12-32，略微扩大
    efc = trial.suggest_int("efc", 150, 512) # 常见范围 200-500，略微扩大
    efs = trial.suggest_int("efs", 32, 512)  # 搜索速度 vs 召回率的核心旋钮

    # 2. 运行测试
    print(f"\nTrial {trial.number}: Testing m={m}, efc={efc}, efs={efs}")
    recall, time_ms = run_test(FIXED_K, m, efc, efs) 
    
    # 3. 记录额外信息
    trial.set_user_attr("avg_recall", recall)
    trial.set_user_attr("avg_time_ms", time_ms)
    
    # 4. 计算奖励分数 (目标：最大化)
    
    # --- 惩罚项 (召回率未达标) ---
    if recall < MIN_RECALL:
        delta = MIN_RECALL - recall
        
        # 奖励公式: -巨大惩罚 - 召回差距惩罚 + (时间效益奖励)
        # 目的：在召回未达标的方案中，优先测试速度快的（即时间效益高）。
        time_bonus = SUCCESS_FACTOR / time_ms 
        reward = -LARGE_PENALTY - (delta * RECALL_SCALE_NEG) + time_bonus
        
        print(f"  Score: Failed constraint ({delta*100:.2f}% gap). Penalty={reward:.{PRECISION_DIGITS}f}")
        return reward
    
    # --- 达标时的奖励 (正奖励) ---
    
    # 核心奖励：时间反比 (主导因素)
    time_component = SUCCESS_FACTOR / time_ms
    
    # 次要奖励：溢出召回率 (避免召回率刚刚好，鼓励略高)
    recall_surplus = recall - MIN_RECALL
    recall_component = RECALL_BONUS_SCALE * recall_surplus
    
    # 最终奖励 = 时间效益 + 溢出召回率奖励
    reward = time_component + recall_component
    
    print(f"  Score: Constraint met. Maximize Reward={reward:.{PRECISION_DIGITS}f}")
    return reward

if __name__ == "__main__":
    
    load_cache_data()

    print("="*50)
    print(f"Starting Optuna optimization for '{BIN_PATH}'...")
    print(f"Target Recall (K={FIXED_K}): >= {MIN_RECALL*100:.2f}%")
    print(f"Optimization Goal: MAXIMIZE Reward (Maximize 1/Time)")
    print(f"CSV log will be written to: {RESULT_CSV}")
    print("="*50)

    study = optuna.create_study(
        direction="maximize", 
        study_name="hnsw_param_tuning",
        # 针对重复参数使用缓存的结果，防止 Optuna 重复建议相同的参数组合
        sampler=optuna.samplers.TPESampler(seed=42) 
    )
    
    # 预先加入一个常见的默认配置进行测试
    initial_key = (35, 395, 35, FIXED_K)
    if initial_key not in CACHE:
        print(f"\nEnqueuing initial trial: m=35, efc=395, efs=35")
        study.enqueue_trial({"m": 35, "efc": 395, "efs": 35})
    
    try:
        study.optimize(
            objective, 
            n_trials=N_TRIALS,
            callbacks=[csv_logger],
            gc_after_trial=True # 每次试验后进行垃圾回收，防止长时间运行内存泄漏
        )
    except Exception as e:
        print(f"\nOptimization interrupted: {e}")

    print("\n" + "="*50)
    print("=== Search Finished ===")
    
    best_trial = study.best_trial
    
    # 确保 best_trial 满足最小召回率
    if best_trial.user_attrs.get('avg_recall', 0.0) < MIN_RECALL:
        print("\n⚠️ 警告: 最佳试验未达到最小召回率约束。")
        print("请检查数据、增加 EFC/EFS 的范围或增加 N_TRIALS。")
    
    print("\nBest Configuration Found:")
    print(f"  m: {best_trial.params['m']}")
    print(f"  efc: {best_trial.params['efc']}")
    print(f"  efs: {best_trial.params['efs']}")
    print(f"  K: {FIXED_K}")
    print("\nMetrics:")
    print(f"  * Best Reward Score: {best_trial.value:.{PRECISION_DIGITS}f}")
    print(f"  * Corresponding Recall: {best_trial.user_attrs['avg_recall']:.4f}")
    print(f"  * Corresponding Time: {best_trial.user_attrs['avg_time_ms']:.{PRECISION_DIGITS}f} ms")
    
    print(f"\nDetailed log saved to {RESULT_CSV}")
    print("="*50)