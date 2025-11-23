import optuna
import subprocess
import re
import csv
import os
import math
from typing import Tuple, Dict
from datetime import datetime

# --- 配置 ---
BIN_PATH = "./testg"
MIN_RECALL = 0.985
NUM_RUNS = 2             
FIXED_I = 16             
INITIAL_C = 8867         
INITIAL_P = 1669         
N_TRIALS = 50000           

RESULT_CSV = "optuna_search_log.csv"
PRECISION_DIGITS = 6     # 统一使用 6 位小数精度

# --- 奖励/惩罚常量 (用于最大化目标) ---
SUCCESS_FACTOR = 1000.0  # 奖励缩放系数，用于放大 1/Time
RECALL_BONUS_SCALE = 1000.0 # 新增：溢出召回率的奖励系数 (次要目标)

LARGE_PENALTY = 1000000.0 # 巨大的负数，确保失败的奖励远低于成功的奖励
RECALL_SCALE_NEG = 100000.0 # 召回率差距的惩罚缩放


# --- 全局缓存字典 ---
CACHE: Dict[Tuple[int, int], Tuple[float, float]] = {}


def load_cache_data():
    """从 CSV 文件加载历史测试结果到内存缓存中。"""
    global CACHE
    if not os.path.exists(RESULT_CSV):
        return

    with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row.get('status') == 'COMPLETE':
                    key = (int(row['C']), int(row['P']))
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
            'C': trial.params.get('c'),
            'P': trial.params.get('p'),
            'I': FIXED_I,
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


def run_test(c: int, p: int, i: int) -> Tuple[float, float]:
    """
    运行外部测试程序并返回平均召回率和平均查询时间，已集成 CSV 缓存和浮点数精度。
    """
    key = (c, p)
    if key in CACHE:
        recall, time_ms = CACHE[key]
        print(f"  [CACHE HIT] C={c}, P={p} -> Recall={recall:.4f}, Time={time_ms:.{PRECISION_DIGITS}f}ms (from CSV)")
        return recall, time_ms
        
    total_recall = 0.0
    total_time_ms = 0.0
    BASE_FAIL_METRIC = 100000.0 
    
    cmd = [BIN_PATH, "--n", str(c), "--k", str(i), "--p", str(p)]
    
    for run_num in range(1, NUM_RUNS + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            
            if result.returncode != 0:
                print(f"  Run {run_num}: Test failed (Code {result.returncode}).")
                return 0.0, BASE_FAIL_METRIC
            
            recall_match = re.search(r"Average recall@.*: *([0-9]*\.?[0-9]+)", result.stdout)
            time_match = re.search(r"Average query time.*: *([0-9]*\.?[0-9]+) *ms", result.stdout)

            if recall_match and time_match:
                recall = float(recall_match.group(1))
                time_ms = float(time_match.group(1))
                total_recall += recall
                total_time_ms += time_ms
            else:
                print(f"  Run {run_num}: Could not parse output.")
                return 0.0, BASE_FAIL_METRIC
                
        except subprocess.TimeoutExpired:
            print(f"  Run {run_num}: Timeout.")
            return 0.0, BASE_FAIL_METRIC
        except FileNotFoundError:
            print(f"Error: Executable '{BIN_PATH}' not found.")
            exit(1)

    avg_recall = total_recall / NUM_RUNS
    avg_time_ms = total_time_ms / NUM_RUNS
    
    # 四舍五入和更新缓存
    rounded_recall = round(avg_recall, PRECISION_DIGITS)
    rounded_time_ms = round(avg_time_ms, PRECISION_DIGITS)

    print(f"  Avg Result: C={c}, P={p} -> Recall={rounded_recall:.4f}, Time={rounded_time_ms:.{PRECISION_DIGITS}f}ms (NEW RUN)")
    
    CACHE[key] = (rounded_recall, rounded_time_ms)
    
    return rounded_recall, rounded_time_ms


def objective(trial: optuna.trial.Trial) -> float:
    
    # 1. 定义搜索空间
    c = trial.suggest_int("c", 1024, 20480) 
    p_max = min(2048, c)
    p = trial.suggest_int("p", 1, p_max)
    
    # 2. 运行测试并获取平均指标
    print(f"\nTrial {trial.number}: Testing C={c}, P={p}")
    recall, time_ms = run_test(c, p, FIXED_I) 
    
    # 3. 记录额外信息
    trial.set_user_attr("avg_recall", recall)
    trial.set_user_attr("avg_time_ms", time_ms)
    
    # 4. 计算奖励分数 (目标：最大化)
    if recall < MIN_RECALL:
        # 惩罚项 (负奖励)：巨大负数 + 召回差距惩罚 + 实际时间惩罚 (时间越短，惩罚越少，奖励越大)
        delta = MIN_RECALL - recall
        reward = -LARGE_PENALTY - (delta * RECALL_SCALE_NEG) - time_ms
        
        print(f"  Score: Failed constraint. Penalty={reward:.{PRECISION_DIGITS}f}")
        return reward
    
    # --- 达标时的奖励 (正奖励) ---
    
    # 核心奖励：时间反比 (主导因素)
    time_component = SUCCESS_FACTOR / time_ms
    
    # 次要奖励：溢出召回率 (次要因素)
    recall_surplus = recall - MIN_RECALL
    recall_component = RECALL_BONUS_SCALE * recall_surplus
    
    # 最终奖励 = 时间效益 + 溢出召回率奖励
    reward = time_component + recall_component
    
    print(f"  Score: Constraint met. Maximize Reward={reward:.{PRECISION_DIGITS}f}")
    return reward

if __name__ == "__main__":
    
    load_cache_data()

    print(f"Starting Optuna optimization for '{BIN_PATH}'...")
    print(f"Optimization Goal: MAXIMIZE Reward (Reward = 1000/Time + 1000 * (Recall - {MIN_RECALL}) if Recall >= {MIN_RECALL})")
    print(f"CSV log will be written to: {RESULT_CSV}")

    study = optuna.create_study(
        direction="maximize", 
        study_name="ivf_param_tuning",
    )
    
    initial_key = (INITIAL_C, INITIAL_P)
    if initial_key not in CACHE:
        print(f"\nEnqueuing initial trial: C={INITIAL_C}, P={INITIAL_P}")
        study.enqueue_trial({"c": INITIAL_C, "p": INITIAL_P})
    
    try:
        study.optimize(
            objective, 
            n_trials=N_TRIALS,
            callbacks=[csv_logger]
        )
    except Exception as e:
        print(f"\nOptimization interrupted: {e}")

    print("\n" + "="*50)
    print("=== Search Finished ===")
    
    best_trial = study.best_trial
    
    print("\nBest Configuration Found:")
    print(f"  C: {best_trial.params['c']}")
    print(f"  P: {best_trial.params['p']}")
    print(f"  I: {FIXED_I}")
    print("\nMetrics (Average of 2 Runs):")
    print(f"  * Best Reward Score: {best_trial.value:.{PRECISION_DIGITS}f}")
    print(f"  * Corresponding Recall: {best_trial.user_attrs['avg_recall']:.4f}")
    print(f"  * Corresponding Time: {best_trial.user_attrs['avg_time_ms']:.{PRECISION_DIGITS}f} ms")
    
    print(f"\nDetailed log saved to {RESULT_CSV}")
    print("="*50)