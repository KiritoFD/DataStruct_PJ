import optuna
import subprocess
import re
import csv
import os
import random
import time
from typing import Tuple, Dict, Any
from datetime import datetime

# ==================== 配置区 ====================
BIN_PATH = "./hng"
MIN_RECALL = 0.985          # 目标召回率
FIXED_K = 10                # Top-K
BATCH_SIZE = 20             # 每次 optimize 运行多少次后进行状态检查和扰动

# ！！！ 补回了漏掉的配置 ！！！
NUM_RUNS = 1                # 每次参数组合运行取平均值的次数 (设为1最快，设为3更准)

# 扩展的搜索范围 (适应高召回率)
M_RANGE = (12, 64)
EFC_RANGE = (200, 1200)     
EFS_RANGE = (100, 2000)     

LOG_DIR = "Log"
RESULT_CSV = "optuna_adaptive_hng.csv"
PRECISION_DIGITS = 6

# --- 奖励/惩罚常量 ---
SUCCESS_FACTOR = 10000.0    
RECALL_BONUS_SCALE = 500.0  
LARGE_PENALTY = 1e9         

# --- 全局缓存 ---
CACHE: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

def load_cache_data():
    """加载历史数据，避免重复计算"""
    global CACHE
    if not os.path.exists(RESULT_CSV):
        return
    
    print(f"Loading history from {RESULT_CSV}...")
    try:
        with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    status = (row.get('status') or '').strip().upper()
                    if status != 'COMPLETE': continue

                    def p(v, t): return t(float(str(v).replace(',', '.'))) if v else None
                    
                    vals = {
                        'm': p(row.get('m'), int), 'efc': p(row.get('efc'), int), 'efs': p(row.get('efs'), int),
                        'rec': p(row.get('avg_recall'), float), 'time': p(row.get('avg_time_ms'), float),
                        'build': p(row.get('build_time_ms'), float)
                    }
                    
                    if all(v is not None for v in vals.values()):
                        CACHE[(vals['m'], vals['efc'], vals['efs'])] = (vals['rec'], vals['time'], vals['build'])
                except: continue
    except Exception as e:
        print(f"Cache load warning: {e}")
    print(f"Loaded {len(CACHE)} unique historical records.")

def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """写入 CSV 日志"""
    if trial.state.is_finished():
        data = {
            'timestamp': datetime.now().isoformat(),
            'trial_id': trial.number,
            'm': trial.params.get('m'), 'efc': trial.params.get('efc'), 'efs': trial.params.get('efs'),
            'K': FIXED_K,
            'avg_recall': trial.user_attrs.get('avg_recall'),
            'avg_time_ms': trial.user_attrs.get('avg_time_ms'),
            'build_time_ms': trial.user_attrs.get('build_time_ms'),
            'reward_score': trial.value if trial.value is not None else -float('inf'),
            'status': trial.state.name,
        }
        
        file_exists = os.path.exists(RESULT_CSV)
        try:
            with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys(), extrasaction='ignore')
                if not file_exists: writer.writeheader()
                writer.writerow(data)
        except IOError: pass

def run_test(m: int, efc: int, efs: int) -> Tuple[float, float, float]:
    """执行测试，含缓存和健壮解析"""
    key = (m, efc, efs)
    if key in CACHE:
        return CACHE[key]

    os.makedirs(LOG_DIR, exist_ok=True)
    cmd = [BIN_PATH, "-m", str(m), "-efc", str(efc), "-efs", str(efs)]
    
    total_rec, total_time, total_build = 0.0, 0.0, 0.0
    
    # 这里需要 NUM_RUNS，现在已在上方定义
    for _ in range(NUM_RUNS):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if res.returncode != 0: 
                print(f"   [Error] Process returned {res.returncode}")
                return 0.0, 1e6, 1e6 
            
            txt = res.stdout or ""
            
            # 解析逻辑 (适配您的格式)
            b_m = re.search(r"(?:Parallel\s*Build|build_time\(internal\))[^0-9]*([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)
            t_m = re.search(r"avg_query_time\s*=\s*([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)
            r_m = re.search(r"avg\s*recall@\d+\s*=\s*([0-9]+(?:[.,][0-9]+)?)", txt, re.I)

            if b_m and t_m and r_m:
                total_build += float(b_m.group(1).replace(',', '.'))
                total_time += float(t_m.group(1).replace(',', '.'))
                total_rec += float(r_m.group(1).replace(',', '.'))
            else:
                print(f"   [Parse Error] Output: {txt[:100]}...")
                return 0.0, 1e6, 1e6 
        except Exception as e:
            print(f"   [Exception] {e}")
            return 0.0, 1e6, 1e6 

    avg_res = (
        round(total_rec/NUM_RUNS, 6), 
        round(total_time/NUM_RUNS, 6), 
        round(total_build/NUM_RUNS, 6)
    )
    
    print(f"   [Exec] m={m} efc={efc} efs={efs} | Rec={avg_res[0]} Time={avg_res[1]}ms")
    CACHE[key] = avg_res
    return avg_res

def objective(trial: optuna.trial.Trial) -> float:
    m = trial.suggest_int("m", M_RANGE[0], M_RANGE[1])
    efc = trial.suggest_int("efc", EFC_RANGE[0], EFC_RANGE[1])
    efs = trial.suggest_int("efs", EFS_RANGE[0], EFS_RANGE[1])

    recall, time_ms, build_ms = run_test(m, efc, efs)
    
    trial.set_user_attr("avg_recall", recall)
    trial.set_user_attr("avg_time_ms", time_ms)
    trial.set_user_attr("build_time_ms", build_ms)

    # 策略：如果 Recall 不达标，根据差距给予巨大惩罚
    if recall < MIN_RECALL:
        gap = MIN_RECALL - recall
        return -LARGE_PENALTY * (1.0 + gap * 10.0)
    
    # 策略：如果 Recall 达标，目标是最小化时间
    return (SUCCESS_FACTOR / time_ms) + (recall - MIN_RECALL) * RECALL_BONUS_SCALE

def inject_exploration_trials(study: optuna.Study):
    """【核心机制】强制注入探索性试验，防止死循环在局部最优"""
    # 1. 随机跳跃
    study.enqueue_trial({
        "m": random.randint(*M_RANGE),
        "efc": random.randint(*EFC_RANGE),
        "efs": random.randint(*EFS_RANGE)
    })

    # 2. 偶尔注入高性能参数 (High Params)
    if random.random() < 0.3: 
        study.enqueue_trial({
            "m": random.randint(32, M_RANGE[1]),      
            "efc": random.randint(500, EFC_RANGE[1]), 
            "efs": random.randint(800, EFS_RANGE[1])  
        })

if __name__ == "__main__":
    load_cache_data()
    os.makedirs(LOG_DIR, exist_ok=True)

    print("="*60)
    print(f"🚀 Adaptive HNSW Optimization (Infinite Loop Mode)")
    print(f"🎯 Target Recall: >= {MIN_RECALL}")
    print(f"🔎 Search Space: M={M_RANGE}, EFC={EFC_RANGE}, EFS={EFS_RANGE}")
    print("="*60)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        n_startup_trials=20,  
        warn_independent_sampling=False
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="hnsw_adaptive_v2",
        sampler=sampler,
        load_if_exists=True 
    )

    # 初始基准点
    if (32, 512, 500) not in CACHE:
        study.enqueue_trial({"m": 32, "efc": 512, "efs": 500})

    batch_count = 0

    try:
        while True:
            batch_count += 1
            print(f"\n\n🔄 Starting Batch #{batch_count} (Size: {BATCH_SIZE})...")
            
            # 运行一个小批次
            study.optimize(objective, n_trials=BATCH_SIZE, callbacks=[csv_logger])
            
            best_trial = study.best_trial
            best_recall = best_trial.user_attrs.get('avg_recall', 0)
            
            print("-" * 40)
            print(f"📊 Batch #{batch_count} Finished.")
            
            if best_recall >= MIN_RECALL:
                print(f"✅ Current Best (Feasible): Recall={best_recall:.4f}")
            else:
                print(f"❌ Current Best (Infeasible): Recall={best_recall:.4f}")
                # 找不到就加重药量，注入最大参数试探
                study.enqueue_trial({
                    "m": M_RANGE[1], "efc": EFC_RANGE[1], "efs": EFS_RANGE[1]
                })

            # 强制扰动，防止收敛停止
            print(">>> 🎲 Injecting exploration trials...")
            inject_exploration_trials(study)
            
            print(f"💾 Data saved to {RESULT_CSV}")
            time.sleep(1) 

    except KeyboardInterrupt:
        print("\n\n🛑 Optimization stopped by user.")
    
    print("="*60)
    qualified = [t for t in study.trials if t.user_attrs.get('avg_recall', 0) >= MIN_RECALL]
    
    if qualified:
        best = max(qualified, key=lambda t: t.value)
        print("🏆 Final Optimal Configuration:")
        print(f"   Recall: {best.user_attrs['avg_recall']:.4f}")
        print(f"   Time:   {best.user_attrs['avg_time_ms']:.4f} ms")
        print(f"   Params: {best.params}")
    else:
        print("⚠️ No configuration met the strict recall requirement.")