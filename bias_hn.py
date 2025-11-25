import optuna
import subprocess
import re
import csv
import os
import random
import time
import threading
from typing import Tuple, Dict, Any
from datetime import datetime

# ==================== 配置区 ====================
BIN_PATH = "./hng"
MIN_RECALL = 0.985          # 目标召回率 (98.5%)
FIXED_K = 10                # Top-K
BATCH_SIZE = 20             # 每批运行多少次 trial

# --- [修复] 补全缺失配置 ---
NUM_RUNS = 1                # 单次参数取平均值的运行次数 (1次最快，3次更稳)

# --- [优化] 合理的参数搜索范围 ---
# M: 16-64 是 HNSW 的黄金区间，太小召回低，太大构建慢且内存大
M_RANGE = (16, 64)          
# EFC: 影响索引质量，通常 200-800 足够，极端情况才需要 1000+
EFC_RANGE = (200, 1000)     
# EFS: 决定搜索时的精细度，为了 0.985 召回，上限设高一点
EFS_RANGE = (100, 2000)     

LOG_DIR = "Log"
RESULT_CSV = "optuna_adaptive_hng.csv"

# --- [优化] 奖励/惩罚常量 (避免数值溢出) ---
# 目标是：Recall不达标 -> 负分；Recall达标 -> 正分 (分数越高代表时间越短)
SUCCESS_BASE = 10000.0      # 基础奖励
RECALL_BONUS = 1000.0       # 召回溢出奖励权重
PENALTY_BASE = 5000.0       # 基础惩罚 (不用 1e9，防止 Optuna 数值不稳定)

# --- 全局缓存 ---
CACHE: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

def load_cache_data():
    """从 CSV 加载历史数据和 trial 信息"""
    global CACHE
    if not os.path.exists(RESULT_CSV):
        return {}
    
    print(f"Loading history from {RESULT_CSV}...")
    trial_records = {}  # trial_id -> (params, user_attrs, value, state)
    
    try:
        with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    status = (row.get('status') or '').strip().upper()
                    
                    def p(v, t): return t(float(str(v).replace(',', '.'))) if v else None
                    m = p(row.get('m'), int)
                    efc = p(row.get('efc'), int)
                    efs = p(row.get('efs'), int)
                    rec = p(row.get('avg_recall'), float)
                    time_ms = p(row.get('avg_time_ms'), float)
                    build_ms = p(row.get('build_time_ms'), float)
                    reward = p(row.get('reward_score'), float)
                    
                    if all(v is not None for v in [m, efc, efs, rec, time_ms, build_ms]):
                        # 加入缓存
                        CACHE[(m, efc, efs)] = (rec, time_ms, build_ms)
                        
                        # 记录 trial 信息
                        trial_id = int(row.get('trial_id', -1)) if row.get('trial_id') else -1
                        if trial_id >= 0:
                            trial_records[trial_id] = {
                                'params': {'m': m, 'efc': efc, 'efs': efs},
                                'user_attrs': {
                                    'avg_recall': rec,
                                    'avg_time_ms': time_ms,
                                    'build_time_ms': build_ms
                                },
                                'value': reward,
                                'state': optuna.trial.TrialState.COMPLETE if status == 'COMPLETE' else optuna.trial.TrialState.FAIL
                            }
                except:
                    continue
    except Exception as e:
        print(f"Cache load warning: {e}")
    
    print(f"Loaded {len(CACHE)} unique cached results, {len(trial_records)} trial records.")
    return trial_records

def restore_trials_to_study(study: optuna.Study, trial_records: Dict):
    """从历史记录恢复 trials 到 study"""
    if not trial_records:
        return
    
    print(f"Restoring {len(trial_records)} trials from history...")
    
    trials_to_add = []
    # 按 trial_id 排序
    for trial_id in sorted(trial_records.keys()):
        record = trial_records[trial_id]
        
        # 为 COMPLETE 状态的 trial 设置时间戳
        now = datetime.now()
        datetime_start = now if record['state'] == optuna.trial.TrialState.COMPLETE else None
        datetime_complete = now if record['state'] == optuna.trial.TrialState.COMPLETE else None
        
        # 构造 FrozenTrial
        frozen_trial = optuna.trial.FrozenTrial(
            number=trial_id,
            state=record['state'],
            value=record['value'] if record['state'] == optuna.trial.TrialState.COMPLETE else None,
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            params=record['params'],
            distributions={
                'm': optuna.distributions.IntDistribution(*M_RANGE),
                'efc': optuna.distributions.IntDistribution(*EFC_RANGE),
                'efs': optuna.distributions.IntDistribution(*EFS_RANGE),
            },
            user_attrs=record['user_attrs'],
            system_attrs={},
            intermediate_values={},
            trial_id=trial_id,
        )
        trials_to_add.append(frozen_trial)
    
    # 批量添加
    study.add_trials(trials_to_add)

def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """写入 CSV 日志"""
    if not trial.state.is_finished():
        return

    data = {
        'timestamp': datetime.now().isoformat(),
        'study_name': getattr(study, "study_name", None) or getattr(study, "name", ""),
        'trial_id': trial.number,
        'm': trial.params.get('m'),
        'efc': trial.params.get('efc'),
        'efs': trial.params.get('efs'),
        'K': FIXED_K,
        'avg_recall': trial.user_attrs.get('avg_recall'),
        'avg_time_ms': trial.user_attrs.get('avg_time_ms'),
        'build_time_ms': trial.user_attrs.get('build_time_ms'),
        'reward_score': trial.value if trial.value is not None else -float('inf'),
        'status': trial.state.name,
    }

    fieldnames = ['timestamp', 'study_name', 'trial_id', 'm', 'efc', 'efs', 'K',
                  'avg_recall', 'avg_time_ms', 'build_time_ms', 'reward_score', 'status']

    file_exists = os.path.exists(RESULT_CSV)
    try:
        with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    except IOError as e:
        print(f"Error writing to CSV: {e}")

    # 另外写一份按 study 名称的日志，方便按 study 查找
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        study_csv = os.path.join(LOG_DIR, f"{data['study_name']}.csv")
        study_file_exists = os.path.exists(study_csv)
        with open(study_csv, 'a', newline='', encoding='utf-8') as sf:
            swriter = csv.DictWriter(sf, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
            if not study_file_exists:
                swriter.writeheader()
            swriter.writerow(data)
    except IOError as e:
        print(f"Warning: cannot write study-specific CSV {study_csv}: {e}")

def run_test(m: int, efc: int, efs: int) -> Tuple[float, float, float]:
    key = (m, efc, efs)
    if key in CACHE:
        print(f"   [CACHE HIT] m={m}, efc={efc}, efs={efs}")
        return CACHE[key]

    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 为本次测试创建日志文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_logfile = os.path.join(LOG_DIR, f"trial_m{m}_efc{efc}_efs{efs}_{ts}.log")
    
    # 立即创建并打开日志文件用于流式写入
    try:
        lf = open(trial_logfile, 'w', encoding='utf-8', buffering=1)
    except IOError as e:
        print(f"   [ERROR] Cannot create log file {trial_logfile}: {e}")
        return 0.0, 99999.0, 99999.0
    
    def log_write(msg: str):
        """同时写入终端和日志文件"""
        print(msg, end='', flush=True)
        lf.write(msg)
        lf.flush()
    
    # --- [关键] 使用双短横线 -- 以匹配 C++ 的 parse_args_g ---
    cmd = [BIN_PATH, "--m", str(m), "--efc", str(efc), "--efs", str(efs)]
    
    log_write(f"   [START] m={m}, efc={efc}, efs={efs}\n")
    log_write(f"   [LOG] {trial_logfile}\n")
    log_write(f"   [RUN 1/{NUM_RUNS}] Executing: {' '.join(cmd)}\n")
    
    total_rec, total_time, total_build = 0.0, 0.0, 0.0
    
    for run_num in range(NUM_RUNS):
        try:
            log_write(f"\n{'='*60}\n")
            log_write(f"Trial: m={m}, efc={efc}, efs={efs}, Run {run_num+1}/{NUM_RUNS}\n")
            log_write(f"Timestamp: {datetime.now().isoformat()}\n")
            log_write(f"Command: {' '.join(cmd)}\n")
            log_write(f"{'='*60}\n\n")
            
            # 使用 Popen 获得实时流
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # 行缓冲
            )
            
            # 实时读取 stdout
            stdout_lines = []
            stderr_lines = []
            
            def read_stream(stream, lines_list, prefix):
                """后台线程读取流并实时写入"""
                try:
                    for line in iter(stream.readline, ''):
                        if line:
                            log_write(f"{prefix}{line}")
                            lines_list.append(line)
                except:
                    pass
            
            # 启动两个后台线程分别读取 stdout 和 stderr
            stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "[OUT] "))
            stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "[ERR] "))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # 等待进程完成（最多 2000 秒）
            start_wait = time.time()
            timeout = 2000
            while proc.poll() is None:
                if time.time() - start_wait > timeout:
                    log_write(f"\n   [TIMEOUT] Process exceeded {timeout} seconds\n")
                    proc.kill()
                    lf.close()
                    return 0.0, 99999.0, 99999.0
                time.sleep(0.1)
            
            # 等待读取线程完成
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            returncode = proc.returncode
            log_write(f"\nReturn Code: {returncode}\n")
            
            if returncode != 0:
                log_write(f"   [ERROR] Process returned {returncode}\n")
                lf.close()
                return 0.0, 99999.0, 99999.0
            
            # 合并 stdout
            txt = ''.join(stdout_lines)
            
            # --- 解析逻辑 (修复正则表达式) ---
            # 格式示例：
            # [OUT] Average recall@10: 0.990500
            # [OUT] Average query time: 9.293846 ms
            # [OUT] Index build time: 1011632.028744 ms
            
            # 1. 解析 recall
            r_m = re.search(r"Average\s+recall@\d+[:\s]+([0-9]+(?:[.,][0-9]+)?)", txt, re.I)
            
            # 2. 解析 query 时间
            t_m = re.search(r"Average\s+query\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)
            
            # 3. 解析 build 时间
            b_m = re.search(r"Index\s+build\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)

            if b_m and t_m and r_m:
                build_val = float(b_m.group(1).replace(',', '.'))
                time_val = float(t_m.group(1).replace(',', '.'))
                rec_val = float(r_m.group(1).replace(',', '.'))
                
                total_build += build_val
                total_time += time_val
                total_rec += rec_val
                
                log_write(f"   [PARSED] Build={build_val:.2f}ms, Query={time_val:.4f}ms, Recall={rec_val:.4f}\n")
            else:
                # 调试输出：显示匹配失败的具体原因
                log_write(f"   [PARSE ERROR] Regex match failed:\n")
                log_write(f"      Build match: {b_m is not None}\n")
                log_write(f"      Time match:  {t_m is not None}\n")
                log_write(f"      Recall match: {r_m is not None}\n")
                log_write(f"      Output preview (last 500 chars):\n{txt[-500:]}\n")
                lf.close()
                return 0.0, 99999.0, 99999.0 
        except Exception as e:
            log_write(f"   [EXCEPTION] {type(e).__name__}: {e}\n")
            lf.close()
            return 0.0, 99999.0, 99999.0 

    avg_res = (
        round(total_rec/NUM_RUNS, 6), 
        round(total_time/NUM_RUNS, 6), 
        round(total_build/NUM_RUNS, 6)
    )
    
    log_write(f"   [RESULT] m={m:<2} efc={efc:<4} efs={efs:<4} | Rec={avg_res[0]:.4f} Time={avg_res[1]:.4f}ms Build={avg_res[2]:.2f}ms\n")
    log_write(f"   [SUCCESS] Logged to {trial_logfile}\n")
    
    lf.close()
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

    # --- 奖励函数优化 ---
    # 场景1：召回率未达标 (Recall < Target)
    if recall < MIN_RECALL:
        gap = MIN_RECALL - recall
        # 惩罚分 = 基础惩罚 + 差距惩罚
        # 结果范围大概在 -5000 到 -15000 之间
        return -PENALTY_BASE * (1.0 + gap * 10.0)
    
    # 场景2：召回率达标 (Recall >= Target)
    # 奖励分 = (基础分 / 时间) + 召回奖励
    # 假设时间是 0.5ms -> 10000/0.5 = 20000 分
    # 假设时间是 1.0ms -> 10000/1.0 = 10000 分 (越快分越高)
    score = (SUCCESS_BASE / max(time_ms, 0.001)) 
    
    # 额外加上召回奖励，鼓励 0.99 > 0.985
    score += (recall - MIN_RECALL) * RECALL_BONUS
    return score

def inject_exploration_trials(study: optuna.Study):
    """注入探索点，防止陷入局部最优"""
    # 1. 随机点
    study.enqueue_trial({
        "m": random.randint(*M_RANGE),
        "efc": random.randint(*EFC_RANGE),
        "efs": random.randint(*EFS_RANGE)
    })
    
    # 2. 高潜力点 (大参数)
    if random.random() < 0.4:
        study.enqueue_trial({
            "m": random.randint(32, M_RANGE[1]),
            "efc": random.randint(500, EFC_RANGE[1]),
            "efs": random.randint(800, EFS_RANGE[1])
        })

if __name__ == "__main__":
    trial_records = load_cache_data()
    os.makedirs(LOG_DIR, exist_ok=True)

    print("="*60)
    print(f"🚀 Adaptive HNSW Optimization (CSV Storage)")
    print(f"🎯 Target Recall: >= {MIN_RECALL}")
    print(f"🔎 Search Space: M={M_RANGE}, EFC={EFC_RANGE}, EFS={EFS_RANGE}")
    print("="*60)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        n_startup_trials=10, 
        warn_independent_sampling=False
    )
    
    # 创建内存 study（不用数据库）
    study = optuna.create_study(
        direction="maximize",
        study_name="hnsw_adaptive_v3",
        sampler=sampler
    )

    # 恢复历史 trials
    restore_trials_to_study(study, trial_records)

    # --- 仅在新 study 时注入初始探测点 ---
    if len(study.trials) == 0:
        print("Enqueuing initial trials...")
        initial_trials = [
            {"m": 32, "efc": 300, "efs": 300},  # 中等配置
            {"m": 48, "efc": 500, "efs": 800}   # 高配配置
        ]
        
        for t in initial_trials:
            if (t['m'], t['efc'], t['efs']) not in CACHE:
                study.enqueue_trial(t)

    batch_count = 0

    try:
        while True:
            batch_count += 1
            print(f"\n🔄 Batch #{batch_count} ...")
            print(f"   Current trials count: {len(study.trials)}")
            
            study.optimize(objective, n_trials=BATCH_SIZE, callbacks=[csv_logger])
            
            # --- 批次分析 ---
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            best_trial = study.best_trial
            best_recall = best_trial.user_attrs.get('avg_recall', 0) if best_trial.state == optuna.trial.TrialState.COMPLETE else 0
            
            print(f"   Completed trials: {len(completed_trials)}")
            
            if best_recall >= MIN_RECALL:
                print(f"✅ Feasible Best: Recall={best_recall:.4f}, Time={best_trial.user_attrs['avg_time_ms']:.4f}ms")
            else:
                print(f"❌ Infeasible Best: Recall={best_recall:.4f} (Target {MIN_RECALL})")
                # 如果依然不达标，强制注入极大参数
                study.enqueue_trial({"m": 48, "efc": 800, "efs": 1500})

            # 注入扰动
            inject_exploration_trials(study)
            print(f"💾 Saved to {RESULT_CSV}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("="*60)
    # 确保所有完成的 trial 都满足条件
    qualified = [t for t in study.trials 
                 if t.state == optuna.trial.TrialState.COMPLETE 
                 and t.user_attrs.get('avg_recall', 0) >= MIN_RECALL]
    
    if qualified:
        best = max(qualified, key=lambda t: t.value if t.value is not None else -float('inf'))
        print("🏆 Final Winner:")
        print(f"   Recall: {best.user_attrs['avg_recall']:.4f}")
        print(f"   Time:   {best.user_attrs['avg_time_ms']:.4f} ms")
        print(f"   Build:  {best.user_attrs.get('build_time_ms', 'N/A')} ms")
        print(f"   Params: {best.params}")
    else:
        print("⚠️ No config met requirements.")
    
    print(f"Total trials: {len(study.trials)}")
    print(f"CSV Log: {RESULT_CSV}")