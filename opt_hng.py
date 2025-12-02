import optuna
from optuna.samplers import CmaEsSampler
import subprocess
import re
import csv
import os
import random
import time
import threading
import shutil
from typing import Tuple, Dict, Any
from datetime import datetime

# ==================== 配置区 ====================
BIN_PATH = "./hng"
MIN_RECALL = 0.9801          # 约束条件下界
FIXED_K = 10
BATCH_SIZE = 20
NUM_RUNS = 1

# --- 初始点 ---
INITIAL_M = 40
INITIAL_MAX_LAYER = 17
INITIAL_EFC = 648
INITIAL_EFS = 457

# --- 参数搜索范围 ---
M_RANGE = (16, 64)
MAX_LAYER_RANGE = (1, 20)
EFC_RANGE = (80, 1000)
EFS_RANGE = (80, 2000)

LOG_DIR = "Log"
RESULT_CSV = "optuna_hng.csv"

# --- 差异化惩罚常量 ---
# 根据 Recall 偏离程度进行不同强度的惩罚
PENALTY_TIER_1 = 100000.0    # 极度不合格 (Recall <= 0.97)
PENALTY_TIER_2 = 50000.0     # 严重不合格 (0.97 < Recall <= 0.975)
PENALTY_TIER_3 = 10000.0     # 中等不合格 (0.975 < Recall <= 0.9801)

# --- 全局缓存 ---
CACHE: Dict[Tuple[int, int, int, int], Tuple[float, float, float]] = {}

def backup_csv():
    """备份当前 CSV，命名为 原名字+时间戳.bak"""
    if not os.path.exists(RESULT_CSV):
        return
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{RESULT_CSV.rsplit('.', 1)[0]}_{ts}.bak"
    try:
        shutil.copy2(RESULT_CSV, backup_name)
        print(f"💾 CSV backed up to: {backup_name}")
    except Exception as e:
        print(f"Warning: failed to backup CSV: {e}")

def recalculate_scores_from_csv():
    """
    从 CSV 重新加载所有数据，按当前奖惩规则重新计算 score
    """
    if not os.path.exists(RESULT_CSV):
        print("No CSV file to recalculate.")
        return
    
    print("🔄 Recalculating scores from CSV...")
    
    # 读取原 CSV
    rows = []
    try:
        with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if not rows:
        print("CSV is empty.")
        return
    
    # 重新计算 score
    recalc_count = 0
    for row in rows:
        try:
            recall = float(row.get('avg_recall', 0))
            time_ms = float(row.get('avg_time_ms', 99999))
            
            # 使用当前惩罚规则
            if recall > MIN_RECALL:
                score = time_ms  # 可行解：最小化时间
            else:
                # 差异化惩罚
                gap = MIN_RECALL - recall
                if gap >= 0.0099:  # Recall <= 0.97
                    score = PENALTY_TIER_1 * (1.0 + gap * 100.0)
                elif gap >= 0.0049:  # 0.97 < Recall <= 0.975
                    score = PENALTY_TIER_2 * (1.0 + gap * 100.0)
                else:  # 0.975 < Recall <= 0.9801
                    score = PENALTY_TIER_3 * (1.0 + gap * 100.0)
            
            row['reward_score'] = f"{score:.2f}"
            recalc_count += 1
        except Exception as e:
            print(f"Error recalculating row: {e}")
            continue
    
    # 写回 CSV（覆盖原文件）
    try:
        fieldnames = ['timestamp', 'study_name', 'trial_id', 'm', 'max_layer', 'efc', 'efs', 'K',
                      'avg_recall', 'avg_time_ms', 'build_time_ms', 'reward_score', 'status']
        with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Recalculated {recalc_count} rows in CSV")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def load_cache_data():
    """从 CSV 加载历史数据"""
    global CACHE
    if not os.path.exists(RESULT_CSV):
        return {}
    
    print(f"Loading history from {RESULT_CSV}...")
    trial_records = {}
    
    try:
        with open(RESULT_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    def p(v, t): return t(float(str(v).replace(',', '.'))) if v else None
                    m = p(row.get('m'), int)
                    ml = p(row.get('max_layer'), int) or 8
                    efc = p(row.get('efc'), int)
                    efs = p(row.get('efs'), int)
                    rec = p(row.get('avg_recall'), float)
                    time_ms = p(row.get('avg_time_ms'), float)
                    build_ms = p(row.get('build_time_ms'), float)
                    
                    if all(v is not None for v in [m, efc, efs, rec, time_ms, build_ms]):
                        CACHE[(m, ml, efc, efs)] = (rec, time_ms, build_ms)
                        
                        trial_id = int(row.get('trial_id', -1)) if row.get('trial_id') else -1
                        if trial_id >= 0:
                            trial_records[trial_id] = {
                                'params': {'m': m, 'max_layer': ml, 'efc': efc, 'efs': efs},
                                'user_attrs': {'avg_recall': rec, 'avg_time_ms': time_ms, 'build_time_ms': build_ms},
                                'state': optuna.trial.TrialState.COMPLETE
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
    now = datetime.now()

    for trial_id in sorted(trial_records.keys()):
        record = trial_records[trial_id]
        try:
            params = record.get('params') or {}
            user_attrs = record.get('user_attrs') or {}
            
            # 根据当前奖惩规则重新计算 value
            recall = user_attrs.get('avg_recall', 0)
            time_ms = user_attrs.get('avg_time_ms', 99999)
            
            if recall > MIN_RECALL:
                value = time_ms
            else:
                gap = MIN_RECALL - recall
                if gap >= 0.0099:
                    value = PENALTY_TIER_1 * (1.0 + gap * 100.0)
                elif gap >= 0.0049:
                    value = PENALTY_TIER_2 * (1.0 + gap * 100.0)
                else:
                    value = PENALTY_TIER_3 * (1.0 + gap * 100.0)

            frozen_trial = optuna.trial.FrozenTrial(
                number=trial_id,
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                datetime_start=now,
                datetime_complete=now,
                params=params,
                distributions={
                    'm': optuna.distributions.IntDistribution(*M_RANGE),
                    'max_layer': optuna.distributions.IntDistribution(*MAX_LAYER_RANGE),
                    'efc': optuna.distributions.IntDistribution(*EFC_RANGE),
                    'efs': optuna.distributions.IntDistribution(*EFS_RANGE),
                },
                user_attrs=user_attrs,
                system_attrs={},
                intermediate_values={},
                trial_id=trial_id,
            )
            trials_to_add.append(frozen_trial)
        except Exception as e:
            print(f"Warning: skipping trial {trial_id}: {e}")
            continue

    if not trials_to_add:
        return

    try:
        study.add_trials(trials_to_add)
    except Exception as e:
        print(f"Warning: batch add_trials failed: {e}")
        for tr in trials_to_add:
            try:
                study.add_trial(tr)
            except:
                pass

def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """写入 CSV 日志"""
    if not trial.state.is_finished():
        return

    data = {
        'timestamp': datetime.now().isoformat(),
        'study_name': getattr(study, "study_name", None) or getattr(study, "name", ""),
        'trial_id': trial.number,
        'm': trial.params.get('m'),
        'max_layer': trial.params.get('max_layer'),
        'efc': trial.params.get('efc'),
        'efs': trial.params.get('efs'),
        'K': FIXED_K,
        'avg_recall': trial.user_attrs.get('avg_recall'),
        'avg_time_ms': trial.user_attrs.get('avg_time_ms'),
        'build_time_ms': trial.user_attrs.get('build_time_ms'),
        'reward_score': trial.value if trial.value is not None else -float('inf'),
        'status': trial.state.name,
    }

    fieldnames = ['timestamp', 'study_name', 'trial_id', 'm', 'max_layer', 'efc', 'efs', 'K',
                  'avg_recall', 'avg_time_ms', 'build_time_ms', 'reward_score', 'status']

    try:
        file_exists = os.path.exists(RESULT_CSV)
        with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    except IOError as e:
        print(f"Error writing to CSV: {e}")

def run_test(m: int, max_layer: int, efc: int, efs: int) -> Tuple[float, float, float]:
    """运行测试"""
    key = (m, max_layer, efc, efs)
    if key in CACHE:
        print(f"   [CACHE HIT] m={m}, ml={max_layer}, efc={efc}, efs={efs}")
        return CACHE[key]

    os.makedirs(LOG_DIR, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_logfile = os.path.join(LOG_DIR, f"trial_m{m}_ml{max_layer}_efc{efc}_efs{efs}_{ts}.log")
    
    try:
        lf = open(trial_logfile, 'w', encoding='utf-8', buffering=1)
    except IOError as e:
        print(f"   [ERROR] Cannot create log file: {e}")
        return 0.0, 99999.0, 99999.0
    
    def log_write(msg: str):
        print(msg, end='', flush=True)
        lf.write(msg)
        lf.flush()
    
    cmd = [BIN_PATH, "--m", str(m), "--max_layer", str(max_layer), "--efc", str(efc), "--efs", str(efs)]
    
    log_write(f"   [START] m={m}, ml={max_layer}, efc={efc}, efs={efs}\n")
    
    total_rec, total_time, total_build = 0.0, 0.0, 0.0
    
    for run_num in range(NUM_RUNS):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            stdout_lines = []
            stderr_lines = []
            
            def read_stream(stream, lines_list, prefix):
                try:
                    for line in iter(stream.readline, ''):
                        if line:
                            log_write(f"{prefix}{line}")
                            lines_list.append(line)
                except:
                    pass
            
            stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "[OUT] "))
            stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "[ERR] "))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            start_wait = time.time()
            timeout = 2000
            while proc.poll() is None:
                if time.time() - start_wait > timeout:
                    log_write(f"\n   [TIMEOUT] Exceeded {timeout}s\n")
                    proc.kill()
                    lf.close()
                    return 0.0, 99999.0, 99999.0
                time.sleep(0.1)
            
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            if proc.returncode != 0:
                log_write(f"   [ERROR] Process returned {proc.returncode}\n")
                lf.close()
                return 0.0, 99999.0, 99999.0
            
            txt = ''.join(stdout_lines)
            
            r_m = re.search(r"Average\s+recall@\d+[:\s]+([0-9]+(?:[.,][0-9]+)?)", txt, re.I)
            t_m = re.search(r"Average\s+query\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)
            b_m = re.search(r"Index\s+build\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)

            if b_m and t_m and r_m:
                build_val = float(b_m.group(1).replace(',', '.'))
                time_val = float(t_m.group(1).replace(',', '.'))
                rec_val = float(r_m.group(1).replace(',', '.'))
                
                total_build += build_val
                total_time += time_val
                total_rec += rec_val
                
                log_write(f"   [PARSED] Rec={rec_val:.4f}, Time={time_val:.4f}ms, Build={build_val:.2f}ms\n")
            else:
                log_write(f"   [PARSE ERROR] Regex match failed\n")
                lf.close()
                return 0.0, 99999.0, 99999.0 
        except Exception as e:
            log_write(f"   [EXCEPTION] {type(e).__name__}: {e}\n")
            lf.close()
            return 0.0, 99999.0, 99999.0 

    avg_res = (round(total_rec/NUM_RUNS, 6), round(total_time/NUM_RUNS, 6), round(total_build/NUM_RUNS, 6))
    
    log_write(f"   [RESULT] Rec={avg_res[0]:.4f} Time={avg_res[1]:.4f}ms\n")
    lf.close()
    
    CACHE[key] = avg_res
    return avg_res

def objective(trial: optuna.trial.Trial) -> float:
    """目标函数 - 差异化惩罚"""
    m = trial.suggest_int("m", M_RANGE[0], M_RANGE[1])
    max_layer = trial.suggest_int("max_layer", MAX_LAYER_RANGE[0], MAX_LAYER_RANGE[1])
    efc = trial.suggest_int("efc", EFC_RANGE[0], EFC_RANGE[1])
    efs = trial.suggest_int("efs", EFS_RANGE[0], EFS_RANGE[1])

    recall, time_ms, build_ms = run_test(m, max_layer, efc, efs)
    
    trial.set_user_attr("avg_recall", recall)
    trial.set_user_attr("avg_time_ms", time_ms)
    trial.set_user_attr("build_time_ms", build_ms)

    # 差异化惩罚
    if recall > MIN_RECALL:
        score = time_ms  # 可行解
    else:
        gap = MIN_RECALL - recall
        if gap >= 0.0099:  # Recall <= 0.97
            score = PENALTY_TIER_1 * (1.0 + gap * 100.0)
        elif gap >= 0.0049:  # 0.97 < Recall <= 0.975
            score = PENALTY_TIER_2 * (1.0 + gap * 100.0)
        else:  # 0.975 < Recall <= 0.9801
            score = PENALTY_TIER_3 * (1.0 + gap * 100.0)
    
    print(f"   => Recall={recall:.6f}, Time={time_ms:.4f}ms, Gap={MIN_RECALL-recall:.6f}, Score={score:.2f}")
    
    if recall > MIN_RECALL:
        print(f"   ✅ FEASIBLE")
    else:
        print(f"   ⚠️  INFEASIBLE (Heavy Penalty)")
    
    return score

READ_FROM_CSV_ONLY = True

if __name__ == "__main__":
    # 1. 备份 CSV
    backup_csv()
    
    # 2. 加载历史数据
    if READ_FROM_CSV_ONLY:
        print("Loading entries from CSV only...")
        trial_records = load_cache_data()
        print(f"Pre-loaded {len(CACHE)} entries from CSV.")
    else:
        trial_records = load_cache_data()

    # 3. 重新计算所有分数
    recalculate_scores_from_csv()

    os.makedirs(LOG_DIR, exist_ok=True)

    print("="*60)
    print(f"🚀 HNSW Optimization with CMA-ES + Tiered Penalties")
    print(f"🎯 Constraint: Recall > {MIN_RECALL}")
    print(f"📊 Tiered Penalties:")
    print(f"   Tier 1 (Recall <= 0.97):      {PENALTY_TIER_1:.0f}")
    print(f"   Tier 2 (0.97 < Recall <= 0.975): {PENALTY_TIER_2:.0f}")
    print(f"   Tier 3 (0.975 < Recall <= {MIN_RECALL}): {PENALTY_TIER_3:.0f}")
    print(f"🔧 Initial: M={INITIAL_M}, ML={INITIAL_MAX_LAYER}, EFC={INITIAL_EFC}, EFS={INITIAL_EFS}")
    print("="*60)

    sampler = CmaEsSampler(seed=42)
    
    study = optuna.create_study(
        direction="minimize",
        study_name="hnsw_cmaes_v5",
        sampler=sampler
    )

    # 恢复历史 trials
    restore_trials_to_study(study, trial_records)

    # 注入初始点
    if len(study.trials) == 0:
        print("Enqueuing initial point...")
        study.enqueue_trial({"m": INITIAL_M, "max_layer": INITIAL_MAX_LAYER, "efc": INITIAL_EFC, "efs": INITIAL_EFS})

    batch_count = 0

    try:
        while True:
            batch_count += 1
            print(f"\n🔄 Batch #{batch_count}")
            print(f"   Trials so far: {len(study.trials)}")
            
            study.optimize(objective, n_trials=BATCH_SIZE, callbacks=[csv_logger])
            
            # 分析
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            feasible = [t for t in completed if t.user_attrs.get('avg_recall', 0) > MIN_RECALL 
                       and (t.value is not None and t.value != float('inf'))]
            
            print(f"   Completed: {len(completed)}, Feasible: {len(feasible)}")
            
            if feasible:
                best = min(feasible, key=lambda t: t.value)
                print(f"✅ Best: Recall={best.user_attrs['avg_recall']:.6f}, Time={best.user_attrs['avg_time_ms']:.4f}ms")
            
            print(f"💾 Saved to {RESULT_CSV}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("="*60)
    feasible = [t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE 
                and t.user_attrs.get('avg_recall', 0) > MIN_RECALL
                and (t.value is not None and t.value != float('inf'))]
    
    if feasible:
        best = min(feasible, key=lambda t: t.value)
        print("🏆 Final Winner:")
        print(f"   Recall: {best.user_attrs['avg_recall']:.6f}")
        print(f"   Time:   {best.user_attrs['avg_time_ms']:.4f} ms")
        print(f"   Params: {best.params}")
    else:
        print("⚠️ No feasible solutions found.")
    
    print(f"Total trials: {len(study.trials)}")