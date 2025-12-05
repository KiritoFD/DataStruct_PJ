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
from typing import Tuple, Dict, Any, List
from datetime import datetime
import sys

# ==================== 配置区 ====================
BIN_PATH = "./hng1"
MIN_RECALL = 0.98         # 约束条件下界
TARGET_RECALL = 0.99         # EFS 二分搜索目标召回率
FIXED_K = 10
BATCH_SIZE = 20
NUM_RUNS = 1

# --- 初始点 (只有前三个参数) ---
INITIAL_M = 59
INITIAL_MAX_LAYER = 5
INITIAL_EFC = 763

# --- 参数搜索范围 ---
M_RANGE = (16, 64)
MAX_LAYER_RANGE = (0, 20)
EFC_RANGE = (80, 1000)
EFS_RANGE = (80, 2000)  # EFS 二分搜索范围

LOG_DIR = "Log"
RESULT_CSV = "optu_hng.csv"

# --- 差异化惩罚常量 ---
# 不可行解的惩罚：使其分数远大于任何可行解的时间
INFEASIBLE_PENALTY_BASE = 1000000.0  # 基础惩罚，确保不可行解永远比可行解差

# --- 全局缓存 ---
# 完整缓存: (m, ml, efc, efs) -> (recall, time_ms, build_ms)
CACHE: Dict[Tuple[int, int, int, int], Tuple[float, float, float]] = {}
# 图参数缓存: (m, ml, efc) -> (best_efs, recall, time_ms) - 存储二分搜索结果
GRAPH_CACHE: Dict[Tuple[int, int, int], Tuple[int, float, float]] = {}

def backup_csv():
    """备份当前 CSV"""
    if not os.path.exists(RESULT_CSV):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{RESULT_CSV.rsplit('.', 1)[0]}_{ts}.bak"
    try:
        shutil.copy2(RESULT_CSV, backup_name)
        print(f"💾 CSV backed up to: {backup_name}")
    except Exception as e:
        print(f"Warning: failed to backup CSV: {e}")

def load_cache_data():
    """从 CSV 加载历史数据"""
    global CACHE, GRAPH_CACHE
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
                                'params': {'m': m, 'max_layer': ml, 'efc': efc},
                                'user_attrs': {'avg_recall': rec, 'avg_time_ms': time_ms, 'build_time_ms': build_ms, 'opt_efs': efs},
                                'state': optuna.trial.TrialState.COMPLETE
                            }
                except:
                    continue
    except Exception as e:
        print(f"Cache load warning: {e}")
    
    # 构建 GRAPH_CACHE: 对每个 (m, ml, efc) 找到满足 TARGET_RECALL 的最小 efs
    for (m, ml, efc, efs), (rec, time_ms, build_ms) in CACHE.items():
        if rec >= TARGET_RECALL:
            key = (m, ml, efc)
            if key not in GRAPH_CACHE or efs < GRAPH_CACHE[key][0]:
                GRAPH_CACHE[key] = (efs, rec, time_ms)
    
    print(f"Loaded {len(CACHE)} cached results, {len(GRAPH_CACHE)} graph configs, {len(trial_records)} trials.")
    return trial_records

def run_test(m: int, max_layer: int, efc: int, efs: int, silent: bool = False) -> Tuple[float, float, float]:
    """运行测试"""
    key = (m, max_layer, efc, efs)
    if key in CACHE:
        if not silent:
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
        if not silent:
            print(msg, end='', flush=True)
        lf.write(msg)
        lf.flush()
    
    cmd = [BIN_PATH, "--m", str(m), "--max_layer", str(max_layer), "--efc", str(efc), "--efs", str(efs)]
    log_write(f"   [START] m={m}, ml={max_layer}, efc={efc}, efs={efs}\n")
    
    total_rec, total_time, total_build = 0.0, 0.0, 0.0
    
    for run_num in range(NUM_RUNS):
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            stdout_lines, stderr_lines = [], []
            
            def read_stream(stream, lines_list, prefix):
                try:
                    for line in iter(stream.readline, ''):
                        if line:
                            log_write(f"{prefix}{line}")
                            lines_list.append(line)
                except: pass
            
            stdout_thread = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "[OUT] "))
            stderr_thread = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "[ERR] "))
            stdout_thread.daemon = stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            start_wait = time.time()
            while proc.poll() is None:
                if time.time() - start_wait > 2000:
                    log_write(f"\n   [TIMEOUT]\n")
                    proc.kill()
                    lf.close()
                    return 0.0, 99999.0, 99999.0
                time.sleep(0.1)
            
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            if proc.returncode != 0:
                lf.close()
                return 0.0, 99999.0, 99999.0
            
            txt = ''.join(stdout_lines)
            r_m = re.search(r"Average\s+recall@\d+[:\s]+([0-9]+(?:[.,][0-9]+)?)", txt, re.I)
            t_m = re.search(r"Average\s+query\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)
            b_m = re.search(r"Index\s+build\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms", txt, re.I)

            if b_m and t_m and r_m:
                total_build += float(b_m.group(1).replace(',', '.'))
                total_time += float(t_m.group(1).replace(',', '.'))
                total_rec += float(r_m.group(1).replace(',', '.'))
            else:
                lf.close()
                return 0.0, 99999.0, 99999.0 
        except Exception as e:
            lf.close()
            return 0.0, 99999.0, 99999.0 

    avg_res = (round(total_rec/NUM_RUNS, 6), round(total_time/NUM_RUNS, 6), round(total_build/NUM_RUNS, 6))
    log_write(f"   [RESULT] Rec={avg_res[0]:.4f} Time={avg_res[1]:.4f}ms\n")
    lf.close()
    
    CACHE[key] = avg_res
    return avg_res

def save_result_to_csv(m: int, ml: int, efc: int, efs: int, recall: float, time_ms: float, build_ms: float, study_name: str = 'efs_search'):
    """保存单条结果到 CSV"""
    # 与 objective 函数一致的评分逻辑
    if recall > MIN_RECALL:
        score = time_ms
    else:
        gap = MIN_RECALL - recall
        score = INFEASIBLE_PENALTY_BASE + gap * 10000.0
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'study_name': study_name,
        'trial_id': -1,
        'm': m, 'max_layer': ml, 'efc': efc, 'efs': efs,
        'K': FIXED_K,
        'avg_recall': recall, 'avg_time_ms': time_ms, 'build_time_ms': build_ms,
        'reward_score': score, 'status': 'COMPLETE',
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

def binary_search_optimal_efs(m: int, max_layer: int, efc: int, target_recall: float = TARGET_RECALL) -> Tuple[int, float, float]:
    """
    二分搜索找到满足 target_recall 的 EFS 范围，
    然后在所有满足召回率的 EFS 中，返回查询时间最短的那个。
    返回: (optimal_efs, recall, time_ms)
    """
    graph_key = (m, max_layer, efc)
    
    # 检查图缓存
    if graph_key in GRAPH_CACHE:
        cached_efs, cached_recall, cached_time = GRAPH_CACHE[graph_key]
        print(f"   [GRAPH CACHE] M={m}, ML={max_layer}, EFC={efc} -> EFS={cached_efs}, Rec={cached_recall:.4f}, Time={cached_time:.4f}ms")
        return cached_efs, cached_recall, cached_time
    
    print(f"   🔍 Binary search EFS for M={m}, ML={max_layer}, EFC={efc}, target={target_recall}")
    
    lo, hi = EFS_RANGE[0], EFS_RANGE[1]
    
    # 收集所有满足召回率的结果: [(efs, recall, time_ms), ...]
    feasible_results = []
    
    # 先测试最大 EFS，确认图能达到目标召回率
    recall, time_ms, build_ms = run_test(m, max_layer, efc, hi, silent=True)
    save_result_to_csv(m, max_layer, efc, hi, recall, time_ms, build_ms)
    
    if recall >= target_recall:
        feasible_results.append((hi, recall, time_ms))
    
    if recall < target_recall:
        print(f"   ❌ Max EFS={hi} only gets Recall={recall:.4f}, graph cannot meet target")
        return hi, recall, time_ms
    
    # 二分搜索找到临界点
    while lo < hi:
        mid = (lo + hi) // 2
        
        recall, time_ms, build_ms = run_test(m, max_layer, efc, mid, silent=True)
        save_result_to_csv(m, max_layer, efc, mid, recall, time_ms, build_ms)
        
        print(f"      EFS={mid} -> Recall={recall:.4f}, Time={time_ms:.4f}ms")
        
        if recall >= target_recall:
            feasible_results.append((mid, recall, time_ms))
            hi = mid
        else:
            lo = mid + 1
    
    # 从缓存中补充：查找该图配置下所有满足召回率的测试结果
    for (cm, cml, cefc, cefs), (crec, ctime, cbuild) in CACHE.items():
        if cm == m and cml == max_layer and cefc == efc and crec >= target_recall:
            # 检查是否已在 feasible_results 中
            if not any(f[0] == cefs for f in feasible_results):
                feasible_results.append((cefs, crec, ctime))
    
    if not feasible_results:
        # 不应该到这里，但保险起见
        print(f"   ❌ No feasible EFS found")
        return hi, 0.0, 99999.0
    
    # 在所有满足召回率的结果中，找查询时间最短的
    best = min(feasible_results, key=lambda x: x[2])  # 按 time_ms 排序
    best_efs, best_recall, best_time = best
    
    # 更新图缓存
    GRAPH_CACHE[graph_key] = (best_efs, best_recall, best_time)
    
    print(f"   ✅ Best feasible: EFS={best_efs} -> Recall={best_recall:.4f}, Time={best_time:.4f}ms")
    print(f"      (Found {len(feasible_results)} feasible EFS values)")
    
    return best_efs, best_recall, best_time

def restore_trials_to_study(study: optuna.Study, trial_records: Dict):
    """从历史记录恢复 trials 到 study (只恢复前三个参数)"""
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
            
            recall = user_attrs.get('avg_recall', 0)
            time_ms = user_attrs.get('avg_time_ms', 99999)
            
            # 与 objective 函数一致的评分逻辑
            if recall > MIN_RECALL:
                value = time_ms
            else:
                gap = MIN_RECALL - recall
                value = INFEASIBLE_PENALTY_BASE + gap * 10000.0

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
                },
                user_attrs=user_attrs,
                system_attrs={},
                intermediate_values={},
                trial_id=trial_id,
            )
            trials_to_add.append(frozen_trial)
        except Exception as e:
            continue

    if trials_to_add:
        try:
            study.add_trials(trials_to_add)
        except:
            for tr in trials_to_add:
                try: study.add_trial(tr)
                except: pass

def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """写入 CSV 日志"""
    if not trial.state.is_finished():
        return

    data = {
        'timestamp': datetime.now().isoformat(),
        'study_name': getattr(study, "study_name", "") or getattr(study, "name", ""),
        'trial_id': trial.number,
        'm': trial.params.get('m'),
        'max_layer': trial.params.get('max_layer'),
        'efc': trial.params.get('efc'),
        'efs': trial.user_attrs.get('opt_efs', -1),
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

def objective(trial: optuna.trial.Trial) -> float:
    """
    目标函数 - 只优化前三个参数 (M, max_layer, efc)
    EFS 通过二分搜索自动找到最小满足召回率的值
    
    目标：在召回率 > MIN_RECALL 的可行解中，最小化查询时间
    - 可行解 (recall > 0.99): 返回 time_ms
    - 不可行解 (recall <= 0.99): 返回大惩罚值，确保永远不会被选中
    """
    m = trial.suggest_int("m", M_RANGE[0], M_RANGE[1])
    max_layer = trial.suggest_int("max_layer", MAX_LAYER_RANGE[0], MAX_LAYER_RANGE[1])
    efc = trial.suggest_int("efc", EFC_RANGE[0], EFC_RANGE[1])

    print(f"\n📊 Trial #{trial.number}: M={m}, ML={max_layer}, EFC={efc}")
    
    # 二分搜索找到最优 EFS
    opt_efs, recall, time_ms = binary_search_optimal_efs(m, max_layer, efc, TARGET_RECALL)
    
    # 获取 build_ms (从缓存)
    build_ms = CACHE.get((m, max_layer, efc, opt_efs), (0, 0, 99999))[2]
    
    trial.set_user_attr("avg_recall", recall)
    trial.set_user_attr("avg_time_ms", time_ms)
    trial.set_user_attr("build_time_ms", build_ms)
    trial.set_user_attr("opt_efs", opt_efs)

    # 核心逻辑：可行解返回时间，不可行解返回大惩罚
    if recall > MIN_RECALL:
        # 可行解：直接返回查询时间作为优化目标
        score = time_ms
        print(f"   ✅ FEASIBLE: EFS={opt_efs}, Recall={recall:.4f}, Time={time_ms:.4f}ms -> Score={score:.4f}")
    else:
        # 不可行解：返回一个足够大的惩罚值
        # 惩罚 = 基础惩罚 + 与目标召回率的差距（使更接近目标的解惩罚稍小，引导搜索方向）
        gap = MIN_RECALL - recall
        score = INFEASIBLE_PENALTY_BASE + gap * 10000.0
        print(f"   ❌ INFEASIBLE: EFS={opt_efs}, Recall={recall:.4f} (gap={gap:.4f}) -> Penalty={score:.0f}")
    
    return score

# ----------------- 新增：确保 stdout/stderr 行缓冲（实时写入重定向文件） -----------------
# 1) 优先使用 Python 3.7+ 的 reconfigure 方法
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    # 2) 作为降级兼容，尝试将 stdout/stderr 置为行缓冲（仅在部分平台/版本工作）
    try:
        import io, os
        fileno_out = sys.stdout.fileno()
        fileno_err = sys.stderr.fileno()
        sys.stdout = io.TextIOWrapper(os.fdopen(fileno_out, 'wb', 0), encoding='utf-8', line_buffering=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(fileno_err, 'wb', 0), encoding='utf-8', line_buffering=True)
    except Exception:
        # 无法强制行缓冲，继续运行（仍可通过 env PYTHONUNBUFFERED=1 或 -u 启动）
        pass
# -----------------------------------------------------------------------------------

READ_FROM_CSV_ONLY = True

if __name__ == "__main__":
    backup_csv()
    
    print("Loading entries from CSV...")
    trial_records = load_cache_data()
    print(f"Pre-loaded {len(CACHE)} entries, {len(GRAPH_CACHE)} graph configs.")

    os.makedirs(LOG_DIR, exist_ok=True)

    print("\n" + "="*60)
    print(f"🚀 HNSW Graph Parameter Optimization")
    print(f"🎯 Target Recall: {TARGET_RECALL} (constraint: > {MIN_RECALL})")
    print(f"📊 Optimizing: M, max_layer, efc (EFS auto-tuned via binary search)")
    print(f"🔧 Initial: M={INITIAL_M}, ML={INITIAL_MAX_LAYER}, EFC={INITIAL_EFC}")
    print("="*60)

    sampler = CmaEsSampler(seed=42)
    
    study = optuna.create_study(
        direction="minimize",
        study_name="hnsw_graph_opt_v1",
        sampler=sampler
    )

    restore_trials_to_study(study, trial_records)

    if len(study.trials) == 0:
        print("Enqueuing initial point...")
        study.enqueue_trial({"m": INITIAL_M, "max_layer": INITIAL_MAX_LAYER, "efc": INITIAL_EFC})

    batch_count = 0

    try:
        while True:
            batch_count += 1
            print(f"\n🔄 Batch #{batch_count}, Trials: {len(study.trials)}")
            
            study.optimize(objective, n_trials=BATCH_SIZE, callbacks=[csv_logger])
            
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            feasible = [t for t in completed if t.user_attrs.get('avg_recall', 0) > MIN_RECALL]
            
            print(f"   Completed: {len(completed)}, Feasible: {len(feasible)}")
            
            if feasible:
                best = min(feasible, key=lambda t: t.value if t.value else float('inf'))
                print(f"✅ Best: M={best.params['m']}, ML={best.params['max_layer']}, EFC={best.params['efc']}, "
                      f"EFS={best.user_attrs.get('opt_efs')}, Recall={best.user_attrs['avg_recall']:.4f}, "
                      f"Time={best.user_attrs['avg_time_ms']:.4f}ms")
            
            print(f"💾 Saved to {RESULT_CSV}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    
    print("\n" + "="*60)
    feasible = [t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE 
                and t.user_attrs.get('avg_recall', 0) > MIN_RECALL]
    
    if feasible:
        best = min(feasible, key=lambda t: t.value if t.value else float('inf'))
        print("🏆 Final Winner:")
        print(f"   M={best.params['m']}, ML={best.params['max_layer']}, EFC={best.params['efc']}")
        print(f"   EFS={best.user_attrs.get('opt_efs')}")
        print(f"   Recall={best.user_attrs['avg_recall']:.4f}, Time={best.user_attrs['avg_time_ms']:.4f}ms")
    else:
        print("⚠️ No feasible solutions found.")
    
    print(f"Total trials: {len(study.trials)}")