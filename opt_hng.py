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
MIN_RECALL = 0.9803          # 目标召回率 (98.5%)
FIXED_K = 10                # Top-K
BATCH_SIZE = 20             # 每批运行多少次 trial

# --- [修复] 补全缺失配置 ---
NUM_RUNS = 1                # 单次参数取平均值的运行次数 (1次最快，3次更稳)

# --- [优化] 合理的参数搜索范围 ---
# M: 16-64 是 HNSW 的黄金区间，太小召回低，太大构建慢且内存大
M_RANGE = (16, 64)          
# MAX_LAYER: 降低层数可以加速搜索，但太低会影响大数据集
MAX_LAYER_RANGE = (0,20)
# EFC: 影响索引质量，通常 200-800 足够，极端情况才需要 1000+
EFC_RANGE = (80, 1000)     
# EFS: 决定搜索时的精细度，为了 0.985 召回，上限设高一点
EFS_RANGE = (80, 2000)     

LOG_DIR = "Log"
RESULT_CSV = "optuna_adaptive_hng.csv"

# --- [优化] 奖励/惩罚常量 (避免数值溢出) ---
# 目标是：Recall不达标 -> 负分；Recall达标 -> 正分 (分数越高代表时间越短)
SUCCESS_BASE = 1000.0      # 基础奖励
RECALL_BONUS = 100.0       # 召回溢出奖励权重
PENALTY_BASE = 500.0       # 基础惩罚 (不用 1e9，防止 Optuna 数值不稳定)

# --- 全局缓存 ---
CACHE: Dict[Tuple[int, int, int, int], Tuple[float, float, float]] = {}  # (m, max_layer, efc, efs) -> (recall, time, build)

def scan_log_directory():
    """扫描 Log 目录下的所有日志，解析图质量信息"""
    graph_stats = []
    if not os.path.exists(LOG_DIR):
        return graph_stats
    
    for filename in os.listdir(LOG_DIR):
        if not filename.endswith('.log'):
            continue
        filepath = os.path.join(LOG_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 解析参数
            m_match = re.search(r'M:\s*(\d+)', content)
            ml_match = re.search(r'Max\s*layer:\s*(\d+)', content, re.I)
            efc_match = re.search(r'EFC:\s*(\d+)', content)
            efs_match = re.search(r'EFS:\s*(\d+)', content)
            
            # 解析图质量
            level_match = re.search(r'max_level[:\s]+(\d+)', content, re.I)
            degree_match = re.search(r'avg.*degree.*l0[:\s]+([0-9.]+)', content, re.I)
            nodes_match = re.search(r'num_nodes[:\s]+(\d+)', content, re.I)
            actual_layer_match = re.search(r'actual.*max.*layer[:\s]+(\d+)', content, re.I)
            
            # 解析性能
            recall_match = re.search(r'Average\s+recall@\d+[:\s]+([0-9.]+)', content, re.I)
            time_match = re.search(r'Average\s+query\s+time[:\s]+([0-9.]+)\s*ms', content, re.I)
            build_match = re.search(r'Index\s+build\s+time[:\s]+([0-9.]+)\s*ms', content, re.I)
            
            stat = {
                'file': filename,
                'm': int(m_match.group(1)) if m_match else None,
                'max_layer': int(ml_match.group(1)) if ml_match else None,
                'efc': int(efc_match.group(1)) if efc_match else None,
                'efs': int(efs_match.group(1)) if efs_match else None,
                'graph_max_level': int(level_match.group(1)) if level_match else None,
                'avg_degree_l0': float(degree_match.group(1)) if degree_match else None,
                'num_nodes': int(nodes_match.group(1)) if nodes_match else None,
                'actual_max_layer': int(actual_layer_match.group(1)) if actual_layer_match else None,
                'recall': float(recall_match.group(1)) if recall_match else None,
                'query_time_ms': float(time_match.group(1)) if time_match else None,
                'build_time_ms': float(build_match.group(1)) if build_match else None,
            }
            
            # 只保留有效记录
            if stat['recall'] is not None and stat['m'] is not None:
                graph_stats.append(stat)
                
        except Exception as e:
            continue
    
    return graph_stats

def analyze_graph_quality(stats):
    """分析图质量与性能的关系"""
    if not stats:
        print("No graph statistics available.")
        return
    
    print("\n" + "="*60)
    print("📊 Graph Quality Analysis from Log Files")
    print("="*60)
    
    # 按召回率排序
    valid_stats = [s for s in stats if s['recall'] is not None and s['recall'] >= MIN_RECALL]
    if valid_stats:
        valid_stats.sort(key=lambda x: x['query_time_ms'] or float('inf'))
        
        print(f"\n✅ {len(valid_stats)} configurations met recall >= {MIN_RECALL}")
        print("\nTop 5 fastest configurations:")
        for i, s in enumerate(valid_stats[:5]):
            print(f"  {i+1}. M={s['m']}, ML={s['max_layer']}, EFC={s['efc']}, EFS={s['efs']}")
            print(f"     Recall={s['recall']:.4f}, Time={s['query_time_ms']:.3f}ms, Build={s['build_time_ms']:.0f}ms")
            if s.get('avg_degree_l0'):
                print(f"     AvgDegree_L0={s['avg_degree_l0']:.1f}, ActualMaxLayer={s.get('actual_max_layer', 'N/A')}")
    
    # 分析图质量与性能关系
    with_degree = [s for s in stats if s.get('avg_degree_l0') is not None]
    if with_degree:
        print(f"\n📈 Graph Quality Insights ({len(with_degree)} samples):")
        avg_degree = sum(s['avg_degree_l0'] for s in with_degree) / len(with_degree)
        print(f"  Average L0 Degree: {avg_degree:.1f}")
        
        # 高度数 vs 低度数
        high_degree = [s for s in with_degree if s['avg_degree_l0'] > avg_degree]
        low_degree = [s for s in with_degree if s['avg_degree_l0'] <= avg_degree]
        
        if high_degree and low_degree:
            high_recall = sum(s['recall'] or 0 for s in high_degree) / len(high_degree)
            low_recall = sum(s['recall'] or 0 for s in low_degree) / len(low_degree)
            print(f"  High Degree (>{avg_degree:.0f}) Avg Recall: {high_recall:.4f}")
            print(f"  Low Degree (<={avg_degree:.0f}) Avg Recall: {low_recall:.4f}")
    
    print("="*60 + "\n")

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
                    ml = p(row.get('max_layer'), int) or 8  # 默认值
                    efc = p(row.get('efc'), int)
                    efs = p(row.get('efs'), int)
                    rec = p(row.get('avg_recall'), float)
                    time_ms = p(row.get('avg_time_ms'), float)
                    build_ms = p(row.get('build_time_ms'), float)
                    reward = p(row.get('reward_score'), float)
                    
                    if all(v is not None for v in [m, efc, efs, rec, time_ms, build_ms]):
                        # 加入缓存
                        CACHE[(m, ml, efc, efs)] = (rec, time_ms, build_ms)
                        
                        # 记录 trial 信息
                        trial_id = int(row.get('trial_id', -1)) if row.get('trial_id') else -1
                        if trial_id >= 0:
                            trial_records[trial_id] = {
                                'params': {'m': m, 'max_layer': ml, 'efc': efc, 'efs': efs},
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
    """从历史记录恢复 trials 到 study，具有更强的容错性。"""
    if not trial_records:
        return

    print(f"Restoring {len(trial_records)} trials from history...")

    trials_to_add = []
    now = datetime.now()

    for trial_id in sorted(trial_records.keys()):
        record = trial_records[trial_id]
        try:
            # 规范化 state（兼容字符串或枚举）
            state = record.get('state')
            if isinstance(state, str):
                try:
                    state = optuna.trial.TrialState[state.upper()]
                except Exception:
                    state = optuna.trial.TrialState.COMPLETE if record.get('value') is not None else optuna.trial.TrialState.FAIL
            elif state is None:
                state = optuna.trial.TrialState.COMPLETE if record.get('value') is not None else optuna.trial.TrialState.FAIL

            # 确保 datetime_start 对于非 WAITING 状态已设置
            if state != optuna.trial.TrialState.WAITING:
                datetime_start = record.get('datetime_start') or now
            else:
                datetime_start = None

            # 确保完成时间对于 COMPLETE trial 已设置（若缺失则回退到 datetime_start 或 now）
            if state == optuna.trial.TrialState.COMPLETE:
                datetime_complete = record.get('datetime_complete') or datetime_start or now
            else:
                datetime_complete = None

            # 构造 FrozenTrial，必要字段校验
            params = record.get('params') or {}
            user_attrs = record.get('user_attrs') or {}
            value = record.get('value') if state == optuna.trial.TrialState.COMPLETE else None

            frozen_trial = optuna.trial.FrozenTrial(
                number=trial_id,
                state=state,
                value=value,
                datetime_start=datetime_start,
                datetime_complete=datetime_complete,
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
            print(f"Warning: skipping invalid trial record {trial_id}: {e}")
            continue

    # 尝试批量添加；若失败则退化为逐个添加并跳过异常 trial
    if not trials_to_add:
        return

    try:
        study.add_trials(trials_to_add)
    except Exception as e:
        print(f"Warning: batch add_trials failed: {e}; trying individual adds...")
        for tr in trials_to_add:
            try:
                study.add_trial(tr)
            except Exception as ex:
                print(f"Warning: failed to add trial {tr.number}: {ex}")
                continue

def csv_logger(study: optuna.Study, trial: optuna.trial.Trial):
    """写入 CSV 日志，自动修复/写入表头"""
    if not trial.state.is_finished():
        return

    data = {
        'timestamp': datetime.now().isoformat(),
        'study_name': getattr(study, "study_name", None) or getattr(study, "name", ""),
        'trial_id': trial.number,
        'm': trial.params.get('m'),
        'max_layer': trial.params.get('max_layer'),
        'efc': trial.params.get('efc',
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

    # 如果存在文件但表头不正确，修复它（读取现有内容并重写带正确表头）
    try:
        if os.path.exists(RESULT_CSV):
            with open(RESULT_CSV, 'r', encoding='utf-8', errors='ignore') as rf:
                first_line = rf.readline()
                rest = rf.read()
            # 如果第一行不是期望的表头，则重写文件并加上表头
            if 'timestamp' not in first_line.lower():
                try:
                    with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as wf:
                        writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
                        writer.writeheader()
                        # 将原始内容（若有）写回，保留旧数据
                        if rest:
                            wf.write(rest)
                except Exception as e:
                    print(f"Warning: failed to repair CSV header: {e}")
    except Exception as e:
        print(f"Warning: error checking CSV file: {e}")

    # 追加写入主 CSV
    try:
        file_exists = os.path.exists(RESULT_CSV)
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
        # 修复 study 文件的表头（与主文件相同）
        if os.path.exists(study_csv):
            try:
                with open(study_csv, 'r', encoding='utf-8', errors='ignore') as rf:
                    first_line = rf.readline()
                    rest = rf.read()
                if 'timestamp' not in first_line.lower():
                    with open(study_csv, 'w', newline='', encoding='utf-8') as wf:
                        swriter = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
                        swriter.writeheader()
                        if rest:
                            wf.write(rest)
            except Exception:
                pass

        study_file_exists = os.path.exists(study_csv)
        with open(study_csv, 'a', newline='', encoding='utf-8') as sf:
            swriter = csv.DictWriter(sf, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
            if not study_file_exists:
                swriter.writeheader()
            swriter.writerow(data)
    except IOError as e:
        print(f"Warning: cannot write study-specific CSV {study_csv}: {e}")

def run_test(m: int, max_layer: int, efc: int, efs: int) -> Tuple[float, float, float]:
    key = (m, max_layer, efc, efs)
    if key in CACHE:
        print(f"   [CACHE HIT] m={m}, ml={max_layer}, efc={efc}, efs={efs}")
        return CACHE[key]

    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 为本次测试创建日志文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_logfile = os.path.join(LOG_DIR, f"trial_m{m}_ml{max_layer}_efc{efc}_efs{efs}_{ts}.log")
    
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
    cmd = [BIN_PATH, "--m", str(m), "--max_layer", str(max_layer), "--efc", str(efc), "--efs", str(efs)]
    
    log_write(f"   [START] m={m}, ml={max_layer}, efc={efc}, efs={efs}\n")
    log_write(f"   [LOG] {trial_logfile}\n")
    log_write(f"   [RUN 1/{NUM_RUNS}] Executing: {' '.join(cmd)}\n")
    
    total_rec, total_time, total_build = 0.0, 0.0, 0.0
    
    for run_num in range(NUM_RUNS):
        try:
            log_write(f"\n{'='*60}\n")
            log_write(f"Trial: m={m}, ml={max_layer}, efc={efc}, efs={efs}, Run {run_num+1}/{NUM_RUNS}\n")
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
    
    log_write(f"   [RESULT] m={m:<2} ml={max_layer:<2} efc={efc:<4} efs={efs:<4} | Rec={avg_res[0]:.4f} Time={avg_res[1]:.4f}ms Build={avg_res[2]:.2f}ms\n")
    log_write(f"   [SUCCESS] Logged to {trial_logfile}\n")
    
    lf.close()
    CACHE[key] = avg_res
    return avg_res

def objective(trial: optuna.trial.Trial) -> float:
    m = trial.suggest_int("m", M_RANGE[0], M_RANGE[1])
    max_layer = trial.suggest_int("max_layer", MAX_LAYER_RANGE[0], MAX_LAYER_RANGE[1])
    efc = trial.suggest_int("efc", EFC_RANGE[0], EFC_RANGE[1])
    efs = trial.suggest_int("efs", EFS_RANGE[0], EFS_RANGE[1])

    recall, time_ms, build_ms = run_test(m, max_layer, efc, efs)
    
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
        "max_layer": random.randint(*MAX_LAYER_RANGE),
        "efc": random.randint(*EFC_RANGE),
        "efs": random.randint(*EFS_RANGE)
    })
    
    # 2. 高潜力点 (大参数)
    if random.random() < 0.4:
        study.enqueue_trial({
            "m": random.randint(40, M_RANGE[1]),
            "max_layer": random.randint(6, MAX_LAYER_RANGE[1]),
            "efc": random.randint(500, EFC_RANGE[1]),
            "efs": random.randint(150, EFS_RANGE[1])
        })

# Script settings: read only from CSV instead of scanning Log/ (set True to skip log scanning)
READ_FROM_CSV_ONLY = True

if __name__ == "__main__":
    if READ_FROM_CSV_ONLY:
        print("Loading entries from CSV only (skip log directory scan)...")
        trial_records = load_cache_data()
        print(f"Pre-loaded {len(CACHE)} entries from CSV.")
    else:
        # 启动时扫描日志并分析图质量
        print("Scanning log directory for graph quality analysis...")
        log_stats = scan_log_directory()
        if log_stats:
            analyze_graph_quality(log_stats)
            # 将有效结果加入缓存
            for s in log_stats:
                if s['m'] is not None and s['max_layer'] is not None and s['efc'] is not None and s['efs'] is not None and s['recall'] is not None and s['query_time_ms'] is not None and s['build_time_ms'] is not None:
                    key = (s['m'], s['max_layer'], s['efc'], s['efs'])
                    if key not in CACHE:
                        CACHE[key] = (s['recall'], s['query_time_ms'], s['build_time_ms'])
            print(f"Pre-loaded {len(CACHE)} entries from log files.")
        trial_records = load_cache_data()

    os.makedirs(LOG_DIR, exist_ok=True)

    print("="*60)
    print(f"🚀 Adaptive HNSW Optimization (CSV Storage)")
    print(f"🎯 Target Recall: >= {MIN_RECALL}")
    print(f"🔎 Search Space: M={M_RANGE}, ML={MAX_LAYER_RANGE}, EFC={EFC_RANGE}, EFS={EFS_RANGE}")
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
            {"m": 48, "max_layer": 8, "efc": 550, "efs": 120},   # 高质量图配置
            {"m": 40, "max_layer": 6, "efc": 500, "efs": 200},   # 中等配置
            {"m": 56, "max_layer": 10, "efc": 600, "efs": 150},  # 激进配置
        ]
        
        for t in initial_trials:
            key = (t['m'], t['max_layer'], t['efc'], t['efs'])
            if key not in CACHE:
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
                study.enqueue_trial({"m": 48, "max_layer": 8, "efc": 800, "efs": 400})

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