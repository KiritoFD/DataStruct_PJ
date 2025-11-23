#!/bin/bash
# gradient_search.sh - 梯度下降式参数搜索

set -euo pipefail

# 初始参数（从您的出发点开始）
CURRENT_C=2560
CURRENT_P=2560
CURRENT_I=16

# 搜索设置
MAX_STEPS=50000
MIN_RECALL=0.990
PENALTY_COEFF=200
NUM_RUNS=2                 # 新增：每个点运行次数

BIN="testg"
RESULTS_DIR="gradient_results"
SUMMARY_CSV="${RESULTS_DIR}/gradient_search.csv"
DETAILED_CSV="${RESULTS_DIR}/gradient_search_detailed.csv"

# 奖励函数：召回率高于阈值时奖励时间短，低于阈值时严厉惩罚
reward_function() {
    local recall=$1
    local time_ms=$2
    # 高于或等于阈值：时间越短奖励越高（基准奖励）
    if (( $(echo "$recall >= $MIN_RECALL" | bc -l) )); then
        echo "scale=6; 1000 / $time_ms" | bc -l
    else
        # 严厉惩罚：根据召回率与阈值的差距进行线性放大作为分母
        # penalty = 1 + PENALTY_COEFF * (MIN_RECALL - recall)
        # reward = (1000 / time_ms) / penalty
        local delta=$(echo "scale=6; $MIN_RECALL - $recall" | bc -l)
        local penalty=$(echo "scale=6; 1 + $PENALTY_COEFF * $delta" | bc -l)
        echo "scale=6; (1000 / $time_ms) / $penalty" | bc -l
    fi
}

# 邻居生成函数
generate_neighbors() {
    local c=$1
    local p=$2
    
    # 定义搜索步长
    local c_step=$(( c / 10 ))
    local p_step=$(( p / 8 ))
    
    #  确保最小步长
    c_step=$(( c_step < 32 ? 32 : c_step ))
    p_step=$(( p_step < 4 ? 4 : p_step ))
    
    # 生成邻居点
    echo "$((c + c_step)) $p"
    echo "$((c - c_step)) $p"
    echo "$c $((p + p_step))"
    echo "$c $((p - p_step))"
    
    # 对角线方向
    echo "$((c + c_step)) $((p + p_step))"
    echo "$((c + c_step)) $((p - p_step))"
    echo "$((c - c_step)) $((p + p_step))"
    echo "$((c - c_step)) $((p - p_step))"
    
    # 添加一些随机扰动
    local random_c_step=$(( RANDOM % 64 + 32 ))
    local random_p_step=$(( RANDOM % 8 + 4 ))
    echo "$((c + random_c_step)) $p"
    echo "$((c - random_c_step)) $p"
    echo "$c $((p + random_p_step))"
    echo "$c $((p - random_p_step))"
}

# 记录详细结果到CSV
log_detailed_result() {
    local step=$1
    local type=$2
    local c=$3
    local p=$4
    local recall=$5
    local time_ms=$6
    local reward=$7
    local neighbor_index=${8:-""}
    local event=${9:-""}
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp,$step,$type,$c,$p,$CURRENT_I,$recall,$time_ms,$reward,$neighbor_index,$event" >> "$DETAILED_CSV"
    
    # 同时打印到控制台
    if [ "$type" = "neighbor" ]; then
        echo "  Neighbor $neighbor_index: C=$c, P=$p -> Recall=$recall, Time=${time_ms}ms, Reward=$reward"
    elif [ "$type" = "current" ]; then
        echo "  Current: C=$c, P=$p -> Recall=$recall, Time=${time_ms}ms, Reward=$reward"
    elif [ "$type" = "best" ]; then
        echo "  *** NEW BEST: C=$c, P=$p -> Recall=$recall, Time=${time_ms}ms, Reward=$reward ***"
    fi
}

# 运行测试并获取指标
run_test() {
    local c=$1
    local p=$2
    local i=$3
    local step=${4:-0}
    local neighbor_index=${5:-""}
    
    local stamp="c${c}_i${i}_p${p}"
    local logfile="${RESULTS_DIR}/run_${stamp}.log"
    
    # 检查缓存
    if grep -q -F "${stamp}," "$SUMMARY_CSV" 2>/dev/null; then
        echo "Using cached result for ${stamp}" >&2
        local line=$(grep -F "${stamp}," "$SUMMARY_CSV")
        local recall=$(echo "$line" | cut -d',' -f6)
        local time_ms=$(echo "$line" | cut -d',' -f7)
        echo "$recall $time_ms"
        return 0
    fi
    
    echo "Running test: C=$c, P=$p, I=$i (${NUM_RUNS} runs)" >&2
    
    # 运行多次并累积结果
    local total_recall=0
    local total_time=0
    
    for run_num in $(seq 1 $NUM_RUNS); do
        local run_logfile="${RESULTS_DIR}/run_${stamp}_run${run_num}.log"
        
        # 运行测试
        if ! ./"$BIN" --n "$c" --k "$i" --p "$p" &> "$run_logfile"; then
            echo "Test run $run_num failed for C=$c, P=$p" >&2
            echo "0.0 1000.0"
            return 1
        fi
        
        # 解析结果
        local recall_line=$(grep -m1 "Average recall@" "$run_logfile" 2>/dev/null || true)
        local avg_query_line=$(grep -m1 "Average query time" "$run_logfile" 2>/dev/null || true)
        
        local recall="0.0"
        local time_ms="1000.0"
        
        if [ -n "$recall_line" ]; then
            recall=$(echo "$recall_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+).*/\1/')
        fi
        
        if [ -n "$avg_query_line" ]; then
            time_ms=$(echo "$avg_query_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *ms.*/\1/')
        fi
        
        echo "  Run $run_num: Recall=$recall, Time=${time_ms}ms" >&2
        total_recall=$(echo "$total_recall + $recall" | bc -l)
        total_time=$(echo "$total_time + $time_ms" | bc -l)
    done
    
    # 计算平均值
    local avg_recall=$(echo "scale=6; $total_recall / $NUM_RUNS" | bc -l)
    local avg_time=$(echo "scale=6; $total_time / $NUM_RUNS" | bc -l)
    
    echo "  Average: Recall=$avg_recall, Time=${avg_time}ms" >&2
    
    # 记录到主CSV（使用平均值）
    echo "${stamp},${c},${i},${p},0,${avg_recall},${avg_time},0" >> "$SUMMARY_CSV"
    
    echo "$avg_recall $avg_time"
}

# 主搜索函数
gradient_search() {
    local current_c=$1
    local current_p=$2
    local current_i=$3

    local best_c=$current_c
    local best_p=$current_p
    local best_reward=-1.0
    local best_recall=0
    local best_time=1000

    echo "Starting gradient search from: C=$current_c, P=$current_p"
    echo "=================================================================="
    echo "timestamp,step,type,C,P,I,recall,time_ms,reward,neighbor_index,event" > "$DETAILED_CSV"

    local c_step_init=$(( current_c / 10 )); c_step_init=$(( c_step_init < 32 ? 32 : c_step_init ))
    local p_step_init=$(( current_p / 8  )); p_step_init=$(( p_step_init < 4  ? 4  : p_step_init ))

    local c_step=$c_step_init
    local p_step=$p_step_init

    local stagnation=0
    local improvements_in_row=0
    local MAX_STAGNATION_BEFORE_JUMP=5
    local MAX_STEP_SCALE=4        # 最大放大倍数
    local LARGE_JUMP_FACTOR=3     # 大跳时放大基础步长
    local T0=1.0                  # 模拟退火初始温度
    local TEMP_DECAY=500          # 温度衰减尺度

    generate_neighbors_custom() {
        local c=$1; local p=$2; local cst=$3; local pst=$4
        echo "$((c + cst)) $p"
        echo "$((c - cst)) $p"
        echo "$c $((p + pst))"
        echo "$c $((p - pst))"
        echo "$((c + cst)) $((p + pst))"
        echo "$((c + cst)) $((p - pst))"
        echo "$((c - cst)) $((p + pst))"
        echo "$((c - cst)) $((p - pst))"
        local rc=$(( RANDOM % (cst*2) - cst ))
        local rp=$(( RANDOM % (pst*2) - pst ))
        echo "$((c + rc)) $((p + rp))"
    }

    for step in $(seq 1 $MAX_STEPS); do
        echo
        echo "=== Step $step ==="
        echo "Pos: C=$current_c P=$current_p | step: c_step=$c_step p_step=$p_step | stagnation=$stagnation improve_row=$improvements_in_row"

        # 测当前点
        local current_result
        current_result=$(run_test $current_c $current_p $current_i $step "current")
        local current_recall=$(echo $current_result | cut -d' ' -f1)
        local current_time=$(echo $current_result | cut -d' ' -f2)
        local current_reward=$(reward_function $current_recall $current_time)
        log_detailed_result $step "current" $current_c $current_p $current_recall $current_time $current_reward ""  # event空

        # 全局最优更新
        if (( $(echo "$current_reward > $best_reward" | bc -l) )); then
            best_reward=$current_reward
            best_c=$current_c
            best_p=$current_p
            best_recall=$current_recall
            best_time=$current_time
            improvements_in_row=$((improvements_in_row + 1))
            stagnation=0
            log_detailed_result $step "best" $best_c $best_p $best_recall $best_time $best_reward "" "***NEW_BEST(current)***"
        else
            improvements_in_row=0
        fi

        # 邻居搜索
        local best_neighbor_c=$current_c
        local best_neighbor_p=$current_p
        local best_neighbor_reward=$current_reward
        local best_neighbor_recall=$current_recall
        local best_neighbor_time=$current_time
        local neighbor_index=0
        local T=$(echo "$T0 * e(-$step / $TEMP_DECAY)" | bc -l)

        while IFS=' ' read -r nc np; do
            neighbor_index=$((neighbor_index + 1))
            # 边界
            if [ $nc -lt 64 ] || [ $nc -gt 65536 ] || [ $np -lt 1 ] || [ $np -gt 2048 ]; then
                continue
            fi
            local nres
            nres=$(run_test $nc $np $current_i $step $neighbor_index 2>/dev/null || echo "0.0 1000.0")
            local nrec=$(echo $nres | cut -d' ' -f1)
            local ntime=$(echo $nres | cut -d' ' -f2)
            local nreward=$(reward_function $nrec $ntime)
            log_detailed_result $step "neighbor" $nc $np $nrec $ntime $nreward $neighbor_index ""

            # 最佳邻居（用于移动）
            if (( $(echo "$nreward > $best_neighbor_reward" | bc -l) )); then
                best_neighbor_reward=$nreward
                best_neighbor_c=$nc
                best_neighbor_p=$np
                best_neighbor_recall=$nrec
                best_neighbor_time=$ntime
            fi

            # 全局最优
            if (( $(echo "$nreward > $best_reward" | bc -l) )); then
                best_reward=$nreward
                best_c=$nc
                best_p=$np
                best_recall=$nrec
                best_time=$ntime
                improvements_in_row=$((improvements_in_row + 1))
                stagnation=0
                log_detailed_result $step "best" $best_c $best_p $best_recall $best_time $best_reward $neighbor_index "***NEW_BEST(neighbor)***"
            fi
        done < <(generate_neighbors_custom $current_c $current_p $c_step $p_step)

        echo "Best neighbor reward=$best_neighbor_reward (C=$best_neighbor_c P=$best_neighbor_p) | current_reward=$current_reward"

        # 移动策略：改进 or 模拟退火接受
        if (( $(echo "$best_neighbor_reward > $current_reward" | bc -l) )); then
            current_c=$best_neighbor_c
            current_p=$best_neighbor_p
            echo "Move: IMPROVED -> C=$current_c P=$current_p"
        else
            # 模拟退火接受差的
            local drop=$(echo "$current_reward - $best_neighbor_reward" | bc -l)
            if (( $(echo "$drop < 0" | bc -l) )); then
                drop=0
            fi
            local prob=$(echo "e(-$drop / ($T+1e-9))" | bc -l)
            local randf=$(echo "$RANDOM / 32767" | bc -l)
            if (( $(echo "$randf < $prob" | bc -l) )); then
                current_c=$best_neighbor_c
                current_p=$best_neighbor_p
                echo "Move: ANNEAL_ACCEPT (prob=$prob rand=$randf) -> C=$current_c P=$current_p"
                log_detailed_result $step "current" $current_c $current_p $best_neighbor_recall $best_neighbor_time $best_neighbor_reward "" "ANNEAL_ACCEPT"
            else
                echo "Move: STAY (no improvement)"
            fi
        fi

        # 步长调整：提高或降低
        if [ "$improvements_in_row" -ge 3 ]; then
            local new_c_step=$(( c_step * 2 ))
            local new_p_step=$(( p_step * 2 ))
            local max_c=$(( c_step_init * MAX_STEP_SCALE ))
            local max_p=$(( p_step_init * MAX_STEP_SCALE ))
            if [ $new_c_step -le $max_c ]; then c_step=$new_c_step; fi
            if [ $new_p_step -le $max_p ]; then p_step=$new_p_step; fi
            echo "STEP_UP -> c_step=$c_step p_step=$p_step"
            log_detailed_result $step "current" $current_c $current_p $current_recall $current_time $current_reward "" "STEP_UP"
            improvements_in_row=0
        fi

        if (( $(echo "$best_neighbor_reward <= $current_reward" | bc -l) )) && [ "$current_c" -eq "$best_c" ] && [ "$current_p" -eq "$best_p" ]; then
            stagnation=$((stagnation + 1))
        else
            stagnation=0
        fi

        if [ $stagnation -gt 0 ] && (( $(echo "$best_neighbor_reward <= $current_reward" | bc -l) )); then
            if [ $c_step -gt 32 ] || [ $p_step -gt 4 ]; then
                c_step=$(( c_step / 2 )); if [ $c_step -lt 32 ]; then c_step=32; fi
                p_step=$(( p_step / 2 )); if [ $p_step -lt 4 ]; then p_step=4; fi
                echo "STEP_DOWN -> c_step=$c_step p_step=$p_step (stagnation=$stagnation)"
                log_detailed_result $step "current" $current_c $current_p $current_recall $current_time $current_reward "" "STEP_DOWN"
            elif [ $stagnation -ge $MAX_STAGNATION_BEFORE_JUMP ]; then
                # 触发大跳
                c_step=$(( c_step_init * LARGE_JUMP_FACTOR ))
                p_step=$(( p_step_init * LARGE_JUMP_FACTOR ))
                # 围绕全局最优随机新位置
                local jump_c=$(( best_c + (RANDOM % (c_step*2) - c_step) ))
                local jump_p=$(( best_p + (RANDOM % (p_step*2) - p_step) ))
                [ $jump_c -lt 64 ] && jump_c=64
                [ $jump_c -gt 65536 ] && jump_c=65536
                [ $jump_p -lt 1 ] && jump_p=1
                [ $jump_p -gt 2048 ] && jump_p=2048
                current_c=$jump_c
                current_p=$jump_p
                stagnation=0
                echo "BIG_JUMP -> new C=$current_c P=$current_p | c_step=$c_step p_step=$p_step"
                log_detailed_result $step "current" $current_c $current_p $current_recall $current_time $current_reward "" "BIG_JUMP"
            fi
        fi
        echo "=================================================================="
    done

    echo
    echo "=== Search Finished (MAX_STEPS reached or loop exhausted) ==="
    echo "Best configuration:"
    echo "  C=$best_c P=$best_p I=$current_i"
    echo "  Recall=$best_recall Time=${best_time}ms Reward=$best_reward"
    echo "Results saved -> $SUMMARY_CSV  /  $DETAILED_CSV"
}

# 初始化
mkdir -p "$RESULTS_DIR"
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "STAMP,NUM_CENTROIDS,KMEANS_ITER,NPROBE,ELAPSED_s,RECALL,AVG_QUERY_TIME_ms,INDEX_BUILD_TIME_s" > "$SUMMARY_CSV"
fi

# 检查bc是否可用
if ! command -v bc &>/dev/null; then
    echo "Error: 'bc' command is required but not found. Please install it."
    exit 1
fi

# 检查二进制文件
if [ ! -x "$BIN" ]; then
    echo "Error: executable '$BIN' not found or not executable."
    exit 1
fi

# 开始搜索
gradient_search $CURRENT_C $CURRENT_P $CURRENT_I