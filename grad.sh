#!/bin/bash
# gradient_search.sh - 梯度下降式参数搜索

set -euo pipefail

# 初始参数（从您的出发点开始）
CURRENT_C=2564
CURRENT_P=42
CURRENT_I=16

# 搜索设置
MAX_STEPS=50000
MIN_RECALL=0.98
PENALTY_COEFF=1000   # 惩罚系数：越大，低于阈值时惩罚越严重

BIN="test"
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
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp,$step,$type,$c,$p,$CURRENT_I,$recall,$time_ms,$reward,$neighbor_index" >> "$DETAILED_CSV"
    
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
    
    echo "Running test: C=$c, P=$p, I=$i" >&2
    
    # 运行测试
    if ! ./"$BIN" --n "$c" --k "$i" --p "$p" &> "$logfile"; then
        echo "Test failed for C=$c, P=$p" >&2
        echo "0.0 1000.0"  # 测试失败返回最差奖励
        return 1
    fi
    
    # 解析结果
    local recall_line=$(grep -m1 "Average recall@" "$logfile" 2>/dev/null || true)
    local avg_query_line=$(grep -m1 "Average query time" "$logfile" 2>/dev/null || true)
    
    local recall="0.0"
    local time_ms="1000.0"
    
    if [ -n "$recall_line" ]; then
        recall=$(echo "$recall_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+).*/\1/')
    fi
    
    if [ -n "$avg_query_line" ]; then
        time_ms=$(echo "$avg_query_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *ms.*/\1/')
    fi
    
    # 记录到主CSV
    echo "${stamp},${c},${i},${p},0,${recall},${time_ms},0" >> "$SUMMARY_CSV"
    
    echo "$recall $time_ms"
}

# 主搜索函数
gradient_search() {
    local current_c=$1
    local current_p=$2
    local current_i=$3
    
    local best_c=$current_c
    local best_p=$current_p
    local best_reward=0
    local best_recall=0
    local best_time=1000
    
    echo "Starting gradient search from: C=$current_c, P=$current_p"
    echo "=================================================================="
    
    # 初始化详细CSV文件头
    echo "timestamp,step,type,C,P,I,recall,time_ms,reward,neighbor_index" > "$DETAILED_CSV"
    
    for step in $(seq 1 $MAX_STEPS); do
        echo
        echo "=== Step $step ==="
        echo "Current position: C=$current_c, P=$current_p"
        
        # 测试当前点
        echo "Testing current point..."
        local current_result
        current_result=$(run_test $current_c $current_p $current_i $step "current")
        local current_recall=$(echo $current_result | cut -d' ' -f1)
        local current_time=$(echo $current_result | cut -d' ' -f2)
        local current_reward=$(reward_function $current_recall $current_time)
        
        # 记录当前点结果
        log_detailed_result $step "current" $current_c $current_p $current_recall $current_time $current_reward
        
        # 更新全局最优
        if (( $(echo "$current_reward > $best_reward" | bc -l) )); then
            best_reward=$current_reward
            best_c=$current_c
            best_p=$current_p
            best_recall=$current_recall
            best_time=$current_time
            log_detailed_result $step "best" $best_c $best_p $best_recall $best_time $best_reward
        fi
        
        echo "Current reward: $current_reward, Best reward: $best_reward"
        
        # 生成并评估邻居
        echo "Generating and testing neighbors..."
        local best_neighbor_c=$current_c
        local best_neighbor_p=$current_p
        local best_neighbor_reward=$current_reward
        local neighbor_index=0
        
        while IFS=' ' read -r neighbor_c neighbor_p; do
            neighbor_index=$((neighbor_index + 1))
            
            # 参数边界检查
            if [ $neighbor_c -lt 64 ] || [ $neighbor_c -gt 65536 ] || \
               [ $neighbor_p -lt 1 ] || [ $neighbor_p -gt 1024 ]; then
                echo "  Neighbor $neighbor_index: C=$neighbor_c, P=$neighbor_p -> SKIPPED (out of bounds)"
                continue
            fi
            
            # 测试邻居
            local neighbor_result
            neighbor_result=$(run_test $neighbor_c $neighbor_p $current_i $step $neighbor_index 2>/dev/null || echo "0.0 1000.0")
            local neighbor_recall=$(echo $neighbor_result | cut -d' ' -f1)
            local neighbor_time=$(echo $neighbor_result | cut -d' ' -f2)
            local neighbor_reward=$(reward_function $neighbor_recall $neighbor_time)
            
            # 记录邻居结果
            log_detailed_result $step "neighbor" $neighbor_c $neighbor_p $neighbor_recall $neighbor_time $neighbor_reward $neighbor_index
            
            # 更新最佳邻居
            if (( $(echo "$neighbor_reward > $best_neighbor_reward" | bc -l) )); then
                best_neighbor_reward=$neighbor_reward
                best_neighbor_c=$neighbor_c
                best_neighbor_p=$neighbor_p
                echo "  -> New best neighbor: C=$best_neighbor_c, P=$best_neighbor_p, Reward=$best_neighbor_reward"
            fi
            
        done < <(generate_neighbors $current_c $current_p)
        
        echo "Best neighbor reward: $best_neighbor_reward, Current reward: $current_reward"
        
        # 移动到最佳邻居
        if (( $(echo "$best_neighbor_reward > $current_reward" | bc -l) )); then
            echo "Moving to better neighbor: C=$best_neighbor_c, P=$best_neighbor_p"
            current_c=$best_neighbor_c
            current_p=$best_neighbor_p
        else
            # 没有更好的邻居，可能到达局部最优
            echo "No better neighbors found - local optimum reached at step $step"
            break
        fi
        
        # 提前终止条件
        if (( $(echo "$best_recall >= $MIN_RECALL && $best_time < 2.0" | bc -l) )); then
            echo "Target performance reached! Recall: $best_recall, Time: ${best_time}ms"
            break
        fi
        
        echo "=================================================================="
    done
    
    echo
    echo "=== Search Finished ==="
    echo "Best configuration found:"
    echo "  C=$best_c, P=$best_p, I=$current_i"
    echo "  Recall: $best_recall, Time: ${best_time}ms"
    echo "  Reward: $best_reward"
    echo
    echo "Results saved to:"
    echo "  - $SUMMARY_CSV (main results)"
    echo "  - $DETAILED_CSV (detailed search progress)"
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