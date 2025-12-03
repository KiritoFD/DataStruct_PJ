#!/usr/bin/env bash
set -euo pipefail

# run_ablation_best_worst.sh
# 从 optuna_adaptive_hng.csv 中：
# 1. 均匀选择 10 个点（从好到坏）
# 2. 随机选择 10 个点
# 总共 20 个点进行消融实验

WORKDIR=$(pwd)
BINARY="hng"
COMPILE_CMD=(g++ MySolution.cpp test_hn_g.cpp -o "$BINARY" -O3 -Ofast -march=znver4 -mtune=native -ffast-math -flto -pthread -std=c++20 -Wall)

# Default parameters
THREADS=${THREADS:-32}
REPEATS=${REPEATS:-1}

# Results dir
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/ablation_uniform_random_${TS}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

# CSV header
echo "variant,point_type,point_index,m,ml,efc,efs,recall_cached,csr,prefetch,simd,pruning,heap,flat_index,build_time_internal_ms,build_time_local_ms,queries_per_sec,avg_recall,avg_query_time_ms,avg_dists,last_query_dists,total_query_time_ms,log_file,run_index" > "$CSV"

# Compile once
echo "Compiling: ${COMPILE_CMD[*]}"
"${COMPILE_CMD[@]}"
echo "Binary compiled: $(readlink -f $BINARY)"

# 提取点的函数
extract_points() {
    local csv_file="optuna_hng.csv"
    if [[ ! -f "$csv_file" ]]; then
        echo "Error: $csv_file not found" >&2
        return 1
    fi
    
    # 总行数
    local total_lines=$(tail -n +2 "$csv_file" | wc -l)
    echo "Total points in cache: $total_lines" >&2
    
    # 1. 按 recall 降序排序，均匀选 10 个点
    echo "=== Extracting 10 uniformly distributed points (best to worst) ===" >&2
    local uniform_points=$(tail -n +2 "$csv_file" | sort -t',' -k5 -rn)
    local step=$((total_lines / 10))
    local idx=0
    echo "$uniform_points" | while read -r line; do
        if (( idx % step == 0 && idx / step < 10 )); then
            echo "uniform_$(printf "%02d" $((idx / step))) $(echo "$line" | cut -d',' -f1-4) $(echo "$line" | cut -d',' -f5)"
        fi
        ((idx++))
    done
    
    # 2. 随机选 10 个点
    echo "=== Extracting 10 random points ===" >&2
    tail -n +2 "$csv_file" | shuf | head -10 | while read -r line; do
        echo "random $(echo "$line" | cut -d',' -f1-4) $(echo "$line" | cut -d',' -f5)"
    done
}

# 获取所有点
all_points=$(extract_points)
if [[ -z "$all_points" ]]; then
    echo "Error: Could not extract points" >&2
    exit 1
fi

# Define ablation variants: name,csr,prefetch,simd,pruning,heap,flat_index
ablation_variants=(
    "baseline,0,0,0,0,0,0"
    "no_csr,1,0,0,0,0,0"
    "dynamic_struct,0,0,0,0,0,1"
    "no_prefetch,0,1,0,0,0,0"
    "no_simd,0,0,1,0,0,0"
)

# Function to parse metrics from a log file
parse_metrics() {
    local logfile="$1"
    local build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms

    build_internal_ms=$(grep -m1 "Index build time:" "$logfile" | awk -F": " '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
    build_local_ms=$(grep -m1 "build_time(local)" "$logfile" | awk -F"=" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
    queries_per_sec=$(grep -m1 "Queries per second" "$logfile" | awk -F":" '{print $2}' | tr -d '\n' || echo "")
    avg_recall=$(grep -m1 "Average recall@" "$logfile" | awk -F":" '{print $2}' | tr -d '\n' || echo "")
    avg_query_time_ms=$(grep -m1 "Average query time" "$logfile" | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
    avg_dists=$(grep -m1 "Average distance ops per query" "$logfile" | awk -F":" '{print $2}' | tr -d '\n' || echo "")
    last_q_dists=$(grep -m1 "Last query distance ops:" "$logfile" | awk -F":" '{print $2}' | tr -d '\n' || echo "")
    total_query_time_ms=$(grep -m1 "Total query time" "$logfile" | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")

    echo "$build_internal_ms,$build_local_ms,$queries_per_sec,$avg_recall,$avg_query_time_ms,$avg_dists,$last_q_dists,$total_query_time_ms"
}

# Run variant on a single point
run_variant_on_point() {
    local point_type="$1"      # "uniform_XX" or "random"
    local point_index="$2"     # sequence number
    local m="$3"
    local ml="$4"
    local efc="$5"
    local efs="$6"
    local recall_cached="$7"   # cached recall value
    local variant_spec="$8"
    
    IFS=',' read -r varname csr prefetch simd pruning heap flat_index <<< "$variant_spec"
    
    local point_name="${point_type}_m${m}_ml${ml}_efc${efc}_efs${efs}"
    local variant_dir="$OUT_DIR/${point_name}/${varname}"
    mkdir -p "$variant_dir"
    
    echo "[RUN] $point_name / $varname (cached_recall=$recall_cached)"
    
    # Clean logs
    rm -f Log/* 2>/dev/null || true
    
    for run_index in $(seq 1 $REPEATS); do
        local log_file_stdout="$variant_dir/run_stdout_${run_index}.log"
        ./hng --m "$m" --max_layer "$ml" --efc "$efc" --efs "$efs" --threads "$THREADS" \
            --ablate_csr "$csr" --ablate_prefetch "$prefetch" --ablate_simd "$simd" --ablate_pruning "$pruning" --ablate_heap "$heap" --ablate_flat_index "$flat_index" > "$log_file_stdout" 2>&1 || true
        
        # Find latest log file
        latest_log=$(ls -t Log 2>/dev/null | head -n1 || true)
        if [[ -n "$latest_log" ]]; then
            cp "Log/$latest_log" "$variant_dir/"
            harness_log="$variant_dir/$latest_log"
        else
            harness_log="$log_file_stdout"
        fi
        
        # Parse metrics
        metrics=$(parse_metrics "$harness_log")
        IFS=',' read -r build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms <<< "$metrics"
        
        # CSV record
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${varname}" "${point_type}" "${point_index}" "$m" "$ml" "$efc" "$efs" "$recall_cached" \
            "$csr" "$prefetch" "$simd" "$pruning" "$heap" "$flat_index" \
            "$build_internal_ms" "$build_local_ms" "$queries_per_sec" "$avg_recall" "$avg_query_time_ms" "$avg_dists" "$last_q_dists" "$total_query_time_ms" "$harness_log" "$run_index" >> "$CSV"
        
        echo "  ✓ $varname run $run_index: recall=$avg_recall, query_time=${avg_query_time_ms}ms"
    done
}

# Process all points
echo ""
echo "========== RUNNING ABLATION ON 20 POINTS =========="
point_counter=0
echo "$all_points" | while read -r point_type m ml efc efs recall_cached; do
    [[ -z "$m" ]] && continue
    ((point_counter++))
    
    echo ""
    echo "[$point_counter/20] Testing $point_type: m=$m ml=$ml efc=$efc efs=$efs (recall=$recall_cached)"
    
    for variant in "${ablation_variants[@]}"; do
        run_variant_on_point "$point_type" "$point_counter" "$m" "$ml" "$efc" "$efs" "$recall_cached" "$variant"
    done
done

echo ""
echo "✅ Ablation experiments completed!"
echo "Results saved to: $CSV"
echo "Experiment directory: $OUT_DIR"
echo "Total experiments: 20 points × 5 variants = 100 runs"

exit 0
