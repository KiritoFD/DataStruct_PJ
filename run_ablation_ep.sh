#!/usr/bin/env bash
set -euo pipefail

# run_ablation_ep.sh
# 简化的消融实验脚本 - 只测试自适应入口点优化
# Usage: ./run_ablation_ep.sh

WORKDIR=$(pwd)
BINARY=${BINARY:-"hng_hm"}

# 数据集路径
BASE_FILE=${BASE_FILE:-data_o/glove/base.txt}
QUERY_FILE=${QUERY_FILE:-data_o/glove/query.txt}
TRUTH_FILE=${TRUTH_FILE:-data_o/glove/truth.txt}

# HNSW 参数
K=${K:-10}
M=${M:-57}
MAX_LAYER=${MAX_LAYER:-7}
EFC=${EFC:-2000}
THREADS=${THREADS:-32}

# EFS 范围
HYPER_EFS_MIN=${HYPER_EFS_MIN:-100}
HYPER_EFS_MAX=${HYPER_EFS_MAX:-1200}
HYPER_EFS_STEP=${HYPER_EFS_STEP:-100}

# 重复次数
REPEATS=${REPEATS:-1}

# 绘图选项 - 使用原来的 plot_result.py
PYTHON=${PYTHON:-python3}
PLOT_SCRIPT=${PLOT_SCRIPT:-"$WORKDIR/plot_result.py"}
GENERATE_PLOTS=${GENERATE_PLOTS:-1}

# 结果目录
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="result-ablation-ep/ablation_${TS}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

# CSV 表头 - 与 plot_result.py 兼容的格式
echo "variant,csr,prefetch,simd,pruning,heap,flat_index,reorder,adaptive_ep,M,ML,EFC,EFS,build_time_internal_ms,build_time_local_ms,queries_per_sec,avg_recall,avg_query_time_ms,avg_dists,last_query_dists,total_query_time_ms,log_file,run_index" > "$CSV"

# 定义消融变体: name,adaptive_ep
# adaptive_ep=0: 启用自适应入口点 (优化版)
# adaptive_ep=1: 禁用自适应入口点 (消融/基线)
variants=(
  "with_adaptive_ep,0"
  "without_adaptive_ep,1"
)

# 生成 EFS 列表
efs_list=()
for (( e=HYPER_EFS_MIN; e<=HYPER_EFS_MAX; e+=HYPER_EFS_STEP )); do
  efs_list+=("$e")
done

echo "================================================"
echo "自适应入口点消融实验"
echo "================================================"
echo "数据集: $BASE_FILE"
echo "参数: M=$M, ML=$MAX_LAYER, EFC=$EFC"
echo "EFS 范围: ${HYPER_EFS_MIN} - ${HYPER_EFS_MAX} (步长 ${HYPER_EFS_STEP})"
echo "共 ${#efs_list[@]} 个 EFS 值 × ${#variants[@]} 个变体 = $((${#efs_list[@]} * ${#variants[@]})) 次实验"
echo "================================================"

# 解析日志文件获取指标
parse_metrics() {
  local logfile="$1"
  local build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms

  build_internal_ms=$(grep -m1 "Index build time:" "$logfile" 2>/dev/null | awk -F": " '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
  build_local_ms=$(grep -m1 "build_time(local)" "$logfile" 2>/dev/null | awk -F"=" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
  queries_per_sec=$(grep -m1 "Queries per second" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d '\n' || echo "")
  avg_recall=$(grep -m1 "Average recall@" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d '\n' || echo "")
  avg_query_time_ms=$(grep -m1 "Average query time" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")
  avg_dists=$(grep -m1 "Average distance ops per query" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d '\n' || echo "")
  last_q_dists=$(grep -m1 "Last query distance ops:" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d '\n' || echo "")
  total_query_time_ms=$(grep -m1 "Total query time" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d '\n' || echo "")

  echo "$build_internal_ms,$build_local_ms,$queries_per_sec,$avg_recall,$avg_query_time_ms,$avg_dists,$last_q_dists,$total_query_time_ms"
}

# 运行实验
run_count=0
total_runs=$((${#efs_list[@]} * ${#variants[@]} * REPEATS))

for set_EFS in "${efs_list[@]}"; do
  for variant_spec in "${variants[@]}"; do
    IFS=',' read -r varname adaptive_ep <<< "$variant_spec"
    
    for run_idx in $(seq 1 $REPEATS); do
      run_count=$((run_count + 1))
      echo "[$run_count/$total_runs] 运行: $varname, EFS=$set_EFS, run=$run_idx"
      
      # 创建日志目录
      log_dir="$OUT_DIR/EFS${set_EFS}/${varname}"
      mkdir -p "$log_dir"
      log_file="$log_dir/run_${run_idx}.log"
      
      # 运行实验 - 其他消融标志都设为0
      ./"$BINARY" --base "$BASE_FILE" --query "$QUERY_FILE" --truth "$TRUTH_FILE" --k "$K" \
        --m "$M" --max_layer "$MAX_LAYER" --efc "$EFC" --efs "$set_EFS" --threads "$THREADS" \
        --ablate_csr 0 --ablate_prefetch 0 --ablate_simd 0 \
        --ablate_pruning 0 --ablate_heap 0 \
        --ablate_adaptive_ep "$adaptive_ep" \
        > "$log_file" 2>&1 || {
          echo "[ERROR] 实验失败: $varname, EFS=$set_EFS"
          continue
        }
      
      # 解析结果
      metrics=$(parse_metrics "$log_file")
      IFS=',' read -r build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms <<< "$metrics"
      
      # 写入 CSV - 与 plot_result.py 兼容的格式
      # variant,csr,prefetch,simd,pruning,heap,flat_index,reorder,adaptive_ep,M,ML,EFC,EFS,...
      echo "$varname,0,0,0,0,0,0,0,$adaptive_ep,$M,$MAX_LAYER,$EFC,$set_EFS,$build_internal_ms,$build_local_ms,$queries_per_sec,$avg_recall,$avg_query_time_ms,$avg_dists,$last_q_dists,$total_query_time_ms,$log_file,$run_idx" >> "$CSV"
    done
  done
done

echo "================================================"
echo "实验完成！结果保存至: $CSV"
echo "================================================"

# 生成绘图
if [[ "${GENERATE_PLOTS}" -eq 1 ]]; then
  echo "生成绘图..."
  mkdir -p "$OUT_DIR/plots"
  
  if [[ -f "$PLOT_SCRIPT" ]]; then
    PLOT_TS=$(date +%Y%m%d_%H%M%S)
    
    # 生成完整图
    "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation_ep_full_${PLOT_TS}.png" || echo "绘图失败"
    
    # 生成不同recall范围的图
    "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation_ep_99_${PLOT_TS}.png" -u 0.99 || echo "绘图失败"
    "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation_ep_995_${PLOT_TS}.png" -d 0.99 -u 0.995 || echo "绘图失败"
    "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation_ep_995plus_${PLOT_TS}.png" -d 0.995 || echo "绘图失败"
    
    echo "绘图保存至: $OUT_DIR/plots/"
  else
    echo "绘图脚本不存在: $PLOT_SCRIPT"
    echo "请创建绘图脚本或设置 GENERATE_PLOTS=0 跳过绘图"
  fi
fi

echo "完成！"
exit 0
