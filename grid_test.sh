#!/usr/bin/env bash
# grid_test.sh - compile & run test over a grid of NUM_CENTROIDS, KMEANS_ITER, NPROBE

set -euo pipefail

# Edit these arrays for the grid you want to try
NUM_CENTROIDS_ARR=(
  # === 破坏性小值 (5组) - 低于常规最小值
  64 96 128 192 256
  
  # === 传统最优区加密 (15组) - 从历史数据提炼
  384 512 640 768 896 
  1024 1280 1536 2048 2560 
  3072 4096 5120 6144 8192
  
  # === 大规模区 (10组) - 覆盖你的原始范围
  10240 12288 14336 16384 18432 
  20480 24576 28672 32768 40960
  
  # === 疯狂大值 (10组) - 挑战内存极限
  49152 57344 65536 81920 
)
KMEANS_ITER_ARR=(6 16  32)
NPROBE_ARR=(
  # === 破坏性小值 (5组) - 低于有效阈值
  1 2 4 8 16
  
  # === 关键过渡区加密 (12组) - 召回率从0.95→0.99
  32 48 64 80 96 112 
  128 144 160 192 224 256
  
  # === 高性能区 (5组) - 低延迟高召回
  324 400 512 648 800
  
  # === 疯狂大值 (3组) - 接近全表扫描
  1024 2048 3000 4096 65535
)

# Other settings
SRC="MySolution.cpp"
TEST_SRC="test_solution.cpp"
BIN="test"
BIN_G="testg"                       # 新增：第二个可执行文件
RESULTS_DIR="results"
TIMEOUT_SEC=${TIMEOUT_SEC:-1800}  # allow override from env
CXX=${CXX:-g++}
CXXFLAGS="-O3 -std=c++17 -mavx -fopenmp -ffast-math"
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-0}  # 默认不 sleep

# Check if timeout is available
if ! command -v timeout &>/dev/null; then
  echo "Warning: 'timeout' command not found. Runs will not be killed on timeout."
  TIMEOUT_CMD=""
else
  TIMEOUT_CMD="timeout"
fi

mkdir -p "$RESULTS_DIR"

# ensure CSV summary exists with header
SUMMARY_CSV="${RESULTS_DIR}/summary.csv"
if [ ! -f "$SUMMARY_CSV" ]; then
  # 移除 STATUS 列
  echo "STAMP,NUM_CENTROIDS,KMEANS_ITER,NPROBE,ELAPSED_s,RECALL,AVG_QUERY_TIME_ms,INDEX_BUILD_TIME_s" > "$SUMMARY_CSV"
fi

# 新增：检查 testg 可执行文件并准备独立的 summary CSV
SUMMARY_CSV_G="${RESULTS_DIR}/summary_g.csv"
if [ ! -x "$BIN_G" ]; then
  echo "Warning: executable '$BIN_G' not found or not executable. Skipping testg runs."
  RUN_TESTG="0"
else
  RUN_TESTG="1"
  if [ ! -f "$SUMMARY_CSV_G" ]; then
    # 同样移除 STATUS 列
    echo "STAMP,NUM_CENTROIDS,KMEANS_ITER,NPROBE,ELAPSED_s,RECALL,AVG_QUERY_TIME_ms,INDEX_BUILD_TIME_s" > "$SUMMARY_CSV_G"
  fi
fi

echo "Grid test starting: $(date)"
echo "NUM_CENTROIDS: ${NUM_CENTROIDS_ARR[*]}"
echo "KMEANS_ITER: ${KMEANS_ITER_ARR[*]}"
echo "NPROBE: ${NPROBE_ARR[*]}"
echo "Timeout per run: ${TIMEOUT_SEC}s"
echo

# 不编译，直接验证可执行文件存在
if [ ! -x "$BIN" ]; then
  echo "Error: executable '$BIN' not found or not executable."
  echo "Please build the binary manually before running this script (e.g. g++ ... -o $BIN)."
  exit 1
fi

for ki in "${KMEANS_ITER_ARR[@]}"; do
  for np in "${NPROBE_ARR[@]}"; do
    for nc in "${NUM_CENTROIDS_ARR[@]}"; do
      stamp="c${nc}_i${ki}_p${np}"
      logfile="${RESULTS_DIR}/run_${stamp}.log"

      # 缓存检查：若在任一 summary CSV 中已存在该 stamp，则跳过运行
      if grep -q -F "${stamp}," "$SUMMARY_CSV" 2>/dev/null || ( [ -f "$SUMMARY_CSV_G" ] && grep -q -F "${stamp}," "$SUMMARY_CSV_G" 2>/dev/null ); then
        echo "=== Skipping ${stamp} (cached) ==="
        continue
      fi
      
      echo "=== Running ${stamp} (test) ==="
      echo "Run: --n ${nc}, --k ${ki}, --p ${np}"
      echo "Log: ${logfile}"

      echo "Running (timeout ${TIMEOUT_SEC}s)..."
      start_ts=$(date +%s)
      # run with timeout; pass parameters to binary as runtime args
      if [ -n "$TIMEOUT_CMD" ]; then
        if $TIMEOUT_CMD "${TIMEOUT_SEC}"s ./"$BIN" --n "${nc}" --k "${ki}" --p "${np}" &>> "$logfile"; then
          status="OK"
        else
          rc=$?
          if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
            status="TIMEOUT_KILLED"
            echo "[RUN] exited with $rc (timeout/kill)" >> "$logfile"
          else
            status="ERROR_${rc}"
            echo "[RUN] exited with $rc" >> "$logfile"
          fi
        fi
      else
        if ./"$BIN" --n "${nc}" --k "${ki}" --p "${np}" &>> "$logfile"; then
          status="OK"
        else
          rc=$?
          status="ERROR_${rc}"
          echo "[RUN] exited with $rc" >> "$logfile"
        fi
      fi
      end_ts=$(date +%s)
      elapsed=$((end_ts - start_ts))
      echo "Result: ${status}, elapsed ${elapsed}s"

      # Parse logfile for metrics (tolerant to missing lines)
      # Example lines in test logs:
      #   Average recall@10: 0.1234
      #   Average query time: 1.23 ms
      #   Index build time: 12 seconds

      recall_line=$(grep -m1 "Average recall@" "$logfile" 2>/dev/null || true)
      if [ -n "$recall_line" ]; then
        recall=$(echo "$recall_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+).*/\1/')
      else
        recall="NA"
      fi

      avg_query_line=$(grep -m1 "Average query time" "$logfile" 2>/dev/null || true)
      if [ -n "$avg_query_line" ]; then
        avg_query_time_ms=$(echo "$avg_query_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *ms.*/\1/')
      else
        avg_query_time_ms="NA"
      fi

      index_build_line=$(grep -m1 "Index build time" "$logfile" 2>/dev/null || true)
      if [ -n "$index_build_line" ]; then
        index_build_time_s=$(echo "$index_build_line" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *seconds.*/\1/')
      else
        index_build_time_s="NA"
      fi

      # Append CSV summary line (去掉 status 列)
      echo "${stamp},${nc},${ki},${np},${elapsed},${recall},${avg_query_time_ms},${index_build_time_s}" >> "$SUMMARY_CSV"

      # Also append a human-readable note to summary.txt for quick glance
      echo "NUM_CENTROIDS=${nc} KMEANS_ITER=${ki} NPROBE=${np} STATUS=${status} ELAPSED=${elapsed}s RECALL=${recall} AVG_QUERY_TIME_ms=${avg_query_time_ms} INDEX_BUILD_s=${index_build_time_s}" >> "${RESULTS_DIR}/summary.txt"

      echo "=== Finished ${stamp} (test) ==="
      echo

      # small pause to let system settle (make configurable)
      if [ "$SLEEP_BETWEEN_RUNS" -gt 0 ]; then
        sleep "$SLEEP_BETWEEN_RUNS"
      fi

      # 如果 testg 可用，则用相同参数运行 testg 并记录到独立的 CSV/log
      if [ "$RUN_TESTG" = "1" ]; then
        logfile_g="${RESULTS_DIR}/run_${stamp}_g.log"
        echo "=== Running ${stamp} (testg) ==="
        start_ts_g=$(date +%s)
        if [ -n "$TIMEOUT_CMD" ]; then
          if $TIMEOUT_CMD "${TIMEOUT_SEC}"s ./"$BIN_G" --n "${nc}" --k "${ki}" --p "${np}" &>> "$logfile_g"; then
            status_g="OK"
          else
            rc=$?
            if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
              status_g="TIMEOUT_KILLED"
              echo "[RUN] exited with $rc (timeout/kill)" >> "$logfile_g"
            else
              status_g="ERROR_${rc}"
              echo "[RUN] exited with $rc" >> "$logfile_g"
            fi
          fi
        else
          if ./"$BIN_G" --n "${nc}" --k "${ki}" --p "${np}" &>> "$logfile_g"; then
            status_g="OK"
          else
            rc=$?
            status_g="ERROR_${rc}"
            echo "[RUN] exited with $rc" >> "$logfile_g"
          fi
        fi
        end_ts_g=$(date +%s)
        elapsed_g=$((end_ts_g - start_ts_g))
        echo "Result (testg): ${status_g}, elapsed ${elapsed_g}s"

        # 解析 testg logfile（与 test 相同的解析策略）
        recall_line_g=$(grep -m1 "Average recall@" "$logfile_g" 2>/dev/null || true)
        if [ -n "$recall_line_g" ]; then
          recall_g=$(echo "$recall_line_g" | sed -E 's/.*: *([0-9]*\.?[0-9]+).*/\1/')
        else
          recall_g="NA"
        fi
        avg_query_line_g=$(grep -m1 "Average query time" "$logfile_g" 2>/dev/null || true)
        if [ -n "$avg_query_line_g" ]; then
          avg_query_time_ms_g=$(echo "$avg_query_line_g" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *ms.*/\1/')
        else
          avg_query_time_ms_g="NA"
        fi
        index_build_line_g=$(grep -m1 "Index build time" "$logfile_g" 2>/dev/null || true)
        if [ -n "$index_build_line_g" ]; then
          index_build_time_s_g=$(echo "$index_build_line_g" | sed -E 's/.*: *([0-9]*\.?[0-9]+) *seconds.*/\1/')
        else
          index_build_time_s_g="NA"
        fi

        # 将 testg 的结果追加到独立 CSV 与 summary 文件（去掉 status 列）
        echo "${stamp},${nc},${ki},${np},${elapsed_g},${recall_g},${avg_query_time_ms_g},${index_build_time_s_g}" >> "$SUMMARY_CSV_G"
        echo "NUM_CENTROIDS=${nc} KMEANS_ITER=${ki} NPROBE=${np} STATUS=${status_g} ELAPSED=${elapsed_g}s RECALL=${recall_g} AVG_QUERY_TIME_ms=${avg_query_time_ms_g} INDEX_BUILD_s=${index_build_time_s_g}" >> "${RESULTS_DIR}/summary_g.txt"
        echo "=== Finished ${stamp} (testg) ==="
        echo
      fi

      # 跳出所有循环（break 3）如果召回率低于0.98
      awk_recall=$(awk "BEGIN {print ($recall < 0.98) ? 1 : 0}")
      if [ "$awk_recall" -eq 1 ]; then
        echo "Recall is below threshold, stopping early."
        break 1
      fi
    done
  done
done

echo "Grid test finished: $(date)"
echo "CSV summary in ${SUMMARY_CSV}"
echo "Text summary in ${RESULTS_DIR}/summary.txt"