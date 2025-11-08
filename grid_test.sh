#!/usr/bin/env bash
# grid_test.sh - compile & run test over a grid of NUM_CENTROIDS, KMEANS_ITER, NPROBE

set -euo pipefail

# Edit these arrays for the grid you want to try
NUM_CENTROIDS_ARR=(1024)
KMEANS_ITER_ARR=(4 6 8 )
NPROBE_ARR=(288 300 324 400)

# Other settings
SRC="MySolution.cpp"
TEST_SRC="test_solution.cpp"
BIN="test"
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
  echo "STAMP,NUM_CENTROIDS,KMEANS_ITER,NPROBE,STATUS,ELAPSED_s,RECALL,AVG_QUERY_TIME_ms,INDEX_BUILD_TIME_s" > "$SUMMARY_CSV"
fi

echo "Grid test starting: $(date)"
echo "NUM_CENTROIDS: ${NUM_CENTROIDS_ARR[*]}"
echo "KMEANS_ITER: ${KMEANS_ITER_ARR[*]}"
echo "NPROBE: ${NPROBE_ARR[*]}"
echo "Timeout per run: ${TIMEOUT_SEC}s"
echo


for ki in "${KMEANS_ITER_ARR[@]}"; do
  for np in "${NPROBE_ARR[@]}"; do
    for nc in "${NUM_CENTROIDS_ARR[@]}"; do
      stamp="c${nc}_i${ki}_p${np}"
      logfile="${RESULTS_DIR}/run_${stamp}.log"
      buildlog="${logfile}.build"
      echo "=== Running ${stamp} ==="
      echo "Compile: NUM_CENTROIDS=${nc}, KMEANS_ITER=${ki}, NPROBE=${np}"
      echo "Log: ${logfile}"
      # compile with macro overrides (fix macro names to match C++ code)
      echo "Compiling..."
      if ! $CXX $CXXFLAGS -DNUM_CENTROID=${nc} -DKMEAN_ITER=${ki} -DNPROB=${np} "$SRC" "$TEST_SRC" -o "$BIN" 2>&1 | tee "$buildlog"; then
        echo "Compile failed for ${stamp}, see ${buildlog}"
        status="COMPILE_FAIL"
        elapsed=0
        recall="NA"
        avg_query_time_ms="NA"
        index_build_time_s="NA"
        echo "${stamp},${nc},${ki},${np},${status},${elapsed},${recall},${avg_query_time_ms},${index_build_time_s}" >> "$SUMMARY_CSV"
        continue
      fi

      echo "Running (timeout ${TIMEOUT_SEC}s)..."
      start_ts=$(date +%s)
      # run with timeout; capture stdout/stderr to logfile
      if [ -n "$TIMEOUT_CMD" ]; then
        if $TIMEOUT_CMD "${TIMEOUT_SEC}"s ./"$BIN" &>> "$logfile"; then
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
        if ./"$BIN" &>> "$logfile"; then
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

      # Append CSV summary line
      echo "${stamp},${nc},${ki},${np},${status},${elapsed},${recall},${avg_query_time_ms},${index_build_time_s}" >> "$SUMMARY_CSV"

      # Also append a human-readable note to summary.txt for quick glance
      echo "NUM_CENTROIDS=${nc} KMEANS_ITER=${ki} NPROBE=${np} STATUS=${status} ELAPSED=${elapsed}s RECALL=${recall} AVG_QUERY_TIME_ms=${avg_query_time_ms} INDEX_BUILD_s=${index_build_time_s}" >> "${RESULTS_DIR}/summary.txt"

      echo "=== Finished ${stamp} ==="
      echo
      # small pause to let system settle (make configurable)
      if [ "$SLEEP_BETWEEN_RUNS" -gt 0 ]; then
        sleep "$SLEEP_BETWEEN_RUNS"
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