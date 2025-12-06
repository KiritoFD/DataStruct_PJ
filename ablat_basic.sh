#!/usr/bin/env bash

# POSIX-safe settings: use -e and -u always; enable pipefail only when running under bash
set -e
set -u
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail
fi

# Simplified runner for HNSW basic experiments
# Usage: ./ablat_basic.sh [--base base_file] [--query query_file] [--truth truth_file] [--k K]
#        [--M M] [--ML max_layer] [--EFC ef_construction] [--EFS ef_search]
#        [--threads THREADS] [--repeats REPEATS] [--parallel PARALLEL_JOBS]

WORKDIR=$(pwd)
BINARY=${BINARY:-"./basic"}

# Default dataset paths
BASE_FILE=${BASE_FILE:-data_o/glove/base.txt}
QUERY_FILE=${QUERY_FILE:-data_o/glove/query.txt}
TRUTH_FILE=${TRUTH_FILE:-data_o/glove/truth.txt}
K=${K:-10}

# HNSW params (single config or you can set HYPER_SETS)
M=${M:-59}
MAX_LAYER=${MAX_LAYER:-7}
EFC=${EFC:-763}
EFS=${EFS:-432}

THREADS=${THREADS:-32}
REPEATS=${REPEATS:-1}
PARALLEL_JOBS=${PARALLEL_JOBS:-1}

# Output
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="basic_results/run_${TS}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

# CSV header: concise set of columns + params_match
echo "M,ML,EFC,EFS,run_index,build_time_ms,queries_per_sec,avg_recall,avg_query_time_ms,avg_dists,last_query_dists,total_query_time_ms,params_match,log_file" > "$CSV"

# Parse metrics from log file
parse_metrics() {
  local logfile="$1"
  local build_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms

  build_ms=$(grep -m1 "Index build time:" "$logfile" 2>/dev/null | awk -F": " '{print $2}' | sed 's/ ms//g' || echo "")
  queries_per_sec=$(grep -m1 "Queries per second" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  avg_recall=$(grep -m1 "Average recall@" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  avg_query_time_ms=$(grep -m1 "Average query time" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d ' ' || echo "")
  # Look for the exact line text: "Average distance ops per query"
  avg_dists=$(grep -m1 "Average distance ops per query" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  last_q_dists=$(grep -m1 "Last query distance ops:" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  total_query_time_ms=$(grep -m1 "Total query time" "$logfile" 2>/dev/null | awk -F":" '{print $2}' | sed 's/ ms//g' | tr -d ' ' || echo "")

  echo "$build_ms,$queries_per_sec,$avg_recall,$avg_query_time_ms,$avg_dists,$last_q_dists,$total_query_time_ms"
}

# Parse effective params from log file (Configuration (effective): lines)
parse_effective_params() {
  local logfile="$1"
  local eff_M eff_ML eff_EFC eff_EFS eff_threads
  eff_M=$(grep -m1 "Configuration (effective):" -A5 "$logfile" 2>/dev/null | grep "M:" | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  eff_ML=$(grep -m1 "Configuration (effective):" -A5 "$logfile" 2>/dev/null | grep "Max layer:" | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  eff_EFC=$(grep -m1 "Configuration (effective):" -A5 "$logfile" 2>/dev/null | grep "EFC:" | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  eff_EFS=$(grep -m1 "Configuration (effective):" -A5 "$logfile" 2>/dev/null | grep "EFS:" | awk -F":" '{print $2}' | tr -d ' ' || echo "")
  echo "$eff_M,$eff_ML,$eff_EFC,$eff_EFS"
}

# Single-run launcher
run_single() {
  local set_M=$1
  local set_ML=$2
  local set_EFC=$3
  local set_EFS=$4
  local run_index=$5
  local debug_flag=${DEBUG_FLAG:-1}   # default to debug on so we can parse effective params
  local log_file="$OUT_DIR/run_M${set_M}_ML${set_ML}_EFC${set_EFC}_EFS${set_EFS}_run${run_index}.log"

  "$BINARY" --base "$BASE_FILE" --query "$QUERY_FILE" --truth "$TRUTH_FILE" --k "$K" \
    --m "$set_M" --max_layer "$set_ML" --efc "$set_EFC" --efs "$set_EFS" --threads "$THREADS" \
    --debug "$debug_flag" \
    > "$log_file" 2>&1 || echo "[RUN ERROR] exit status from binary, see $log_file" >&2

  # parse metrics and append CSV
  IFS=',' read -r build_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms <<< "$(parse_metrics "$log_file")"

  # verify effective params vs requested
  IFS=',' read -r effM effML effEFC effEFS <<< "$(parse_effective_params "$log_file")"
  params_match="match"
  if [[ -n "$effM" && "$set_M" -gt 0 && "$effM" != "$set_M" ]]; then params_match="mismatch"; fi
  if [[ -n "$effML" && "$set_ML" -gt 0 && "$effML" != "$set_ML" ]]; then params_match="mismatch"; fi
  if [[ -n "$effEFC" && "$set_EFC" -gt 0 && "$effEFC" != "$set_EFC" ]]; then params_match="mismatch"; fi
  if [[ -n "$effEFS" && "$set_EFS" -gt 0 && "$effEFS" != "$set_EFS" ]]; then params_match="mismatch"; fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$set_M" "$set_ML" "$set_EFC" "$set_EFS" "$run_index" \
    "$build_ms" "$queries_per_sec" "$avg_recall" "$avg_query_time_ms" "$avg_dists" "$last_q_dists" "$total_query_time_ms" "$params_match" "$log_file" \
    >> "$CSV"
}

# HYPER_EFS configuration: specify a list via min,max,step (integers)
# Example: export HYPER_EFS_MIN=425 HYPER_EFS_MAX=1000 HYPER_EFS_STEP=25
HYPER_EFS_MIN=${HYPER_EFS_MIN:-2000}
HYPER_EFS_MAX=${HYPER_EFS_MAX:-10000}
HYPER_EFS_STEP=${HYPER_EFS_STEP:-1000}


# ---------------------------------------------------------------------
# Generate HYPER_EFS_LIST from min/max/step if provided
# ---------------------------------------------------------------------
if [[ -n "${HYPER_EFS_MIN}" && -n "${HYPER_EFS_MAX}" && -n "${HYPER_EFS_STEP}" ]]; then
  # Validate ints
  if ! [[ "${HYPER_EFS_MIN}" =~ ^-?[0-9]+$ && "${HYPER_EFS_MAX}" =~ ^-?[0-9]+$ && "${HYPER_EFS_STEP}" =~ ^-?[0-9]+$ ]]; then
    echo "Invalid HYPER_EFS_MIN/HYPER_EFS_MAX/HYPER_EFS_STEP: must be integers"
    exit 1
  fi
  if (( HYPER_EFS_STEP <= 0 )); then
    echo "HYPER_EFS_STEP must be a positive integer"
    exit 1
  fi
  if (( HYPER_EFS_MIN > HYPER_EFS_MAX )); then
    echo "HYPER_EFS_MIN must be <= HYPER_EFS_MAX"
    exit 1
  fi

  echo "Generating HYPER_EFS_LIST from min=${HYPER_EFS_MIN} max=${HYPER_EFS_MAX} step=${HYPER_EFS_STEP}"
  tmp_list=""
  for (( e=HYPER_EFS_MIN; e<=HYPER_EFS_MAX; e+=HYPER_EFS_STEP )); do
    tmp_list+="${e},"
  done
  # remove trailing comma
  HYPER_EFS_LIST="${tmp_list%,}"
  echo "Generated HYPER_EFS_LIST=${HYPER_EFS_LIST}"
fi

# ---------------------------------------------------------------------
# Build parameter sets: M,ML,EFC fixed; EFS vary across HYPER_EFS_LIST (or HYPER_SETS / defaults)
# ---------------------------------------------------------------------
param_sets=()

# If explicit HYPER_SETS provided, use them (each line M,ML,EFC,EFS)
if [[ -n "${HYPER_SETS:-}" && -n "$(echo "$HYPER_SETS" | sed -n '1p')" ]]; then
  while IFS= read -r line; do
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$line" || "${line:0:1}" == "#" ]] && continue
    IFS=',' read -r m ml efc efs <<< "$line"
    param_sets+=("$m,$ml,$efc,$efs")
  done <<< "$HYPER_SETS"
elif [[ -n "${HYPER_EFS_LIST}" ]]; then
  # Construct param sets by combining fixed M, MAX_LAYER, EFC with each EFS
  IFS=',' read -r -a efs_array <<< "$HYPER_EFS_LIST"
  for e in "${efs_array[@]}"; do
    e="$(echo "$e" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ -z "$e" ]] && continue
    param_sets+=("${M},${MAX_LAYER},${EFC},${e}")
  done
else
  # default single config
  param_sets+=("${M},${MAX_LAYER},${EFC},${EFS}")
fi

echo "Found ${#param_sets[@]} parameter set(s) to run. Running $REPEATS repeats each with concurrency $PARALLEL_JOBS."

# Run tasks in parallel, limit concurrency
pids=()
for param in "${param_sets[@]}"; do
  IFS=',' read -r set_M set_ML set_EFC set_EFS <<< "$param"
  for (( r=1; r<=REPEATS; r++ )); do
    if [[ "$PARALLEL_JOBS" -gt 1 ]]; then
      run_single "$set_M" "$set_ML" "$set_EFC" "$set_EFS" "$r" &
      pids+=($!)
      # limit background jobs
      while [[ ${#pids[@]} -ge "$PARALLEL_JOBS" ]]; do
        sleep 1
        newpids=()
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            newpids+=("$pid")
          fi
        done
        pids=("${newpids[@]}")
      done
    else
      run_single "$set_M" "$set_ML" "$set_EFC" "$set_EFS" "$r"
    fi
  done
done

# Wait for remaining background jobs
if [[ ${#pids[@]} -gt 0 ]]; then
  echo "Waiting for remaining ${#pids[@]} tasks..."
  for pid in "${pids[@]}"; do wait "$pid" || true; done
fi

echo "All runs finished. Results CSV: $CSV"

# Simple summary: avg_dists column is 10, avg_recall col 8, queries_per_sec col 7
echo "Averages summary:"
awk -F, 'NR>1 {sum_dist += ($10==""?0:$10); sum_recall += ($8==""?0:$8); sum_qps += ($7==""?0:$7); n++}
END { if (n>0) printf("avg_dists=%.6f avg_recall=%.6f avg_qps=%.6f\n", sum_dist/n, sum_recall/n, sum_qps/n); else print "No runs" }' "$CSV"

exit 0
