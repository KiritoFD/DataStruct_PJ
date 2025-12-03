#!/usr/bin/env bash
set -euo pipefail

# run_ablation.sh
# Compile and run ablation experiments for MySolution + test_hn_g
# Usage: ./run_ablation.sh [--base base_file] [--query query_file] [--truth truth_file] [--k K] [--variants VARIANT1,VARIANT2,...]

WORKDIR=$(pwd)
BINARY="hng1"

# Default dataset paths
BASE_FILE=${BASE_FILE:-data_o/glove/base.txt}
QUERY_FILE=${QUERY_FILE:-data_o/glove/query.txt}
TRUTH_FILE=${TRUTH_FILE:-data_o/glove/truth.txt}
K=${K:-10}
M=${M:-51}
MAX_LAYER=${MAX_LAYER:-7}
EFC=${EFC:-648}
EFS=${EFS:-432}
THREADS=${THREADS:-32}

# repetitions per variant (set via env var REPEATS, default to 1)
REPEATS=${REPEATS:-1}

# New: Accept multiple hyperparameter sets via env var HYPER_SETS or default to current values
# Format: each line: "M,ML,EFC,EFS" (K is constant = 10)
HYPER_SETS=${HYPER_SETS:-$'
51,7,648,648
51,7,648,660
51,7,648,680
'}

# Results dir
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/ablation_${TS}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

# CSV header
# Add hyperparameters and flat_index and reorder columns
echo "variant,csr,prefetch,simd,pruning,heap,flat_index,reorder,M,ML,EFC,EFS,build_time_internal_ms,build_time_local_ms,queries_per_sec,avg_recall,avg_query_time_ms,avg_dists,last_query_dists,total_query_time_ms,log_file,run_index" > "$CSV"


# define variants: name,csr,prefetch,simd,pruning,heap,flat_index,reorder
# 说明：
#   - 0 = 启用该优化 (baseline)
#   - 1 = 禁用该优化 (ablation)
# 注意：flat_index=1 时强制使用动态结构，此时 csr 和 reorder 不适用于查询
variants=(
  "baseline,0,0,0,0,0,0,0"
  "no_simd,0,0,1,0,0,0,0"
  "no_pruning,0,0,0,1,0,0,0"
  "heap,0,0,0,0,1,0,0"
  "dynamic_struct,0,0,0,0,0,1,0"
  "no_reorder,0,0,0,0,0,0,1"
  "dynamic_with_opts,0,0,0,0,0,1,0"
)

# function to parse metrics from a log file
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

# run variant
run_variant() {
  local variant_spec="$1"
  local param_dir="$2"        # New: directory for this hyperparameter set
  local set_M="$3"            # New: M for this param set
  local set_ML="$4"
  local set_EFC="$5"
  local set_EFS="$6"

  IFS=',' read -r varname csr prefetch simd pruning heap flat_index reorder <<< "$variant_spec"
  echo "Running variant: $varname (csr=$csr prefetch=$prefetch simd=$simd pruning=$pruning heap=$heap flat_index=$flat_index reorder=$reorder) on M=$set_M ML=$set_ML EFC=$set_EFC EFS=$set_EFS"

  local vdir="$param_dir/$varname"
  mkdir -p "$vdir"

  # Do not remove Log/* - preserve all harness logs
  local pre_logs=""
  if [[ -d Log ]]; then pre_logs=$(ls -1 Log); fi

  for run_index in $(seq 1 $REPEATS); do
    local log_file_stdout="$vdir/run_stdout_${run_index}.log"
    ./hng1 --base "$BASE_FILE" --query "$QUERY_FILE" --truth "$TRUTH_FILE" --k "$K" --m "$set_M" --max_layer "$set_ML" --efc "$set_EFC" --efs "$set_EFS" --threads "$THREADS" \
      --ablate_csr "$csr" --ablate_prefetch "$prefetch" --ablate_simd "$simd" --ablate_pruning "$pruning" --ablate_heap "$heap" --ablate_flat_index "$flat_index" --ablate_reorder "$reorder" > "$log_file_stdout" 2>&1

    # detect post run logs
    local post_logs=""
    if [[ -d Log ]]; then post_logs=$(ls -1 Log); fi

    local harness_log="$log_file_stdout"
    if [[ -n "$post_logs" ]]; then
      while IFS= read -r file; do
        if ! grep -Fxq "$file" <<< "$pre_logs"; then
          cp "Log/$file" "$vdir/"
          harness_log="$vdir/$file"
        fi
      done <<< "$post_logs"
    fi

    # parse metrics
    metrics=$(parse_metrics "$harness_log")
    IFS=',' read -r build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms <<< "$metrics"

    # CSV record includes hyperparameters and flat_index and reorder
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$varname" "$csr" "$prefetch" "$simd" "$pruning" "$heap" "$flat_index" "$reorder" \
      "$set_M" "$set_ML" "$set_EFC" "$set_EFS" \
      "$build_internal_ms" "$build_local_ms" "$queries_per_sec" "$avg_recall" "$avg_query_time_ms" "$avg_dists" "$last_q_dists" "$total_query_time_ms" "$harness_log" "$run_index" >> "$CSV"

    echo "Saved log & metrics for $varname (run $run_index) in $vdir"

    if [[ -d Log ]]; then pre_logs=$(ls -1 Log); else pre_logs=""; fi
  done
}

# Main: loop over hyperparameter sets (now read HYPER_SETS as lines)
# mapfile safely reads lines into an array
mapfile -t param_sets <<< "$HYPER_SETS"

# Trim and filter out empty/commented lines
clean_param_sets=()
for p in "${param_sets[@]}"; do
  # Trim whitespace
  ps="$(echo "$p" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  # Skip empty or commented lines
  if [[ -z "$ps" || "${ps:0:1}" == "#" ]]; then
    continue
  fi
  clean_param_sets+=("$ps")
done

if [[ ${#clean_param_sets[@]} -eq 0 ]]; then
  echo "No hyperparameter sets to run (HYPER_SETS is empty). Exiting."
  exit 1
fi

echo "Parsed ${#clean_param_sets[@]} hyperparameter sets to run."

for param_set in "${clean_param_sets[@]}"; do
  # parse comma list M,ML,EFC,EFS
  IFS=',' read -r set_M set_ML set_EFC set_EFS <<< "$param_set"
  if [[ -z "$set_M" || -z "$set_ML" || -z "$set_EFC" || -z "$set_EFS" ]]; then
    echo "Invalid param set: $param_set, skipping"
    continue
  fi

  param_dir="$OUT_DIR/params_M${set_M}_ML${set_ML}_EFC${set_EFC}_EFS${set_EFS}"
  mkdir -p "$param_dir"

  echo "==========================================="
  echo "Running experiments for M=$set_M ML=$set_ML EFC=$set_EFC EFS=$set_EFS (K=$K)"
  echo "Logs and outputs under: $param_dir"
  echo "==========================================="

  for v in "${variants[@]}"; do
    run_variant "$v" "$param_dir" "$set_M" "$set_ML" "$set_EFC" "$set_EFS"
  done
done

echo "All experiments done. Results saved to: $CSV"
echo "List of run directories:"
ls -1 "$OUT_DIR"

exit 0
