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
M=${M:-80}
MAX_LAYER=${MAX_LAYER:-11}
EFC=${EFC:-1301}
EFS=${EFS:-432}
THREADS=${THREADS:-32}

# repetitions per variant (set via env var REPEATS, default to 1)
REPEATS=${REPEATS:-1}

# Parallel jobs control: set via env var PARALLEL_JOBS (default 8). If 0 or 1 -> no parallelism.
PARALLEL_JOBS=${PARALLEL_JOBS:-1}
# Plotting options (set GENERATE_PLOTS=0 to skip)
PYTHON=${PYTHON:-python3}
# Default to the script in the current working directory for compatibility on WSL / Unix
PLOT_SCRIPT=${PLOT_SCRIPT:-"$WORKDIR/plot_result.py"}
GENERATE_PLOTS=${GENERATE_PLOTS:-0}
# Excludes for the three ablation plots (default exclude no_csr and nothing)
PLOT_EXCLUDES=${PLOT_EXCLUDES:-"no_csr,nothing"}

# New: Accept multiple hyperparameter sets via env var HYPER_SETS or default to current values
# Format: each line: "M,ML,EFC,EFS" (K is constant = 10)
HYPER_SETS=${HYPER_SETS:-$'
'}



# New: Allow specifying an EFS range via min/max/step (integers).
# If provided, we generate HYPER_EFS_LIST from this range and prefer it to any existing HYPER_EFS_LIST.
# Example: HYPER_EFS_MIN=425 HYPER_EFS_MAX=1000 HYPER_EFS_STEP=25
HYPER_EFS_MIN=${HYPER_EFS_MIN:-300}
HYPER_EFS_MAX=${HYPER_EFS_MAX:-1500}
HYPER_EFS_STEP=${HYPER_EFS_STEP:-20}

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

# New: If HYPER_EFS_LIST provided, generate HYPER_SETS by combining current M, MAX_LAYER, EFC with the given EFS values.
if [[ -n "${HYPER_EFS_LIST}" ]]; then
  echo "Generating param sets from HYPER_EFS_LIST: ${HYPER_EFS_LIST} with fixed M=${M}, ML=${MAX_LAYER}, EFC=${EFC}"
  HYPER_SETS=""
  # Convert comma separated to array safely
  IFS=',' read -r -a efs_arr <<< "$HYPER_EFS_LIST"
  for e in "${efs_arr[@]}"; do
    # Trim whitespace
    e="$(echo "$e" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ -n "$e" ]]; then
      HYPER_SETS+="$M,$MAX_LAYER,$EFC,$e"$'\n'
    fi
  done
fi

# Results dir
TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/ablation_${TS}"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/results.csv"

# CSV header
echo "variant,csr,prefetch,simd,pruning,heap,flat_index,reorder,M,ML,EFC,EFS,build_time_internal_ms,build_time_local_ms,queries_per_sec,avg_recall,avg_query_time_ms,avg_dists,last_query_dists,total_query_time_ms,log_file,run_index" > "$CSV"

# define variants: name,csr,prefetch,simd,pruning,heap,flat_index,reorder
# 说明：
#   - 0 = 启用该优化 (baseline)
#   - 1 = 禁用该优化 (ablation)
# 注意：flat_index=1 时强制使用动态结构，此时 csr 和 reorder 不适用于查询
variants=(
  "full,0,0,0,0,0,0,0"
  "no_csr,1,0,0,0,0,0,0"
  "no_prefetch,0,1,0,0,0,0,0"
  "no_simd,0,0,1,0,0,0,0"
  "no_pruning,0,0,0,1,0,0,0"
  "heap,0,0,0,0,1,0,0"
  "dynamic_struct,0,0,0,0,0,1,0"
  "no_reorder,0,0,0,0,0,0,1"
  "dynamic_with_opts,0,0,0,0,0,1,0"
  "nothing,1,1,1,1,1,1,1"
)


# function to parse metrics from a log file
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

# run variant - only runs the experiment and saves logs, does NOT write CSV
run_variant() {
  local variant_spec="$1"
  local param_dir="$2"
  local set_M="$3"
  local set_ML="$4"
  local set_EFC="$5"
  local set_EFS="$6"

  IFS=',' read -r varname csr prefetch simd pruning heap flat_index reorder <<< "$variant_spec"
  echo "[START] variant=$varname M=$set_M ML=$set_ML EFC=$set_EFC EFS=$set_EFS"

  local vdir="$param_dir/$varname"
  mkdir -p "$vdir"

  for run_index in $(seq 1 $REPEATS); do
    local log_file_stdout="$vdir/run_stdout_${run_index}.log"
    
    # Save metadata for later CSV extraction
    local meta_file="$vdir/meta_${run_index}.txt"
    echo "variant_spec=$variant_spec" > "$meta_file"
    echo "set_M=$set_M" >> "$meta_file"
    echo "set_ML=$set_ML" >> "$meta_file"
    echo "set_EFC=$set_EFC" >> "$meta_file"
    echo "set_EFS=$set_EFS" >> "$meta_file"
    echo "run_index=$run_index" >> "$meta_file"
    echo "log_file=$log_file_stdout" >> "$meta_file"

    ./hng1 --base "$BASE_FILE" --query "$QUERY_FILE" --truth "$TRUTH_FILE" --k "$K" \
      --m "$set_M" --max_layer "$set_ML" --efc "$set_EFC" --efs "$set_EFS" --threads "$THREADS" \
      --ablate_csr "$csr" --ablate_prefetch "$prefetch" --ablate_simd "$simd" \
      --ablate_pruning "$pruning" --ablate_heap "$heap" --ablate_flat_index "$flat_index" \
      --ablate_reorder "$reorder" > "$log_file_stdout" 2>&1 || {
        echo "[ERROR] variant=$varname run=$run_index failed"
        echo "status=failed" >> "$meta_file"
      }

    echo "status=done" >> "$meta_file"
  done

  echo "[DONE] variant=$varname M=$set_M ML=$set_ML EFC=$set_EFC EFS=$set_EFS"
}

# Export functions and variables for subshells
export -f run_variant parse_metrics
export BASE_FILE QUERY_FILE TRUTH_FILE K THREADS REPEATS

# Main: loop over hyperparameter sets
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
echo "Parallel jobs limit: ${PARALLEL_JOBS}"

# Collect all tasks to run
tasks=()
for param_set in "${clean_param_sets[@]}"; do
  IFS=',' read -r set_M set_ML set_EFC set_EFS <<< "$param_set"
  if [[ -z "$set_M" || -z "$set_ML" || -z "$set_EFC" || -z "$set_EFS" ]]; then
    echo "Invalid param set: $param_set, skipping"
    continue
  fi

  param_dir="$OUT_DIR/params_M${set_M}_ML${set_ML}_EFC${set_EFC}_EFS${set_EFS}"
  mkdir -p "$param_dir"

  for v in "${variants[@]}"; do
    # Store task as: variant_spec|param_dir|M|ML|EFC|EFS
    tasks+=("$v|$param_dir|$set_M|$set_ML|$set_EFC|$set_EFS")
  done
done

echo "Total tasks to run: ${#tasks[@]}"

# Run tasks in parallel
pids=()
for task in "${tasks[@]}"; do
  IFS='|' read -r variant_spec param_dir set_M set_ML set_EFC set_EFS <<< "$task"

  if [[ "${PARALLEL_JOBS}" -gt 1 ]]; then
    # Start in background
    run_variant "$variant_spec" "$param_dir" "$set_M" "$set_ML" "$set_EFC" "$set_EFS" &
    pids+=($!)

    # Limit concurrent jobs
    while [[ ${#pids[@]} -ge ${PARALLEL_JOBS} ]]; do
      # Wait for any job to finish
      wait -n 2>/dev/null || wait "${pids[0]}" || true
      # Prune finished PIDs
      newpids=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          newpids+=("$pid")
        fi
      done
      pids=("${newpids[@]}")
    done
  else
    # Sequential
    run_variant "$variant_spec" "$param_dir" "$set_M" "$set_ML" "$set_EFC" "$set_EFS"
  fi
done

# Wait for all remaining jobs
if [[ ${#pids[@]} -gt 0 ]]; then
  echo "Waiting for remaining ${#pids[@]} background jobs..."
  wait
fi

echo "==========================================="
echo "All experiments finished. Collecting CSV..."
echo "==========================================="

# Collect results from all meta files into CSV
find "$OUT_DIR" -name "meta_*.txt" | sort | while read -r meta_file; do
  # Read metadata
  source "$meta_file"

  # Skip failed runs
  if [[ "${status:-}" == "failed" ]]; then
    echo "Skipping failed: $meta_file"
    continue
  fi

  # Parse variant_spec
  IFS=',' read -r varname csr prefetch simd pruning heap flat_index reorder <<< "$variant_spec"

  # Parse metrics from log
  if [[ -f "$log_file" ]]; then
    metrics=$(parse_metrics "$log_file")
    IFS=',' read -r build_internal_ms build_local_ms queries_per_sec avg_recall avg_query_time_ms avg_dists last_q_dists total_query_time_ms <<< "$metrics"
  else
    build_internal_ms="" build_local_ms="" queries_per_sec="" avg_recall=""
    avg_query_time_ms="" avg_dists="" last_q_dists="" total_query_time_ms=""
  fi

  # Write CSV line
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$varname" "$csr" "$prefetch" "$simd" "$pruning" "$heap" "$flat_index" "$reorder" \
    "$set_M" "$set_ML" "$set_EFC" "$set_EFS" \
    "$build_internal_ms" "$build_local_ms" "$queries_per_sec" "$avg_recall" \
    "$avg_query_time_ms" "$avg_dists" "$last_q_dists" "$total_query_time_ms" \
    "$log_file" "$run_index" >> "$CSV"

done

echo "All experiments done. Results saved to: $CSV"
echo "List of run directories:"
ls -1 "$OUT_DIR"

generate_plots() {
  if [[ "${GENERATE_PLOTS:-1}" -eq 0 ]]; then
    echo "Skipping generated plots (GENERATE_PLOTS=0)."
    return
  fi
  
  mkdir -p "$OUT_DIR/plots"
  echo "Generating default plots in $OUT_DIR/plots ..."
  
  # Use a timestamp for plot filenames
  PLOT_TS=$(date +%Y%m%d_%H%M%S)
  
  # Full plot (no excludes)
  echo " - Full plot (all variants)"
  "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/full_${PLOT_TS}.png" || echo "Failed to generate full plot"
  
  # Helper to append exclude flags
  IFS=',' read -r -a excl_arr <<< "$PLOT_EXCLUDES"
  excl_flags=()
  for e in "${excl_arr[@]}"; do
    if [[ -n "$e" ]]; then
      excl_flags+=("--exclude" "$e")
    fi
  done
  
  # Plot 1: up to 0.99
  echo " - ablation-98-99 (upper <= 0.99), excluding: ${PLOT_EXCLUDES}"
  "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation-98-99_${PLOT_TS}.png" -u 0.99 "${excl_flags[@]}" || echo "Failed to generate ablation-98-99"
  
  # Plot 2: [0.993, 0.9955]
  echo " - ablation-9935-9955 (0.993..0.9955), excluding: ${PLOT_EXCLUDES}"
  "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation-9935-9955_${PLOT_TS}.png" -d 0.993 -u 0.9955 "${excl_flags[@]}" || echo "Failed to generate ablation-9935-9955"
  
  # Plot 3: [0.990, 0.9935] (we preserve user's ordering; plot script will swap if needed)
  echo " - ablation-990-9935 (0.990..0.9935), excluding: ${PLOT_EXCLUDES}"
  "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation-990-9935_${PLOT_TS}.png" -d 0.990 -u 0.9935 "${excl_flags[@]}" || echo "Failed to generate ablation-990-9935"

  echo " - ablation-above0.995+, excluding: ${PLOT_EXCLUDES}"
  "$PYTHON" "$PLOT_SCRIPT" "$CSV" -o "$OUT_DIR/plots/ablation-995+_${PLOT_TS}.png"  -d 0.995 "${excl_flags[@]}" || echo "Failed to generate ablation-990-9935"
  
  echo "Plots saved to: $OUT_DIR/plots"
}

# Only run plots when requested (default enabled)
generate_plots

exit 0
