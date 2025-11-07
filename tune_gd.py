#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PARAM_ORDER = ["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE"]
PARAM_BOUNDS: Dict[str, Tuple[int, int]] = {
    "NUM_CENTROIDS": (32, 8192),
    "KMEANS_ITER": (1, 64),
    "NPROBE": (4, 1024),
}
LOG_BOUNDS: Dict[str, Tuple[float, float]] = {
    key: (math.log(bounds[0]), math.log(bounds[1])) for key, bounds in PARAM_BOUNDS.items()
}
STEP_SIZES_LOG = {
    "NUM_CENTROIDS": 0.25,  # ≈ +28% multiplicative perturbation
    "KMEANS_ITER": 0.18,    # ≈ +20%
    "NPROBE": 0.32,         # ≈ +38%
}
LEARNING_RATE_INIT = 0.6
LEARNING_RATE_MIN = 0.05
LEARNING_RATE_MAX = 2.0
LEARNING_RATE_DECAY = 0.5
LEARNING_RATE_GROW = 1.15
PENALTY_WEIGHT = 5000.0           # strong penalty if recall drops below target
HEURISTIC_TARGETS = {"NUM_CENTROIDS": 80, "KMEANS_ITER": 16, "NPROBE": 32}
HEURISTIC_WEIGHTS = {"NUM_CENTROIDS": 0.4, "KMEANS_ITER": 1.8, "NPROBE": 4.5}
CACHE_DIR = Path("results") / "tune_gd"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = CACHE_DIR / "history.csv"
TRACE_PATH = CACHE_DIR / "trace.txt"
BEST_JSON = CACHE_DIR / "best_config.json"

CXX = os.environ.get("CXX", "g++")
CXXFLAGS = os.environ.get("CXXFLAGS", "-O2 -std=c++17 -pthread -march=native")
SRC = "MySolution.cpp"
TEST_SRC = "test_solution.cpp"
BIN = "test"

class RunResult:
    __slots__ = ("params", "status", "elapsed", "recall", "avg_time", "build_time",
                 "logfile", "build_logfile")

    def __init__(self, params: Dict[str, int], status: str, elapsed: Optional[float],
                 recall: Optional[float], avg_time: Optional[float], build_time: Optional[float],
                 logfile: Path, build_logfile: Path):
        self.params = params
        self.status = status
        self.elapsed = elapsed
        self.recall = recall
        self.avg_time = avg_time
        self.build_time = build_time
        self.logfile = logfile
        self.build_logfile = build_logfile

    def meets_recall(self, target: float) -> bool:
        return self.recall is not None and self.recall >= target

    def to_history_row(self) -> List[str]:
        def fmt(value: Optional[float], digits: int = 2) -> str:
            if value is None or math.isnan(value):
                return "NA"
            return f"{value:.{digits}f}"
        return [
            stamp_from_params(self.params),
            str(self.params["NUM_CENTROIDS"]),
            str(self.params["KMEANS_ITER"]),
            str(self.params["NPROBE"]),
            self.status,
            fmt(self.elapsed, 2),
            fmt(self.recall, 4),
            fmt(self.avg_time, 2),
            fmt(self.build_time, 2),
        ]

def stamp_from_params(params: Dict[str, int]) -> str:
    return f"c{params['NUM_CENTROIDS']}_i{params['KMEANS_ITER']}_p{params['NPROBE']}"

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def project_vec(vec: List[float]) -> List[float]:
    projected = []
    for coord, key in zip(vec, PARAM_ORDER):
        lo, hi = LOG_BOUNDS[key]
        projected.append(clamp(coord, lo, hi))
    return projected

def vec_to_params(vec: List[float]) -> Dict[str, int]:
    params = {}
    for idx, key in enumerate(PARAM_ORDER):
        value = math.exp(vec[idx])
        lb, ub = PARAM_BOUNDS[key]
        value = clamp(value, lb, ub)
        params[key] = max(lb, min(ub, int(round(value))))
    return params

def params_to_vec(params: Dict[str, int]) -> List[float]:
    return [math.log(max(1, params[key])) for key in PARAM_ORDER]

def parse_float_safe(token: str) -> Optional[float]:
    token = token.strip()
    if not token or token == "NA":
        return None
    try:
        return float(token)
    except ValueError:
        return None

class ResultCache:
    """
    Simple CSV-backed cache.
    Loads HISTORY_CSV if present (one-row-per-run).
    Keeps an in-memory dict: stamp -> RunResult.
    On add(), appends a row to HISTORY_CSV.
    """

    def __init__(self):
        self.cache: Dict[str, RunResult] = {}
        if HISTORY_CSV.exists():
            try:
                with HISTORY_CSV.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            stamp = row.get("STAMP") or ""
                            nc = int(float(row.get("NUM_CENTROIDS", "0") or 0))
                            ki = int(float(row.get("KMEANS_ITER", "0") or 0))
                            npv = int(float(row.get("NPROBE", "0") or 0))
                            params = {"NUM_CENTROIDS": nc, "KMEANS_ITER": ki, "NPROBE": npv}
                            status = row.get("STATUS", "OK")
                            elapsed = parse_float_safe(row.get("ELAPSED_s", "") or "NA")
                            recall = parse_float_safe(row.get("RECALL", "") or "NA")
                            avg_time = parse_float_safe(row.get("AVG_QUERY_TIME_ms", "") or "NA")
                            build_time = parse_float_safe(row.get("INDEX_BUILD_TIME_s", "") or "NA")
                            rr = RunResult(
                                params=params,
                                status=status,
                                elapsed=elapsed,
                                recall=recall,
                                avg_time=avg_time,
                                build_time=build_time,
                                logfile=CACHE_DIR / f"{stamp}.log",
                                build_logfile=CACHE_DIR / f"{stamp}.build.log",
                            )
                            self.cache[stamp] = rr
                        except Exception:
                            # skip malformed row
                            continue
            except Exception as e:
                print(f"[cache] warning: cannot read {HISTORY_CSV}: {e}")

    def get(self, params: Dict[str, int]) -> Optional[RunResult]:
        stamp = stamp_from_params(params)
        return self.cache.get(stamp)

    def add(self, result: RunResult) -> None:
        stamp = stamp_from_params(result.params)
        self.cache[stamp] = result
        header_needed = not HISTORY_CSV.exists()
        try:
            with HISTORY_CSV.open("a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow([
                        "STAMP", "NUM_CENTROIDS", "KMEANS_ITER", "NPROBE",
                        "STATUS", "ELAPSED_s", "RECALL", "AVG_QUERY_TIME_ms", "INDEX_BUILD_TIME_s"
                    ])
                writer.writerow(result.to_history_row())
        except Exception as e:
            print(f"[cache] warning: cannot write to {HISTORY_CSV}: {e}")

    def best_feasible(self, target: float) -> Optional[RunResult]:
        best: Optional[RunResult] = None
        for rr in self.cache.values():
            if rr.status != "OK" or rr.recall is None or rr.avg_time is None:
                continue
            if rr.recall >= target:
                if best is None or rr.avg_time < best.avg_time:
                    best = rr
        return best

def compile_run(params: Dict[str, int], timeout: int) -> RunResult:
    stamp = stamp_from_params(params)
    logfile = CACHE_DIR / f"{stamp}.log"
    build_logfile = CACHE_DIR / f"{stamp}.build.log"
    defines = [
        f"-DNUM_CENTROIDS={params['NUM_CENTROIDS']}",
        f"-DKMEANS_ITER={params['KMEANS_ITER']}",
        f"-DNPROBE={params['NPROBE']}",
    ]
    compile_cmd = " ".join([CXX, CXXFLAGS, *defines, SRC, TEST_SRC, "-o", BIN])
    compile_proc = subprocess.run(compile_cmd, shell=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
    build_logfile.write_text(compile_proc.stdout)
    if compile_proc.returncode != 0:
        return RunResult(params, "COMPILE_FAIL", None, None, None, None, logfile, build_logfile)

    start = time.time()
    with logfile.open("w") as f:
        run = subprocess.run(["timeout", str(timeout), f"./{BIN}"], stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if run.returncode not in (0,):
        status = "TIMEOUT" if run.returncode in (124, 137) else f"ERROR_{run.returncode}"
        return RunResult(params, status, elapsed, None, None, None, logfile, build_logfile)

    recall, avg_ms, build_sec = parse_metrics(logfile)
    status = "OK" if recall is not None and avg_ms is not None else "PARSE_FAIL"
    return RunResult(params, status, elapsed, recall, avg_ms, build_sec, logfile, build_logfile)

def parse_metrics(logfile: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    recall = avg_time = build_time = None
    with logfile.open() as f:
        for line in f:
            line = line.strip()
            if "Average recall@" in line:
                try:
                    recall = float(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif "Average query time" in line:
                try:
                    avg_time = float(line.split(":")[-1].split()[0])
                except ValueError:
                    pass
            elif "Index build time" in line:
                try:
                    build_time = float(line.split(":")[-1].split()[0])
                except ValueError:
                    pass
    return recall, avg_time, build_time

def compute_objective(result: RunResult, params: Dict[str, int], target: float) -> float:
    # Minimize average query time while enforcing recall >= target via a large penalty.
    # Do NOT penalize build time here; build can be long.
    if result.status != "OK" or result.avg_time is None or result.recall is None:
        return float("inf")
    base = result.avg_time
    deficit = target - result.recall
    penalty = PENALTY_WEIGHT * deficit * deficit if deficit > 0 else 0.0
    return base + penalty

def evaluate_vector(vec: List[float], cache: ResultCache, timeout: int,
                    target: float) -> Tuple[RunResult, float, Dict[str, int]]:
    params = vec_to_params(vec)
    cached = cache.get(params)
    if cached is None:
        cached = compile_run(params, timeout)
        cache.add(cached)
    obj = compute_objective(cached, params, target)
    return cached, obj, params

def evaluate_direction(base_vec: List[float], idx: int, base_params: Dict[str, int],
                       cache: ResultCache, timeout: int, target: float,
                       step: float, sign: int) -> Tuple[List[float], RunResult, float, Dict[str, int], float]:
    base_key = PARAM_ORDER[idx]
    base_value = base_params[base_key]
    shift = step
    for _ in range(6):
        trial_vec = base_vec[:]
        trial_vec[idx] += sign * shift
        trial_vec = project_vec(trial_vec)
        trial_params = vec_to_params(trial_vec)
        if trial_params != base_params:
            result, obj, params = evaluate_vector(trial_vec, cache, timeout, target)
            delta = trial_vec[idx] - base_vec[idx]
            return trial_vec, result, obj, params, delta
        shift *= 1.6
    # forced discrete move if rounding never left the base point
    forced_params = dict(base_params)
    lo, hi = PARAM_BOUNDS[base_key]
    if sign > 0:
        forced_val = min(hi, max(base_value + 1, int(round(base_value * 1.25))))
    else:
        forced_val = max(lo, min(base_value - 1, int(round(base_value * 0.80))))
    if forced_val == base_value:
        forced_val = hi if sign > 0 else lo
    forced_params[base_key] = forced_val
    forced_vec = params_to_vec(forced_params)
    forced_vec = project_vec(forced_vec)
    result, obj, params = evaluate_vector(forced_vec, cache, timeout, target)
    delta = forced_vec[idx] - base_vec[idx]
    return forced_vec, result, obj, params, delta

def stamp_to_params(stamp: str) -> Optional[Dict[str,int]]:
    s = stamp.strip()
    if not s:
        return None
    # accept c1024_i4_p256 or 1024,4,256 or "1024 4 256"
    if s.startswith("c"):
        try:
            parts = s.split("_")
            nc = int(parts[0][1:])
            ki = int(parts[1][1:])
            npv = int(parts[2][1:])
            return {"NUM_CENTROIDS": nc, "KMEANS_ITER": ki, "NPROBE": npv}
        except Exception:
            return None
    else:
        toks = [t for t in s.replace(",", " ").split() if t]
        if len(toks) == 3:
            try:
                return {"NUM_CENTROIDS": int(toks[0]), "KMEANS_ITER": int(toks[1]), "NPROBE": int(toks[2])}
            except Exception:
                return None
    return None

def preprobe_neighbors(cache: ResultCache, base_params: Dict[str,int], timeout: int, target: float):
    # compile a small set of nearby configurations so search has real neighbors to consider
    to_try = []
    for key in PARAM_ORDER:
        val = base_params[key]
        lo, hi = PARAM_BOUNDS[key]
        # discrete +/- 1
        for nv in (val - 1, val + 1):
            if lo <= nv <= hi:
                p = dict(base_params); p[key] = nv; to_try.append(p)
        # multiplicative probes
        mul_up = min(hi, max(lo, int(round(val * 1.25))))
        mul_down = max(lo, min(hi, int(round(val * 0.80))))
        to_try.append({**base_params, key: mul_up})
        to_try.append({**base_params, key: mul_down})
    # unique
    seen = set()
    for p in to_try:
        key = (p["NUM_CENTROIDS"], p["KMEANS_ITER"], p["NPROBE"])
        if key in seen: continue
        seen.add(key)
        if cache.get(p) is None:
            print(f"[preprobe] evaluating neighbor {stamp_from_params(p)}")
            r = compile_run(p, timeout)
            cache.add(r)

# Add SPSA hyperparameters (joint optimization)
SPSA_A0 = 0.6
SPSA_C0 = {"NUM_CENTROIDS": 0.25, "KMEANS_ITER": 0.18, "NPROBE": 0.32}
SPSA_A_DECAY = 0.602
SPSA_C_DECAY = 0.101
SPSA_A_MIN = 1e-3

def gradient_descent(cache: ResultCache, init_params: Dict[str, int],
                     target: float, max_iters: int, timeout: int) -> RunResult:
    """
    Joint optimizer using SPSA (simultaneous perturbation).
    Minimizes compute_objective (avg query time + heavy penalty if recall < target).
    """
    # initial evaluation
    vec = params_to_vec(init_params)
    current_res, current_obj, current_params = evaluate_vector(vec, cache, timeout, target)
    vec = params_to_vec(current_params)

    # ensure neighbors are available (force at least some runs)
    preprobe_neighbors(cache, current_params, timeout, target)

    trace_lines = [f"start: {current_params} recall={current_res.recall} avg_ms={current_res.avg_time} obj={current_obj:.3f}"]
    a0 = SPSA_A0
    # use per-dim c0 from table
    c0 = [SPSA_C0[key] for key in PARAM_ORDER]

    for k in range(1, max_iters + 1):
        # gain sequences
        a_k = a0 / (k ** SPSA_A_DECAY)
        c_k = [c0_i / (k ** SPSA_C_DECAY) for c0_i in c0]

        # draw perturbation vector delta_i in {-1, +1}
        delta = [1 if (os.urandom(1)[0] & 1) == 1 else -1 for _ in PARAM_ORDER]

        # form perturbation in log-space
        c_vec = [c_k[i] * delta[i] for i in range(len(PARAM_ORDER))]

        plus_vec = project_vec([vec[i] + c_vec[i] for i in range(len(vec))])
        minus_vec = project_vec([vec[i] - c_vec[i] for i in range(len(vec))])

        plus_res, plus_obj, plus_params = evaluate_vector(plus_vec, cache, timeout, target)
        minus_res, minus_obj, minus_params = evaluate_vector(minus_vec, cache, timeout, target)

        # if both evaluations failed to produce finite objectives, stop
        if not math.isfinite(plus_obj) and not math.isfinite(minus_obj):
            trace_lines.append(f"iter {k}: both plus/minus invalid, stopping")
            break

        # SPSA gradient estimate
        g_hat = []
        for i in range(len(PARAM_ORDER)):
            denom = 2.0 * c_vec[i]
            if denom == 0:
                g_hat.append(0.0)
            else:
                # prefer finite values; if one side inf use the other difference
                if math.isfinite(plus_obj) and math.isfinite(minus_obj):
                    g_hat.append((plus_obj - minus_obj) / denom)
                elif math.isfinite(plus_obj):
                    g_hat.append((plus_obj - current_obj) / (c_vec[i]))
                elif math.isfinite(minus_obj):
                    g_hat.append((current_obj - minus_obj) / (c_vec[i]))
                else:
                    g_hat.append(0.0)

        # candidate update (joint)
        candidate_vec = project_vec([vec[i] - a_k * g_hat[i] for i in range(len(vec))])
        candidate_res, candidate_obj, candidate_params = evaluate_vector(candidate_vec, cache, timeout, target)

        improved = math.isfinite(candidate_obj) and candidate_obj < current_obj - 1e-4
        if improved:
            vec = params_to_vec(candidate_params)
            current_res, current_obj, current_params = candidate_res, candidate_obj, candidate_params
            trace_lines.append(f"iter {k}: accepted -> {current_params} recall={current_res.recall:.4f} avg_ms={current_res.avg_time:.2f} obj={current_obj:.3f} a_k={a_k:.4f}")
            # optionally increase a0 a bit (mildly)
            a0 = min(a0 * 1.03, 5.0)
            if current_res.recall is not None and current_res.recall < target:
                # if accepted but not feasible, continue searching (penalty forces objective)
                trace_lines.append(f"iter {k}: accepted but recall {current_res.recall:.4f} < target {target:.4f}")
            continue

        # no improvement: try greedy selection among plus/minus and small coordinate moves
        candidate_list = []
        if math.isfinite(plus_obj):
            candidate_list.append((plus_obj, plus_params, plus_res))
        if math.isfinite(minus_obj):
            candidate_list.append((minus_obj, minus_params, minus_res))
        # also probe integer neighbors around current_params (small)
        for key in PARAM_ORDER:
            p_up = dict(current_params); p_down = dict(current_params)
            lo, hi = PARAM_BOUNDS[key]
            if current_params[key] + 1 <= hi:
                p_up[key] = current_params[key] + 1
                res_up = cache.get(p_up) or compile_run(p_up, timeout); cache.add(res_up)
                obj_up = compute_objective(res_up, p_up, target)
                candidate_list.append((obj_up, p_up, res_up))
            if current_params[key] - 1 >= lo:
                p_down[key] = current_params[key] - 1
                res_down = cache.get(p_down) or compile_run(p_down, timeout); cache.add(res_down)
                obj_down = compute_objective(res_down, p_down, target)
                candidate_list.append((obj_down, p_down, res_down))

        # pick best candidate among tried ones
        if candidate_list:
            candidate_list.sort(key=lambda x: x[0])
            best_obj, best_params, best_res = candidate_list[0]
            if best_obj < current_obj - 1e-4:
                vec = params_to_vec(best_params)
                current_res, current_obj, current_params = best_res, best_obj, best_params
                trace_lines.append(f"iter {k}: fallback adopt -> {current_params} recall={current_res.recall:.4f} avg_ms={current_res.avg_time:.2f} obj={current_obj:.3f}")
                # slightly reduce a0 to be conservative
                a0 = max(a0 * 0.8, SPSA_A_MIN)
                continue

        # no improvement found, reduce step sizes and possibly stop
        a0 = max(a0 * 0.7, SPSA_A_MIN)
        trace_lines.append(f"iter {k}: no improvement, reduce a0 -> {a0:.4f}")
        if a0 <= SPSA_A_MIN * 1.5:
            trace_lines.append(f"stop: a0 below threshold at iter {k}")
            break

    TRACE_PATH.write_text("\n".join(trace_lines) + "\n")
    BEST_JSON.write_text(json.dumps({
        "params": current_params,
        "status": current_res.status,
        "recall": current_res.recall,
        "avg_query_time_ms": current_res.avg_time,
        "elapsed_s": current_res.elapsed,
        "objective": current_obj,
        "logfile": str(current_res.logfile),
    }, indent=2))
    return current_res

def main():
    parser = argparse.ArgumentParser(description="Gradient-style tuner for ANN parameters")
    parser.add_argument("--target-recall", type=float, default=0.98, help="Minimum recall constraint")
    parser.add_argument("--max-iters", type=int, default=30, help="Maximum gradient iterations")
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("TUNE_TIMEOUT", "900")),
                        help="Per-run timeout (seconds)")
    parser.add_argument("--init", type=str,
                        help="Initial guess, e.g. NUM_CENTROIDS=512,KMEANS_ITER=16,NPROBE=96")
    parser.add_argument("--start-stamp", type=str,
                        help='Start from an existing stamp like "c1024_i4_p256" or "1024,4,256"')
    args = parser.parse_args()

    cache = ResultCache()

    if args.start_stamp:
        parsed = stamp_to_params(args.start_stamp)
        if parsed is None:
            print(f"Invalid start-stamp: {args.start_stamp}")
            return
        init_params = parsed
    elif args.init:
        init_params = {}
        for segment in args.init.split(","):
            key, value = segment.split("=")
            key = key.strip().upper()
            if key not in PARAM_BOUNDS:
                raise ValueError(f"Unknown parameter '{key}' in --init")
            init_params[key] = int(value)
        for key, default in zip(PARAM_ORDER, [512, 16, 96]):
            init_params.setdefault(key, default)
    else:
        seeded = cache.best_feasible(args.target_recall)
        if seeded:
            init_params = dict(seeded.params)
        else:
            init_params = {"NUM_CENTROIDS": 512, "KMEANS_ITER": 16, "NPROBE": 96}

    print(f"Tuning target recall >= {args.target_recall}")
    print(f"Initial parameters: {init_params}")
    best = gradient_descent(cache, init_params, args.target_recall, args.max_iters, args.timeout)

    print("\n=== Final best configuration ===")
    print("params:", best.params)
    print("status:", best.status)
    print("recall:", best.recall)
    print("avg_query_time_ms:", best.avg_time)
    print("elapsed_s:", best.elapsed)
    print(f"Logs: {best.logfile}")
    print(f"Trace: {TRACE_PATH}")
    print(f"Best summary saved to: {BEST_JSON}")

if __name__ == "__main__":
    main()
