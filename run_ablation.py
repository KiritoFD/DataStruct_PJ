#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import random
from datetime import datetime
import shutil

# Configuration
BINARY = "./hng"
K = 10
THREADS = 32
REPEATS = 1

# Ablation variants: (name, csr, prefetch, simd, pruning, heap, flat_index)
ABLATION_VARIANTS = [
    ("baseline", 0, 0, 0, 0, 0, 0),
    ("no_csr", 1, 0, 0, 0, 0, 0),
    ("no_prefetch", 0, 1, 0, 0, 0, 0),
    ("no_simd", 0, 0, 1, 0, 0, 0),
    ("no_pruning", 0, 0, 0, 1, 0, 0),
    ("heap", 0, 0, 0, 0, 1, 0),
    ("dynamic_struct", 0, 0, 0, 0, 0, 1),
    ("nothing", 1, 1, 1, 1, 1, 1),
]

def extract_points():
    """Extract 10 uniform points (best to worst) + 10 random points from cache"""
    csv_file = "optuna_hng.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found", file=sys.stderr)
        return None
    
    # Read all points
    points = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 5:
                m, ml, efc, efs, recall = row[0], row[1], row[2], row[3], row[4]
                points.append((m, ml, efc, efs, recall))
    
    total = len(points)
    print(f"Total points in cache: {total}")
    
    # Sort by recall (descending)
    points.sort(key=lambda x: float(x[4]), reverse=True)
    
    # Extract 10 uniform points
    uniform_points = []
    step = max(1, total // 10)
    for i in range(10):
        idx = i * step
        if idx < total:
            point_type = f"uniform_{i:02d}"
            uniform_points.append((point_type, *points[idx]))
    
    print(f"=== Extracted {len(uniform_points)} uniformly distributed points (best to worst) ===")
    
    # Extract 10 random points
    random_points = []
    all_points = points.copy()
    random.shuffle(all_points)
    for i, point in enumerate(all_points[:10]):
        point_type = f"random_{i:02d}"
        random_points.append((point_type, *point))
    
    print(f"=== Extracted {len(random_points)} random points ===")
    
    return uniform_points + random_points

def parse_metrics(logfile):
    """Parse metrics from log file"""
    metrics = {
        'build_internal_ms': '',
        'build_local_ms': '',
        'queries_per_sec': '',
        'avg_recall': '',
        'avg_query_time_ms': '',
        'avg_dists': '',
        'last_query_dists': '',
        'total_query_time_ms': '',
    }
    
    if not os.path.exists(logfile):
        return metrics
    
    try:
        with open(logfile, 'r') as f:
            content = f.read()
            
        # Extract metrics
        for line in content.split('\n'):
            if 'Index build time:' in line:
                try:
                    metrics['build_internal_ms'] = line.split('Index build time:')[1].split('ms')[0].strip()
                except:
                    pass
            elif 'build_time(local)=' in line:
                try:
                    metrics['build_local_ms'] = line.split('build_time(local)=')[1].split('ms')[0].strip()
                except:
                    pass
            elif 'Queries per second' in line:
                try:
                    metrics['queries_per_sec'] = line.split(':')[1].strip()
                except:
                    pass
            elif 'Average recall@' in line:
                try:
                    metrics['avg_recall'] = line.split(':')[1].strip()
                except:
                    pass
            elif 'Average query time' in line:
                try:
                    metrics['avg_query_time_ms'] = line.split(':')[1].split('ms')[0].strip()
                except:
                    pass
            elif 'Average distance ops per query' in line:
                try:
                    metrics['avg_dists'] = line.split(':')[1].strip()
                except:
                    pass
            elif 'Last query distance ops:' in line:
                try:
                    metrics['last_query_dists'] = line.split(':')[1].strip()
                except:
                    pass
            elif 'Total query time' in line:
                try:
                    metrics['total_query_time_ms'] = line.split(':')[1].split('ms')[0].strip()
                except:
                    pass
    except Exception as e:
        print(f"Error parsing {logfile}: {e}", file=sys.stderr)
    
    return metrics

def run_variant_on_point(point_type, point_index, m, ml, efc, efs, recall_cached, 
                         variant_name, csr, prefetch, simd, pruning, heap, flat_index,
                         out_dir, csv_writer):
    """Run a single ablation variant on a single point"""
    
    point_name = f"{point_type}_m{m}_ml{ml}_efc{efc}_efs{efs}"
    variant_dir = os.path.join(out_dir, point_name, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    
    print(f"  [RUN] {variant_name}")
    
    # Clean logs
    log_dir = "Log"
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            try:
                os.remove(os.path.join(log_dir, f))
            except:
                pass
    
    for run_index in range(1, REPEATS + 1):
        log_file_stdout = os.path.join(variant_dir, f"run_stdout_{run_index}.log")
        
        # Build command
        cmd = [
            BINARY,
            "--k", str(K),
            "--m", str(m),
            "--max_layer", str(ml),
            "--efc", str(efc),
            "--efs", str(efs),
            "--threads", str(THREADS),
            "--ablate_csr", str(csr),
            "--ablate_prefetch", str(prefetch),
            "--ablate_simd", str(simd),
            "--ablate_pruning", str(pruning),
            "--ablate_heap", str(heap),
            "--ablate_flat_index", str(flat_index),
        ]
        
        # Run command
        try:
            with open(log_file_stdout, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=f, timeout=3600)
            print(f"    Run {run_index}: completed")
        except subprocess.TimeoutExpired:
            print(f"    Run {run_index}: TIMEOUT")
            continue
        except Exception as e:
            print(f"    Run {run_index}: ERROR - {e}")
            continue
        
        # Find and copy harness log
        harness_log = log_file_stdout
        if os.path.exists(log_dir):
            logs = sorted(os.listdir(log_dir), key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
            if logs:
                latest_log = os.path.join(log_dir, logs[0])
                harness_log = os.path.join(variant_dir, logs[0])
                try:
                    shutil.copy(latest_log, harness_log)
                except:
                    pass
        
        # Parse metrics
        metrics = parse_metrics(harness_log)
        
        # Write CSV record
        csv_writer.writerow([
            variant_name, point_type, point_index, m, ml, efc, efs, recall_cached,
            csr, prefetch, simd, pruning, heap, flat_index,
            metrics['build_internal_ms'],
            metrics['build_local_ms'],
            metrics['queries_per_sec'],
            metrics['avg_recall'],
            metrics['avg_query_time_ms'],
            metrics['avg_dists'],
            metrics['last_query_dists'],
            metrics['total_query_time_ms'],
            harness_log,
            run_index
        ])
        
        print(f"      recall={metrics['avg_recall']}, qtime={metrics['avg_query_time_ms']}ms")

def main():
    """Main function"""
    # Check binary exists
    if not os.path.exists(BINARY):
        print(f"Error: Binary {BINARY} not found. Please compile first.", file=sys.stderr)
        return 1
    
    print(f"Using binary: {os.path.abspath(BINARY)}")
    
    # Extract points
    points = extract_points()
    if not points:
        print("Error: Could not extract points", file=sys.stderr)
        return 1
    
    print(f"Extracted {len(points)} points\n")
    
    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results/ablation_uniform_random_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    
    # Open CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow([
            "variant", "point_type", "point_index", "m", "ml", "efc", "efs", "recall_cached",
            "csr", "prefetch", "simd", "pruning", "heap", "flat_index",
            "build_time_internal_ms", "build_time_local_ms", "queries_per_sec", "avg_recall",
            "avg_query_time_ms", "avg_dists", "last_query_dists", "total_query_time_ms",
            "log_file", "run_index"
        ])
        
        # Process all points
        print("========== RUNNING ABLATION ON 20 POINTS ==========\n")
        total_runs = len(points) * len(ABLATION_VARIANTS)
        current_run = 0
        
        for point_counter, (point_type, m, ml, efc, efs, recall_cached) in enumerate(points, 1):
            print(f"[{point_counter}/20] Point: {point_type} m={m} ml={ml} efc={efc} efs={efs}")
            
            for var_name, csr, prefetch, simd, pruning, heap, flat_index in ABLATION_VARIANTS:
                current_run += 1
                run_variant_on_point(
                    point_type, point_counter, m, ml, efc, efs, recall_cached,
                    var_name, csr, prefetch, simd, pruning, heap, flat_index,
                    out_dir, csv_writer
                )
                print(f"  [{current_run}/{total_runs}] ✓")
            print()
    
    print("\n✅ Ablation experiments completed!")
    print(f"Results saved to: {csv_path}")
    print(f"Experiment directory: {out_dir}")
    print(f"Total experiments: {len(points)} points × {len(ABLATION_VARIANTS)} variants = {total_runs} runs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
