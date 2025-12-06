#!/usr/bin/env python3
"""
draw_dist.py

Usage:
  python draw_dist.py file1.csv [file2.csv ...] [-d DIST_COL] [-r RECALL_COL] [-o out.png] [--show]

Defaults:
  DIST_COL: avg_dists
  RECALL_COL: avg_recall

The script supports reading multiple CSV files and will plot recall vs distance curves on the same figure.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def detect_col(df, preferred, candidates):
    """Find a column name in df (case-insensitive) from a list of candidates; fallback to preferred if present."""
    lcmap = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lcmap:
            return lcmap[cand.lower()]
    # fallback to preferred if exists exactly
    return df.columns[df.columns.str.lower() == preferred.lower()].tolist()[0] if preferred.lower() in lcmap else None


def read_csv_file(path):
    # try pandas to be robust against stray commas, blank rows etc.
    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback, try to read ignoring comment lines
        df = pd.read_csv(path, comment='#', engine='python')
    return df


def label_from_df(df, fname):
    # try to extract common params for label
    parts = []
    for key in ("M", "ML", "EFC", "EFS"):
        if key in df.columns:
            try:
                # take unique value if scalar across file, else skip
                vals = pd.unique(df[key].dropna().astype(str))
                if len(vals) == 1:
                    parts.append(f"{key}={vals[0]}")
            except Exception:
                pass
    if parts:
        return f"{os.path.basename(fname)}: " + ",".join(parts)
    return os.path.basename(fname)


def main(argv):
    parser = argparse.ArgumentParser(description="Draw recall vs distance from results CSV files.")
    parser.add_argument("files", nargs="+", help="CSV file(s) to read")
    parser.add_argument("-d", "--dists", default="avg_dists", help="Distance column name (default avg_dists)")
    parser.add_argument("-r", "--recall", default="avg_recall", help="Recall column (default avg_recall)")
    parser.add_argument("-o", "--out", default="draw_dist.png", help="Output image file (default draw_dist.png)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--xlabel", default=None, help="Custom X-axis label")
    args = parser.parse_args(argv[1:])

    dfs = []
    dists_cols = []
    recall_cols = []
    valid_files = []
    min_vals = []
    max_vals = []

    # 先收集所有文件的距离列区间
    for f in args.files:
        if not os.path.exists(f):
            print(f"Warning: file not found: {f}; skipping")
            continue
        df = read_csv_file(f)
        if df.empty:
            print(f"Warning: empty file {f}; skipping")
            continue

        columns_lower = {c.lower(): c for c in df.columns}
        dists_col = None
        for cand in (args.dists, "avg_dists", "avg_dists_per_query", "avg_dists_per_q", "dists", "dist"):
            if cand and cand.lower() in columns_lower:
                dists_col = columns_lower[cand.lower()]
                break
        if not dists_col:
            print(f"Cannot detect distance column in {f}; skipping.")
            continue

        recall_col = None
        for cand in (args.recall, "avg_recall", "recall", "AVG_RECALL"):
            if cand and cand.lower() in columns_lower:
                recall_col = columns_lower[cand.lower()]
                break
        if not recall_col:
            print(f"Cannot detect recall column in {f}; skipping.")
            continue

        # 记录距离区间
        try:
            dist_vals = df[dists_col].astype(float)
            min_vals.append(dist_vals.min())
            max_vals.append(dist_vals.max())
        except Exception:
            print(f"Cannot parse distance values in {f}; skipping.")
            continue

        dfs.append(df)
        dists_cols.append(dists_col)
        recall_cols.append(recall_col)
        valid_files.append(f)

    if not dfs:
        print("No valid data plotted. Exiting.")
        return 1

    # 计算重叠区间
    overlap_min = max(min_vals)
    overlap_max = min(max_vals)
    if overlap_min >= overlap_max:
        print("No overlapping distance region among all files.")
        return 1

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    for df, dists_col, recall_col, f in zip(dfs, dists_cols, recall_cols, valid_files):
        # 不再限制为重叠区间，直接使用所有数据
        df_sorted = df.sort_values(by=dists_col)
        xs = df_sorted[dists_col].astype(float).values
        ys_rec = df_sorted[recall_col].astype(float).values

        label = label_from_df(df, f)
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

        ax1.plot(xs, ys_rec, label=label, color=color, marker='o', linestyle='-')

    ax1.set_xlabel(args.xlabel if args.xlabel else "Distance")
    ax1.set_ylabel(args.recall if args.recall else "Recall")
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)
    ax1.legend(loc='lower right', fontsize='small')  # 修改图例位置为右下角

    plt.title(f"Recall vs {args.xlabel if args.xlabel else 'Distance'}")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")
    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
