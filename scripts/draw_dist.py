#!/usr/bin/env python3
"""
draw_dist.py

Usage:
  python draw_dist.py file1.csv [file2.csv ...] [-x XCOL] [-r RECALL_COL] [-d DIST_COL] [-o out.png] [--show]

Defaults:
  XCOL: EFS
  RECALL_COL: avg_recall
  DIST_COL: avg_dists

The script supports reading multiple CSV files and will plot recall and average distance op curves
on the same figure (left y-axis: recall; right y-axis: avg_dists).
"""

import sys
import os
import argparse

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: pip install pandas matplotlib")
    sys.exit(1)


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
    parser = argparse.ArgumentParser(description="Draw recall and avg_dists from results CSV files.")
    parser.add_argument("files", nargs="+", help="CSV file(s) to read")
    parser.add_argument("-x", "--xcol", default="EFS", help="X-axis column name (default EFS)")
    parser.add_argument("-r", "--recall", default="avg_recall", help="Recall column (default avg_recall)")
    parser.add_argument("-d", "--dists", default="avg_dists", help="Average distances column (default avg_dists)")
    parser.add_argument("-o", "--out", default="draw_dist.png", help="Output image file (default draw_dist.png)")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--xlabel", default=None, help="Custom X-axis label")
    args = parser.parse_args(argv[1:])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_idx = 0

    any_plotted = False
    for f in args.files:
        if not os.path.exists(f):
            print(f"Warning: file not found: {f}; skipping")
            continue
        df = read_csv_file(f)
        if df.empty:
            print(f"Warning: empty file {f}; skipping")
            continue

        # Try to detect columns case-insensitively
        columns_lower = {c.lower(): c for c in df.columns}
        # X column detection
        xcol = None
        for cand in (args.xcol, "EFS", "efs", "EfS", "EFS", "EFSEARCH", "ef_search"):
            if cand and cand.lower() in columns_lower:
                xcol = columns_lower[cand.lower()]
                break
        if not xcol:
            # pick first numeric parameter-like column
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) >= 1:
                xcol = numeric_cols[0]
            else:
                print(f"Cannot detect X column in {f}; skipping.")
                continue

        # recall column
        recall_col = None
        for cand in (args.recall, "avg_recall", "recall", "AVG_RECALL"):
            if cand and cand.lower() in columns_lower:
                recall_col = columns_lower[cand.lower()]
                break
        if not recall_col:
            print(f"Cannot detect recall column in {f}; skipping.")
            continue

        # dists column
        dists_col = None
        for cand in (args.dists, "avg_dists", "avg_dists_per_query", "avg_dists_per_q", "avg_dists"):
            if cand and cand.lower() in columns_lower:
                dists_col = columns_lower[cand.lower()]
                break
        if not dists_col:
            print(f"Cannot detect dists column in {f}; skipping.")
            continue

        # sort by x
        try:
            df_sorted = df.sort_values(by=xcol)
        except Exception:
            df_sorted = df

        xs = df_sorted[xcol].astype(float).values
        ys_rec = df_sorted[recall_col].astype(float).values
        ys_dist = df_sorted[dists_col].astype(float).values

        label = label_from_df(df, f)
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

        ax1.plot(xs, ys_rec, label=f"{label} (recall)", color=color, marker='o', linestyle='-')
        ax2.plot(xs, ys_dist, label=f"{label} (avg_dists)", color=color, marker='x', linestyle='--')
        any_plotted = True

    if not any_plotted:
        print("No valid data plotted. Exiting.")
        return 1

    ax1.set_xlabel(args.xlabel if args.xlabel else args.xcol)
    ax1.set_ylabel("avg_recall", color='tab:blue')
    ax2.set_ylabel("avg_dists (ops)", color='tab:orange')
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)

    # merge legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize='small')

    plt.title("Recall and Avg Distance Ops vs " + (args.xlabel if args.xlabel else args.xcol))
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")
    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
