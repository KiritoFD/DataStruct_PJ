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
import math
import re
import unicodedata
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FixedLocator
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: pip install pandas matplotlib numpy")
    sys.exit(1)

# Add helper to clean strange chars from column names
def sanitize_columns(cols):
    # strip leading/trailing whitespace, remove BOM/zero-width/control characters and surrounding quotes
    cleaned = []
    for c in cols:
        s = str(c)
        # normalize unicode to avoid weird composed characters
        s = unicodedata.normalize('NFKC', s)
        # remove BOM, zero-width, non-breaking space, CR/LF, tab
        s = re.sub(r'[\ufeff\u200b-\u200f\u00a0\r\n\t]', '', s)
        # remove content inside parentheses (units), e.g. "(ms)"
        s = re.sub(r'\([^)]*\)', '', s)
        s = s.strip()
        # remove surrounding quotes if any
        if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
            s = s[1:-1].strip()
        # strip stray punctuation from ends
        s = s.strip(' \'"`.,;:')
        cleaned.append(s)
    return cleaned

def detect_col(df, preferred, candidates):
    """Find a column name in df (case-insensitive, punctuation-insensitive, whitespace-insensitive) from a list of candidates; fallback to preferred if present."""
    def norm(s):
        # Remove all non-alphanumeric, lowercase
        return re.sub(r'[^0-9a-z]', '', str(s).lower())
    # Build normalized map for all columns
    nmmap = {norm(c): c for c in df.columns}
    # Try candidates
    for cand in candidates:
        if not cand:
            continue
        nc = norm(cand)
        if nc in nmmap:
            return nmmap[nc]
    # fallback to preferred if present
    if preferred:
        npref = norm(preferred)
        if npref in nmmap:
            return nmmap[npref]
    return None

# Add helper to find a column containing a substring in its normalized name
def find_col_contains(df, substr):
    """Return first column whose normalized name contains the substring (case-insensitive)."""
    def norm(s): return re.sub(r'[^0-9a-z]', '', str(s).lower())
    substr_norm = norm(substr)
    for c in df.columns:
        if substr_norm in norm(c):
            return c
    return None


def read_csv_file(path):
    # try pandas to be robust against stray commas, blank rows, BOMs, different encodings etc.
    encodings = [None, "utf-8-sig", "utf-8", "latin1", "utf-16"]
    for enc in encodings:
        try:
            if enc is None:
                df = pd.read_csv(path)
            else:
                df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            # try with python engine if default fails
            try:
                if enc is None:
                    df = pd.read_csv(path, engine='python')
                else:
                    df = pd.read_csv(path, encoding=enc, engine='python')
                break
            except Exception:
                df = None
    if df is None:
        # last-chance: read as latin1 with python engine
        df = pd.read_csv(path, encoding='latin1', engine='python')

    # sanitize column names to remove BOM or control characters that can break detection
    df.columns = sanitize_columns(df.columns)

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
    recall_values = []  # collect recall values across files
    for f in args.files:
        if not os.path.exists(f):
            print(f"Warning: file not found: {f}; skipping")
            continue
        df = read_csv_file(f)
        if df.empty:
            print(f"Warning: empty file {f}; skipping")
            continue

        # Try to detect columns case-insensitively and robustly (ignore spaces/punctuation)
        # X column detection
        xcol = detect_col(df, args.xcol, (
            args.xcol, "EFS", "EFSEARCH", "ef_search", "ef", "e_f_s", "efsearch"
        ))
        if not xcol:
            # pick first numeric parameter-like column
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) >= 1:
                xcol = numeric_cols[0]
            else:
                print(f"Cannot detect X column in {f}; skipping.")
                continue

        # recall column
        recall_col = detect_col(df, args.recall, (
            args.recall, "avg_recall", "average_recall", "recall", "avgRecall", "Recall", "AVG_RECALL"
        ))
        if not recall_col:
            print(f"Cannot detect recall column in {f}; skipping.")
            continue

        # dists column
        dists_col = detect_col(df, args.dists, (
            args.dists,
            "avg_dists",
            "avg_dists_per_query",
            "avg_dists_per_q",
            "AvgDists",
            "DistOpsPerQuery",
            "distopsperquery",
            "dist_ops_per_query",
            "dist_ops_per_q",
            "distops_per_q",
            "dist ops per query",
            "distopsperquery(ms)",
            "distance",
            "distances",
            "dist_per_query",
        ))
        # Fallback: any column whose normalized name contains 'dist'
        if not dists_col:
            dists_col = find_col_contains(df, "dist")
            if dists_col:
                print(f"Note: using detected dists column '{dists_col}' in {f} (normalized contains 'dist')")
        if not dists_col:
            # helpful debug: show raw column names and normalized mapping
            cols_repr = ", ".join([repr(c) for c in df.columns])
            nmmap = {re.sub(r'[^0-9a-z]', '', str(c).lower()): c for c in df.columns}
            nmmap_repr = ", ".join([f"{k}:{repr(v)}" for k,v in nmmap.items()])
            print(f"Cannot detect dists column in {f}; skipping. Found columns: {cols_repr}; normalized map: {nmmap_repr}")
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

        # collect recall values for tick-setting logic later
        recall_values.extend(ys_rec.tolist())

    if not any_plotted:
        print("No valid data plotted. Exiting.")
        return 1

    # Determine appropriate Y-scale and ticks for recall axis
    if recall_values:
        y_min = float(np.min(recall_values))
        y_max = float(np.max(recall_values))

        # lower bound with a small margin
        y_lower = max(0.0, y_min - 0.01)
        # upper bound with small margin
        y_upper = min(1.0, y_max + 0.005)

        # If recall goes >= 0.98, ensure fine-grained ticks above 0.98 and ensure 0.99 is visible
        if y_max >= 0.98:
            if y_upper < 0.99:
                y_upper = min(1.0, 0.99 + 0.002)
            # major ticks below 0.98 every 0.01
            major_start = math.floor(y_lower * 100.0) / 100.0
            major_end = min(0.98, y_upper)
            if major_end <= major_start:
                major_ticks = np.array([])
            else:
                major_ticks = np.arange(major_start, major_end + 1e-9, 0.01)

            # Use finer ticks from 0.98 upward
            fine_step = 0.001 if (y_upper - 0.98) <= 0.02 else 0.002
            fine_ticks = np.arange(0.98, y_upper + 1e-12, fine_step)

            ticks = np.unique(np.concatenate((major_ticks, fine_ticks)))
            # ensure 0.99 included
            if 0.99 not in ticks and 0.99 <= y_upper:
                ticks = np.sort(np.append(ticks, 0.99))

            ax1.set_yticks(ticks)
        else:
            # Default tick every 0.01 across the range
            maj_start = math.floor(y_lower * 100.0) / 100.0
            maj_end = math.ceil(y_upper * 100.0) / 100.0
            ticks = np.arange(maj_start, maj_end + 1e-9, 0.01)
            ax1.set_yticks(ticks)

        ax1.set_ylim(y_lower, y_upper)

        # If 0.99 is inside the visible Y-limits, draw a dashed horizontal line and annotation to mark it
        if 0.99 >= y_lower and 0.99 <= y_upper:
            ax1.axhline(0.99, color='red', linestyle=':', linewidth=1)
            # place the annotation slightly to the right of the axes (fraction x coordinate)
            ax1.annotate('Recall 0.99', xy=(1.01, 0.99), xycoords=('axes fraction', 'data'),
                         color='red', fontsize='small', va='center')

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
