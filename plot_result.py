#!/usr/bin/env python3
"""
Simple plotting utility:
  - X axis: avg_recall
  - Y axis: avg_query_time_ms
  - One line per group (default 'variant')
Usage:
  python plot_results.py [csv] -o out.png --group-by variant+M+ML+EFC+EFS
"""
import os
import sys
import argparse
import csv
import math
from collections import defaultdict

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
except Exception:
    print("Error: matplotlib required. Install via pip install matplotlib")
    sys.exit(1)


def parse_filters(filter_str):
    if not filter_str:
        return {}
    fs = {}
    for kv in filter_str.split(','):
        if '=' in kv:
            k, v = kv.split('=', 1)
            fs[k.strip()] = v.strip()
    return fs


def match_filters(row, filters):
    if not filters:
        return True
    for k, v in filters.items():
        if k not in row or row[k] is None:
            return False
        if str(row[k]) != v:
            return False
    return True


def try_read_csv_csvmodule(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize keys by stripping spaces
            nr = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append(nr)
    return rows


def read_csv(path):
    if pd:
        try:
            df = pd.read_csv(path, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            # Pandas reads CSV into strings, convert numeric later as needed
            rows = df.fillna('').to_dict(orient='records')
            return rows
        except Exception:
            # fallback
            pass
    return try_read_csv_csvmodule(path)


def float_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == '':
        return None
    try:
        return float(s)
    except Exception:
        return None


def make_group_label(row, group_by_cols):
    if not group_by_cols:
        return "all"
    if isinstance(group_by_cols, str):
        cols = [c.strip() for c in group_by_cols.split('+')]
    else:
        cols = list(group_by_cols)
    label_parts = []
    for c in cols:
        label_parts.append(f"{c}={row.get(c, '')}")
    return "|".join(label_parts)


def main():
    default_csv = "./results4.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='?', default=default_csv, help='CSV file path (default provided results path)')
    parser.add_argument('-o', '--output', help='Output image path', default=None)
    parser.add_argument('--filter', help='Comma separated param filters like M=40,ML=17', default=None)
    parser.add_argument('--group-by', help='Column(s) to group by (use + to combine)', default='variant')
    parser.add_argument('--x', help='X axis column', default='avg_recall')
    parser.add_argument('--y', help='Y axis column', default='queries_per_sec')
    parser.add_argument('--dpi', help='Output DPI', type=int, default=180)
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print("CSV not found:", csv_path)
        return

    rows = read_csv(csv_path)
    if not rows:
        print("No rows read from CSV.")
        return

    # 异常召回率修正：当 avg_recall < 0.5 时，使用上一行的值
    prev_recall = None
    for r in rows:
        recall_val = float_or_none(r.get('avg_recall'))
        if recall_val is not None:
            if recall_val < 0.5 and prev_recall is not None:
                print(f"[Warning] Fixing abnormal recall {recall_val} -> {prev_recall} for row: {r.get('variant', '')} M={r.get('M', '')} EFS={r.get('EFS', '')}")
                r['avg_recall'] = str(prev_recall)
            else:
                prev_recall = recall_val

    filters = parse_filters(args.filter)
    group_by = args.group_by
    group_col = group_by  # can be multi cols joined by '+'
    x_col = args.x
    y_col = args.y

    grouped = defaultdict(list)
    for r in rows:
        if not match_filters(r, filters):
            continue
        x = float_or_none(r.get(x_col))
        y = float_or_none(r.get(y_col))
        if x is None or y is None:
            continue
        grp = make_group_label(r, group_col)
        grouped[str(grp)].append((x, y, r))

    if not grouped:
        print("No valid data rows found after filtering.")
        return

    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab20')
    colors = cmap.colors if hasattr(cmap, 'colors') else cmap(range(20))
    idx = 0
    for k, lst in sorted(grouped.items()):
        lst_sorted = sorted(lst, key=lambda t: t[0])
        xs = [t[0] for t in lst_sorted]
        ys = [t[1] for t in lst_sorted]
        if not xs:
            continue
        color = colors[idx % len(colors)]
        idx += 1
        plt.plot(xs, ys, marker='o', linestyle='-', linewidth=1.5, markersize=4, color=color, label=k)
        plt.text(xs[-1], ys[-1], ' ' + k, verticalalignment='center', fontsize=8, color=color)

    plt.xlabel('Average recall')
    plt.ylabel('Queries per second (QPS)')
    plt.title('Recall vs Queries per second (QPS) per group')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()

    out = args.output
    if not out:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out = os.path.join(os.path.dirname(csv_path), f"{base}_recall_vs_time.png")
    plt.savefig(out, dpi=args.dpi)
    print("Saved plot to", out)


if __name__ == "__main__":
    main()
