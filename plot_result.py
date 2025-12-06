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
import re
from matplotlib import colors as mcolors

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

# --- default excludes: edit this list to hardcode excludes in the script ---
# 支持三种格式：
#   "no_csr"          -> 子串匹配
#   "=full"           -> 精确匹配 "full"
#   "re:variant=.*42" -> 正则匹配
DEFAULT_EXCLUDE_PATTERNS = ["no_csr", "nothing","no_prefetch"]
# --- end defaults ---

# helper: compile list of pattern strings into usable patterns
def compile_exclude_patterns(items):
    """
    items: iterable of pattern strings, e.g. ["no_csr", "=full", "re:variant=.*"]
    returns list of tuples: ("re", compiled_regex) or ("eq", string) or ("sub", string)
    """
    if not items:
        return []
    parts = []
    for it in items:
        if it is None:
            continue
        # allow comma-separated entries in each item
        for p in str(it).split(','):
            p = p.strip()
            if not p:
                continue
            if p.startswith("re:"):
                try:
                    parts.append(("re", re.compile(p[3:])))
                except re.error:
                    # invalid regex -> treat as substring
                    parts.append(("sub", p[3:]))
            elif p.startswith("="):
                parts.append(("eq", p[1:]))
            else:
                parts.append(("sub", p))
    return parts

def is_excluded(label, compiled_patterns):
    if not compiled_patterns:
        return False
    for typ, pat in compiled_patterns:
        if typ == "re":
            if pat.search(label):
                return True
        elif typ == "eq":
            if label == pat:
                return True
        else:  # sub
            if pat in label:
                return True
    return False


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
    default_csv = "./results_1206.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', nargs='?', default=default_csv, help='CSV file path (default provided results path)')
    parser.add_argument('-o', '--output', help='Output image path', default=None)
    parser.add_argument('--filter', help='Comma separated param filters like M=40,ML=17', default=None)
    parser.add_argument('--group-by', help='Column(s) to group by (use + to combine)', default='variant')
    parser.add_argument('--x', help='X axis column', default='avg_recall')
    parser.add_argument('--y', help='Y axis column', default='queries_per_sec')
    parser.add_argument('--dpi', help='Output DPI', type=int, default=441)
    # lower/upper bounds for avg_recall (inclusive). Use -d for lower, -u for upper.
    parser.add_argument('-d', '--lower', type=float, help='Lower bound for avg_recall (inclusive)', default=None)
    parser.add_argument('-u', '--upper', type=float, help='Upper bound for avg_recall (inclusive)', default=None)
    # compatibility: -high sets lower bound to 0.99 if lower not provided
    parser.add_argument('-high', '--high', action='store_true', help='Compatibility: set lower bound to 0.99 if --lower not provided')
    parser.add_argument('--low-recall-threshold', type=float, default=0.5, help='Threshold below which avg_recall is considered abnormal and will be patched')
    # allow multiple --exclude flags; each may also contain comma-separated patterns
    parser.add_argument('--exclude', help='Comma-separated patterns (or repeat option). Prefix "re:" for regex, "=" for exact, otherwise substring match', default=None, action='append')
    args = parser.parse_args()
    LOW_RECALL_THRESHOLD = args.low_recall_threshold

    # group_col needed for exclusion checks during recall fix stage
    group_by = args.group_by
    group_col = group_by  # can be multi cols joined by '+'

    # combine excludes from environment and CLI (no hardcoded defaults)
    combined_excludes = []
    env_excl = os.environ.get('PLOT_EXCLUDES')
    if env_excl:
        for p in str(env_excl).split(','):
            p = p.strip()
            if p:
                combined_excludes.append(p)
    if args.exclude:
        combined_excludes.extend(args.exclude)
    combined_excludes = [c for c in (dict.fromkeys([x.strip() for x in combined_excludes if x and str(x).strip()]))]
    print("Excluding groups:", ", ".join(combined_excludes) if combined_excludes else "(none)")
    compiled_excludes = compile_exclude_patterns(combined_excludes)

    # validate and normalize recall bounds: swap if lower > upper
    if args.lower is not None and args.upper is not None and args.lower > args.upper:
        # swap to avoid filtering out everything
        old_lower, old_upper = args.lower, args.upper
        args.lower, args.upper = args.upper, args.lower
        print(f"[Warning] lower ({old_lower}) > upper ({old_upper}); swapped to lower={args.lower}, upper={args.upper}")

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print("CSV not found:", csv_path)
        return

    rows = read_csv(csv_path)
    if not rows:
        print("No rows read from CSV.")
        return

    # 异常召回率修正（改进）：按 variant 维护上一次有效召回率，仅在相同 variant 内填补
    last_recall_by_variant = {}
    # New: track last recall per hyperparameter set (M, ML, EFC, EFS)
    last_recall_by_hyper = {}

    for r in rows:
        # skip excluded groups entirely during preprocessing
        grp = make_group_label(r, group_col)
        if is_excluded(grp, compiled_excludes):
            continue
        variant = (r.get('variant') or '').strip()
        # build hyperparam key, fallback to empty string if missing parts
        m_key = r.get('M', '')
        ml_key = r.get('ML', r.get('max_layer', ''))
        efc_key = r.get('EFC', r.get('efc', ''))
        efs_key = r.get('EFS', r.get('efs', ''))
        hyper_key = f"M={m_key}|ML={ml_key}|EFC={efc_key}|EFS={efs_key}"

        recall_val = float_or_none(r.get('avg_recall'))
        if recall_val is not None:
            if recall_val < LOW_RECALL_THRESHOLD:
                # prefer hyperparam-level fill if available
                if hyper_key and hyper_key in last_recall_by_hyper:
                    prev = last_recall_by_hyper[hyper_key]
                    print(f"[Warning] Fixing abnormal recall {recall_val} -> {prev} for hyper={hyper_key} (row: {variant} M={m_key} EFS={efs_key})")
                    r['avg_recall'] = str(prev)
                elif variant and variant in last_recall_by_variant:
                    prev = last_recall_by_variant[variant]
                    print(f"[Warning] Fixing abnormal recall {recall_val} -> {prev} for row: {variant} M={m_key} EFS={efs_key} (fallback variant)")
                    r['avg_recall'] = str(prev)
                else:
                    print(f"[Warning] Abnormal recall {recall_val} for row: {variant} M={m_key} EFS={efs_key} (no previous for hyper/variant, left unchanged)")
            else:
                # normal recall: update both hyperparam and variant states
                last_recall_by_hyper[hyper_key] = recall_val
                last_recall_by_variant[variant] = recall_val
            # end recall handling
            continue

    filters = parse_filters(args.filter)
    x_col = args.x
    y_col = args.y

    grouped = defaultdict(list)

    # Counters to help debug empty filtering result
    total_rows = len(rows)
    pre_filters_count = 0
    match_filters_skipped = 0
    recall_pass_count = 0
    recall_skipped = 0
    excluded_count = 0
    xy_missing_skipped = 0
    kept_count = 0

    for r in rows:
        # compute group label and apply excludes immediately (skip reading/processing)
        grp = make_group_label(r, group_col)
        if is_excluded(grp, compiled_excludes):
            excluded_count += 1
            continue

        # apply user-provided match filters
        if not match_filters(r, filters):
            match_filters_skipped += 1
            continue
        pre_filters_count += 1

        # optionally filter by recall bounds (lower/upper). -high sets lower=0.99 if lower not provided.
        recall_val = float_or_none(r.get('avg_recall'))
        lower = args.lower
        upper = args.upper
        if args.high and lower is None:
            lower = 0.99
        if lower is not None or upper is not None:
            # if recall missing, skip
            if recall_val is None:
                recall_skipped += 1
                continue
            if lower is not None and recall_val < lower:
                recall_skipped += 1
                continue
            if upper is not None and recall_val > upper:
                recall_skipped += 1
                continue
        recall_pass_count += 1

        x = float_or_none(r.get(x_col))
        y = float_or_none(r.get(y_col))
        if x is None or y is None:
            xy_missing_skipped += 1
            continue

        # grp already computed above
        # --- exclude at read-time: skip rows whose group label matches compiled_excludes ---
        # if is_excluded(grp, compiled_excludes):
        #     # optional: uncomment the next line to log excluded rows
        #     # print(f"Skipping excluded group {grp}")
        #     excluded_count += 1
        #     continue
        # --- end exclusion check ---

        kept_count += 1
        grouped[str(grp)].append((x, y, r))

    # Debug summary
    print(f"Rows total={total_rows}, after_match_filters={pre_filters_count}, recall_pass={recall_pass_count}, "
          f"excluded={excluded_count}, match_skipped={match_filters_skipped}, recall_skipped={recall_skipped}, "
          f"xy_missing={xy_missing_skipped}, kept={kept_count}")

    if not grouped:
        print("No valid data rows found after filtering. Saving an empty placeholder plot.")
        # Create an empty plot with explanatory message
        plt.figure(figsize=(12, 7))
        plt.text(0.5, 0.5, "No data after filters/excludes", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        out = args.output
        if not out:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            out = os.path.join(os.path.dirname(csv_path), f"{base}_recall_vs_time.png")
        plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print("Saved empty plot to", out)
        return

    # --- 绘图优化部分 ---
    # 更大画布便于展示标签和图例
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Helper: generate N visually distinct colors across the hue space, optionally excluding some hex colors
    def distinct_colors(n, exclude=None, s=0.65, v=0.95):
        if n <= 0:
            return []
        exclude_set = set([e.lower() for e in (exclude or [])])
        cols = []
        attempts = 0
        while len(cols) < n and attempts < n * 10:
            # spread hues evenly, small jitter to avoid clashes if exclusion present
            h = ((len(cols) + 0.0) / n + 0.02 * attempts) % 1.0
            rgb = mcolors.hsv_to_rgb((h, s, v))
            hexc = mcolors.to_hex(rgb)
            if hexc.lower() in exclude_set:
                attempts += 1
                continue
            cols.append(hexc)
            attempts += 1
        # if we somehow didn't fill, fallback to repeating distinct ones
        while len(cols) < n:
            cols.append(mcolors.to_hex(mcolors.hsv_to_rgb(((len(cols) % n)/n, s, v))))
        return cols

    color_map = {
        # 【顶层】性能最优的几条线，需要极高对比度
        "full": "#D62728",            # **深红** - 作为基准线，代表最优解，醒目且权威

        "dynamic_struct": "#17BECF",  # **经典道奇蓝** - 与深红形成完美互补，清晰易辨

        "heap": "#FF7F0E",            # **橙色** - 非常醒目，在白底上与深红和蓝色都形成强烈对比

        "no_pruning": "#9467BD",      # **紫色** - 中等饱和度，与红、蓝、绿都不同，易于区分

        "no_reorder": "#E377C2",      # **洋红** - 明亮但不刺眼，用于表示“排序”功能的缺失

        "no_simd": "#FFD700",         # **金色/黄色** - 非常醒目，在白底上与深红和蓝色都形成强烈对比

        "dynamic_with_opts": "#8C564B", # **棕色** - 沉稳的暖色，表示“带优化的动态结构”

        # 【中层】性能中等，用稍暗或不同的色调
        "no_csr": "#2CA02C",          # **绿色** - 标准森林绿，与红色、蓝色形成经典三原色对比

        "no_prefetch": "#1F77B4",     # **青色** - 清晰明亮，与绿色和蓝色都不同

        # 【底层】性能最差，用黑色或深灰
        "nothing": "#000000",         # **黑色** - 作为最低性能基线，颜色最深，视觉上最弱，与 `full` 的深红形成最强对比
    }

    HARD_VARIANTS = list(color_map.keys())

    # Build ordered_keys: only include keys whose variant is in HARD_VARIANTS, preserve order by HARD_VARIANTS
    def extract_variant_from_label(label):
        # label looks like "variant=full|M=43|ML=..." or "full"
        m = re.search(r'(?:^|[|])variant=([^|]+)(?:$|[|])', label)
        if m:
            return m.group(1)
        # fallback: maybe label is just "full"
        if label in HARD_VARIANTS:
            return label
        return None

    all_keys = sorted(grouped.keys())
    # Create ordered_keys by expanding HARD_VARIANTS: for each hard variant, include matching keys in all_keys (in original order)
    ordered_keys = []
    for hv in HARD_VARIANTS:
        for k in all_keys:
            v = extract_variant_from_label(k)
            if v == hv:
                ordered_keys.append(k)
    # Now 'ordered_keys' only contains groups from HARD_VARIANTS (in requested order), other groups are ignored.

    # label vertical offsets expressed as fraction of y_range (percentages).
    LABEL_Y_OFFSET = {
        "full": 0.00,
        "no_prefetch": -0.02,
        "no_simd": -0.05,
        "no_pruning": 0.02,
        "heap": 0.00,
        "dynamic_struct": 0.04,
        "no_reorder": 0.01,
        "dynamic_with_opts": -0.01,
        "no_csr": 0,
        "nothing": 0,
    }

    # Prepare collision avoidance variables:
    all_ys = [p[1] for grp in grouped.values() for p in grp] if grouped else []
    if all_ys:
        y_min = min(all_ys)
        y_max = max(all_ys)
        y_range = (y_max - y_min) if (y_max > y_min) else 1.0
    else:
        y_range = 1.0

    collision_threshold = max(1e-6, 0.02 * y_range)
    used_label_ys = []

    # Prepare colors for non-specified mapped variants using distinct palette if needed
    highlight_color = color_map.get("full", "#d62728")
    # compute how many keys remain that don't have an entry in VARIANT_COLOR_MAP
    unmapped_keys = [k for k in ordered_keys if (extract_variant_from_label(k) not in color_map)]
    nonfull_palette = distinct_colors(len(unmapped_keys), exclude=[highlight_color, color_map.get("no_simd", "#000000")])
    # count non-full keys for non-full palette; avoid using undefined full_key
    nonfull_count = sum(1 for k in ordered_keys if extract_variant_from_label(k) != "full")
    # Ensure palette length matches expected
    if len(nonfull_palette) < nonfull_count:
        nonfull_palette = distinct_colors(nonfull_count, exclude=[highlight_color, color_map.get("no_simd", "#000000")])
    idx = 0
    for k in ordered_keys:
        lst = grouped[k]
        lst_sorted = sorted(lst, key=lambda t: t[0])
        xs = [t[0] for t in lst_sorted]
        ys = [t[1] for t in lst_sorted]
        if not xs:
            continue
        # Determine base variant and color
        bv = extract_variant_from_label(k)
        if bv is None:
            # Shouldn't happen: skip
            continue
        # Use the hardcoded color if present; otherwise use generated palette
        if bv in color_map:
            color = color_map[bv]
        else:
            color = nonfull_palette[idx % len(nonfull_palette)]
            idx += 1
        # full gets highlighted style
        if bv == "full":
            lw = 3.5
            ms = 6
            z = 12
        else:
            lw = 2.0
            ms = 4   # 缩小非 full 的点半径以降低视觉干扰
            z = 5
        # Special: ensure no_simd uses black and stands out a bit
        if bv == "no_simd":
            color = color_map.get("no_simd", "#000000")
            lw = 2.5
            ms = 4   # 缩小 no_simd 点半径但保持其较高可见性
            z = 9

        # ensure marker edge (linewidth) for visibility
        ax.plot(xs, ys, marker='o', linestyle='-', linewidth=lw, markersize=ms, color=color, label=f"{bv}", zorder=z,
                 markeredgewidth=0.5, markeredgecolor='white')

        x_last, y_last = xs[-1], ys[-1]
        offset = LABEL_Y_OFFSET.get(bv, 0)
        # offset is a fraction of the total y_range
        y_text = y_last + (offset * y_range)

        # Collision avoidance: if too close to any used label y, shift up/down until safe
        def is_conflicting(y):
            for yy in used_label_ys:
                if abs(y - yy) < collision_threshold:
                    return True
            return False

        # initial strategy: apply offset sign; attempt small steps to separate
        step = collision_threshold * 0.5 * (1 if offset >= 0 else -1)
        adj_sign = 1 if offset >= 0 else -1
        attempts = 0
        while is_conflicting(y_text) and attempts < 50:
            y_text += step
            attempts += 1
        used_label_ys.append(y_text)

        # Text bbox to ensure readability when overlapping with markers or grid.
        bbox = dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.6)
        ax.text(
            x_last, 
            y_text, 
            ' ' + bv, 
            horizontalalignment='left', 
            verticalalignment='center', 
            fontsize=9, 
            color=color, 
            zorder=z + 1,
            bbox=bbox
        )

    ax.set_xlabel('Average recall', fontsize=12)
    ax.set_ylabel('Queries per second (QPS)', fontsize=12)
    ax.set_title('Recall vs Queries per second (QPS) per group', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    # Force legend, place outside to the right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9, title='Variant', frameon=True)

    plt.tight_layout()

    # determine output file path and save
    out = args.output
    if not out:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out = os.path.join(os.path.dirname(csv_path), f"{base}_recall_vs_time.png")
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print("Saved plot to", out)


if __name__ == "__main__":
    main()
