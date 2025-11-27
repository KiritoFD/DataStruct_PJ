import os
import re
import csv
import shutil
from datetime import datetime

LOG_DIR = 'Log'
OUT_CSV = 'optuna_adaptive_hng.csv'
BACKUP_SUFFIX = '.bak'

# Scoring constants (same as bias_hn.py)
MIN_RECALL = 0.9803
SUCCESS_BASE = 1000.0
RECALL_BONUS = 100.0
PENALTY_BASE = 500.0

FIELDNAMES = ['timestamp','trial_id','m','max_layer','efc','efs','K','avg_recall','avg_time_ms','build_time_ms','reward_score','status']

# Script-level options
WRITE_ONLY_COMPLETE = False  # If True, rebuild will write only rows with status 'COMPLETE'

def parse_log_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    # try multiple patterns
    def find_int(pattern):
        m = re.search(pattern, txt, re.I)
        return int(m.group(1)) if m else None

    def find_float(pattern):
        m = re.search(pattern, txt, re.I)
        return float(m.group(1).replace(',', '.')) if m else None

    # command-line style params
    m = find_int(r"--m\s+(\d+)") or find_int(r"M:\s*(\d+)") or find_int(r"\bM=(\d+)\b")
    max_layer = find_int(r"--max_layer\s+(\d+)") or find_int(r"Max\s*layer[:=]?\s*(\d+)") or find_int(r"max_layer[:=]?\s*(\d+)")
    efc = find_int(r"--efc\s+(\d+)") or find_int(r"EFC[:=]?\s*(\d+)")
    efs = find_int(r"--efs\s+(\d+)") or find_int(r"EFS[:=]?\s*(\d+)")

    # metrics
    recall = find_float(r"Average\s+recall@\d+[:\s]+([0-9]+(?:[.,][0-9]+)?)")
    avg_time = find_float(r"Average\s+query\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms")
    build_time = find_float(r"Index\s+build\s+time[:\s]+([0-9]+(?:[.,][0-9]+)?)\s*ms")

    # timestamp: try to get from log content or file mtime
    tstamp = None
    mtime_match = re.search(r"Timestamp:\s*([0-9T:\-\. ]+)", txt)
    if mtime_match:
        tstamp = mtime_match.group(1).strip()
    else:
        try:
            ts = os.path.getmtime(path)
            tstamp = datetime.fromtimestamp(ts).isoformat()
        except Exception:
            tstamp = datetime.now().isoformat()

    # default values
    if max_layer is None:
        max_layer = 16

    status = 'COMPLETE' if (recall is not None and avg_time is not None and build_time is not None) else 'PARSE_ERROR'

    # compute reward_score using same logic
    if recall is None:
        reward = -PENALTY_BASE
    elif recall < MIN_RECALL:
        gap = MIN_RECALL - recall
        reward = -PENALTY_BASE * (1.0 + gap * 10.0)
    else:
        reward = (SUCCESS_BASE / max(avg_time if avg_time is not None else 0.001, 0.001)) + (recall - MIN_RECALL) * RECALL_BONUS

    # K: attempt to parse from content
    K = find_int(r"@?(\d+):" )
    if K is None:
        K = 10

    return {
        'timestamp': tstamp,
        'm': m if m is not None else '',
        'max_layer': max_layer,
        'efc': efc if efc is not None else '',
        'efs': efs if efs is not None else '',
        'K': K,
        'avg_recall': round(recall, 6) if recall is not None else '',
        'avg_time_ms': round(avg_time, 6) if avg_time is not None else '',
        'build_time_ms': round(build_time, 6) if build_time is not None else '',
        'reward_score': reward,
        'status': status
    }


def rebuild_csv_from_logs():
    if not os.path.isdir(LOG_DIR):
        print(f"Log dir '{LOG_DIR}' not found.")
        return

    files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    files.sort()
    print(f"Found {len(files)} log files in {LOG_DIR}")
    parsed_entries = []
    for p in files:
        parsed = parse_log_file(p)
        if not parsed:
            continue
        parsed['source'] = os.path.basename(p)
        parsed_entries.append(parsed)

    total_parsed = len(parsed_entries)
    complete_parsed = sum(1 for e in parsed_entries if e.get('status') == 'COMPLETE')
    print(f"Parsed {total_parsed} log entries, {complete_parsed} complete entries.")

    if not parsed_entries:
        print('No log entries parsed.')
        return

    # Deduplicate by key (m, max_layer, efc, efs) to align with CACHE key used by load_cache_data
    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            return None

    best_map = {}
    for e in parsed_entries:
        key = (to_int_safe(e.get('m')), to_int_safe(e.get('max_layer')), to_int_safe(e.get('efc')), to_int_safe(e.get('efs')))
        if any(k is None for k in key):
            uniq = ("_BAD_", e.get('source'), e.get('timestamp'))
            best_map[uniq] = e
            continue

        existing = best_map.get(key)
        if existing is None:
            best_map[key] = e
            continue

        def status_rank(rec):
            return 1 if rec.get('status') == 'COMPLETE' else 0

        if status_rank(e) > status_rank(existing):
            best_map[key] = e
            continue
        if status_rank(e) < status_rank(existing):
            continue

        try:
            nr = float(e.get('reward_score') or -1e18)
        except Exception:
            nr = -1e18
        try:
            er = float(existing.get('reward_score') or -1e18)
        except Exception:
            er = -1e18
        if nr > er:
            best_map[key] = e
            continue

        try:
            new_ts = datetime.fromisoformat(e.get('timestamp'))
        except Exception:
            new_ts = datetime.min
        try:
            exist_ts = datetime.fromisoformat(existing.get('timestamp'))
        except Exception:
            exist_ts = datetime.min
        if new_ts > exist_ts:
            best_map[key] = e
            continue

    # Build list and sort by timestamp
    rows = list(best_map.values())

    def parse_ts(x):
        try:
            return datetime.fromisoformat(x.get('timestamp'))
        except Exception:
            return datetime.min

    rows.sort(key=parse_ts)
    deduped_count = len(rows)

    # Optionally filter only COMPLETE rows
    if WRITE_ONLY_COMPLETE:
        rows = [r for r in rows if r.get('status') == 'COMPLETE']

    written_count = len(rows)

    print(f"After dedup: {deduped_count} unique combos, writing {written_count} rows (WRITE_ONLY_COMPLETE={WRITE_ONLY_COMPLETE}).")

    # Backup existing CSV and write
    if os.path.exists(OUT_CSV):
        bak = OUT_CSV + BACKUP_SUFFIX
        print(f'Backing up {OUT_CSV} -> {bak}')
        shutil.copy2(OUT_CSV, bak)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as wf:
        writer = csv.writer(wf)
        writer.writerow(FIELDNAMES)
        for i, e in enumerate(rows):
            ts = e.get('timestamp') or datetime.now().isoformat()
            m = e.get('m') if e.get('m') is not None else ''
            max_layer = e.get('max_layer') if e.get('max_layer') is not None else 16
            efc = e.get('efc') if e.get('efc') is not None else ''
            efs = e.get('efs') if e.get('efs') is not None else ''
            K = e.get('K') if e.get('K') is not None else 10
            avg_recall = e.get('avg_recall', '')
            avg_time_ms = e.get('avg_time_ms', '')
            build_time_ms = e.get('build_time_ms', '')
            reward = e.get('reward_score', '')
            status = e.get('status', '')
            writer.writerow([ts, i, m, max_layer, efc, efs, K, avg_recall, avg_time_ms, build_time_ms, reward, status])

    print(f'Wrote {written_count} rows to {OUT_CSV} (total logs parsed: {total_parsed}, complete parsed: {complete_parsed}').format())

def recalc_csv_scores(in_csv, out_csv, min_recall, success_base, recall_bonus, penalty_base, backup=True, dedupe=False):
    if not os.path.exists(in_csv):
        print(f"CSV {in_csv} not found.")
        return

    if backup:
        bak = in_csv + BACKUP_SUFFIX
        print(f'Backing up {in_csv} -> {bak}')
        shutil.copy2(in_csv, bak)

    rows = []
    with open(in_csv, 'r', newline='', encoding='utf-8') as rf:
        reader = csv.DictReader(rf)
        headers = reader.fieldnames or FIELDNAMES
        for row in reader:
            rows.append(row)

    def pval(v):
        try:
            return float(str(v).replace(',', '.'))
        except Exception:
            return None

    if dedupe:
        # group by (m, max_layer, efc, efs)
        groups = {}
        for row in rows:
            try:
                key = (int(str(row.get('m'))), int(str(row.get('max_layer'))), int(str(row.get('efc'))), int(str(row.get('efs'))))
            except Exception:
                key = (row.get('m'), row.get('max_layer'), row.get('efc'), row.get('efs'))
            groups.setdefault(key, []).append(row)

        new_rows = []
        for key, members in groups.items():
            # aggregate means over numeric values
            recs = [pval(r.get('avg_recall')) for r in members if pval(r.get('avg_recall')) is not None]
            times = [pval(r.get('avg_time_ms')) for r in members if pval(r.get('avg_time_ms')) is not None]
            builds = [pval(r.get('build_time_ms')) for r in members if pval(r.get('build_time_ms')) is not None]

            avg_recall = sum(recs)/len(recs) if recs else None
            avg_time = sum(times)/len(times) if times else None
            avg_build = sum(builds)/len(builds) if builds else None

            # pick representative m, max_layer, efc, efs
            m, max_layer, efc, efs = key
            # if key contains non-int (bad), fallback to first member
            if not all(isinstance(x, int) for x in key):
                first = members[0]
                m = first.get('m')
                max_layer = first.get('max_layer')
                efc = first.get('efc')
                efs = first.get('efs')

            # compute reward
            if avg_recall is None:
                new_reward = -penalty_base
                new_status = 'PARSE_ERROR'
            elif avg_recall < min_recall:
                gap = min_recall - avg_recall
                new_reward = -penalty_base * (1.0 + gap * 10.0)
                new_status = 'INFEASIBLE'
            else:
                new_reward = (success_base / max(avg_time if avg_time is not None else 0.001, 0.001)) + (avg_recall - min_recall) * recall_bonus
                new_status = 'COMPLETE'

            new_rows.append({
                'timestamp': members[0].get('timestamp') or datetime.now().isoformat(),
                'm': m,
                'max_layer': max_layer if max_layer != '' else 16,
                'efc': efc,
                'efs': efs,
                'K': members[0].get('K') or 10,
                'avg_recall': round(avg_recall, 6) if avg_recall is not None else '',
                'avg_time_ms': round(avg_time, 6) if avg_time is not None else '',
                'build_time_ms': round(avg_build, 6) if avg_build is not None else '',
                'reward_score': new_reward,
                'status': new_status
            })

        rows = new_rows
    else:
        for row in rows:
            avg_recall = pval(row.get('avg_recall'))
            avg_time = pval(row.get('avg_time_ms'))
            build_time = pval(row.get('build_time_ms'))

            if avg_recall is None:
                new_reward = -penalty_base
                new_status = row.get('status') or 'PARSE_ERROR'
            elif avg_recall < min_recall:
                gap = min_recall - avg_recall
                new_reward = -penalty_base * (1.0 + gap * 10.0)
                new_status = row.get('status') or 'INFEASIBLE'
            else:
                new_reward = (success_base / max(avg_time if avg_time is not None else 0.001, 0.001)) + (avg_recall - min_recall) * recall_bonus
                new_status = 'COMPLETE'

            row['reward_score'] = str(new_reward)
            row['status'] = new_status

    # Write back using the standard FIELDNAMES
    with open(out_csv, 'w', newline='', encoding='utf-8') as wf:
        writer = csv.DictWriter(wf, fieldnames=FIELDNAMES, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for i, row in enumerate(rows):
            # ensure trial id is coherent
            row['trial_id'] = i
            # fill missing max_layer
            if row.get('max_layer') in (None, ''):
                row['max_layer'] = 16
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})

    print(f'Updated reward scores for {len(rows)} entries in {out_csv}')

def compare_logs_csv(in_csv=OUT_CSV):
    # Parse logs
    files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.log')]
    logs = []
    for p in files:
        parsed = parse_log_file(p)
        if parsed and parsed.get('m') not in ('', None):
            logs.append(parsed)

    def key_of_entry(e):
        try:
            return (int(e.get('m')), int(e.get('max_layer')), int(e.get('efc')), int(e.get('efs')))
        except Exception:
            return (e.get('m'), e.get('max_layer'), e.get('efc'), e.get('efs'))

    log_keys = set(key_of_entry(e) for e in logs if e is not None)

    # Parse csv
    csv_keys = set()
    csv_rows = []
    if os.path.exists(in_csv):
        with open(in_csv, 'r', newline='', encoding='utf-8') as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                try:
                    k = (int(str(row.get('m'))), int(str(row.get('max_layer'))), int(str(row.get('efc'))), int(str(row.get('efs'))))
                except Exception:
                    k = (row.get('m'), row.get('max_layer'), row.get('efc'), row.get('efs'))
                csv_keys.add(k)
                csv_rows.append(row)

    overlap = log_keys.intersection(csv_keys)
    logs_only = log_keys - csv_keys
    csv_only = csv_keys - log_keys

    print("Compare Logs vs CSV")
    print("=====================")
    print(f"Log files parsed: {len(files)}, valid entries: {len(logs)}")
    print(f"Unique combos in logs: {len(log_keys)}")
    print(f"Unique combos in CSV: {len(csv_keys)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Only in logs: {len(logs_only)}")
    print(f"Only in CSV: {len(csv_only)}")

    # optional: print a few examples
    if len(logs_only) > 0:
        print('\nExample combos in logs but not in CSV:')
        for i, k in enumerate(list(logs_only)[:10]):
            print('  ', k)
    if len(csv_only) > 0:
        print('\nExample combos in CSV but not in logs:')
        for i, k in enumerate(list(csv_only)[:10]):
            print('  ', k)

# Script mode configuration (edit here)
MODE = 'rebuild'  # 'rebuild' or 'recalc'
INFILE = OUT_CSV
OUTFILE = OUT_CSV
DEDUPE = False

if __name__ == '__main__':
    if MODE == 'rebuild':
        rebuild_csv_from_logs()
        if DEDUPE:
            # rebuild already performs deduplication
            pass
    elif MODE == 'recalc':
        recalc_csv_scores(INFILE, OUTFILE, MIN_RECALL, SUCCESS_BASE, RECALL_BONUS, PENALTY_BASE, dedupe=DEDUPE)
    elif MODE == 'compare':
        compare_logs_csv(INFILE)
    else:
        print(f"Unknown MODE: {MODE}")
