#!/usr/bin/env python3
"""
compute_ground_truth.py

读取数据集目录下的 base.txt（每行是向量，可能以 id 开头），随机选取若干 query（流式水塘抽样），
对每个 query 对整个数据集做流式遍历计算距离（L2 或 cosine），维护 top-k 最近邻（最小距离），
并把结果写成 JSON。

可选参数：--dataset 指定单个文件，未指定则尝试处理 data_o/glove/base.txt 和 data_o/sift/base.txt。
支持 --num_queries, --k, --metric (l2|cosine), --seed, --out, --max_base (仅处理前 N 个向量，便于 smoke-test)

实现目标：对大文件也能工作（不把整个文件一次性读入内存）。
"""

from __future__ import annotations

import argparse
import math
import heapq
import json
import random
from typing import List, Optional, Tuple, Iterable
import threading
from queue import Queue


def parse_vector_line(line: str) -> Optional[Tuple[Optional[str], List[float]]]:
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    # try to convert all to float
    try:
        vals = [float(x) for x in parts]
        # no id
        return None, vals
    except ValueError:
        # assume first token is id, rest are floats
        if len(parts) < 2:
            return None
        id0 = parts[0]
        try:
            vals = [float(x) for x in parts[1:]]
            return id0, vals
        except ValueError:
            return None


def l2_distance(a: List[float], b: List[float]) -> float:
    # assume same length
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return math.sqrt(s)


def cosine_distance(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 1.0
    cosine_sim = dot / (math.sqrt(na) * math.sqrt(nb))
    # convert similarity to distance in [0,2]
    return 1.0 - cosine_sim


def reservoir_sample_vectors(path: str, num: int, seed: int = 1) -> List[Tuple[int, Optional[str], List[float]]]:
    """流式从文件中随机抽取 num 个向量，返回 (line_index, id_or_None, vector) 的列表"""
    random.seed(seed)
    reservoir: List[Tuple[int, Optional[str], List[float]]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            parsed = parse_vector_line(line)
            if parsed is None:
                continue
            id0, vec = parsed
            item = (idx, id0, vec)
            if len(reservoir) < num:
                reservoir.append(item)
            else:
                j = random.randrange(idx + 1)
                if j < num:
                    reservoir[j] = item
    return reservoir


def topk_bruteforce_stream(path: str, queries: List[Tuple[int, Optional[str], List[float]]], k: int, metric: str = "l2", max_base: Optional[int] = None, threads: int = 4, batch_size: int = 1000, show_progress: bool = True):
    """多线程生产者-消费者实现的流式暴力 top-k 搜索。
    主线程分批读取文件（每 batch_size 行为一块），放入队列；工作线程消费块并更新每个 query 的堆。
    """
    metric_fn = l2_distance if metric == "l2" else cosine_distance

    # 每个 query 有自己的堆和锁
    heaps: List[List[Tuple[float, int, Optional[str], float]]] = [ [] for _ in range(len(queries)) ]
    locks = [threading.Lock() for _ in queries]

    q = Queue(maxsize=threads * 2)
    stop_sentinels = threads

    # try to import tqdm for nicer progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except Exception:
        tqdm = None
        use_tqdm = False

    processed = 0
    processed_lock = threading.Lock()

    def worker():
        nonlocal processed
        while True:
            batch = q.get()
            if batch is None:
                q.task_done()
                break
            for idx, id0, vec in batch:
                for qi, (_, qid, qvec) in enumerate(queries):
                    try:
                        dist = metric_fn(qvec, vec)
                    except Exception:
                        dist = float('inf')
                    # update heap for query qi
                    with locks[qi]:
                        if len(heaps[qi]) < k:
                            heapq.heappush(heaps[qi], (-dist, idx, id0, dist))
                        else:
                            if -heaps[qi][0][0] > dist:
                                heapq.heapreplace(heaps[qi], (-dist, idx, id0, dist))
                with processed_lock:
                    processed += 1
            q.task_done()

    # start worker threads
    workers = []
    for _ in range(max(1, threads)):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        workers.append(t)

    # Reading producer
    total_estimate = max_base if max_base is not None else None
    pbar = None
    if show_progress:
        if use_tqdm:
            pbar = tqdm(total=total_estimate, unit='vec')

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            batch = []
            for idx, line in enumerate(f):
                if max_base is not None and idx >= max_base:
                    break
                parsed = parse_vector_line(line)
                if parsed is None:
                    continue
                id0, vec = parsed
                batch.append((idx, id0, vec))
                if len(batch) >= batch_size:
                    q.put(batch)
                    if pbar is not None:
                        pbar.update(len(batch))
                    batch = []
            if batch:
                q.put(batch)
                if pbar is not None:
                    pbar.update(len(batch))
    finally:
        # send stop sentinels
        for _ in range(stop_sentinels):
            q.put(None)
        # wait for queue to be fully processed
        q.join()
        if pbar is not None:
            pbar.close()

    # collect results
    results = []
    for qi, (qidx, qid, qvec) in enumerate(queries):
        h = heaps[qi]
        neighbors = sorted([{'index': item[1], 'id': item[2], 'distance': item[3]} for item in h], key=lambda x: x['distance'])
        results.append({'query_index': qidx, 'query_id': qid, 'query_vector_len': len(qvec), 'neighbors': neighbors})
    return results


def process_dataset(path: str, num_queries: int, k: int, metric: str, seed: int, max_base: Optional[int], out: Optional[str], threads: int = 4, batch_size: int = 1000, show_progress: bool = True):
    print(f"Processing {path} (num_queries={num_queries}, k={k}, metric={metric}, max_base={max_base}, threads={threads}, batch_size={batch_size})")
    queries = reservoir_sample_vectors(path, num_queries, seed=seed)
    if not queries:
        print("No valid vectors found in", path)
        return None
    results = topk_bruteforce_stream(path, queries, k, metric=metric, max_base=max_base, threads=threads, batch_size=batch_size, show_progress=show_progress)
    payload = {
        'dataset': path,
        'metric': metric,
        'k': k,
        'num_queries': num_queries,
        'results': results,
    }
    if out:
        with open(out, 'w', encoding='utf-8') as fo:
            json.dump(payload, fo, ensure_ascii=False, indent=2)
        print(f"Wrote output to {out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='单个 base.txt 文件路径')
    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--metric', type=str, choices=['l2', 'cosine'], default='l2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', type=str, help='输出 JSON 文件路径（如果不指定则打印）')
    parser.add_argument('--max_base', type=int, default=None, help='仅处理前 N 个基向量（用于快速 smoke-test）')
    parser.add_argument('--threads', type=int, default=32, help='工作线程数')
    parser.add_argument('--batch_size', type=int, default=1000, help='主线程每次读入的向量批大小')
    parser.add_argument('--no_progress', action='store_true', help='禁用进度条（默认显示）')
    args = parser.parse_args()

    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [
            'data_o/glove/base.txt',
            'data_o/sift/base.txt'
        ]

    for ds in datasets:
        out_path = None
        if args.out:
            # if multiple datasets and single out specified, append dataset name
            if len(datasets) > 1:
                out_path = args.out.rsplit('.', 1)[0] + '_' + ds.replace('/', '_').replace('\\', '_').replace(':','') + '.json'
            else:
                out_path = args.out
        else:
            out_path = None
        try:
            process_dataset(ds, args.num_queries, args.k, args.metric, args.seed, args.max_base, out_path, threads=args.threads, batch_size=args.batch_size, show_progress=not args.no_progress)
        except FileNotFoundError:
            print(f"File not found: {ds}")
        except Exception as e:
            print(f"Error processing {ds}: {e}")


if __name__ == '__main__':
    main()
