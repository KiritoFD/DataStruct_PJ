import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_df(df):
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # coerce numeric columns, allow "NA"
    for col in ["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE", "ELAPSED_s", "RECALL", "AVG_QUERY_TIME_ms", "INDEX_BUILD_TIME_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def pareto_front(df, score_col="RECALL", cost_col="ELAPSED_s"):
    # higher score better, lower cost better
    arr = df[[score_col, cost_col]].to_numpy()
    is_pareto = np.ones(arr.shape[0], dtype=bool)
    for i, (s, c) in enumerate(arr):
        if not is_pareto[i]:
            continue
        # any j that has s_j >= s and c_j <= c, and one strictly better -> dominate
        dominated = ( (arr[:,0] >= s) & (arr[:,1] <= c) & ((arr[:,0] > s) | (arr[:,1] < c)) )
        dominated[i] = False
        is_pareto[dominated] = False
    return df[is_pareto]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_recall_vs_nprobe(df, out_png):
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="NPROBE", y="RECALL",
        hue="NUM_CENTROIDS", style="KMEANS_ITER",
        markers=True, dashes=False
    )
    plt.title("Recall vs NPROBE")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_recall_vs_centroids(df, out_png):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="NUM_CENTROIDS", y="RECALL", hue="NPROBE", style="KMEANS_ITER", markers=True, dashes=False)
    plt.title("Recall vs NUM_CENTROIDS")
    plt.xlabel("NUM_CENTROIDS")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_recall_box_by_iter(df, out_png):
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x="KMEANS_ITER", y="RECALL", palette="Set2")
    sns.stripplot(data=df, x="KMEANS_ITER", y="RECALL", color="black", size=3, jitter=True, alpha=0.6)
    plt.title("Recall distribution by KMEANS_ITER")
    plt.xlabel("KMEANS_ITER")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_heatmaps(df, out_dir):
    # heatmap per KMEANS_ITER: rows NUM_CENTROIDS, cols NPROBE
    for ki, g in df.groupby("KMEANS_ITER"):
        pivot = g.pivot_table(index="NUM_CENTROIDS", columns="NPROBE", values="RECALL", aggfunc="mean")
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label':'RECALL'})
        plt.title(f"Recall heatmap (KMEANS_ITER={ki})")
        plt.xlabel("NPROBE")
        plt.ylabel("NUM_CENTROIDS")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"recall_heatmap_iter_{ki}.png"), dpi=150)
        plt.close()

def plot_recall_vs_time(df, out_png):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(df["ELAPSED_s"], df["RECALL"], c=df["NPROBE"], s=np.clip(df["NUM_CENTROIDS"]/4, 10, 400), cmap="viridis", alpha=0.8)
    plt.colorbar(sc, label="NPROBE")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Recall")
    plt.title("Recall vs Elapsed time (point size ~ NUM_CENTROIDS)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_query_time_vs_nprobe(df, out_png):
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="NPROBE", y="AVG_QUERY_TIME_ms",
        hue="NUM_CENTROIDS", style="KMEANS_ITER",
        markers=True, dashes=False
    )
    plt.title("Avg Query Time vs NPROBE")
    plt.ylabel("Avg Query Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_query_time_vs_centroids(df, out_png):
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="NUM_CENTROIDS", y="AVG_QUERY_TIME_ms",
        hue="NPROBE", style="KMEANS_ITER",
        markers=True, dashes=False
    )
    plt.title("Avg Query Time vs NUM_CENTROIDS")
    plt.xlabel("NUM_CENTROIDS")
    plt.ylabel("Avg Query Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_recall_vs_query_time(df, out_png):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(
        df["AVG_QUERY_TIME_ms"], df["RECALL"],
        c=df["NPROBE"], s=np.clip(df["NUM_CENTROIDS"]/4, 10, 400),
        cmap="viridis", alpha=0.8
    )
    plt.colorbar(sc, label="NPROBE")
    plt.xlabel("Avg Query Time (ms)")
    plt.ylabel("Recall")
    plt.title("Recall vs Avg Query Time (point size ~ NUM_CENTROIDS)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_high_recall_time_stats(high_recall, outdir):
    # 直方图：ELAPSED_s
    plt.figure(figsize=(7,4))
    plt.hist(high_recall["ELAPSED_s"], bins=20, color='skyblue', edgecolor='k')
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Count")
    plt.title("Elapsed time distribution (RECALL > 0.98)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_elapsed_hist.png"), dpi=150)
    plt.close()

    # 直方图：AVG_QUERY_TIME_ms
    plt.figure(figsize=(7,4))
    plt.hist(high_recall["AVG_QUERY_TIME_ms"], bins=20, color='orange', edgecolor='k')
    plt.xlabel("Avg Query Time (ms)")
    plt.ylabel("Count")
    plt.title("Avg Query Time distribution (RECALL > 0.98)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_query_time_hist.png"), dpi=150)
    plt.close()

    # 散点图：ELAPSED_s vs AVG_QUERY_TIME_ms
    plt.figure(figsize=(6,5))
    plt.scatter(high_recall["ELAPSED_s"], high_recall["AVG_QUERY_TIME_ms"], alpha=0.7)
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Avg Query Time (ms)")
    plt.title("Elapsed vs Query Time (RECALL > 0.98)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_elapsed_vs_query_time.png"), dpi=150)
    plt.close()

    # 散点图：ELAPSED_s vs AVG_QUERY_TIME_ms，颜色为NPROBE，点大小为NUM_CENTROIDS
    plt.figure(figsize=(7,5))
    sc = plt.scatter(
        high_recall["ELAPSED_s"], high_recall["AVG_QUERY_TIME_ms"],
        c=high_recall["NPROBE"], s=np.clip(high_recall["NUM_CENTROIDS"]/4, 10, 400),
        cmap="viridis", alpha=0.8
    )
    plt.colorbar(sc, label="NPROBE")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Avg Query Time (ms)")
    plt.title("Elapsed vs Query Time (RECALL > 0.98)\n(color=NPROBE, size=NUM_CENTROIDS)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_elapsed_vs_query_time_vs.png"), dpi=150)
    plt.close()

    # 可选：NPROBE分布
    plt.figure(figsize=(7,4))
    plt.hist(high_recall["NPROBE"], bins=20, color='green', edgecolor='k')
    plt.xlabel("NPROBE")
    plt.ylabel("Count")
    plt.title("NPROBE distribution (RECALL > 0.98)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_nprobe_hist.png"), dpi=150)
    plt.close()

def plot_high_recall_query_time_vs_hyperparams(high_recall, outdir):
    # NPROBE vs AVG_QUERY_TIME_ms
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=high_recall,
        x="NPROBE", y="AVG_QUERY_TIME_ms",
        hue="NUM_CENTROIDS", style="KMEANS_ITER",
        markers=True, dashes=False
    )
    plt.title("High Recall (RECALL>0.98): Avg Query Time vs NPROBE")
    plt.ylabel("Avg Query Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_query_time_vs_nprobe.png"), dpi=150)
    plt.close()

    # NUM_CENTROIDS vs AVG_QUERY_TIME_ms
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=high_recall,
        x="NUM_CENTROIDS", y="AVG_QUERY_TIME_ms",
        hue="NPROBE", style="KMEANS_ITER",
        markers=True, dashes=False
    )
    plt.title("High Recall (RECALL>0.98): Avg Query Time vs NUM_CENTROIDS")
    plt.xlabel("NUM_CENTROIDS")
    plt.ylabel("Avg Query Time (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_query_time_vs_centroids.png"), dpi=150)
    plt.close()

    # KMEANS_ITER vs AVG_QUERY_TIME_ms（可选）
    plt.figure(figsize=(8,6))
    sns.boxplot(data=high_recall, x="KMEANS_ITER", y="AVG_QUERY_TIME_ms", palette="Set2")
    sns.stripplot(data=high_recall, x="KMEANS_ITER", y="AVG_QUERY_TIME_ms", color="black", size=3, jitter=True, alpha=0.6)
    plt.title("High Recall (RECALL>0.98): Query Time by KMEANS_ITER")
    plt.xlabel("KMEANS_ITER")
    plt.ylabel("Avg Query Time (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "high_recall_query_time_box_by_iter.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--csv", type=str, default="results/summary.csv", help="Path to summary CSV")
    parser.add_argument("--outdir", type=str, default="results/plots", help="Output directory for plots")
    parser.add_argument("--topk", type=int, default=20, help="Top-K rows to save by recall")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print("CSV not found:", args.csv)
        return

    ensure_dir(args.outdir)

    # Try reading as comma-separated first, then tab-separated if STATUS not found
    df = pd.read_csv(args.csv)
    df = clean_df(df)
    if "STATUS" not in df.columns:
        # Try reading as tab-separated
        df = pd.read_csv(args.csv, sep='\t')
        df = clean_df(df)
        if "STATUS" not in df.columns:
            print("ERROR: 'STATUS' column not found in CSV. Columns are:", list(df.columns))
            return

    # drop failed runs for numeric analyses
    good = df[df["STATUS"].str.startswith("OK", na=False)].copy()
    if good.empty:
        print("No successful runs to analyze.")
        return

    # PRIORITY 1: Recall-focused analyses
    # Top-K by recall (global)
    topk_recall = good.sort_values("RECALL", ascending=False).head(args.topk)
    topk_recall.to_csv(os.path.join(args.outdir, f"top_{args.topk}_by_recall.csv"), index=False)

    # recall aggregated by parameter combinations (mean)
    recall_stats = good.groupby(["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE"]).agg(
        RECALL_mean=("RECALL","mean"),
        RECALL_std=("RECALL","std"),
        ELAPSED_s_mean=("ELAPSED_s", "mean"),  # 新增
        AVG_QUERY_TIME_MS_mean=("AVG_QUERY_TIME_ms", "mean"),  # 新增
        runs=("RECALL","count")
    ).reset_index().sort_values("RECALL_mean", ascending=False)
    recall_stats.to_csv(os.path.join(args.outdir, "recall_aggregated.csv"), index=False)

    # Recall-first plots
    try:
        plot_recall_vs_nprobe(good, os.path.join(args.outdir, "recall_vs_nprobe.png"))
        plot_recall_vs_centroids(good, os.path.join(args.outdir, "recall_vs_centroids.png"))
        plot_recall_box_by_iter(good, os.path.join(args.outdir, "recall_box_by_iter.png"))
        plot_heatmaps(good, args.outdir)  # existing heatmaps per iteration
        # 新增时间相关可视化
        plot_query_time_vs_nprobe(good, os.path.join(args.outdir, "query_time_vs_nprobe.png"))
        plot_query_time_vs_centroids(good, os.path.join(args.outdir, "query_time_vs_centroids.png"))
        plot_recall_vs_query_time(good, os.path.join(args.outdir, "recall_vs_query_time.png"))
    except Exception as e:
        print("Plotting error (recall/time plots):", e)

    # keep the other analyses (time / combined) but place them after recall
    # basic stats (time-focused)
    stats = good.groupby(["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE"]).agg(
        RECALL_mean=("RECALL","mean"),
        ELAPSED_mean=("ELAPSED_s","mean"),
        AVG_QUERY_TIME_MS_mean=("AVG_QUERY_TIME_ms","mean"),
        INDEX_BUILD_TIME_s_mean=("INDEX_BUILD_TIME_s","mean"),
        runs=("RECALL","count")
    ).reset_index()
    stats.to_csv(os.path.join(args.outdir, "aggregated_stats.csv"), index=False)

    # other plots
    try:
        plot_recall_vs_time(good, os.path.join(args.outdir, "recall_vs_time.png"))
    except Exception as e:
        print("Plotting error (time plots):", e)

    # Pareto front preferring recall first (filter by recall then time)
    pareto = pareto_front(good, score_col="RECALL", cost_col="ELAPSED_s")
    pareto = pareto.sort_values(["RECALL","ELAPSED_s"], ascending=[False, True])
    pareto.to_csv(os.path.join(args.outdir, "pareto_front.csv"), index=False)

    # recall per second ranking
    good["RECALL_PER_SEC"] = good["RECALL"] / (good["ELAPSED_s"].replace(0, np.nan))
    topk_ratio = good.sort_values("RECALL_PER_SEC", ascending=False).head(args.topk)
    topk_ratio.to_csv(os.path.join(args.outdir, f"top_{args.topk}_by_recall_per_sec.csv"), index=False)

    # 统计召回率高于0.98的点的时间情况
    high_recall = good[good["RECALL"] > 0.98]
    high_recall_stats = {
        "count": len(high_recall),
        "ELAPSED_s_mean": high_recall["ELAPSED_s"].mean(),
        "ELAPSED_s_std": high_recall["ELAPSED_s"].std(),
        "ELAPSED_s_min": high_recall["ELAPSED_s"].min(),
        "ELAPSED_s_max": high_recall["ELAPSED_s"].max(),
        "AVG_QUERY_TIME_MS_mean": high_recall["AVG_QUERY_TIME_ms"].mean(),
        "AVG_QUERY_TIME_MS_std": high_recall["AVG_QUERY_TIME_ms"].std(),
        "AVG_QUERY_TIME_MS_min": high_recall["AVG_QUERY_TIME_ms"].min(),
        "AVG_QUERY_TIME_MS_max": high_recall["AVG_QUERY_TIME_ms"].max(),
    }
    # 保存详细点
    high_recall.to_csv(os.path.join(args.outdir, "high_recall_time_stats.csv"), index=False)

    # 可视化高召回率点的时间分布
    if not high_recall.empty:
        plot_high_recall_time_stats(high_recall, args.outdir)
        plot_high_recall_query_time_vs_hyperparams(high_recall, args.outdir)

    # Write a short recall-first text summary
    with open(os.path.join(args.outdir, "analysis_summary.txt"), "w") as f:
        f.write("Recall-first Analysis summary\n")
        f.write("=============================\n\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write(f"Successful runs: {len(good)}\n\n")
        best = good.loc[good["RECALL"].idxmax()]
        f.write("Best recall run (global):\n")
        f.write(best.to_string() + "\n")
        f.write(f"\nElapsed time (s): {best['ELAPSED_s']}, AVG_QUERY_TIME_ms: {best['AVG_QUERY_TIME_ms']}\n\n")
        f.write("Top configurations by mean recall (recall_aggregated.csv)\n")
        # 只显示部分列，突出召回率和平均时间
        top10 = recall_stats.head(10)[["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE", "RECALL_mean", "ELAPSED_s_mean", "AVG_QUERY_TIME_MS_mean", "runs"]]
        f.write(top10.to_string(index=False, float_format="%.4f") + "\n\n")
        f.write("Pareto front (recall then time) saved to: pareto_front.csv\n")
        f.write(f"Top-{args.topk} runs by recall saved to: top_{args.topk}_by_recall.csv\n")
        f.write(f"Top-{args.topk} runs by recall/sec saved to: top_{args.topk}_by_recall_per_sec.csv\n")
        f.write("统计召回率高于0.98的点的时间情况：\n")
        f.write(f"数量: {high_recall_stats['count']}\n")
        f.write(f"ELAPSED_s: mean={high_recall_stats['ELAPSED_s_mean']:.4f}, std={high_recall_stats['ELAPSED_s_std']:.4f}, min={high_recall_stats['ELAPSED_s_min']:.4f}, max={high_recall_stats['ELAPSED_s_max']:.4f}\n")
        f.write(f"AVG_QUERY_TIME_MS: mean={high_recall_stats['AVG_QUERY_TIME_MS_mean']:.4f}, std={high_recall_stats['AVG_QUERY_TIME_MS_std']:.4f}, min={high_recall_stats['AVG_QUERY_TIME_MS_min']:.4f}, max={high_recall_stats['AVG_QUERY_TIME_MS_max']:.4f}\n")
        f.write(f"详细点保存于: high_recall_time_stats.csv\n\n")

    print("Recall-first analysis complete. Plots and CSVs in", args.outdir)

if __name__ == "__main__":
    main()