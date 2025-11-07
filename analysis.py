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
    df = pd.read_csv(args.csv)
    df = clean_df(df)

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
        runs=("RECALL","count")
    ).reset_index().sort_values("RECALL_mean", ascending=False)
    recall_stats.to_csv(os.path.join(args.outdir, "recall_aggregated.csv"), index=False)

    # Recall-first plots
    try:
        plot_recall_vs_nprobe(good, os.path.join(args.outdir, "recall_vs_nprobe.png"))
        plot_recall_vs_centroids(good, os.path.join(args.outdir, "recall_vs_centroids.png"))
        plot_recall_box_by_iter(good, os.path.join(args.outdir, "recall_box_by_iter.png"))
        plot_heatmaps(good, args.outdir)  # existing heatmaps per iteration
    except Exception as e:
        print("Plotting error (recall plots):", e)

    # keep the other analyses (time / combined) but place them after recall
    # basic stats (time-focused)
    stats = good.groupby(["NUM_CENTROIDS", "KMEANS_ITER", "NPROBE"]).agg(
        RECALL_mean=("RECALL","mean"),
        ELAPSED_mean=("ELAPSED_s","mean"),
        AVG_QUERY_TIME_ms_mean=("AVG_QUERY_TIME_ms","mean"),
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

    # Write a short recall-first text summary
    with open(os.path.join(args.outdir, "analysis_summary.txt"), "w") as f:
        f.write("Recall-first Analysis summary\n")
        f.write("=============================\n\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write(f"Successful runs: {len(good)}\n\n")
        best = good.loc[good["RECALL"].idxmax()]
        f.write("Best recall run (global):\n")
        f.write(best.to_string() + "\n\n")
        f.write("Top configurations by mean recall (recall_aggregated.csv)\n")
        f.write(recall_stats.head(10).to_string(index=False) + "\n\n")
        f.write("Pareto front (recall then time) saved to: pareto_front.csv\n")
        f.write(f"Top-{args.topk} runs by recall saved to: top_{args.topk}_by_recall.csv\n")
        f.write(f"Top-{args.topk} runs by recall/sec saved to: top_{args.topk}_by_recall_per_sec.csv\n")

    print("Recall-first analysis complete. Plots and CSVs in", args.outdir)

if __name__ == "__main__":
    main()