import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings("ignore", category=UserWarning)

REQUIRED_HYPERS = ["m", "max_layer", "efc", "efs"]
METRICS = ["avg_recall", "avg_time_ms"]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=",")
    except Exception:
        try:
            df = pd.read_csv(csv_path, sep="\t")
        except Exception:
            df = pd.read_csv(csv_path, engine="python")

    df.columns = [c.strip() for c in df.columns]
    required = REQUIRED_HYPERS + METRICS
    keep = [c for c in df.columns if c in required]
    df = df[keep].copy()

    for col in required:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def safe_save(fig_name, outdir):
    path = os.path.join(outdir, fig_name)
    try:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print("Warning: failed to save", path, "-", e)
    finally:
        plt.close()

def plot_hyper_dual_axis(df, hyper, outdir):
	"""
	Plot one hyperparameter vs two metrics on dual y-axes.
	Left: avg_recall (blue), Right: avg_time_ms (orange)
	"""
	if hyper not in df.columns:
		return
	# Group by hyper, compute means
	grp = df.groupby(hyper)[METRICS].mean().reset_index()
	grp = grp.sort_values(hyper)
	if len(grp) < 2:
		return

	fig, ax1 = plt.subplots(figsize=(10, 6))
	color1 = 'tab:blue'
	ax1.set_xlabel(hyper, fontsize=12)
	ax1.set_ylabel('avg_recall', color=color1, fontsize=12)
	line1 = ax1.plot(grp[hyper], grp['avg_recall'], marker='o', color=color1, linewidth=2, label='avg_recall')
	ax1.tick_params(axis='y', labelcolor=color1)
	ax1.grid(True, alpha=0.3)

	ax2 = ax1.twinx()
	color2 = 'tab:orange'
	ax2.set_ylabel('avg_time_ms', color=color2, fontsize=12)
	line2 = ax2.plot(grp[hyper], grp['avg_time_ms'], marker='s', color=color2, linewidth=2, label='avg_time_ms')
	ax2.tick_params(axis='y', labelcolor=color2)

	fig.suptitle(f'Effect of {hyper} on recall and query time', fontsize=14, fontweight='bold')
	fig.tight_layout()
	safe_save(f'dual_axis_{hyper}.png', outdir)

def plot_efc_vs_recall_by_efs(df, outdir):
	"""
	X-axis: efc
	Y-axis (left): avg_recall
	Multiple lines: different efs values (sampled if too many)
	"""
	if not {"efc", "efs", "avg_recall"}.issubset(df.columns):
		return
	# Group by (efc, efs), compute mean recall
	grp = df.groupby(["efc", "efs"])['avg_recall'].mean().reset_index()
	if grp.empty:
		return

	# Sample efs values if too many (max 8 lines)
	efs_vals = sorted(grp['efs'].unique())
	if len(efs_vals) > 8:
		idx = np.linspace(0, len(efs_vals) - 1, 8, dtype=int)
		selected_efs = [efs_vals[i] for i in idx]
	else:
		selected_efs = efs_vals

	plt.figure(figsize=(10, 6))
	colors = plt.cm.tab10(np.linspace(0, 1, len(selected_efs)))
	for i, efs_val in enumerate(selected_efs):
		line_data = grp[grp['efs'] == efs_val].sort_values('efc')
		if len(line_data) > 0:
			plt.plot(line_data['efc'], line_data['avg_recall'], 
					marker='o', linewidth=2, label=f'efs={int(efs_val)}', color=colors[i])

	plt.xlabel('efc', fontsize=12)
	plt.ylabel('avg_recall', fontsize=12)
	plt.title('Effect of efc on recall (lines = different efs)', fontsize=14, fontweight='bold')
	plt.legend(title='efs', fontsize=10, title_fontsize=11, loc='best')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	safe_save(f'multiline_efc_vs_recall_by_efs.png', outdir)

def aggregate_all(df, outdir):
	keys = [k for k in REQUIRED_HYPERS if k in df.columns]
	metrics = [m for m in METRICS if m in df.columns]
	if not keys or not metrics:
		return
	agg = df.groupby(keys)[metrics].agg(['mean','std','count']).reset_index()
	agg.to_csv(os.path.join(outdir, "aggregated_hyper_metrics.csv"), index=False)

def main():
	parser = argparse.ArgumentParser(description="Focused visualization for hyperparams")
	parser.add_argument("--csv", type=str, default="optuna_hng1.csv")
	parser.add_argument("--outdir", type=str, default="viz_results")
	args = parser.parse_args()

	if not os.path.exists(args.csv):
		print("CSV not found:", args.csv)
		return
	ensure_dir(args.outdir)

	df = load_data(args.csv)

	# If CSV didn't contain required columns, print and exit
	missing = [c for c in REQUIRED_HYPERS + METRICS if c not in df.columns]
	if missing:
		print("CSV missing expected columns:", missing)
		return

	# Drop completely invalid rows upfront
	total_before = len(df)
	df = df.dropna(subset=REQUIRED_HYPERS + METRICS, how='all')
	df = df.replace([np.inf, -np.inf], np.nan)
	df = df.dropna(subset=[c for c in REQUIRED_HYPERS + METRICS if c in df.columns], how='all')
	# drop clearly invalid recall rows
	if "avg_recall" in df.columns:
		df = df[df["avg_recall"] > 0]
	# sensible cap on avg_time_ms to remove sentinel values (like 99999)
	if "avg_time_ms" in df.columns:
		df = df[df["avg_time_ms"] < 1e5]

	total_after = len(df)
	if total_before != total_after:
		print(f"Info: removed {total_before - total_after} rows with invalid values")

	# keep only relevant columns to avoid clutter
	keep = [c for c in df.columns if c in REQUIRED_HYPERS + METRICS]
	df = df[keep].copy()

	# require at least one metric column
	if not any((m in df.columns) for m in METRICS):
		print("No metrics (avg_recall or avg_time_ms) found.")
		return

	# drop rows without hyper values entirely
	if not any(h in df.columns for h in REQUIRED_HYPERS):
		print("No hyperparameters found.")
		return

	# drop rows missing both metrics (we still want some metric)
	metric_cols = [m for m in METRICS if m in df.columns]
	df = df.dropna(subset=metric_cols, how='all')
	if df.empty:
		print("No rows with metrics after dropping NaNs.")
		return

	print("Creating focused visualizations:")
	print("  1. Dual-axis plots for m, max_layer")
	# Plot dual-axis for m and max_layer
	plot_hyper_dual_axis(df, "m", args.outdir)
	plot_hyper_dual_axis(df, "max_layer", args.outdir)

	print("  2. Multi-line plot: efc vs recall (lines=efs)")
	# Plot multi-line: efc vs recall, lines are efs
	plot_efc_vs_recall_by_efs(df, args.outdir)

	print("  3. Aggregated statistics CSV")
	aggregate_all(df, args.outdir)
	print("Done — outputs saved to", args.outdir)

if __name__ == "__main__":
	main()
