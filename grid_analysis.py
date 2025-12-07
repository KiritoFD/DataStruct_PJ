#!/usr/bin/env python3
"""
Grid Search Results Analysis Tool
Analyzes HNSW parameter grid search results and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

def load_latest_results(results_dir="grid_results"):
    """Load the most recent grid search CSV file"""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return None
    
    csv_files = list(results_path.glob("grid_search_*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in '{results_dir}'")
        return None
    
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df, latest_file

def analyze_results(df):
    """Generate comprehensive analysis of grid search results"""
    print("\n" + "="*70)
    print("GRID SEARCH ANALYSIS")
    print("="*70)
    
    # Basic statistics
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"\nParameter ranges:")
    print(f"  M: {df['M'].min()} - {df['M'].max()}")
    print(f"  EFC: {df['EFC'].min()} - {df['EFC'].max()}")
    print(f"  EFS: {df['EFS'].unique()}")
    
    # Filter valid results
    df_valid = df[df['Recall'] != 'N/A'].copy()
    df_valid['Recall'] = pd.to_numeric(df_valid['Recall'])
    df_valid['AvgQueryTime_ms'] = pd.to_numeric(df_valid['AvgQueryTime_ms'])
    df_valid['DistOpsPerQuery'] = pd.to_numeric(df_valid['DistOpsPerQuery'])
    df_valid['BuildTime_ms'] = pd.to_numeric(df_valid['BuildTime_ms'])
    df_valid['QPS'] = pd.to_numeric(df_valid['QPS'])
    
    # Best configurations
    print("\n" + "-"*70)
    print("TOP 5 CONFIGURATIONS BY RECALL")
    print("-"*70)
    top_recall = df_valid.nlargest(5, 'Recall')
    for idx, row in top_recall.iterrows():
        print(f"M={row['M']}, EFC={row['EFC']}: "
              f"Recall={row['Recall']:.6f}, "
              f"QTime={row['AvgQueryTime_ms']:.3f}ms, "
              f"DistOps={row['DistOpsPerQuery']:.0f}")
    
    print("\n" + "-"*70)
    print("TOP 5 CONFIGURATIONS BY QUERY SPEED")
    print("-"*70)
    top_speed = df_valid.nsmallest(5, 'AvgQueryTime_ms')
    for idx, row in top_speed.iterrows():
        print(f"M={row['M']}, EFC={row['EFC']}: "
              f"QTime={row['AvgQueryTime_ms']:.3f}ms, "
              f"Recall={row['Recall']:.6f}, "
              f"QPS={row['QPS']:.2f}")
    
    print("\n" + "-"*70)
    print("TOP 5 CONFIGURATIONS BY DISTANCE EFFICIENCY")
    print("-"*70)
    top_dist = df_valid.nsmallest(5, 'DistOpsPerQuery')
    for idx, row in top_dist.iterrows():
        print(f"M={row['M']}, EFC={row['EFC']}: "
              f"DistOps={row['DistOpsPerQuery']:.0f}, "
              f"Recall={row['Recall']:.6f}, "
              f"QTime={row['AvgQueryTime_ms']:.3f}ms")
    
    # Pareto frontier analysis
    print("\n" + "-"*70)
    print("PARETO OPTIMAL CONFIGURATIONS (Recall vs Query Time)")
    print("-"*70)
    pareto = find_pareto_frontier(df_valid[['Recall', 'AvgQueryTime_ms', 'M', 'EFC']].values)
    pareto_configs = df_valid.iloc[pareto]
    for idx, row in pareto_configs.iterrows():
        print(f"M={row['M']}, EFC={row['EFC']}: "
              f"Recall={row['Recall']:.6f}, "
              f"QTime={row['AvgQueryTime_ms']:.3f}ms")
    
    # Statistical summary
    print("\n" + "-"*70)
    print("STATISTICAL SUMMARY")
    print("-"*70)
    print(f"\nRecall:")
    print(f"  Mean: {df_valid['Recall'].mean():.6f}")
    print(f"  Std:  {df_valid['Recall'].std():.6f}")
    print(f"  Min:  {df_valid['Recall'].min():.6f}")
    print(f"  Max:  {df_valid['Recall'].max():.6f}")
    
    print(f"\nQuery Time (ms):")
    print(f"  Mean: {df_valid['AvgQueryTime_ms'].mean():.3f}")
    print(f"  Std:  {df_valid['AvgQueryTime_ms'].std():.3f}")
    print(f"  Min:  {df_valid['AvgQueryTime_ms'].min():.3f}")
    print(f"  Max:  {df_valid['AvgQueryTime_ms'].max():.3f}")
    
    print(f"\nDistance Ops per Query:")
    print(f"  Mean: {df_valid['DistOpsPerQuery'].mean():.0f}")
    print(f"  Std:  {df_valid['DistOpsPerQuery'].std():.0f}")
    print(f"  Min:  {df_valid['DistOpsPerQuery'].min():.0f}")
    print(f"  Max:  {df_valid['DistOpsPerQuery'].max():.0f}")
    
    return df_valid

def find_pareto_frontier(costs):
    """Find Pareto optimal points (minimize cost[1], maximize cost[0])"""
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # Keep points where we have better recall AND better/equal time
            # or equal recall and better time
            is_pareto[is_pareto] = np.any(
                (costs[is_pareto, 0] >= c[0]) & (costs[is_pareto, 1] < c[1]) |
                (costs[is_pareto, 0] > c[0]) & (costs[is_pareto, 1] <= c[1]),
                axis=0
            )
            is_pareto[i] = True
    return np.where(is_pareto)[0]

def create_visualizations(df, output_dir="grid_results"):
    """Generate visualization plots"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Recall vs Query Time scatter
    plt.figure()
    for m in df['M'].unique():
        df_m = df[df['M'] == m]
        plt.scatter(df_m['AvgQueryTime_ms'], df_m['Recall'], 
                   label=f'M={m}', s=100, alpha=0.7)
    plt.xlabel('Average Query Time (ms)', fontsize=12)
    plt.ylabel('Recall@10', fontsize=12)
    plt.title('Recall vs Query Time Trade-off', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'recall_vs_time_{timestamp}.png', dpi=300)
    print(f"Saved: recall_vs_time_{timestamp}.png")
    
    # 2. Heatmap: Recall by M and EFC
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(values='Recall', index='EFC', columns='M')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Recall@10'})
    plt.title('Recall Heatmap: EFC vs M', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / f'recall_heatmap_{timestamp}.png', dpi=300)
    print(f"Saved: recall_heatmap_{timestamp}.png")
    
    # 3. Distance Operations comparison
    plt.figure()
    df_grouped = df.groupby(['M', 'EFC'])['DistOpsPerQuery'].mean().reset_index()
    for m in df['M'].unique():
        df_m = df_grouped[df_grouped['M'] == m]
        plt.plot(df_m['EFC'], df_m['DistOpsPerQuery'], 
                marker='o', label=f'M={m}', linewidth=2)
    plt.xlabel('EF Construction', fontsize=12)
    plt.ylabel('Distance Ops per Query', fontsize=12)
    plt.title('Distance Computation Efficiency', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'dist_ops_{timestamp}.png', dpi=300)
    print(f"Saved: dist_ops_{timestamp}.png")
    
    # 4. Build time comparison
    plt.figure()
    df_grouped = df.groupby('M')['BuildTime_ms'].mean()
    plt.bar(df_grouped.index.astype(str), df_grouped.values, 
           color='steelblue', alpha=0.7)
    plt.xlabel('M Parameter', fontsize=12)
    plt.ylabel('Average Build Time (ms)', fontsize=12)
    plt.title('Index Build Time by M', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / f'build_time_{timestamp}.png', dpi=300)
    print(f"Saved: build_time_{timestamp}.png")
    
    print("\nAll visualizations saved successfully!")

def main():
    # Load results
    result = load_latest_results()
    if result is None:
        return 1
    
    df, csv_file = result
    
    # Analyze
    df_valid = analyze_results(df)
    
    # Create visualizations
    create_visualizations(df_valid)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
