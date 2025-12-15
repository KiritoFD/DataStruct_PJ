#!/usr/bin/env python3
"""
plot_ablation_ep.py
绘制自适应入口点消融实验结果
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='绘制自适应入口点消融实验结果')
    parser.add_argument('csv_file', help='CSV结果文件路径')
    parser.add_argument('-o', '--output', default='ablation_ep.png', help='输出图片路径')
    parser.add_argument('-d', '--recall_min', type=float, default=None, help='Recall下限')
    parser.add_argument('-u', '--recall_max', type=float, default=None, help='Recall上限')
    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.csv_file)
    
    # 清理数据
    df['avg_recall'] = pd.to_numeric(df['avg_recall'], errors='coerce')
    df['avg_dists'] = pd.to_numeric(df['avg_dists'], errors='coerce')
    df['queries_per_sec'] = pd.to_numeric(df['queries_per_sec'], errors='coerce')
    df = df.dropna(subset=['avg_recall', 'avg_dists'])
    
    # 过滤 Recall 范围
    if args.recall_min is not None:
        df = df[df['avg_recall'] >= args.recall_min]
    if args.recall_max is not None:
        df = df[df['avg_recall'] <= args.recall_max]
    
    if df.empty:
        print("警告: 过滤后没有数据")
        return
    
    # 按变体分组
    variants = df['variant'].unique()
    
    # 设置颜色和标记
    colors = {'with_adaptive_ep': '#2ecc71', 'without_adaptive_ep': '#e74c3c'}
    markers = {'with_adaptive_ep': 'o', 'without_adaptive_ep': 's'}
    labels = {'with_adaptive_ep': '启用自适应入口点', 'without_adaptive_ep': '禁用自适应入口点 (Baseline)'}
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: Recall vs 平均距离计算次数
    ax1 = axes[0]
    for variant in variants:
        data = df[df['variant'] == variant].sort_values('avg_recall')
        color = colors.get(variant, 'gray')
        marker = markers.get(variant, 'x')
        label = labels.get(variant, variant)
        ax1.plot(data['avg_recall'], data['avg_dists'], 
                marker=marker, color=color, label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Recall@10', fontsize=12)
    ax1.set_ylabel('平均距离计算次数 (NDC)', fontsize=12)
    ax1.set_title('Recall vs 距离计算次数', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 图2: Recall vs QPS
    ax2 = axes[1]
    for variant in variants:
        data = df[df['variant'] == variant].sort_values('avg_recall')
        color = colors.get(variant, 'gray')
        marker = markers.get(variant, 'x')
        label = labels.get(variant, variant)
        ax2.plot(data['avg_recall'], data['queries_per_sec'], 
                marker=marker, color=color, label=label, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Recall@10', fontsize=12)
    ax2.set_ylabel('查询吞吐量 (QPS)', fontsize=12)
    ax2.set_title('Recall vs 查询性能', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"图表已保存至: {args.output}")
    
    # 打印统计摘要
    print("\n========== 统计摘要 ==========")
    for variant in variants:
        data = df[df['variant'] == variant]
        print(f"\n{labels.get(variant, variant)}:")
        print(f"  Recall 范围: {data['avg_recall'].min():.4f} - {data['avg_recall'].max():.4f}")
        print(f"  平均 NDC: {data['avg_dists'].mean():.1f}")
        print(f"  平均 QPS: {data['queries_per_sec'].mean():.1f}")
    
    # 计算相对改进
    if 'with_adaptive_ep' in variants and 'without_adaptive_ep' in variants:
        print("\n========== 相对改进 ==========")
        for efs in df['EFS'].unique():
            with_ep = df[(df['variant'] == 'with_adaptive_ep') & (df['EFS'] == efs)]
            without_ep = df[(df['variant'] == 'without_adaptive_ep') & (df['EFS'] == efs)]
            
            if not with_ep.empty and not without_ep.empty:
                recall_with = with_ep['avg_recall'].values[0]
                recall_without = without_ep['avg_recall'].values[0]
                ndc_with = with_ep['avg_dists'].values[0]
                ndc_without = without_ep['avg_dists'].values[0]
                qps_with = with_ep['queries_per_sec'].values[0]
                qps_without = without_ep['queries_per_sec'].values[0]
                
                if ndc_without > 0:
                    ndc_reduction = (ndc_without - ndc_with) / ndc_without * 100
                else:
                    ndc_reduction = 0
                    
                if qps_without > 0:
                    qps_improvement = (qps_with - qps_without) / qps_without * 100
                else:
                    qps_improvement = 0
                
                print(f"EFS={efs}: Recall {recall_with:.4f} vs {recall_without:.4f}, "
                      f"NDC减少 {ndc_reduction:.1f}%, QPS提升 {qps_improvement:.1f}%")

if __name__ == '__main__':
    main()
