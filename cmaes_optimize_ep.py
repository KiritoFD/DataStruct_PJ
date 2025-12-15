#!/usr/bin/env python3
"""
Optuna CMA-ES 优化器：寻找让自适应入口点 (Adaptive EP) 优化效果最明显的参数组合

优化目标：最大化 with_adaptive_ep 和 without_adaptive_ep 两条曲线的差距
- 在 recall 0.98-0.998 范围内扫描多个 EFS 值
- 计算两条曲线在相同 recall 下的 QPS 差距
- 最大化平均差距

使用方法：
    python cmaes_optimize_ep.py [--max_iter 30]
"""

import argparse
import subprocess
import os
import re
import time
import json
import csv
from datetime import datetime
from pathlib import Path
import shutil

try:
    import optuna
    from optuna.samplers import CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("警告: optuna 库未安装，使用 'pip install optuna' 安装")

import numpy as np

# 默认配置
DEFAULT_CONFIG = {
    'binary': 'hnga',
    'base_file': 'data_o/glove/base.txt',
    'query_file': 'data_o/glove/query.txt',
    'truth_file': 'data_o/glove/truth.txt',
    'k': 10,
    'threads': 32,
    'recall_min': 0.98,
    'recall_max': 0.998,
    'max_layer': 7,
}

# 参数搜索范围
PARAM_BOUNDS = {
    'M': (16, 100),
    'efc': (200, 800),
    'cluster_k': (64, 384),
}

# EFS 扫描范围
EFS_SCAN_LIST = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]

# 全局汇总CSV路径
GLOBAL_SUMMARY_CSV = Path('optimization_summary.csv')


class Logger:
    """搜索日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'optimization_{timestamp}.log'
        self.trial_dir = self.log_dir / 'trials'
        self.trial_dir.mkdir(exist_ok=True)
        
        self.write(f"========== Optuna CMA-ES 优化启动 ==========\n")
        self.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.write(f"日志目录: {self.log_dir}\n")
        self.write(f"=" * 60 + "\n\n")
    
    def write(self, msg):
        """写入日志"""
        with open(self.log_file, 'a') as f:
            f.write(msg)
        print(msg, end='')
    
    def log_trial(self, trial_num, params, metrics, eval_dir, qps_improvement):
        """记录单次试验"""
        trial_log = self.trial_dir / f'trial_{trial_num:04d}.json'
        
        record = {
            'trial': trial_num,
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'metrics': metrics,
            'qps_improvement': qps_improvement,
            'eval_dir': str(eval_dir),
        }
        
        with open(trial_log, 'w') as f:
            json.dump(record, f, indent=2)
        
        summary = (f"\n[Trial #{trial_num}] M={params['M']}, EFC={params['efc']}, "
                  f"ClusterK={params['cluster_k']}\n"
                  f"  QPS 提升: {qps_improvement*100:.2f}%\n")
        self.write(summary)
    
    def log_best(self, best_trial, best_value):
        """记录最优结果"""
        msg = (f"\n{'='*60}\n"
               f"最优结果\n"
               f"{'='*60}\n"
               f"Trial #{best_trial.number}\n"
               f"参数:\n"
               f"  M: {best_trial.params['M']}\n"
               f"  EFC: {best_trial.params['efc']}\n"
               f"  ClusterK: {best_trial.params['cluster_k']}\n"
               f"最优 QPS 提升: {-best_value*100:.2f}%\n"
               f"{'='*60}\n")
        self.write(msg)


class GlobalSummary:
    """全局汇总记录"""
    
    def __init__(self):
        self.csv_path = GLOBAL_SUMMARY_CSV
        self.load_or_create()
    
    def load_or_create(self):
        """加载或创建汇总CSV"""
        if self.csv_path.exists():
            print(f"[续搜] 加载已有汇总: {self.csv_path}")
        else:
            # 创建新CSV
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'trial_num', 'timestamp', 'M', 'EFC', 'cluster_k',
                    'qps_improvement', 'eval_dir', 'result_dir'
                ])
    
    def add_trial(self, trial_num, timestamp, M, efc, cluster_k, qps_improvement, eval_dir, result_dir):
        """添加试验记录"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_num, timestamp, M, efc, cluster_k,
                f"{qps_improvement:.6f}", str(eval_dir), str(result_dir)
            ])
    
    def get_last_trial_num(self):
        """获取最后一个试验编号"""
        if not self.csv_path.exists():
            return -1
        
        max_num = -1
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['trial_num']:
                        max_num = max(max_num, int(row['trial_num']))
        except Exception as e:
            print(f"[警告] 读取汇总CSV失败: {e}")
        
        return max_num


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config, results_dir, logger):
        self.config = config
        self.binary = config['binary']
        self.base_file = config['base_file']
        self.query_file = config['query_file']
        self.truth_file = config['truth_file']
        self.k = config['k']
        self.threads = config['threads']
        self.recall_min = config['recall_min']
        self.recall_max = config['recall_max']
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.eval_count = 0
        
    def run_single_experiment(self, M, max_layer, efc, efs, cluster_k, adaptive_ep, log_file):
        """运行单次实验"""
        M = int(round(M))
        max_layer = int(round(max_layer))
        efc = int(round(efc))
        efs = int(round(efs))
        cluster_k = int(round(cluster_k))
        
        cmd = [
            f'./{self.binary}',
            '--base', self.base_file,
            '--query', self.query_file,
            '--truth', self.truth_file,
            '--k', str(self.k),
            '--m', str(M),
            '--max_layer', str(max_layer),
            '--efc', str(efc),
            '--efs', str(efs),
            '--threads', str(self.threads),
            '--ablate_adaptive_ep', str(adaptive_ep),
            '--ablate_csr', '0',
            '--ablate_prefetch', '0',
            '--ablate_simd', '0',
            '--ablate_pruning', '0',
            '--ablate_heap', '0',
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
            
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
            
            if result.returncode != 0:
                return None
            
            return self.parse_output(result.stdout)
            
        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            self.logger.write(f"    异常: {e}\n")
            return None
    
    def parse_output(self, output):
        """解析实验输出"""
        metrics = {}
        
        match = re.search(r'Average recall@\d+:\s*([\d.]+)', output)
        if match:
            metrics['recall'] = float(match.group(1))
        
        match = re.search(r'Queries per second:\s*([\d.]+)', output)
        if match:
            metrics['qps'] = float(match.group(1))
        
        match = re.search(r'Average distance ops per query:\s*([\d.]+)', output)
        if match:
            metrics['avg_dists'] = float(match.group(1))
        
        return metrics if 'recall' in metrics and 'avg_dists' in metrics else None
    
    def scan_efs_range(self, M, max_layer, efc, cluster_k):
        """扫描 EFS 范围"""
        self.eval_count += 1
        eval_dir = self.results_dir / f"eval_{self.eval_count:04d}_M{int(M)}_ML{int(max_layer)}_EFC{int(efc)}_K{int(cluster_k)}"
        eval_dir.mkdir(exist_ok=True)
        
        results_with = []
        results_without = []
        
        self.logger.write(f"\n  扫描 EFS 范围...\n")
        
        for efs in EFS_SCAN_LIST:
            log_with = eval_dir / f"efs{efs}_with_ep.log"
            metrics_with = self.run_single_experiment(M, max_layer, efc, efs, cluster_k, 0, log_with)
            
            log_without = eval_dir / f"efs{efs}_without_ep.log"
            metrics_without = self.run_single_experiment(M, max_layer, efc, efs, cluster_k, 1, log_without)
            
            if metrics_with and metrics_without:
                results_with.append((metrics_with['recall'], metrics_with['avg_dists'], 
                                    metrics_with.get('qps', 0), efs))
                results_without.append((metrics_without['recall'], metrics_without['avg_dists'],
                                       metrics_without.get('qps', 0), efs))
                self.logger.write(f"    EFS={efs}: recall={metrics_with['recall']:.4f}/{metrics_without['recall']:.4f}, "
                                 f"QPS={metrics_with['qps']:.1f}/{metrics_without['qps']:.1f}\n")
        
        # 保存扫描结果到 CSV
        csv_file = eval_dir / "scan_results.csv"
        with open(csv_file, 'w') as f:
            f.write("variant,efs,recall,avg_dists,qps\n")
            for r, ndc, qps, efs in results_with:
                f.write(f"with_adaptive_ep,{efs},{r},{ndc},{qps}\n")
            for r, ndc, qps, efs in results_without:
                f.write(f"without_adaptive_ep,{efs},{r},{ndc},{qps}\n")
        
        return results_with, results_without, eval_dir
    
    def compute_curve_gap(self, results_with, results_without):
        """计算两条曲线的QPS差距"""
        if len(results_with) < 3 or len(results_without) < 3:
            return 1e10
        
        with_in_range = [(r, qps) for r, _, qps, _ in results_with 
                         if self.recall_min <= r <= self.recall_max]
        without_in_range = [(r, qps) for r, _, qps, _ in results_without 
                            if self.recall_min <= r <= self.recall_max]
        
        if len(with_in_range) < 2 or len(without_in_range) < 2:
            self.logger.write(f"  警告: recall 范围内数据点不足\n")
            return 1e10
        
        with_sorted = sorted(with_in_range)
        without_sorted = sorted(without_in_range)
        
        recall_points = np.linspace(self.recall_min, min(with_sorted[-1][0], without_sorted[-1][0]), 10)
        
        def interpolate_qps(data, recall_target):
            recalls = [d[0] for d in data]
            qps_vals = [d[1] for d in data]
            if recall_target < min(recalls) or recall_target > max(recalls):
                return None
            return np.interp(recall_target, recalls, qps_vals)
        
        gaps = []
        for r in recall_points:
            qps_with = interpolate_qps(with_sorted, r)
            qps_without = interpolate_qps(without_sorted, r)
            if qps_with is not None and qps_without is not None and qps_without > 0:
                improvement = (qps_with - qps_without) / qps_without
                gaps.append(improvement)
        
        if not gaps:
            return 1e10
        
        avg_gap = np.mean(gaps)
        self.logger.write(f"  曲线差距: 平均 QPS 提升 {avg_gap*100:.2f}%\n")
        
        return -avg_gap


class OptunaOptimizer:
    """Optuna CMA-ES 优化器"""
    
    def __init__(self, config):
        self.config = config
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path('result-ablation') / f'optuna_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(self.results_dir)
        self.runner = ExperimentRunner(config, self.results_dir, self.logger)
        self.global_summary = GlobalSummary()
        
        self.param_names = ['M', 'efc', 'cluster_k']
        self.best_trial = None
        self.best_value = float('inf')
    
    def objective(self, trial):
        """目标函数"""
        M = trial.suggest_int('M', 16, 100)
        efc = trial.suggest_int('efc', 200, 800)
        cluster_k = trial.suggest_int('cluster_k', 64, 384)
        
        max_layer = int(self.config.get('max_layer', 7))
        
        results_with, results_without, eval_dir = self.runner.scan_efs_range(
            M, max_layer, efc, cluster_k
        )
        
        gap = self.runner.compute_curve_gap(results_with, results_without)
        
        self.generate_plot(results_with, results_without, eval_dir, gap)
        
        params = {'M': M, 'efc': efc, 'cluster_k': cluster_k}
        metrics = {
            'with_results': len(results_with),
            'without_results': len(results_without),
        }
        self.logger.log_trial(trial.number, params, metrics, eval_dir, -gap)
        
        # 记录到全局汇总CSV
        self.global_summary.add_trial(
            trial.number,
            datetime.now().isoformat(),
            M, efc, cluster_k,
            -gap,
            eval_dir,
            self.results_dir
        )
        
        if gap < self.best_value:
            self.best_value = gap
            self.best_trial = trial
        
        return gap
    
    def generate_plot(self, results_with, results_without, eval_dir, gap):
        """生成对比图"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 图1: Recall vs NDC（距离计算次数）
            if results_with:
                recalls_w = [r[0] for r in results_with]
                ndcs_w = [r[1] for r in results_with]
                ax1.plot(recalls_w, ndcs_w, 'o-', color='#2ecc71', label='With Adaptive EP', linewidth=2, markersize=6)
            
            if results_without:
                recalls_wo = [r[0] for r in results_without]
                ndcs_wo = [r[1] for r in results_without]
                ax1.plot(recalls_wo, ndcs_wo, 's-', color='#e74c3c', label='Without Adaptive EP', linewidth=2, markersize=6)
            
            ax1.set_xlabel('Recall@10', fontsize=12)
            ax1.set_ylabel('Average Distance Computations', fontsize=12)
            ax1.set_title(f'Recall vs NDC', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 图2: Recall vs QPS（查询吞吐量）- 这是优化目标
            if results_with:
                qps_w = [r[2] for r in results_with]
                ax2.plot(recalls_w, qps_w, 'o-', color='#2ecc71', label='With Adaptive EP', linewidth=2.5, markersize=7)
            
            if results_without:
                qps_wo = [r[2] for r in results_without]
                ax2.plot(recalls_wo, qps_wo, 's-', color='#e74c3c', label='Without Adaptive EP', linewidth=2.5, markersize=7)
            
            ax2.set_xlabel('Recall@10', fontsize=12)
            ax2.set_ylabel('Queries per Second (QPS)', fontsize=12)
            ax2.set_title(f'Recall vs QPS (Gap: {-gap*100:.2f}%)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(eval_dir / 'comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.write(f"  绘图失败: {e}\n")
    
    def optimize(self, max_iter=3000):
        """运行优化"""
        if not HAS_OPTUNA:
            self.logger.write("错误: 需要安装 optuna 库\n")
            return None
        
        max_layer = int(self.config.get('max_layer', 7))
        self.logger.write(f"参数搜索配置:\n")
        self.logger.write(f"  M: [16, 100]\n")
        self.logger.write(f"  EFC: [200, 800]\n")
        self.logger.write(f"  ClusterK: [64, 384]\n")
        self.logger.write(f"  max_layer (fixed): {max_layer}\n")
        self.logger.write(f"  最大迭代次数: {max_iter}\n")
        self.logger.write(f"  全局汇总: {GLOBAL_SUMMARY_CSV}\n\n")
        
        # 检查是否可以续搜
        last_trial = self.global_summary.get_last_trial_num()
        if last_trial >= 0:
            self.logger.write(f"\n[续搜] 发现已有试验，最后编号: {last_trial}\n")
            self.logger.write(f"[续搜] 将从试验 {last_trial + 1} 继续搜索\n\n")
        
        sampler = CmaEsSampler(seed=42)
        study = optuna.create_study(
            sampler=sampler,
            direction='minimize',
            study_name='Adaptive-EP-Optimization'
        )
        
        start_time = time.time()
        study.optimize(self.objective, n_trials=max_iter)
        elapsed = time.time() - start_time
        
        if self.best_trial:
            self.logger.log_best(self.best_trial, self.best_value)
        
        self.logger.write(f"\n总耗时: {elapsed/60:.1f} 分钟\n")
        self.logger.write(f"总试验次数: {len(study.trials)}\n")
        
        self.save_results(study, elapsed)
        
        return self.best_trial, self.best_value
    
    def save_results(self, study, elapsed):
        """保存优化结果"""
        result_file = self.results_dir / 'optimization_result.json'
        
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
            })
        
        result = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'config': self.config,
            'best_trial': self.best_trial.number if self.best_trial else None,
            'best_params': {
                name: self.best_trial.params[name]
                for name in self.param_names
            } if self.best_trial else None,
            'best_qps_improvement': float(-self.best_value) if self.best_value != float('inf') else None,
            'elapsed_seconds': elapsed,
            'total_trials': len(study.trials),
            'global_summary_csv': str(GLOBAL_SUMMARY_CSV),
        }
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.logger.write(f"\n结果已保存到: {result_file}\n")
        self.logger.write(f"全局汇总CSV: {GLOBAL_SUMMARY_CSV}\n")
        
        if self.best_trial:
            max_layer = int(self.config.get('max_layer', 7))
            params = self.best_trial.params
            self.logger.write(f"\n推荐的消融实验命令:\n")
            self.logger.write(f"M={params['M']} MAX_LAYER={max_layer} EFC={params['efc']} "
                            f"ADAPTIVE_EP_K={params['cluster_k']} ./run_ablation_ep.sh\n")


def main():
    parser = argparse.ArgumentParser(description='Optuna CMA-ES 优化器：寻找最佳自适应入口点参数')
    parser.add_argument('--binary', default=DEFAULT_CONFIG['binary'], help='可执行文件路径')
    parser.add_argument('--base', default=DEFAULT_CONFIG['base_file'], help='Base 文件路径')
    parser.add_argument('--query', default=DEFAULT_CONFIG['query_file'], help='Query 文件路径')
    parser.add_argument('--truth', default=DEFAULT_CONFIG['truth_file'], help='Truth 文件路径')
    parser.add_argument('--max_iter', type=int, default=30, help='最大迭代次数')
    parser.add_argument('--threads', type=int, default=DEFAULT_CONFIG['threads'], help='线程数')
    parser.add_argument('--recall_min', type=float, default=DEFAULT_CONFIG['recall_min'], help='最小 recall')
    parser.add_argument('--recall_max', type=float, default=DEFAULT_CONFIG['recall_max'], help='最大 recall')
    
    args = parser.parse_args()
    
    config = {
        'binary': args.binary,
        'base_file': args.base,
        'query_file': args.query,
        'truth_file': args.truth,
        'k': 10,
        'threads': args.threads,
        'recall_min': args.recall_min,
        'recall_max': args.recall_max,
    }
    
    optimizer = OptunaOptimizer(config)
    optimizer.optimize(max_iter=args.max_iter)


if __name__ == '__main__':
    main()
