import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import re
import time

print("=== 超大规模参数测试 ===")

# 正确解析数据
def parse_data(filename):
    records = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("STAMP"):
                continue
            
            # 使用正则表达式提取数字
            numbers = re.findall(r'[\d.]+', line)
            if len(numbers) >= 7:
                try:
                    C = int(numbers[0])
                    I = int(numbers[1])
                    P = int(numbers[2])
                    recall = float(numbers[4])
                    time_ms = float(numbers[5])
                    records.append((C, I, P, recall, time_ms))
                except:
                    continue
    return pd.DataFrame(records, columns=["C", "I", "P", "recall", "time_ms"])

# 读取数据
df = parse_data("results/summary.txt")
print(f"加载 {len(df)} 条数据")

# 训练模型
X = df[['C', 'P', 'I']].values
y_time = df['time_ms'].values
y_recall = df['recall'].values

time_gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
recall_gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)

time_gbm.fit(X, y_time)
recall_gbm.fit(X, y_recall)

print("模型训练完成")

# 大规模搜索 - 只找召回率>0.98且时间最短的配置
print("开始大规模搜索 (目标: recall>0.98, 时间最短)...")

results = []
C_range = range(64, 16000, 32)  # 64到32768，步长32
P_range = range(1, 5120, 2)      # 1到512，步长2

total = len(C_range) * len(P_range)
count = 0
start_time = time.time()

for C in C_range:
    for P in P_range:
        config = [[C, P, 6]]
        time_pred = time_gbm.predict(config)[0]
        recall_pred = recall_gbm.predict(config)[0]
        
        # 只保留召回率>0.98的配置
        if recall_pred >= 0.98:
            results.append({
                'C': C, 'P': P, 'I': 6,
                'time_pred': round(time_pred, 2),
                'recall_pred': round(recall_pred, 4),
                'P_C_ratio': round(P/C, 4)
            })
        
        count += 1
        if count % 200000 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {count}/{total} ({count/total*100:.1f}%), 已找到 {len(results)} 个候选, 耗时: {elapsed:.1f}s")

# 按时间排序，找出最快的配置
if results:
    results_df = pd.DataFrame(results)
    
    # 按时间排序，取前1000个最快的
    fastest_configs = results_df.nsmallest(1000, 'time_pred')
    fastest_configs.to_csv("fastest_high_recall_configs.csv", index=False)
    
    print(f"\n搜索完成! 找到 {len(results)} 个召回率>0.98的配置")
    print(f"保存前1000个最快的配置到 fastest_high_recall_configs.csv")
    
    # 显示前20个最快配置
    print("\n=== 前20个最快配置 (recall>0.98) ===")
    top20 = fastest_configs.head(20)[['C', 'P', 'time_pred', 'recall_pred', 'P_C_ratio']]
    print(top20.to_string(index=False))
    
    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"最快配置: {fastest_configs['time_pred'].min():.2f}ms")
    print(f"平均时间: {fastest_configs['time_pred'].mean():.2f}ms") 
    print(f"平均召回率: {fastest_configs['recall_pred'].mean():.4f}")
    print(f"配置数量: {len(fastest_configs)}")
    
else:
    print("未找到召回率>0.98的配置")