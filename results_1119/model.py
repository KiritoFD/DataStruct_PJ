import re
import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

# 读取实验数据（改为稳健的 key=value 抽取）
data = []
summary_path = './summary.txt'
with open(summary_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # 抽取形如 KEY=VALUE 的对（KEY 允许大小写字母、数字和下划线）
        # 修复：支持像 "AVG_QUERY_TIME_ms"、"INDEX_BUILD_s" 这类包含小写的键名
        pairs = re.findall(r'([A-Za-z0-9_]+)=([^ ]+)', line)
        if not pairs:
            continue
        record = {}
        for key, val in pairs:
            # 去掉末尾的秒单位 "s" 或其它简单后缀
            if val.endswith('s') and re.match(r'^[0-9\.]+s$', val):
                val = val[:-1]
            # 提取开头的数字部分（如 "12.3", "34"），否则保留原始字符串
            m = re.match(r'^([-+]?[0-9]*\.?[0-9]+)', val)
            if m:
                num = m.group(1)
                # 优先尝试 int，再尝试 float
                if re.match(r'^-?\d+$', num):
                    record[key] = int(num)
                else:
                    record[key] = float(num)
            else:
                record[key] = val
        # 如果想只保留 STATUS=OK 的记录，可打开下一行过滤
        # if record.get('STATUS') != 'OK': continue
        data.append(record)

df = pd.DataFrame(data)
print(f"Total valid records: {len(df)}")
if df.empty:
    print("No data parsed from summary.txt. Exiting.")
    # 退出前给出已读取的几行以便诊断
    print("First 20 raw lines preview:")
    with open(summary_path, 'r') as f:
        for i, l in enumerate(f):
            if i >= 20: break
            print(l.rstrip())
    raise SystemExit(0)

# 规范列名（大写已在解析中保持），确保需要列存在并转换类型
int_cols = ['NUM_CENTROIDS', 'KMEANS_ITER', 'NPROBE']
float_cols = ['RECALL', 'AVG_QUERY_TIME_ms', 'INDEX_BUILD_s']

for c in int_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
for c in float_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

print(df.head())

# 后续建模使用 df 中存在的列，若缺失某些列则提前退出
required = ['NUM_CENTROIDS', 'KMEANS_ITER', 'NPROBE', 'AVG_QUERY_TIME_ms', 'RECALL']
missing = [c for c in required if c not in df.columns]
if missing:
    print("Missing required columns for modeling:", missing)
    raise SystemExit(0)

# 设置已知参数
N = 1000000  # SIFT1M数据集大小
d = 128      # SIFT特征维度
k = 10       # 返回的最近邻数量

# 仅选择KMEANS_ITER=1的数据拟合时间模型(简化)
df_iter1 = df[df['KMEANS_ITER'] == 1]
print(f"Records with KMEANS_ITER=1: {len(df_iter1)}")
if df_iter1.empty:
    print("No records with KMEANS_ITER==1, exiting.")
    raise SystemExit(0)

# 首先拟合时间模型 - 简化版
# T(c,p) = a * p * log(c) + b * p * (N/c) + d
def time_model(params, c, p):
    a, b, d = params
    return a * p * np.log(c) + b * p * (N / c) + d

def time_residuals(params, c, p, t_obs):
    t_pred = time_model(params, c, p)
    return t_pred - t_obs

# 准备数据
c_vals = df_iter1['NUM_CENTROIDS'].astype(float).values
p_vals = df_iter1['NPROBE'].astype(float).values
t_obs = df_iter1['AVG_QUERY_TIME_ms'].astype(float).values

# 初始猜测
initial_params = [0.01, 0.0001, 1.0]

# 拟合
result = least_squares(time_residuals, initial_params, args=(c_vals, p_vals, t_obs))
a_opt, b_opt, d_opt = result.x
print(f"Time model parameters: a={a_opt:.6f}, b={b_opt:.8f}, d={d_opt:.4f}")

# 评估拟合质量
t_pred = time_model([a_opt, b_opt, d_opt], c_vals, p_vals)
rmse = np.sqrt(np.mean((t_pred - t_obs)**2))
r2 = 1 - np.sum((t_obs - t_pred)**2) / np.sum((t_obs - np.mean(t_obs))**2)
print(f"Time model RMSE: {rmse:.4f} ms, R²: {r2:.4f}")

# 拟合召回率模型
# R(p,c) = 1 - (1 - p/c)^theta
def recall_model(params, c, p):
    theta = params[0]
    return 1 - (1 - p / c) ** theta

def recall_residuals(params, c, p, r_obs):
    r_pred = recall_model(params, c, p)
    return r_pred - r_obs

# 仅使用较大的NUM_CENTROIDS (>512)和KMEANS_ITER=1的数据
df_large_c = df_iter1[df_iter1['NUM_CENTROIDS'] > 512]
if df_large_c.empty:
    print("No large-centroid records for recall fit, skipping recall model.")
else:
    c_vals_r = df_large_c['NUM_CENTROIDS'].astype(float).values
    p_vals_r = df_large_c['NPROBE'].astype(float).values
    r_obs = df_large_c['RECALL'].astype(float).values

    # 初始猜测
    initial_theta = [2.0]

    # 拟合
    result_r = least_squares(recall_residuals, initial_theta, args=(c_vals_r, p_vals_r, r_obs))
    theta_opt = result_r.x[0]
    print(f"Recall model parameter: theta={theta_opt:.4f}")

    # 评估拟合质量
    r_pred = recall_model([theta_opt], c_vals_r, p_vals_r)
    rmse_r = np.sqrt(np.mean((r_pred - r_obs)**2))
    r2_r = 1 - np.sum((r_obs - r_pred)**2) / np.sum((r_obs - np.mean(r_obs))**2)
    print(f"Recall model RMSE: {rmse_r:.4f}, R²: {r2_r:.4f}")

# 使用KMEANS_ITER>1的数据拟合聚类质量参数
df_iter_gt1 = df[df['KMEANS_ITER'] > 1]
print(f"Records with KMEANS_ITER>1: {len(df_iter_gt1)}")

# 聚类质量模型: quality = 1 - exp(-kappa * i)
def quality_model(params, i):
    kappa = params[0]
    return 1 - np.exp(-kappa * i)

# 提取固定c和p下，不同i的recall
# 选择c=64, p=64的实验组
df_c64_p64 = df[(df['NUM_CENTROIDS'] == 64) & (df['NPROBE'] == 64)]
i_vals = df_c64_p64['KMEANS_ITER'].values
r_vals = df_c64_p64['RECALL'].values

# 需要归一化到base_recall (i=1时的recall)
base_recall = r_vals[i_vals == 1][0]
r_norm = r_vals / base_recall

# 拟合
initial_kappa = [0.1]
result_k = least_squares(lambda k, i, r: quality_model(k, i) - r, 
                         initial_kappa, args=(i_vals, r_norm))
kappa_opt = result_k.x[0]
print(f"Quality model parameter: kappa={kappa_opt:.4f}")

# 完整召回率模型: R(i,p,c) = (1 - exp(-kappa*i)) * (1 - (1-p/c)^theta)
def full_recall_model(i, p, c, kappa, theta, base):
    quality = 1 - np.exp(-kappa * i)
    coverage = 1 - (1 - p/c)**theta
    return base * quality * coverage

# 拟合完整模型
c_all = df['NUM_CENTROIDS'].values
p_all = df['NPROBE'].values
i_all = df['KMEANS_ITER'].values
r_all = df['RECALL'].values

# 初始参数
initial_params = [kappa_opt, theta_opt, 1.0]

def full_recall_residuals(params, i, p, c, r_obs):
    kappa, theta, base = params
    r_pred = full_recall_model(i, p, c, kappa, theta, base)
    return r_pred - r_obs

result_full = least_squares(full_recall_residuals, initial_params, 
                           args=(i_all, p_all, c_all, r_all))
kappa_full, theta_full, base_full = result_full.x
print(f"Full recall model parameters: kappa={kappa_full:.4f}, theta={theta_full:.4f}, base={base_full:.4f}")

# 评估完整模型
r_pred_full = full_recall_model(i_all, p_all, c_all, kappa_full, theta_full, base_full)
rmse_full = np.sqrt(np.mean((r_pred_full - r_all)**2))
r2_full = 1 - np.sum((r_all - r_pred_full)**2) / np.sum((r_all - np.mean(r_all))**2)
print(f"Full recall model RMSE: {rmse_full:.4f}, R²: {r2_full:.4f}")