import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import json

# 1. 读取并解析数据
print("=== 数据加载 ===")
with open("results/summary.txt", "r") as f:
    data_lines = f.readlines()

# 解析新的数据格式
records = []
for line in data_lines:
    line = line.strip()
    if not line or line.startswith("STAMP") or line.startswith("--"):
        continue
    
    # 使用正则表达式提取关键信息
    pattern = r'NUM_CENTROIDS=(\d+)\s+KMEANS_ITER=(\d+)\s+NPROBE=(\d+).*?RECALL=([\d.]+)\s+AVG_QUERY_TIME_ms=([\d.]+)'
    match = re.search(pattern, line)
    
    if match:
        try:
            C = int(match.group(1))
            I = int(match.group(2))
            P = int(match.group(3))
            recall = float(match.group(4))
            time_ms = float(match.group(5))
            
            records.append((C, I, P, recall, time_ms))
            print(f"解析成功: C={C}, I={I}, P={P}, R={recall}, T={time_ms}")
            
        except (ValueError, IndexError) as e:
            print(f"解析失败: {line}")
            continue
    else:
        # 尝试备用解析方法
        parts = line.split()
        try:
            # 查找包含数字的字段
            nums = []
            for part in parts:
                if '=' in part:
                    key_value = part.split('=')
                    if len(key_value) == 2:
                        try:
                            nums.append(float(key_value[1]))
                        except:
                            continue
                else:
                    try:
                        nums.append(float(part))
                    except:
                        continue
            
            if len(nums) >= 5:
                C, I, P, recall, time_ms = int(nums[0]), int(nums[1]), int(nums[2]), nums[3], nums[4]
                records.append((C, I, P, recall, time_ms))
                print(f"解析成功(备用): C={C}, I={I}, P={P}, R={recall}, T={time_ms}")
                
        except Exception as e:
            continue

if not records:
    print("错误: 没有解析到任何数据!")
    print("请检查数据格式")
    exit(1)

df = pd.DataFrame(records, columns=["C", "I", "P", "recall", "time_ms"])
print(f"\n加载 {len(df)} 条数据")
print(f"C范围: {df['C'].min()} - {df['C'].max()}")
print(f"P范围: {df['P'].min()} - {df['P'].max()}")
print(f"I值: {sorted(df['I'].unique())}")
print(f"时间范围: {df['time_ms'].min():.2f} - {df['time_ms'].max():.2f} ms")
print(f"召回率范围: {df['recall'].min():.3f} - {df['recall'].max():.3f}")

# 准备数据
X = df[['C', 'P', 'I']].values
y_time = df['time_ms'].values
y_recall = df['recall'].values

print(f"\n数据形状: X={X.shape}, y_time={y_time.shape}, y_recall={y_recall.shape}")

# 数据标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_time = StandardScaler()
y_time_scaled = scaler_time.fit_transform(y_time.reshape(-1, 1)).ravel()

scaler_recall = StandardScaler()  
y_recall_scaled = scaler_recall.fit_transform(y_recall.reshape(-1, 1)).ravel()

# 2. 多项式回归
print("\n=== 多项式回归拟合 ===")

def evaluate_polynomial(X, y, degree, target_name):
    """评估多项式回归模型"""
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2')
    
    # 整体拟合
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"{target_name} (degree {degree}): R²={r2:.3f}, RMSE={rmse:.3f}, CV R²={cv_scores.mean():.3f}")
    
    return model, r2, rmse

# 时间模型
print("时间模型:")
time_poly2, time_r2_2, time_rmse_2 = evaluate_polynomial(X, y_time, 2, "时间")
time_poly3, time_r2_3, time_rmse_3 = evaluate_polynomial(X, y_time, 3, "时间")

# 召回率模型  
print("召回率模型:")
recall_poly2, recall_r2_2, recall_rmse_2 = evaluate_polynomial(X, y_recall, 2, "召回率")
recall_poly3, recall_r2_3, recall_rmse_3 = evaluate_polynomial(X, y_recall, 3, "召回率")

# 3. 梯度提升回归
print("\n=== 梯度提升回归拟合 ===")

def evaluate_gbm(X, y, target_name):
    """评估梯度提升模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gbm = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        subsample=0.8
    )
    
    gbm.fit(X_train, y_train)
    
    # 测试集性能
    y_pred = gbm.predict(X_test)
    r2_test = r2_score(y_test, y_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 全数据性能
    y_pred_all = gbm.predict(X)
    r2_all = r2_score(y, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
    
    print(f"{target_name} GBM: 测试集 R²={r2_test:.3f}, RMSE={rmse_test:.3f}")
    print(f"             全数据 R²={r2_all:.3f}, RMSE={rmse_all:.3f}")
    
    # 特征重要性
    importance = gbm.feature_importances_
    print(f"特征重要性: C={importance[0]:.3f}, P={importance[1]:.3f}, I={importance[2]:.3f}")
    
    return gbm, r2_test, rmse_test

# 时间模型
print("时间模型:")
time_gbm, time_gbm_r2, time_gbm_rmse = evaluate_gbm(X, y_time, "时间")

# 召回率模型
print("召回率模型:")
recall_gbm, recall_gbm_r2, recall_gbm_rmse = evaluate_gbm(X, y_recall, "召回率")

# 4. 神经网络回归
print("\n=== 神经网络回归拟合 ===")

class SimpleNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=[32, 16, 8]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_neural_net(X, y, target_name, epochs=1000):
    """训练神经网络"""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 分割训练测试集
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
    
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        y_pred_all = model(X_tensor).squeeze().numpy()
        r2_all = r2_score(y, y_pred_all)
        rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
        
        y_pred_test = model(X_test).squeeze().numpy()
        r2_test = r2_score(y_test.numpy(), y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test.numpy(), y_pred_test))
    
    print(f"{target_name} NeuralNet: 测试集 R²={r2_test:.3f}, RMSE={rmse_test:.3f}")
    print(f"                    全数据 R²={r2_all:.3f}, RMSE={rmse_all:.3f}")
    
    return model, r2_test, rmse_test

# 时间模型（标准化数据）
print("时间模型:")
time_nn, time_nn_r2, time_nn_rmse = train_neural_net(X_scaled, y_time_scaled, "时间")

# 召回率模型（标准化数据）
print("召回率模型:")
recall_nn, recall_nn_r2, recall_nn_rmse = train_neural_net(X_scaled, y_recall_scaled, "召回率")

# 5. 模型比较和保存
print("\n=== 模型性能总结 ===")

results = {
    "time_models": {
        "polynomial_degree2": {"r2": float(time_r2_2), "rmse": float(time_rmse_2)},
        "polynomial_degree3": {"r2": float(time_r2_3), "rmse": float(time_rmse_3)},
        "gradient_boosting": {"r2": float(time_gbm_r2), "rmse": float(time_gbm_rmse)},
        "neural_network": {"r2": float(time_nn_r2), "rmse": float(time_nn_rmse)}
    },
    "recall_models": {
        "polynomial_degree2": {"r2": float(recall_r2_2), "rmse": float(recall_rmse_2)},
        "polynomial_degree3": {"r2": float(recall_r2_3), "rmse": float(recall_rmse_3)},
        "gradient_boosting": {"r2": float(recall_gbm_r2), "rmse": float(recall_gbm_rmse)},
        "neural_network": {"r2": float(recall_nn_r2), "rmse": float(recall_nn_rmse)}
    },
    "best_models": {
        "time": {
            "method": "",
            "r2": 0,
            "rmse": 0
        },
        "recall": {
            "method": "", 
            "r2": 0,
            "rmse": 0
        }
    },
    "data_info": {
        "total_samples": len(df),
        "C_range": [int(df['C'].min()), int(df['C'].max())],
        "P_range": [int(df['P'].min()), int(df['P'].max())],
        "I_values": [int(i) for i in sorted(df['I'].unique())]
    }
}

# 找出最佳模型
time_models = [
    ("poly2", time_r2_2), ("poly3", time_r2_3), 
    ("gbm", time_gbm_r2), ("nn", time_nn_r2)
]
best_time_method = max(time_models, key=lambda x: x[1])[0]
results["best_models"]["time"]["method"] = best_time_method
results["best_models"]["time"]["r2"] = max([r2 for _, r2 in time_models])
results["best_models"]["time"]["rmse"] = min([
    time_rmse_2, time_rmse_3, time_gbm_rmse, time_nn_rmse
])

recall_models = [
    ("poly2", recall_r2_2), ("poly3", recall_r2_3),
    ("gbm", recall_gbm_r2), ("nn", recall_nn_r2)  
]
best_recall_method = max(recall_models, key=lambda x: x[1])[0]
results["best_models"]["recall"]["method"] = best_recall_method
results["best_models"]["recall"]["r2"] = max([r2 for _, r2 in recall_models])
results["best_models"]["recall"]["rmse"] = min([
    recall_rmse_2, recall_rmse_3, recall_gbm_rmse, recall_nn_rmse
])

print(f"最佳时间模型: {best_time_method}, R²={results['best_models']['time']['r2']:.3f}")
print(f"最佳召回率模型: {best_recall_method}, R²={results['best_models']['recall']['r2']:.3f}")

# 保存结果
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== 模型预测示例 ===")
# 用最佳模型预测一些配置
test_configs = [
    [512, 16, 6],
    [640, 16, 6], 
    [5120, 80, 6],
    [1024, 32, 6],
    [2560, 48, 6]
]

print("配置预测示例 (使用GBM模型):")
for config in test_configs:
    C, P, I = config
    # 使用GBM预测
    time_pred = time_gbm.predict([config])[0]
    recall_pred = recall_gbm.predict([config])[0]
    print(f"C={C:4d}, P={P:3d}, I={I}: T≈{time_pred:5.2f}ms, R≈{recall_pred:.3f}")

print(f"\n所有结果已保存到 model_comparison_results.json")

# 6. 网格搜索建议
print("\n=== 网格搜索建议 ===")
print("基于数据分布，建议测试这些配置:")
print("中等C值 + 适当P值:")
for C in [384, 512, 640, 768]:
    for P in [12, 16, 20, 24]:
        if 0.02 <= P/C <= 0.04:
            print(f"  C={C}, P={P}, I=6")

print("\n大C值 + 相对小P值:")
for C in [2048, 3072, 4096, 5120]:
    for P in [48, 64, 80, 96]:
        if 0.015 <= P/C <= 0.025:
            print(f"  C={C}, P={P}, I=6")