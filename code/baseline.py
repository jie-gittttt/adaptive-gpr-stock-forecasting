import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time

# ==========================================
# 1. 加载 NASDAQ 数据
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file = os.path.join(BASE_DIR, 'data', 'HistoricalData_MSFT_1Y.csv') 

try:
    df = pd.read_csv(csv_file)
    df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    close_prices = df['Close/Last'].dropna()
    print(f"✅ 成功加载 NASDAQ 数据：{len(close_prices)} 行")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# 选取数据
prices = close_prices.values[-300:].reshape(-1, 1)
days = np.arange(len(prices)).reshape(-1, 1)

train_size = int(len(days) * 0.85)
X_train, X_test = days[:train_size], days[train_size:]
y_train, y_test = prices[:train_size], prices[train_size:]

# ==========================================
# 2. GPR 建模与预测 (记录效率)
# ==========================================
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)

# 记录开始时间以评估“效率”
start_time = time.time()

# 基础核函数配置
kernel = C(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)

gp.fit(X_train, y_train_scaled)
y_pred_scaled, sigma_scaled = gp.predict(X_test, return_std=True)

train_time = time.time() - start_time

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
sigma = (sigma_scaled * scaler_y.scale_[0]).reshape(-1, 1)

# ==========================================
# 3. 验证与评价 (核心改进)
# ==========================================
# A. 基础误差计算
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# B. 置信区间覆盖率验证 (GPR 特有验证)
lower_bound = y_pred - 1.96 * sigma
upper_bound = y_pred + 1.96 * sigma
# 检查真实值是否在范围内
covered = np.logical_and(y_test >= lower_bound, y_test <= upper_bound)
coverage_rate = np.mean(covered) * 100

print("\n" + "="*30)
print(f"📊 模型验证结果 ({len(y_test)} 天测试集):")
print(f"1. 预测偏差 (MAE): ${mae:.2f}")
print(f"2. 预测误差率 (MAPE): {mape*100:.2f}%")
print(f"3. 均方根误差 (RMSE): {rmse:.2f}")
print(f"4. 置信区间覆盖率: {coverage_rate:.2f}%")
print(f"5. 训练预测总耗时: {train_time:.4f} 秒")
print("="*30 + "\n")

# ==========================================
# 4. 绘图展示
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(days[:train_size], y_train, 'k-', alpha=0.3, linewidth=1, label='Historical Price')
plt.plot(days[train_size:], y_test, 'r-', label='Actual Price')
plt.plot(days[train_size:], y_pred, 'b--', label='GPR Forecast')
plt.fill_between(days[train_size:].flatten(), 
                 lower_bound.flatten(), 
                 upper_bound.flatten(), 
                 color='blue', alpha=0.15, label='95% Confidence Interval')
plt.title(f'GPR Stock Prediction (MSFT)\nMAPE: {mape*100:.2f}% | Coverage: {coverage_rate:.1f}%')
plt.legend()
plt.show()