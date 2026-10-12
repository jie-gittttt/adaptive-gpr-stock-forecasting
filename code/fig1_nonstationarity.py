import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 数据加载与预处理
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_name = os.path.join(BASE_DIR, 'data', 'HistoricalData_NDX_10Y.csv')
df = pd.read_csv(file_name)

# 转换日期并清洗价格列（去除 $ 符号）
df['Date'] = pd.to_datetime(df['Date'])
df['Close/Last'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
df = df.sort_values('Date')

# 2. 选取关键时间段（2021年中至2022年中）以展示非平稳性
mask = (df['Date'] >= '2021-06-01') & (df['Date'] <= '2022-06-01')
plot_df = df[mask].copy()

# 设置论文风格
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 300
})

fig, ax = plt.subplots(figsize=(12, 6))

# 绘制股价曲线
ax.plot(plot_df['Date'], plot_df['Close/Last'], color='#2c3e50', linewidth=1.8, label='NVDA Closing Price')

# 3. 标注“平稳期”与“剧烈波动期”
y_min, y_max = plot_df['Close/Last'].min(), plot_df['Close/Last'].max()
y_lim_min, y_lim_max = y_min * 0.9, y_max * 1.1

# 绿色区域：平稳增长期 (Stationary Phase)
stable_start, stable_end = pd.to_datetime('2021-06-01'), pd.to_datetime('2021-12-31')
ax.add_patch(patches.Rectangle((stable_start, y_lim_min), stable_end - stable_start, 
                               y_lim_max - y_lim_min, facecolor='#27ae60', alpha=0.15))
ax.text(pd.to_datetime('2021-09-15'), y_max, "Stable Growth Phase\n(Stationary Assumption Holds)", 
        color='#1b5e20', fontsize=10, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# 红色区域：结构性转折期 (Regime Shift)
crash_start, crash_end = pd.to_datetime('2022-01-01'), pd.to_datetime('2022-05-31')
ax.add_patch(patches.Rectangle((crash_start, y_lim_min), crash_end - crash_start, 
                               y_lim_max - y_lim_min, facecolor='#c0392b', alpha=0.15))
ax.text(pd.to_datetime('2022-03-15'), y_max, "Regime Shift / Volatile Phase\n(Non-stationary Violation)", 
        color='#b71c1c', fontsize=10, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# 4. 挑战标注：固定核函数的失效
target_date = pd.to_datetime('2022-01-10')
target_price = plot_df[plot_df['Date'] == target_date]['Close/Last'].values[0]
ax.annotate('Fixed Kernel Fails to Adapt\n(Length-scale Misalignment)', 
            xy=(target_date, target_price), xytext=(pd.to_datetime('2021-10-01'), y_min),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10, color='#e67e22', fontweight='bold')

# 图表美化
ax.set_title("Fig. 1: Analysis of Non-stationarity and Regime Shift in NDX Price Series", pad=20)
ax.set_ylabel("Stock Price (USD)")
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(y_lim_min, y_lim_max)
plt.tight_layout()

# 保存
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(FIG_DIR, 'figure1_nonstationarity.png'))