import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 1. 加载数据 - 增加鲁棒的清洗逻辑
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_name = os.path.join(BASE_DIR, 'data', 'HistoricalData_MSFT_10Y.csv')
df = pd.read_csv(file_name)

# 确保日期格式正确
df['Date'] = pd.to_datetime(df['Date'])

# 核心修复：清洗价格列中的 $ 和 , 符号，并强制转换为 float
if df['Close/Last'].dtype == object:
    df['Close'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
else:
    df['Close'] = df['Close/Last'].astype(float)

df = df.sort_values('Date')

# 选取包含平稳与崩盘的典型区间
mask = (df['Date'] >= '2021-06-01') & (df['Date'] <= '2022-05-01')
plot_df = df[mask].copy()

# 2. 绘图设置
plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制灰色背景曲线
ax.plot(plot_df['Date'], plot_df['Close'], color='lightgray', linewidth=1.5, label='MSFT Price Context')

# 获取全局最小值用于阴影底界
y_bottom = plot_df['Close'].min() * 0.95

# --- 场景 A: 平稳期 (Stable) -> 长窗口 ---
t1_date = pd.to_datetime('2021-11-15')
long_window = 90
t1_start = t1_date - pd.Timedelta(days=long_window)
mask_long = (plot_df['Date'] >= t1_start) & (plot_df['Date'] <= t1_date)
df_long = plot_df[mask_long]

# 高亮长窗口
ax.plot(df_long['Date'], df_long['Close'], color='#27ae60', linewidth=2.5, label='Active (Long Window)')
ax.fill_between(df_long['Date'], df_long['Close'], y_bottom, color='#27ae60', alpha=0.1)

# --- 场景 B: 波动期 (Volatile) -> 短窗口 ---
t2_date = pd.to_datetime('2022-03-01')
short_window = 30
t2_start = t2_date - pd.Timedelta(days=short_window)
mask_short = (plot_df['Date'] >= t2_start) & (plot_df['Date'] <= t2_date)
df_short = plot_df[mask_short]

# 高亮短窗口
ax.plot(df_short['Date'], df_short['Close'], color='#c0392b', linewidth=2.5, label='Active (Short Window)')
ax.fill_between(df_short['Date'], df_short['Close'], y_bottom, color='#c0392b', alpha=0.1)

# --- 3. 标注 ---
t1_price = df_long['Close'].iloc[-1]
ax.vlines(t1_date, y_bottom, t1_price, color='#27ae60', linestyle='--', alpha=0.8)
ax.annotate('', xy=(t1_start, t1_price + 600), xytext=(t1_date, t1_price + 600),
            arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
ax.text(t1_start + (t1_date - t1_start)/2, t1_price + 800, f"Long Window ({long_window}d)", 
        color='#27ae60', ha='center', fontweight='bold')

t2_price = df_short['Close'].iloc[-1]
ax.vlines(t2_date, y_bottom, t2_price, color='#c0392b', linestyle='--', alpha=0.8)
ax.annotate('', xy=(t2_start, t2_price + 600), xytext=(t2_date, t2_price + 600),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.5))
ax.text(t2_start + (t2_date - t2_start)/2, t2_price + 800, f"Short Window ({short_window}d)", 
        color='#c0392b', ha='center', fontweight='bold')

# 标题美化
ax.set_title("Fig. 3: Schematic of Adaptive Window Strategy", pad=20, fontsize=14)
ax.set_ylabel("Index Value")
ax.set_ylim(y_bottom, plot_df['Close'].max() * 1.15)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper left')

plt.tight_layout()
FIG_DIR = os.path.join(BASE_DIR, 'figures')
plt.savefig(os.path.join(FIG_DIR, 'figure2_adaptive_window.png'))
print("Fix applied successfully. Image saved.")