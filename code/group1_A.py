import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import time

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, 'preprocessed_data')
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'group1_A')
os.makedirs(FIG_DIR, exist_ok=True)
TRAIN_END_DATE = '2023-12-31'
tickers = ['GOOGL', 'TSLA', 'NDX', 'MSFT', 'NVDA']

class BaselineExperiment:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = os.path.join(INPUT_FOLDER, f"{ticker}_preprocessed.csv")
        self.results = {}
    
    def load_and_prep_data(self):
        """
        数据加载与特定于 Baseline 的预处理
        - 特征策略: 仅使用单变量 (Lag-1 Price)
        - 标准化: 重新进行 Fit/Transform
        """
        if not os.path.exists(self.data_path):
            print(f"Error: 找不到文件 {self.data_path}")
            return False

        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. 数据切分
        train_df = df[df['date'] <= TRAIN_END_DATE].copy()
        test_df  = df[df['date'] > TRAIN_END_DATE].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            return False

        # 2. 特征选择 (Strict Baseline: 只用昨天价格预测明天)
        # 这种单变量方法是早期 GPR 论文的典型做法
        feature_col = ['lag_log_price_1'] 
        target_col = ['target_log_price']
        
        self.dates_test = test_df['date'].values
        
        # 3. 标准化 (只在训练集上 Fit)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_train = self.scaler_X.fit_transform(train_df[feature_col])
        self.y_train = self.scaler_y.fit_transform(train_df[target_col]).flatten()
        
        self.X_test = self.scaler_X.transform(test_df[feature_col])
        self.y_test_scaled = self.scaler_y.transform(test_df[target_col]).flatten()
        self.y_test_raw = test_df[target_col].values.flatten() # 真实的 Log Price
        
        return True

    def run(self):
        """执行全量静态训练"""
        print(f"[{self.ticker}] 正在运行 Group 1 (Baseline)...")
        start_time = time.time()
        
        # --- 核心定义：平稳核函数 ---
        # ConstantKernel * RBF + WhiteKernel
        # 这种组合假设数据特性在整个10年内是不变的 (平稳性假设)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        
        # 创建模型
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=42)
        
        # 全量训练 (Global Fit)
        gp.fit(self.X_train, self.y_train)
        
        # 预测 (一次性预测整个测试集)
        y_pred_scaled, y_std_scaled = gp.predict(self.X_test, return_std=True)
        
        elapsed = time.time() - start_time
        
        # --- 结果处理与反归一化 ---
        # 1. 还原预测值 (Log Scale -> Original Price Scale)
        pred_log = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        # 注意: 之前是对 Log Price 做标准化，所以 inverse 后得到 Log Price，再 exp 得到 Price
        pred_price = np.exp(pred_log)
        
        # 2. 还原真实值
        true_log = self.scaler_y.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten()
        true_price = np.exp(true_log)
        
        # 3. 还原置信区间 (近似)
        # Log 空间的 95% 区间: mu +/- 1.96*sigma
        lower_log = pred_log - 1.96 * y_std_scaled
        upper_log = pred_log + 1.96 * y_std_scaled
        lower_price = np.exp(lower_log)
        upper_price = np.exp(upper_log)
        
        # 4. 计算指标
        rmse = np.sqrt(mean_squared_error(true_price, pred_price))
        mae = mean_absolute_error(true_price, pred_price)
        
        print(f"  -> 完成! RMSE: {rmse:.2f}, Time: {elapsed:.2f}s")
        print(f"  -> Kernel Learned: {gp.kernel_}")
        
        self.results = {
            'RMSE': rmse,
            'MAE': mae,
            'Time': elapsed,
            'Pred': pred_price,
            'True': true_price,
            'Lower': lower_price,
            'Upper': upper_price,
            'Dates': self.dates_test
        }
        
        return self.results

    def plot_results(self):
        """画出 Baseline 的表现图"""
        res = self.results
        dates = res['Dates']
        
        plt.figure(figsize=(12, 6))
        
        # 真实股价
        plt.plot(dates, res['True'], 'k-', label='Actual Price', alpha=0.7)
        
        # 预测股价
        plt.plot(dates, res['Pred'], 'r--', label=f'Baseline Pred (RMSE={res["RMSE"]:.2f})')
        
        # 置信区间
        plt.fill_between(dates, res['Lower'], res['Upper'], color='red', alpha=0.1, label='95% Confidence (Global)')
        
        plt.title(f"Group 1 Baseline: {self.ticker} (Global Stationary GPR)\nSingle Feature Input", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        save_file = os.path.join(FIG_DIR, f"Group1_Baseline_{self.ticker}.png")
        plt.savefig(save_file)
        print(f"  -> 图表已保存: {save_file}\n")
        # plt.show() # 如果在 Notebook 中运行可取消注释

# --- 主执行入口 ---
if __name__ == "__main__":
    baseline_results = {}
    
    print("=== 开始运行第一组实验: 基准复现 (Group 1: Baseline) ===\n")
    
    for ticker in tickers:
        exp = BaselineExperiment(ticker)
        if exp.load_and_prep_data():
            res = exp.run()
            baseline_results[ticker] = res
            exp.plot_results()
        else:
            print(f"跳过 {ticker} (数据加载失败)")
            
    # 打印汇总成绩单
    print("\n" + "="*40)
    print(f"{'Ticker':<10} | {'RMSE':<10} | {'MAE':<10} | {'Time(s)':<10}")
    print("-" * 40)
    for ticker, res in baseline_results.items():
        print(f"{ticker:<10} | {res['RMSE']:<10.2f} | {res['MAE']:<10.2f} | {res['Time']:<10.2f}")
    print("="*40)

"""
=== 开始运行第一组实验: 基准复现 (Group 1: Baseline) ===

[GOOGL] 正在运行 Group 1 (Baseline)...
  -> 完成! RMSE: 7.69, Time: 41.66s
  -> Kernel Learned: 6.86**2 * RBF(length_scale=12.5) + WhiteKernel(noise_level=0.00324)
  -> 图表已保存: Group1_Baseline_GOOGL.png

[TSLA] 正在运行 Group 1 (Baseline)...
  -> 完成! RMSE: 16.94, Time: 37.46s
  -> Kernel Learned: 4.57**2 * RBF(length_scale=8.51) + WhiteKernel(noise_level=0.00173)
  -> 图表已保存: Group1_Baseline_TSLA.png

[NDX] 正在运行 Group 1 (Baseline)...
  -> 完成! RMSE: 414.62, Time: 36.61s
  -> Kernel Learned: 9.68**2 * RBF(length_scale=17.3) + WhiteKernel(noise_level=0.00216)
  -> 图表已保存: Group1_Baseline_NDX.png

[MSFT] 正在运行 Group 1 (Baseline)...
  -> 完成! RMSE: 8.65, Time: 42.43s
  -> Kernel Learned: 13.1**2 * RBF(length_scale=23.1) + WhiteKernel(noise_level=0.00134)
  -> 图表已保存: Group1_Baseline_MSFT.png

[NVDA] 正在运行 Group 1 (Baseline)...
  -> 完成! RMSE: 5.30, Time: 40.97s
  -> Kernel Learned: 18.5**2 * RBF(length_scale=32.5) + WhiteKernel(noise_level=0.00179)
  -> 图表已保存: Group1_Baseline_NVDA.png


========================================
Ticker     | RMSE       | MAE        | Time(s)
----------------------------------------
GOOGL      | 7.69       | 5.40       | 41.66
TSLA       | 16.94      | 12.46      | 37.46
NDX        | 414.62     | 329.85     | 36.61
MSFT       | 8.65       | 6.72       | 42.43
NVDA       | 5.30       | 4.10       | 40.97
========================================"""