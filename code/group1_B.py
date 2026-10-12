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
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'group1_B')
os.makedirs(FIG_DIR, exist_ok=True)

# [核心修改] 采用极具针对性的时间窗口
# 训练集: 2021 牛市 (让模型学到单边上涨趋势)
TRAIN_START_DATE = '2021-01-01'
TRAIN_END_DATE   = '2021-12-31'

# 测试集: 2022 初熊市开启 (观察模型如何崩盘)
TEST_END_DATE    = '2022-02-28' 

tickers = ['GOOGL', 'TSLA', 'NDX', 'MSFT', 'NVDA']

class BaselineExperiment:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = os.path.join(INPUT_FOLDER, f"{ticker}_preprocessed.csv")
        self.results = {}
    
    def load_and_prep_data(self):
        if not os.path.exists(self.data_path):
            print(f"Error: 找不到文件 {self.data_path}")
            return False

        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # 1. 严格的时间切分 (Training vs Testing)
        train_mask = (df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)
        # 注意：这里增加了测试集的结束时间限制
        test_mask  = (df['date'] > TRAIN_END_DATE) & (df['date'] <= TEST_END_DATE)
        
        train_df = df[train_mask].copy()
        test_df  = df[test_mask].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"[{self.ticker}] 数据不足 (Train:{len(train_df)}, Test:{len(test_df)})")
            return False

        # 2. 特征选择 (Baseline: 仅单变量 Lag-1 Price)
        feature_col = ['lag_log_price_1'] 
        target_col = ['target_log_price']
        
        # 保存原始数据
        self.test_dates = test_df['date'].values
        self.test_current_log_price = test_df['log_price'].values
        self.test_target_log_price = test_df['target_log_price'].values
        
        # 3. 标准化 (只在训练集上 Fit)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_train = self.scaler_X.fit_transform(train_df[feature_col])
        self.y_train = self.scaler_y.fit_transform(train_df[target_col]).flatten()
        
        self.X_test = self.scaler_X.transform(test_df[feature_col])
        self.y_test_scaled = self.scaler_y.transform(test_df[target_col]).flatten()
        
        print(f"[{self.ticker}] 数据加载成功: Train={len(train_df)}, Test={len(test_df)}")
        return True

    def calculate_metrics(self, pred_log_price):
        # 还原价格
        current_price = np.exp(self.test_current_log_price)
        true_next_price = np.exp(self.test_target_log_price)
        pred_next_price = np.exp(pred_log_price)
        
        # --- 基础误差 ---
        rmse = np.sqrt(mean_squared_error(true_next_price, pred_next_price))
        mae = mean_absolute_error(true_next_price, pred_next_price)
        mape = np.mean(np.abs((true_next_price - pred_next_price) / true_next_price)) * 100
        
        # --- 方向准确率 (DA) ---
        true_diff = true_next_price - current_price
        pred_diff = pred_next_price - current_price
        da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
        
        # --- Regret (遗憾值) ---
        actual_return = (true_next_price - current_price) / current_price
        
        # 策略: 预测涨则买，否则空仓
        model_signal = np.where(pred_diff > 0, 1, 0)
        model_returns = model_signal * actual_return
        
        # 上帝视角: 只要涨就买
        optimal_signal = np.where(actual_return > 0, 1, 0)
        optimal_returns = optimal_signal * actual_return
        
        cumulative_regret = np.cumsum(optimal_returns - model_returns)
        
        return {
            'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'DA': da,
            'Regret': cumulative_regret[-1], 'CumRegret': cumulative_regret,
            'Pred': pred_next_price, 'True': true_next_price
        }

    def run(self):
        print(f"[{self.ticker}] 启动训练 (2021全量静态拟合)...")
        start_time = time.time()
        
        # RBF 核函数
        kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=42)
        gp.fit(self.X_train, self.y_train)
        
        # 预测 2022 初
        y_pred_scaled, y_std_scaled = gp.predict(self.X_test, return_std=True)
        elapsed = time.time() - start_time
        
        # 还原
        pred_log = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 计算
        metrics = self.calculate_metrics(pred_log)
        metrics['Time'] = elapsed
        metrics['Dates'] = self.test_dates
        self.results = metrics
        
        print(f"  -> RMSE: {metrics['RMSE']:.2f} | Regret: {metrics['Regret']:.2%}")
        return self.results

    def plot_results(self):
        res = self.results
        dates = res['Dates']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 图1: 股价趋势 (展示滞后性)
        ax1.plot(dates, res['True'], 'k-', label='True Price', alpha=0.6)
        ax1.plot(dates, res['Pred'], 'r--', label='Baseline Pred (Static)')
        ax1.set_title(f"{self.ticker}: Static GPR Failure (Train 2021 -> Test 2022)\nRMSE={res['RMSE']:.2f}, MAPE={res['MAPE']:.2f}%")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: Regret (展示亏损累积)
        ax2.plot(dates, res['CumRegret'], 'r-', label='Cumulative Regret', linewidth=2)
        ax2.fill_between(dates, res['CumRegret'], 0, color='red', alpha=0.1)
        ax2.set_title(f"Cumulative Regret (Total: {res['Regret']:.2%})")
        ax2.set_ylabel("Regret")
        ax2.set_xlabel("Date (2022 Jan-Feb)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"Group1_Baseline_{self.ticker}_2022Test.png"))
        print(f"  -> 图表已保存: Group1_Baseline_{self.ticker}_2022Test.png\n")

# --- 执行 ---
if __name__ == "__main__":
    baseline_results = {}
    print(f"=== Group 1: Strategic Baseline (Train: 2021, Test: 2022 Jan-Feb) ===\n")
    
    for ticker in tickers:
        exp = BaselineExperiment(ticker)
        if exp.load_and_prep_data():
            baseline_results[ticker] = exp.run()
            exp.plot_results()
    
    print("\n" + "="*80)
    print(f"{'Ticker':<8} | {'RMSE':<8} | {'MAPE':<8} | {'DA(%)':<8} | {'Regret':<10} | {'Time':<6}")
    print("-" * 80)
    for t, r in baseline_results.items():
        print(f"{t:<8} | {r['RMSE']:<8.2f} | {r['MAPE']:<8.2f} | {r['DA']:<8.1f} | {r['Regret']:<10.2%} | {r['Time']:<6.2f}")
    print("="*80)

"""
=== Group 1: Strategic Baseline (Train: 2021, Test: 2022 Jan-Feb) ===

[GOOGL] 数据加载成功: Train=252, Test=39
[GOOGL] 启动训练 (2021全量静态拟合)...
  -> RMSE: 4.30 | Regret: 36.05%
  -> 图表已保存: Group1_Baseline_GOOGL_2022Test.png

[TSLA] 数据加载成功: Train=252, Test=39
[TSLA] 启动训练 (2021全量静态拟合)...
  -> RMSE: 20.23 | Regret: 78.38%
  -> 图表已保存: Group1_Baseline_TSLA_2022Test.png

[NDX] 数据加载成功: Train=252, Test=39
[NDX] 启动训练 (2021全量静态拟合)...
  -> RMSE: 436.14 | Regret: 31.71%
  -> 图表已保存: Group1_Baseline_NDX_2022Test.png

[MSFT] 数据加载成功: Train=252, Test=39
[MSFT] 启动训练 (2021全量静态拟合)...
  -> RMSE: 8.52 | Regret: 30.05%
  -> 图表已保存: Group1_Baseline_MSFT_2022Test.png

[NVDA] 数据加载成功: Train=252, Test=39
[NVDA] 启动训练 (2021全量静态拟合)...
  -> RMSE: 1.80 | Regret: 70.57%
  -> 图表已保存: Group1_Baseline_NVDA_2022Test.png


================================================================================
Ticker   | RMSE     | MAPE     | DA(%)    | Regret     | Time
--------------------------------------------------------------------------------
GOOGL    | 4.30     | 2.64     | 46.2     | 36.05%     | 0.27
TSLA     | 20.23    | 5.81     | 46.2     | 78.38%     | 0.27
NDX      | 436.14   | 2.57     | 51.3     | 31.71%     | 0.28
MSFT     | 8.52     | 2.34     | 51.3     | 30.05%     | 0.30
NVDA     | 1.80     | 6.08     | 48.7     | 70.57%     | 0.29
================================================================================"""