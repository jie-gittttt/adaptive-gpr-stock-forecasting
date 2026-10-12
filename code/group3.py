import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import os
import time
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, 'preprocessed_data')
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'group3')
os.makedirs(FIG_DIR, exist_ok=True)

# [战场选择] 2022 年初的崩盘期 (Regime Shift)
TEST_START_DATE = '2022-01-01'
TEST_END_DATE   = '2022-02-28'

# [自适应逻辑参数 - 激进调优版]
# 1. 突破下限: 允许窗口缩到 15 天 (比 Group 2B 的 30 天更快)
MIN_WINDOW = 15   
# 2. 正常上限: 平稳期保持 200 天
MAX_WINDOW = 200  
# 3. 极速感知: 只看过去 5 天的波动率，不再看 20 天
VOL_LOOKBACK = 5 

tickers = ['GOOGL', 'TSLA', 'NDX', 'MSFT', 'NVDA']

class AdaptiveExperiment:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = os.path.join(INPUT_FOLDER, f"{ticker}_preprocessed.csv")
        self.df = None
        self.feature_cols = None
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Error: 找不到文件 {self.data_path}")
            return False

        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        self.df = df
        
        # 特征选择
        exclude_cols = ['date', 'close', 'volume', 'open', 'high', 'low', 'ma50', 'target_log_price']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.target_col = 'target_log_price'
        
        # 确定测试范围
        mask = (df['date'] >= TEST_START_DATE) & (df['date'] <= TEST_END_DATE)
        if mask.sum() == 0: 
            print(f"[{self.ticker}] 警告: 测试时间段内无数据")
            return False
        self.test_indices = df[mask].index.tolist()
        
        return True

    def calculate_volatility_and_window(self, current_idx):
        """
        [调优核心] 激进型自适应窗口算法
        """
        # 1. 获取过去 5 天的极短期波动
        start_vol = max(0, current_idx - VOL_LOOKBACK)
        recent_data = self.df.iloc[start_vol:current_idx]
        
        if 'log_ret' in recent_data.columns:
            # 加上极小值防止除零
            vol = recent_data['log_ret'].std() + 1e-6
        else:
            vol = 0.02
            
        # 2. 非线性映射函数 (Sigmoid-like response)
        # 只要波动率稍微抬头 (>1.5%)，马上进入恐慌模式
        # 阈值设定: 
        #   Vol < 0.01 (1%) -> Stable (Max Window)
        #   Vol > 0.025 (2.5%) -> Panic (Min Window)
        
        low_vol_thresh = 0.010
        high_vol_thresh = 0.025 
        
        if vol <= low_vol_thresh:
            window = MAX_WINDOW
        elif vol >= high_vol_thresh:
            window = MIN_WINDOW
        else:
            # 线性插值
            ratio = (vol - low_vol_thresh) / (high_vol_thresh - low_vol_thresh)
            # 加速衰减: 让窗口缩得更快
            ratio = np.sqrt(ratio) 
            window = MAX_WINDOW - ratio * (MAX_WINDOW - MIN_WINDOW)
            
        return int(window), vol

    def calculate_metrics(self, trues, preds, currents, elapsed_time):
        true_price = np.exp(trues)
        pred_price = np.exp(preds)
        curr_price = np.exp(currents)
        
        rmse = np.sqrt(mean_squared_error(true_price, pred_price))
        mape = np.mean(np.abs((true_price - pred_price) / true_price)) * 100
        
        true_diff = true_price - curr_price
        pred_diff = pred_price - curr_price
        da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
        
        actual_ret = (true_price - curr_price) / curr_price
        model_signal = np.where(pred_diff > 0, 1, 0)
        oracle_signal = np.where(actual_ret > 0, 1, 0)
        
        cumulative_regret = np.cumsum(oracle_signal * actual_ret - model_signal * actual_ret)
        total_regret = cumulative_regret[-1] if len(cumulative_regret) > 0 else 0
        
        return {'RMSE': rmse, 'MAPE': mape, 'DA': da, 'Regret': total_regret, 'Time': elapsed_time, 'Pred': pred_price, 'True': true_price}

    def run_adaptive(self):
        if not self.load_data(): return None

        print(f"[{self.ticker}] 启动 Group 3 (Aggressive): 极速自适应模型...")
        start_time = time.time()
        
        X_all = self.df[self.feature_cols].values
        y_all = self.df[self.target_col].values
        curr_all = self.df['log_price'].values
        dates_all = self.df['date'].values
        
        preds, trues, currs, out_dates = [], [], [], []
        window_history = [] 
        
        # [核函数调优] 回归 RationalQuadratic
        # 文献[124]提到 RQ 核是不同尺度 RBF 的加权和，非常适合多尺度波动
        # 相比 DotProduct，它不会因为之前的长期上涨而产生过强的"动量惯性"
        kernel = C(1.0) * RBF(length_scale=1.0) + \
                 C(0.5) * RationalQuadratic(alpha=0.5, length_scale=1.0) + \
                 WhiteKernel(noise_level=0.1)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        
        for i in self.test_indices:
            # 1. 激进的窗口计算
            window_size, _ = self.calculate_volatility_and_window(i)
            window_history.append(window_size)
            
            # 2. 极短窗口训练
            train_start = max(0, i - window_size)
            X_train = X_all[train_start:i]
            y_train = y_all[train_start:i]
            
            # 保证至少有少量数据
            if len(X_train) < 5: 
                # 如果窗口太小导致没数据，强制回溯一点点
                X_train = X_all[i-10:i]
                y_train = y_all[i-10:i]
            
            try:
                gp.fit(X_train, y_train)
                y_pred = gp.predict(X_all[i:i+1])
                
                preds.append(y_pred[0])
                trues.append(y_all[i])
                currs.append(curr_all[i])
                out_dates.append(dates_all[i])
            except:
                pass
        
        elapsed = time.time() - start_time
        metrics = self.calculate_metrics(np.array(trues), np.array(preds), np.array(currs), elapsed)
        
        metrics['Dates'] = out_dates
        metrics['WindowHist'] = window_history
        
        self.plot_adaptive_behavior(metrics)
        return metrics

    def plot_adaptive_behavior(self, res):
        dates = res['Dates']
        windows = res['WindowHist']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 图1: 价格预测
        ax1.plot(dates, res['True'], 'k-', label='True Price', alpha=0.6, linewidth=2)
        ax1.plot(dates, res['Pred'], 'r--', label='Adaptive Pred', linewidth=1.5)
        ax1.set_title(f"Group 3 (Aggressive): {self.ticker}\nRMSE={res['RMSE']:.2f}, Regret={res['Regret']:.2%}")
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: 窗口变化 (观察是否经常触底到 15)
        ax2.plot(dates, windows, 'b-', label='Window Size', linewidth=2)
        ax2.fill_between(dates, windows, MIN_WINDOW, color='blue', alpha=0.1)
        ax2.set_ylabel("Window Size (Days)")
        ax2.set_title("Adaptive Mechanism: Quick response to Volatility")
        ax2.axhline(y=MIN_WINDOW, color='r', linestyle=':', label='Min (15d)')
        ax2.axhline(y=30, color='gray', linestyle='--', label='Group 2B Ref (30d)')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fname = os.path.join(FIG_DIR, f"Group3_Aggressive_{self.ticker}.png")
        plt.savefig(fname)
        print(f"  -> 图表已保存: {fname}")
        plt.close()

# --- 主程序 ---
if __name__ == "__main__":
    summary = {}
    
    print("="*80)
    print(" >>> Group 3: Aggressive Adaptive Model (Tuned)")
    print(f" >>> Params: Min_Window={MIN_WINDOW}, Lookback={VOL_LOOKBACK}d")
    print("="*80)
    
    for ticker in tickers:
        exp = AdaptiveExperiment(ticker)
        res = exp.run_adaptive()
        if res:
            summary[ticker] = res
            
    print("\n" + "="*80)
    print("TABLE 3 (Tuned): Adaptive Model Performance")
    print("-" * 80)
    print(f"{'Ticker':<8} | {'RMSE':<8} | {'MAPE':<8} | {'DA(%)':<8} | {'Regret':<10} | {'Time':<6}")
    print("-" * 80)
    for t, r in summary.items():
        print(f"{t:<8} | {r['RMSE']:<8.2f} | {r['MAPE']:<8.2f} | {r['DA']:<8.1f} | {r['Regret']:<10.2%} | {r['Time']:<6.1f}s")
    print("="*80)

"""

================================================================================
TABLE 3 (Tuned): Adaptive Model Performance
--------------------------------------------------------------------------------
Ticker   | RMSE     | MAPE     | DA(%)    | Regret     | Time
--------------------------------------------------------------------------------
GOOGL    | 3.29     | 1.86     | 48.7     | 29.70%     | 5.4   s
TSLA     | 15.24    | 3.93     | 53.8     | 73.50%     | 2.5   s
NDX      | 304.05   | 1.73     | 48.7     | 34.27%     | 6.4   s
MSFT     | 6.27     | 1.67     | 56.4     | 24.96%     | 5.0   s
NVDA     | 1.06     | 3.46     | 51.3     | 66.01%     | 3.1   s
================================================================================"""