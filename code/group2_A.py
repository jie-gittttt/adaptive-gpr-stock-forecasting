import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import time
import warnings

# 忽略 GPR 优化过程中的收敛警告
warnings.filterwarnings('ignore')

# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FOLDER = os.path.join(BASE_DIR, 'preprocessed_data')
# [时间对齐 Group 1A] 测试 2024 年至今的表现 (宏观稳定期)
TEST_START_DATE = '2024-01-01' 

# 窗口设置 (核心对比: 短期 vs 长期)
WINDOW_SHORT = 30   # 短期记忆 (约1.5个月, 灵敏但噪)
WINDOW_LONG  = 200  # 长期记忆 (约10个月, 稳健但滞后)

tickers = ['GOOGL', 'TSLA', 'NDX', 'MSFT', 'NVDA']

class RollingExperimentA:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = os.path.join(INPUT_FOLDER, f"{ticker}_preprocessed.csv")
        self.results = {}
        self.df = None
        self.feature_cols = None
        
    def load_data(self):
        """加载数据并准备"""
        if not os.path.exists(self.data_path):
            print(f"Error: 找不到文件 {self.data_path}")
            return False

        df = pd.read_csv(self.data_path)
        df['date'] = pd.to_datetime(df['date'])
        self.df = df
        
        # 特征选择 (排除非特征列)
        exclude_cols = ['date', 'close', 'volume', 'open', 'high', 'low', 'ma50', 'target_log_price']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.target_col = 'target_log_price'
        
        # 确定测试集开始的索引
        # 我们从 TEST_START_DATE 开始预测，但训练窗口会在这个日期之前滑动
        test_data = df[df['date'] >= TEST_START_DATE]
        if len(test_data) == 0:
            print(f"[{self.ticker}] 无测试数据 (日期 >= {TEST_START_DATE})")
            return False
            
        self.test_start_idx = test_data.index[0]
        # 还要获取“当前价格”用于计算方向和收益率
        # 注意: df['log_price'] 是 T 时刻价格, df['target_log_price'] 是 T+1
        self.current_log_price_col = 'log_price'
        
        return True

    def calculate_metrics(self, trues, preds, currents, elapsed_time):
        """计算全套指标: RMSE, MAPE, DA, Regret"""
        # 还原为真实价格
        true_price = np.exp(trues)
        pred_price = np.exp(preds)
        curr_price = np.exp(currents)
        
        # 1. 基础误差
        rmse = np.sqrt(mean_squared_error(true_price, pred_price))
        
        # 2. MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((true_price - pred_price) / true_price)) * 100
        
        # 3. DA (方向准确率)
        # 真实涨跌: T+1 - T
        true_diff = true_price - curr_price
        # 预测涨跌: Pred(T+1) - T
        pred_diff = pred_price - curr_price
        # 符号相同则方向正确
        da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
        
        # 4. Regret (遗憾值)
        # 实际收益率
        actual_ret = (true_price - curr_price) / curr_price
        
        # 模型策略: 预测涨则买(1), 否则空仓(0)
        model_signal = np.where(pred_diff > 0, 1, 0)
        model_ret = model_signal * actual_ret
        
        # 上帝策略: 实际涨则买(1), 否则空仓(0)
        oracle_signal = np.where(actual_ret > 0, 1, 0)
        oracle_ret = oracle_signal * actual_ret
        
        # 累积遗憾
        regret_curve = np.cumsum(oracle_ret - model_ret)
        total_regret = regret_curve[-1] if len(regret_curve) > 0 else 0
        
        return {
            'RMSE': rmse,
            'MAPE': mape,
            'DA': da,
            'Regret': total_regret,
            'Time': elapsed_time,
            'Pred': pred_price,
            'True': true_price,
            'Dates': None # 在外层赋值
        }

    def run_rolling(self, window_size, step=5):
        """执行滑动窗口预测"""
        print(f"  > 启动滑动预测 (Window={window_size})...")
        start_time = time.time()
        
        # 准备数据矩阵
        X_all = self.df[self.feature_cols].values
        y_all = self.df[self.target_col].values
        curr_all = self.df[self.current_log_price_col].values # T时刻价格
        dates_all = self.df['date'].values
        
        preds_list = []
        trues_list = []
        currs_list = []
        dates_list = []
        
        # 组合核函数: RBF(趋势) + RationalQuadratic(细节) + Noise
        kernel = C(1.0) * RBF(length_scale=1.0) + \
                 C(0.5) * RationalQuadratic(alpha=1.0, length_scale=1.0) + \
                 WhiteKernel(noise_level=0.1)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        
        total_len = len(self.df)
        
        # 滑动循环
        for i in range(self.test_start_idx, total_len, step):
            # 1. 训练窗口: [i-window : i]
            train_start = max(0, i - window_size)
            train_end = i
            
            X_train = X_all[train_start:train_end]
            y_train = y_all[train_start:train_end]
            
            # 2. 预测窗口: [i : i+step]
            pred_end = min(i + step, total_len)
            X_test_batch = X_all[i:pred_end]
            
            if len(X_train) < 10: continue # 保护机制
            
            # 3. Fit & Predict
            try:
                gp.fit(X_train, y_train)
                y_pred_batch = gp.predict(X_test_batch)
                
                # 收集结果
                preds_list.extend(y_pred_batch)
                trues_list.extend(y_all[i:pred_end])
                currs_list.extend(curr_all[i:pred_end]) # 获取对应的 T 时刻价格
                dates_list.extend(dates_all[i:pred_end])
                
            except Exception as e:
                pass 
                
        elapsed = time.time() - start_time
        
        # 计算指标
        metrics = self.calculate_metrics(
            np.array(trues_list), 
            np.array(preds_list), 
            np.array(currs_list), 
            elapsed
        )
        metrics['Dates'] = dates_list
        return metrics

    def run_experiment(self):
        if not self.load_data(): return None
        
        print(f"[{self.ticker}] 运行 Group 2A: 宏观稳定期 (2024+)...")
        
        # 运行两组窗口
        res_short = self.run_rolling(WINDOW_SHORT)
        res_long  = self.run_rolling(WINDOW_LONG)
        
        return res_short, res_long

# --- 主程序 ---
if __name__ == "__main__":
    summary_short = {}
    summary_long = {}
    
    print("="*80)
    print(" >>> Group 2A: Optimization with Rolling Window (Macro Stability)")
    print(f" >>> Test Period: {TEST_START_DATE} to Present")
    print(f" >>> Strategy: Short Window ({WINDOW_SHORT}d) vs Long Window ({WINDOW_LONG}d)")
    print("="*80)
    
    for ticker in tickers:
        exp = RollingExperimentA(ticker)
        res = exp.run_experiment()
        
        if res:
            summary_short[ticker] = res[0]
            summary_long[ticker]  = res[1]
    
    # --- 输出结果表格 ---
    
    # 1. 短期窗口结果
    print("\n" + "="*80)
    print(f"TABLE 2A-1: Short Window ({WINDOW_SHORT} Days) - High Sensitivity")
    print("-" * 80)
    print(f"{'Ticker':<8} | {'RMSE':<8} | {'MAPE':<8} | {'DA(%)':<8} | {'Regret':<10} | {'Time':<6}")
    print("-" * 80)
    for t, r in summary_short.items():
        print(f"{t:<8} | {r['RMSE']:<8.2f} | {r['MAPE']:<8.2f} | {r['DA']:<8.1f} | {r['Regret']:<10.2%} | {r['Time']:<6.1f}s")
        
    # 2. 长期窗口结果
    print("\n" + "="*80)
    print(f"TABLE 2A-2: Long Window ({WINDOW_LONG} Days) - Trend Following")
    print("-" * 80)
    print(f"{'Ticker':<8} | {'RMSE':<8} | {'MAPE':<8} | {'DA(%)':<8} | {'Regret':<10} | {'Time':<6}")
    print("-" * 80)
    for t, r in summary_long.items():
        print(f"{t:<8} | {r['RMSE']:<8.2f} | {r['MAPE']:<8.2f} | {r['DA']:<8.1f} | {r['Regret']:<10.2%} | {r['Time']:<6.1f}s")
    
    print("="*80)
    print("提示: 若 Long Window 的 DA(%) 高于 Short Window，说明在宏观稳定期，保留长期记忆有助于捕捉主升浪。")

"""
================================================================================
TABLE 2A-1: Short Window (30 Days) - High Sensitivity
--------------------------------------------------------------------------------
Ticker   | RMSE     | MAPE     | DA(%)    | Regret     | Time
--------------------------------------------------------------------------------
GOOGL    | 7.52     | 2.95     | 47.5     | 375.23%    | 7.8   s
TSLA     | 22.79    | 5.66     | 49.3     | 723.52%    | 8.6   s
NDX      | 494.29   | 1.80     | 47.1     | 225.83%    | 9.7   s
MSFT     | 11.40    | 1.90     | 47.5     | 251.80%    | 7.8   s
NVDA     | 6.96     | 4.19     | 50.3     | 549.39%    | 7.9   s

================================================================================
TABLE 2A-2: Long Window (200 Days) - Trend Following
--------------------------------------------------------------------------------
Ticker   | RMSE     | MAPE     | DA(%)    | Regret     | Time
--------------------------------------------------------------------------------
GOOGL    | 6.69     | 2.66     | 47.3     | 362.31%    | 43.8  s
TSLA     | 19.37    | 5.11     | 52.3     | 708.20%    | 38.8  s
NDX      | 474.81   | 1.72     | 45.9     | 221.89%    | 41.3  s
MSFT     | 10.82    | 1.89     | 44.8     | 265.99%    | 34.9  s
NVDA     | 7.05     | 4.27     | 49.3     | 622.76%    | 38.9  s
================================================================================"""