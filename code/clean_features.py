import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# 路径解析：相对脚本位置，从任意工作目录运行都成立
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed_data')

# 1. 配置文件列表
files = {t: os.path.join(DATA_DIR, f'HistoricalData_{t}_10Y.csv')
         for t in ('GOOGL', 'TSLA', 'NDX', 'MSFT', 'NVDA')}

def clean_data(file_path):
    """
    基础清洗模块：读取、统一列名、去符号、排序
    """
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None

    df = pd.read_csv(file_path)
    
    # 统一列名格式
    df.columns = [c.strip().lower().replace('/', '_') for c in df.columns]
    
    # 日期处理
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # 数值处理：去除 '$' 符号并转为 float
    cols_to_clean = ['close_last', 'open', 'high', 'low']
    for col in cols_to_clean:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 成交量处理
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    else:
        df['volume'] = 0
        
    df = df.rename(columns={'close_last': 'close'})
    return df[['date', 'close', 'volume', 'open', 'high', 'low']]

def feature_engineering(df):
    """
    特征工程模块：构建多维特征
    """
    df = df.copy()
    
    # --- 1. 基础变换 ---
    df['log_price'] = np.log(df['close'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log1p(df['volume'])
    
    # --- 2. 非平稳性与趋势特征 ---
    df['rolling_std_20'] = df['log_ret'].rolling(window=20).std()
    df['ma50'] = df['close'].rolling(window=50).mean()
    df['trend_dev'] = (df['close'] - df['ma50']) / df['ma50']
    df['day_idx'] = np.arange(len(df))
    
    # --- 3. 滞后特征 (Lags) ---
    for lag in [1, 2, 3, 5]:
        df[f'lag_log_price_{lag}'] = df['log_price'].shift(lag)
        
    df['lag_rolling_std'] = df['rolling_std_20'].shift(1)
    df['lag_log_vol'] = df['log_vol'].shift(1)
    
    # --- 4. 目标变量 (Target) ---
    df['target_log_price'] = df['log_price'].shift(-1)
    
    # 去除 NaN
    df = df.dropna()
    
    return df

def save_and_split(ticker, df, output_folder, train_end_date='2023-12-31'):
    """
    保存预处理文件 + 切分数据集 + 标准化
    """
    # -------------------------------------------------------
    # [新增功能] 保存带有特征的完整 CSV 到指定文件夹
    # -------------------------------------------------------
    save_path = os.path.join(output_folder, f"{ticker}_preprocessed.csv")
    df.to_csv(save_path, index=False)
    print(f"  -> [保存成功] 已保存特征文件至: {save_path}")
    
    # 接下来继续做模型需要的切分和标准化...
    train_df = df[df['date'] <= train_end_date].copy()
    test_df  = df[df['date'] > train_end_date].copy()
    
    print(f"  - 训练集: {len(train_df)} | 测试集: {len(test_df)}")
    
    # 标准化 (Standardization)
    feature_cols = [c for c in df.columns if c not in ['date', 'close', 'volume', 'open', 'high', 'low', 'ma50', 'target_log_price']]
    target_col = 'target_log_price'
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit on Train ONLY
    X_train = scaler_X.fit_transform(train_df[feature_cols])
    y_train = scaler_y.fit_transform(train_df[[target_col]]).flatten()
    
    # Transform Test
    X_test = scaler_X.transform(test_df[feature_cols])
    y_test = scaler_y.transform(test_df[[target_col]]).flatten()
    
    return {
        'ticker': ticker,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test,   'y_test': y_test,
        'dates_train': train_df['date'],
        'dates_test': test_df['date'],
        'scaler_y': scaler_y, 
        'feature_names': feature_cols
    }

# --- 主执行流程 ---

# 1. 创建文件夹
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建文件夹: {OUTPUT_DIR}")

processed_datasets = {}

print("开始批量处理...")
for ticker, path in files.items():
    print(f"\n正在处理股票: {ticker} ...")
    
    # 1. 清洗
    df_clean = clean_data(path)
    if df_clean is None: continue
    
    # 2. 特征工程
    df_features = feature_engineering(df_clean)
    
    # 3. 保存并切分 (调用修改后的函数)
    data = save_and_split(ticker, df_features, OUTPUT_DIR)
    
    if data:
        processed_datasets[ticker] = data

print(f"\n>>> 所有处理完成！请查看文件夹 '{OUTPUT_DIR}' 中的 CSV 文件。")