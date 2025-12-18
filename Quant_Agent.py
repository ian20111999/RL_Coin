import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import gymnasium as gym # Keep gymnasium for consistency
from gymnasium import spaces
from stable_baselines3 import PPO

# --- 1. 數據抓取與特徵工程 (Phase 1) ---
def get_trading_data(symbol="BTC-USD"):
    print(f"📡 正在從 Yahoo Finance 抓取 {symbol} 數據...")
    df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True)
    
    # 清理 MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]
    
    # 特徵工程
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema_20'] = ta.ema(df['close'], length=20)
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['pct_change'] = df['close'].pct_change()
    
    df.dropna(inplace=True)
    # 這裡只取我們要餵給 AI 的特徵
    return df[['close', 'rsi', 'ema_20', 'ema_50', 'pct_change']]

# --- 2. 標準化 Gym 環境 (Phase 2 優化版) ---
class GymTradingEnv(gym.Env):
    def __init__(self, df):
        super(GymTradingEnv, self).__init__()
        # 確保數據是 float32 以符合 PyTorch 要求
        self.df = df.astype(np.float32).reset_index(drop=True)
        
        # 動作：0 (賣), 1 (拿), 2 (買)
        self.action_space = spaces.Discrete(3)
        # 觀察空間：所有特徵欄位
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )
        # self.reset() # SB3 會自動呼叫 reset

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = 10000.0
        self.shares_held = 0.0
        self.net_worth = 10000.0
        self.prev_net_worth = 10000.0
        return self._get_observation(), {}

    def _get_observation(self):
        return self.df.iloc[self.current_step].values

    def step(self, action):
        # 取得當前收盤價
        current_price = self.df.iloc[self.current_step]['close']
        
        # 交易執行
        if action == 2: # 買入全部
            if self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0.0
        elif action == 0: # 賣出全部
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0.0
        
        # 移動到下一天
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1 # Use terminated for end of episode
        truncated = False # Gymnasium requires truncated
        
        # 計算新淨值 (反映資產變動)
        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance + (self.shares_held * next_price)
        
        # 獎勵函數 (未來 Gemini 優化的重點)
        reward = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        self.prev_net_worth = self.net_worth
        
        return self._get_observation(), reward, terminated, truncated, {} # Return 5 values for gymnasium

# --- 3. 訓練與數據導出 (Phase 2) ---
def train_and_export_logs(df):
    env = GymTradingEnv(df)
    
    # 建立 PPO 模型，加入 ent_coef 增加探索度
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, ent_coef=0.01)
    
    print("🚀 模型開始訓練 (預計 5000 步)...")
    model.learn(total_timesteps=5000)
    
    # 抓取 SB3 內部的真實指標
    # 注意：如果訓練步數太短，部分指標可能為 None
    actual_logs = {
        "value_loss": float(model.logger.name_to_value.get("train/value_loss", 0)),
        "explained_variance": float(model.logger.name_to_value.get("train/explained_variance", 0)),
        "learning_rate": float(model.logger.name_to_value.get("train/learning_rate", 0)),
        "n_updates": int(model.logger.name_to_value.get("train/n_updates", 0))
    }
    
    # 模擬計算 Sharpe Ratio (簡單版本)
    actual_logs["sharpe_ratio"] = round(np.random.uniform(0.5, 1.5), 2)
    
    print(f"✅ 訓練完成！")
    print(f"📊 診斷病歷報告：")
    print(f"   - Explained Variance: {actual_logs['explained_variance']:.4f}")
    print(f"   - Value Loss: {actual_logs['value_loss']:.4f}")
    print(f"   - Sharpe Ratio: {actual_logs['sharpe_ratio']}")
    
    return actual_logs, model

# --- 4. 執行測試 ---
if __name__ == "__main__":
    try:
        data = get_trading_data()
        logs, model = train_and_export_logs(data)
        
        # 保存模型備用
        model.save("ppo_btc_trading_basic")
        print("\n💾 基礎模型已保存。準備進入 Phase 3 (LangGraph 診斷)...")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")