import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 匯入你的實驗室
from Trading_lab import GymTradingEnv, get_trading_data

# 全局變數：快取數據，避免每次 Trial 都重新下載
CACHED_DF = None

def get_data():
    global CACHED_DF
    if CACHED_DF is None:
        CACHED_DF = get_trading_data() # 下載並快取
    return CACHED_DF

def optimize_reward_logic(net_worth, prev_net_worth, action, shares_held, params):
    """
    這是一個「參數化」的獎勵函數。
    Optuna 會傳入 params 字典，嘗試不同的數值組合。
    """
    # 1. 基礎收益 (放大倍率由 Optuna 決定)
    profit = (net_worth - prev_net_worth) / prev_net_worth
    reward = profit * params['profit_multiplier']
    
    # 2. 持倉獎勵 (鼓勵或懲罰持倉)
    if action == 1: # Hold
        reward += params['hold_reward']
        
    # 3. 回撤懲罰 (如果淨值下跌，給予額外懲罰)
    if net_worth < prev_net_worth:
        reward -= params['drawdown_penalty']
        
    return reward

def objective(trial):
    """
    Optuna 的核心迴圈：
    1. 建議參數 -> 2. 訓練模型 -> 3. 回傳分數
    """
    
    # --- A. 定義要優化的超參數空間 ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
        'gamma': trial.suggest_categorical('gamma', [0.9, 0.95, 0.99]),
        # 獎勵函數的參數
        'profit_multiplier': trial.suggest_float('profit_multiplier', 10.0, 200.0),
        'hold_reward': trial.suggest_float('hold_reward', -0.01, 0.01), # 可以是負的(懲罰)或正的(獎勵)
        'drawdown_penalty': trial.suggest_float('drawdown_penalty', 0.0, 0.5)
    }
    
    # --- B. 建立帶有動態獎勵的環境 ---
    # 使用 lambda 函式將 params 注入到我們的獎勵邏輯中
    df = get_data()
    
    # 我們定義一個 wrapper 讓環境能呼叫帶參數的獎勵函數
    def current_reward_wrapper(nw, pnw, act, sh):
        return optimize_reward_logic(nw, pnw, act, sh, params)
        
    env = GymTradingEnv(df, reward_func=current_reward_wrapper)
    
    # --- C. 訓練模型 (快速試錯模式) ---
    # 這裡步數設少一點 (例如 5000-10000)，目的是快速篩選，不用練到完美
    model = PPO("MlpPolicy", env, 
                verbose=0, 
                learning_rate=params['learning_rate'],
                ent_coef=params['ent_coef'],
                gamma=params['gamma'])
    
    try:
        model.learn(total_timesteps=5000)
    except Exception as e:
        print(f"Trial failed: {e}")
        return -float('inf') # 訓練失敗給極低分

    # --- D. 評估模型表現 ---
    # 我們不看訓練時的 Reward (因為那被我們改過)，我們看「最終淨值」或「夏普比率」
    # 這裡簡單跑一次完整的 episode 來算最終淨值
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    # 重新跑一次回測確認成效
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
    final_net_worth = env.net_worth
    
    # Optuna 會嘗試最大化這個回傳值
    return final_net_worth

if __name__ == "__main__":
    print("🚀 啟動 Optuna 量化參數搜尋引擎...")
    
    # 建立 Study，目標是最大化 (maximize) 最終淨值
    study = optuna.create_study(direction="maximize")
    
    # 開始跑 20 輪實驗 (你可以隨意增加)
    # n_jobs=1 代表單線程跑 (比較穩定)，如果你電腦強可以設 -1 (全速運轉)
    study.optimize(objective, n_trials=20, n_jobs=1)
    
    print("\n" + "="*50)
    print("🏆 搜尋完成！最佳參數組合：")
    print(study.best_params)
    print(f"💰 對應的最終淨值: {study.best_value:.2f}")
    print("="*50)
    
    # 你可以把最佳參數存起來，之後用來訓練最終模型