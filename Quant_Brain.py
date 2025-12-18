import os
import json
import re
import asyncio
import operator
import time
from typing import Annotated, Sequence, TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 匯入 Phase 1 & 2
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Standardize import to the gymnasium-compliant version
from Trading_lab import train_and_export_logs, get_trading_data

load_dotenv()

# --- 1. 定義狀態 ---
class AgentState(TypedDict):
    iteration: int
    train_logs: Dict
    diagnostic_report: str
    generated_code: str
    is_satisfied: bool
    history: Annotated[List[str], operator.add]

# --- 2. 診斷節點 (Pathologist) ---
async def ai_pathologist_node(state: AgentState):
    iter_num = state['iteration']
    print(f"\n🧐 [Node: Pathologist] 第 {iter_num} 輪診斷中...")
    
    # 使用 Groq (速度快)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=0)
    
    prompt = f"""
    你是 RL 量化交易診斷專家。當前模型指標：
    - Explained Variance: {state['train_logs']['explained_variance']} (目標 > 0.5)
    - Value Loss: {state['train_logs']['value_loss']}
    - Sharpe Ratio: {state['train_logs']['sharpe_ratio']}
    
    如果 Explained Variance 很低 (<0.1)，代表獎勵函數沒學到東西。
    請判斷是否滿意 (is_satisfied)。
    回傳純 JSON: {{ "diagnosis": "簡短分析", "is_satisfied": true/false }}
    """
    
    result = {"diagnosis": "API 或解析錯誤", "is_satisfied": False}
    max_retries = 5
    base_wait_time = 60  # 基礎等待時間 (秒)

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            res_text = response.content.replace("```json", "").replace("```", "").strip()
            
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            result = json.loads(match.group()) if match else json.loads(res_text)
            break # 成功則跳出迴圈
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = base_wait_time * (attempt + 1)
                print(f"⚠️ API Rate Limit (429). 休息 {wait_time} 秒後重試 ({attempt+1}/{max_retries})...")
                await asyncio.sleep(wait_time)
            else:
                print(f"⚠️ 診斷失敗: {e}，判定為不滿意")
                result = {"diagnosis": f"錯誤: {str(e)}", "is_satisfied": False}
                break
        
    print(f"   📊 診斷: {result['diagnosis']} (Pass: {result['is_satisfied']})")
    
    return {
        "diagnostic_report": result["diagnosis"],
        "is_satisfied": result["is_satisfied"],
        "history": [f"Iter {iter_num}: {result['diagnosis']}"]
    }

# --- 3. 代碼生成節點 (Refiner) ---
async def strategy_refiner_node(state: AgentState):
    print("\n💡 [Node: Refiner] 正在撰寫 Python 獎勵函數...")
    
    # 使用 Groq (能力強)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_retries=0)
    
    prompt = f"""
    診斷：{state['diagnostic_report']}
    
    請重寫 Python 獎勵函數 `calculate_reward` 來改善模型。
    函數簽名：def calculate_reward(net_worth, prev_net_worth, action, shares_held):
    
    邏輯建議：
    1. **獎勵縮放 (Scaling)**: 原始收益率數值太小 (e.g., 0.001)，請將收益率 * 100 或 * 1000，讓模型更容易學習。
    2. **動作獎勵**: 
       - 如果 action==1 (Hold) 且趨勢向上，給予微小獎勵。
       - 如果 action==2 (Buy) 且隨後 net_worth 上升，給予大獎勵。
    3. **風險懲罰**: 如果 net_worth < prev_net_worth，給予更大的懲罰 (e.g., 損失 * 1.5)。
    4. **語法安全**: 
       - 不要引用外部未定義變數。
       - 確保除法不為零。
    
    只回傳 Python 代碼區塊 (```python ... ```)。
    """
    
    code = "def calculate_reward(net_worth, prev_net_worth, action, shares_held):\n    return (net_worth - prev_net_worth) / prev_net_worth"
    max_retries = 5
    base_wait_time = 60

    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            code_match = re.search(r"```python(.*?)```", response.content, re.DOTALL)
            code = code_match.group(1).strip() if code_match else response.content.strip()
            break
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait_time = base_wait_time * (attempt + 1)
                print(f"⚠️ API Rate Limit (429). 休息 {wait_time} 秒後重試 ({attempt+1}/{max_retries})...")
                await asyncio.sleep(wait_time)
            else:
                print(f"⚠️ 生成失敗: {e}")
                break
    
    print(f"   💻 AI 已生成新策略 (長度: {len(code)} chars)")
    return {
        "generated_code": code,
        "iteration": state["iteration"] + 1,
        "history": [f"Iter {state['iteration']}: Code Generated"]
    }

# --- 4. 執行訓練節點 (Executor) ---
def execution_node(state: AgentState):
    print("\n⚙️ [Node: Executor] 注入代碼並重啟訓練...")
    
    # 動態執行代碼
    local_scope = {}
    try:
        exec(state["generated_code"], globals(), local_scope)
        reward_func = local_scope.get("calculate_reward")
        if not reward_func: raise ValueError("函數名稱錯誤")
    except Exception as e:
        print(f"❌ 代碼注入失敗: {e}，使用預設訓練")
        reward_func = None
        
    # 呼叫 Trading_Lab
    if 'data_cache' not in globals():
        globals()['data_cache'] = get_trading_data()
    
    logs, _ = train_and_export_logs(globals()['data_cache'], custom_reward_func=reward_func)
    
    return {"train_logs": logs}

# --- 5. 構建圖形 ---
def build_quant_brain():
    workflow = StateGraph(AgentState)
    workflow.add_node("pathologist", ai_pathologist_node)
    workflow.add_node("refiner", strategy_refiner_node)
    workflow.add_node("executor", execution_node)
    
    workflow.set_entry_point("pathologist")
    
    def router(state):
        if state["is_satisfied"] or state["iteration"] > 3: # 最多跑 3 輪
            return END
        return "refiner"
    
    workflow.add_conditional_edges("pathologist", router)
    workflow.add_edge("refiner", "executor")
    workflow.add_edge("executor", "pathologist")
    
    return workflow.compile()

if __name__ == "__main__":
    # 初始狀態
    initial_state = {
        "iteration": 1,
        "train_logs": {"explained_variance": -1.0, "value_loss": 1.0, "sharpe_ratio": 0.0},
        "diagnostic_report": "",
        "generated_code": "",
        "is_satisfied": False,
        "history": []
    }
    
    print("🚀 啟動 RL 量化交易 Agent...")
    app = build_quant_brain()
    asyncio.run(app.ainvoke(initial_state))
