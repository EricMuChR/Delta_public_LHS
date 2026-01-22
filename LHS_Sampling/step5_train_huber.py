import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt

# 引入支持 Huber Loss 的模型
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ================= 导入运动学库 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import Delta_3 as kinematics
except ImportError:
    print("❌ 无法导入 Delta_3.py")
    sys.exit(1)

# ================= 配置 =================
FILE_TRAIN_CMD = "lhs_final.csv"       
FILE_TRAIN_MEAS = "tracker_final.csv"  
FILE_OFFSET_JSON = "tool_offset.json"  # <--- 新增：读取 Offset

MODEL_FILE = "delta_error_model.pkl" 
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

ROBOT_PARAMS = [100, 250, 35, 23.4] 

HUBER_ALPHA = 0.95 
HARD_OUTLIER_THRESHOLD = 15.0 
# =======================================

def calculate_physics_features(xyz_points):
    """ 计算物理特征 """
    arm = kinematics.arm(l=ROBOT_PARAMS)
    features = []
    for point in xyz_points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        try:
            arm.inverse_kinematics(tip_x_y_z=[x, y, z])
            m1, m2, m3 = arm.theta
        except:
            m1, m2, m3 = 0, 0, 0 
        features.append([x, y, z, r, np.cos(theta), np.sin(theta), m1, m2, m3])
    return np.array(features)

def main():
    print("="*60)
    print("   Step 5 (Robust): 训练抗干扰模型 (含 Offset 修正)")
    print("="*60)

    # 1. 加载 Offset (关键修复!)
    if not os.path.exists(FILE_OFFSET_JSON):
        print("❌ 找不到 tool_offset.json，请先运行 Step 4。")
        return
    
    with open(FILE_OFFSET_JSON, 'r') as f:
        offset_data = json.load(f)
        tool_offset = np.array(offset_data["tool_offset"])
    
    print(f"✅ 已加载 Tool Offset: {tool_offset}")

    # 2. 加载训练数据
    if not os.path.exists(FILE_TRAIN_CMD) or not os.path.exists(FILE_TRAIN_MEAS):
        print("❌ 找不到训练数据 (lhs_final.csv / tracker_final.csv)")
        return

    df_cmd = pd.read_csv(FILE_TRAIN_CMD)
    df_meas = pd.read_csv(FILE_TRAIN_MEAS)

    X_raw = df_cmd[['x', 'y', 'z']].values     # 机器人指令 (法兰中心)
    Y_meas = df_meas[['x', 'y', 'z']].values   # 跟踪仪实测 (工具尖端)

    # 3. 对齐数据 (扣除 Offset)
    # 真正的误差 = (实测值 - Offset) - 指令值
    Y_aligned = Y_meas - tool_offset
    residuals = Y_aligned - X_raw
    error_norms = np.linalg.norm(residuals, axis=1)
    
    print(f"📊 原始数据量: {len(X_raw)}")
    print(f"   修正 Offset 后平均误差: {np.mean(error_norms):.4f} mm")
    print(f"   修正 Offset 后最大误差: {np.max(error_norms):.4f} mm")

    # 4. 数据清洗 (Pre-cleaning)
    # 现在剩下的误差才是真正的“非线性误差”，通常在 3-10mm 之间
    clean_mask = error_norms < HARD_OUTLIER_THRESHOLD
    
    X_clean = X_raw[clean_mask]
    Y_residuals_clean = residuals[clean_mask]
    
    dropped_count = len(X_raw) - len(X_clean)
    print("-" * 40)
    print(f"🧹 数据清洗 (阈值 {HARD_OUTLIER_THRESHOLD}mm):")
    print(f"   保留数据: {len(X_clean)} (剔除 {dropped_count} 个离群点)")
    if len(X_clean) == 0:
        print("❌ 错误：数据全部被过滤！请检查 Offset 是否正确。")
        return
    print(f"   清洗后最大误差: {np.max(np.linalg.norm(Y_residuals_clean, axis=1)):.4f} mm")
    print("-" * 40)

    # 5. 特征工程
    print("fe 正在计算物理特征...")
    X_features = calculate_physics_features(X_clean)

    # 6. 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X_features)
    Y_scaled = scaler_y.fit_transform(Y_residuals_clean)

    # 划分测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.1, random_state=42)

    # 7. 训练 Huber 模型
    print(f"🚀 开始训练 (Loss='huber', Alpha={HUBER_ALPHA})...")
    
    gbr = GradientBoostingRegressor(
        loss='huber',             
        alpha=HUBER_ALPHA,        
        n_estimators=500,         
        learning_rate=0.1,
        max_depth=5,              
        random_state=42
    )
    
    model = MultiOutputRegressor(gbr)
    model.fit(X_train, y_train)
    
    # 8. 评估
    score = model.score(X_test, y_test)
    print(f"✅ 训练完成。R² Score: {score:.4f}")
    
    y_pred_test = model.predict(X_test)
    y_pred_real = scaler_y.inverse_transform(y_pred_test)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    test_errors = np.linalg.norm(y_test_real - y_pred_real, axis=1)
    print(f"   测试集 RMSE: {np.sqrt(np.mean(test_errors**2)):.4f} mm")

    # 9. 保存模型
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler_x, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)
    
    print(f"\n💾 模型已保存至: {MODEL_FILE}")
    print("👉 请继续运行 Step 6 进行验证。")

if __name__ == "__main__":
    main()