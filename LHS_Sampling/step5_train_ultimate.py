import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt

# 回归本源：使用神经网络 (MLP) 以追求极致精度
from sklearn.neural_network import MLPRegressor
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
FILE_OFFSET_JSON = "tool_offset.json" 

MODEL_FILE = "delta_error_model.pkl" 
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

ROBOT_PARAMS = [100, 250, 35, 23.4] 

# 阈值：超过这个值的点被认为是物理坏点，直接剔除
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
    print("   Step 5 (Ultimate): 神经网络 + 强力清洗")
    print("="*60)

    # 1. 加载 Offset
    if not os.path.exists(FILE_OFFSET_JSON):
        print("❌ 找不到 tool_offset.json")
        return
    with open(FILE_OFFSET_JSON, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])
    print(f"✅ 已加载 Offset: {tool_offset}")

    # 2. 加载数据
    if not os.path.exists(FILE_TRAIN_CMD):
        print("❌ 找不到训练数据")
        return
    df_cmd = pd.read_csv(FILE_TRAIN_CMD)
    df_meas = pd.read_csv(FILE_TRAIN_MEAS)

    X_raw = df_cmd[['x', 'y', 'z']].values     
    Y_meas = df_meas[['x', 'y', 'z']].values   

    # 3. 对齐数据 (扣除 Offset)
    Y_aligned = Y_meas - tool_offset
    residuals = Y_aligned - X_raw
    error_norms = np.linalg.norm(residuals, axis=1)
    
    print(f"📊 原始数据: {len(X_raw)}")
    print(f"   平均误差: {np.mean(error_norms):.4f} mm")

    # 4. 数据清洗 (剔除那 90 个害群之马)
    clean_mask = error_norms < HARD_OUTLIER_THRESHOLD
    X_clean = X_raw[clean_mask]
    Y_residuals_clean = residuals[clean_mask]
    
    print("-" * 40)
    print(f"🧹 数据清洗 (阈值 {HARD_OUTLIER_THRESHOLD}mm):")
    print(f"   剔除点数: {len(X_raw) - len(X_clean)}")
    print(f"   保留点数: {len(X_clean)}")
    print("-" * 40)

    # 5. 特征工程 & 标准化
    print("fe 计算特征...")
    X_features = calculate_physics_features(X_clean)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_features)
    Y_scaled = scaler_y.fit_transform(Y_residuals_clean)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.1, random_state=42)

    # 6. 训练神经网络 (MLP)
    # 使用较深的网络结构来捕捉非线性
    print(f"🚀 开始训练神经网络 (Hidden Layers: 128x128x128)...")
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128, 128), # 加深网络
        activation='relu',
        solver='adam',
        alpha=1e-4,              # L2正则化，防止过拟合
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=2000,           # 给足迭代次数
        early_stopping=True,     # 如果不动了就提前停
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    
    # 神经网络本身支持多输出，不需要 MultiOutputRegressor 包装
    mlp.fit(X_train, y_train)
    
    # 7. 评估
    score = mlp.score(X_test, y_test)
    print(f"✅ 训练完成。R² Score: {score:.4f}")
    
    y_pred_test = mlp.predict(X_test)
    y_pred_real = scaler_y.inverse_transform(y_pred_test)
    y_test_real = scaler_y.inverse_transform(y_test)
    
    rmse = np.sqrt(np.mean(np.linalg.norm(y_test_real - y_pred_real, axis=1)**2))
    print(f"   测试集 RMSE: {rmse:.4f} mm")

    # 8. 保存
    joblib.dump(mlp, MODEL_FILE)
    joblib.dump(scaler_x, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)
    
    print(f"\n💾 最终模型已保存: {MODEL_FILE}")

if __name__ == "__main__":
    main()