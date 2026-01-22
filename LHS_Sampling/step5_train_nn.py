import pandas as pd
import numpy as np
import os
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ================= 配置区域 =================
FILE_CMD = "lhs_final.csv"       
FILE_MEAS = "tracker_final.csv"  

MODEL_FILE = "delta_error_model.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

FILE_VALIDATION_OUTPUT = "points_with_compensation.csv"
VALIDATION_COUNT = 50  
# ===========================================

def generate_random_test_points(n=50):
    """ 生成测试用的随机圆柱空间点 """
    z_min, z_max = -240, -190
    radius = 125
    points = []
    for _ in range(n):
        r = radius * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        z = z_min + np.random.random() * (z_max - z_min)
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(points)

def main():
    print("="*60)
    print("   Step 5: 神经网络误差补偿训练")
    print("="*60)

    if not os.path.exists(FILE_CMD) or not os.path.exists(FILE_MEAS):
        print("❌ 找不到输入文件 (Step 4)。")
        return

    # 1. 准备数据
    X = pd.read_csv(FILE_CMD)[['x', 'y', 'z']].values
    Y_meas = pd.read_csv(FILE_MEAS)[['x', 'y', 'z']].values
    Y_error = Y_meas - X  # 目标: 预测误差

    # 2. 预处理
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y_error)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # 3. 训练 MLP
    print("🧠 正在训练神经网络...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 128, 128), 
        activation='relu', solver='adam', 
        max_iter=2000, early_stopping=True, random_state=42
    )
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"✅ 训练完成。R^2 Score: {score:.4f}")

    # 4. 保存模型
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler_x, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)

    # 5. 生成验证点
    print(f"\n📝 生成验证文件: {FILE_VALIDATION_OUTPUT}")
    test_targets = generate_random_test_points(VALIDATION_COUNT)
    
    # 预测误差并反向补偿
    inputs_scaled = scaler_x.transform(test_targets)
    pred_error = scaler_y.inverse_transform(model.predict(inputs_scaled))
    compensated_cmds = test_targets - pred_error # 补偿核心: Target - Error

    df_out = pd.DataFrame({
        'orig_x': test_targets[:, 0], 'orig_y': test_targets[:, 1], 'orig_z': test_targets[:, 2],
        'comp_x': compensated_cmds[:, 0], 'comp_y': compensated_cmds[:, 1], 'comp_z': compensated_cmds[:, 2]
    }).round(4)
    df_out.to_csv(FILE_VALIDATION_OUTPUT, index=False)
    print("👉 请在移动跟踪仪后，运行 Step 5.5，然后运行 Step 6。")

if __name__ == "__main__":
    main()