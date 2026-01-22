import pandas as pd
import numpy as np
import joblib
import os
import sys

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
MODEL_FILE = "delta_error_model.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"
OUTPUT_FILE = "points_with_compensation.csv"
ROBOT_PARAMS = [100, 250, 35, 23.4] 

# 🎯 验证配置
VALIDATION_COUNT = 500 
MAX_RADIUS_GEN = 100.0  # 生成半径 R=100mm
Z_RANGE = [-240, -190]
# =======================================

def calculate_physics_features(xyz_points):
    """ 复用 Step 5 的特征计算逻辑 """
    arm = kinematics.arm(l=ROBOT_PARAMS)
    features = []
    for point in xyz_points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # 逆解计算关节角
        arm.inverse_kinematics(tip_x_y_z=[x, y, z])
        m1, m2, m3 = arm.theta
        features.append([x, y, z, r, np.cos(theta), np.sin(theta), m1, m2, m3])
    return np.array(features)

def main():
    print("="*60)
    print(f"   Step 5 Extra: 生成 {VALIDATION_COUNT} 个全血验证点 (无钳制)")
    print("="*60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 没找到模型文件，请先跑 Step 5 训练。")
        return

    # 1. 生成 500 个随机点 (完全覆盖 R=125mm)
    print(f"1. 随机生成目标点 (R <= {MAX_RADIUS_GEN}mm)...")
    points = []
    for _ in range(VALIDATION_COUNT):
        r = MAX_RADIUS_GEN * np.sqrt(np.random.random())
        theta = np.random.random() * 2 * np.pi
        z = Z_RANGE[0] + np.random.random() * (Z_RANGE[1] - Z_RANGE[0])
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    
    test_targets = np.array(points)

    # 2. 加载模型预测
    print("2. 神经网络计算补偿量...")
    model = joblib.load(MODEL_FILE)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    
    # 计算物理特征
    test_features = calculate_physics_features(test_targets)
    inputs_scaled = scaler_x.transform(test_features)
    
    # 预测残差
    pred_residuals = scaler_y.inverse_transform(model.predict(inputs_scaled))
    
    # 3. 计算补偿指令 (Target - Residual)
    # ⚠️ 无钳制模式：完全信任神经网络
    compensated_cmds = test_targets - pred_residuals

    # 4. 保存
    df_out = pd.DataFrame({
        'orig_x': test_targets[:, 0], 'orig_y': test_targets[:, 1], 'orig_z': test_targets[:, 2],
        'comp_x': compensated_cmds[:, 0], 'comp_y': compensated_cmds[:, 1], 'comp_z': compensated_cmds[:, 2]
    }).round(4)
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 已生成文件: {OUTPUT_FILE}")
    print(f"📊 包含点数: {len(df_out)}")
    print("👉 请重新运行 Step 6，机器人将执行这 500 个验证点。")

if __name__ == "__main__":
    main()