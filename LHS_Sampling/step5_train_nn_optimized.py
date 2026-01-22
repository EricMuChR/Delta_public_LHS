import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ================= 跨文件夹导入运动学库 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import Delta_3 as kinematics 
except ImportError:
    print("❌ 无法导入 Delta_3.py，请检查文件路径")
    sys.exit(1)
# ======================================================

# ================= 配置区域 =================
FILE_CMD = "lhs_final.csv"       
FILE_MEAS = "tracker_final.csv"  

MODEL_FILE = "delta_error_model.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"

FILE_VALIDATION_OUTPUT = "points_with_compensation.csv"
VALIDATION_COUNT = 50 

# 机器人几何参数 [l1, l2, R, r]
ROBOT_PARAMS = [100, 250, 35, 23.4] 
# ===========================================

def calculate_physics_features(xyz_points):
    """ 计算物理特征: [x, y, z, r, cos, sin, m1, m2, m3] """
    arm = kinematics.arm(l=ROBOT_PARAMS)
    features = []
    
    print("⚙️ 正在计算物理特征 (逆运动学)...")
    for point in xyz_points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # 逆解计算关节角
        arm.inverse_kinematics(tip_x_y_z=[x, y, z])
        m1, m2, m3 = arm.theta 
        
        features.append([x, y, z, r, np.cos(theta), np.sin(theta), m1, m2, m3])
        
    return np.array(features)

def generate_random_test_points(n=50):
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
    print("   Step 5 (Final): 去偏 + 物理增强神经网络训练")
    print("="*60)

    if not os.path.exists(FILE_CMD) or not os.path.exists(FILE_MEAS):
        print("❌ 找不到输入文件 (Step 4)。")
        return

    # 1. 准备原始数据
    df_cmd = pd.read_csv(FILE_CMD)
    df_meas = pd.read_csv(FILE_MEAS)
    
    X_raw = df_cmd[['x', 'y', 'z']].values
    Y_meas = df_meas[['x', 'y', 'z']].values
    
    # === 核心修改：分离 Offset ===
    # 计算原始误差
    Y_error_raw = Y_meas - X_raw
    
    # 计算全局 Offset (系统偏差)
    systematic_offset = np.mean(Y_error_raw, axis=0)
    print(f"📏 检测到系统 Offset: {np.round(systematic_offset, 4)}")
    
    # 计算残差 (去偏后的误差)
    # 这才是神经网络真正需要学习的"非线性部分"
    Y_residuals = Y_error_raw - systematic_offset

    # 2. 数据清洗 (基于残差进行清洗)
    res_norms = np.linalg.norm(Y_residuals, axis=1)
    mean_res = np.mean(res_norms)
    std_res = np.std(res_norms)
    
    # 阈值策略：平均波动 + 3倍标准差 (或者硬阈值 10mm)
    # 因为已经去掉了 Offset，这里的波动应该很小 (比如 0.x mm)
    threshold = min(mean_res + 3 * std_res, 10.0) 
    
    mask = res_norms < threshold
    n_dropped = len(X_raw) - np.sum(mask)
    
    print(f"🧹 数据清洗: 剔除 {n_dropped} 个离群点 (残差阈值 {threshold:.4f} mm)")
    
    if np.sum(mask) == 0:
        print("❌ 错误: 所有数据都被剔除了，请检查阈值逻辑。")
        return
        
    X_clean = X_raw[mask]
    # 训练目标：只训练残差！让 NN 专注拟合非线性误差
    Y_target_clean = Y_residuals[mask]
    
    # 3. 特征工程
    X_enhanced = calculate_physics_features(X_clean)
    print(f"📊 最终训练样本数: {len(X_clean)}")

    # 4. 预处理
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X_enhanced)
    Y_scaled = scaler_y.fit_transform(Y_target_clean)

    # 5. 训练 MLP (加深网络以拟合精细残差)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.15, random_state=42)

    print("\n🧠 正在训练神经网络 (Residual Learning)...")
    model = MLPRegressor(
        hidden_layer_sizes=(512, 512, 512), # 深层网络
        activation='tanh',
        solver='adam',
        alpha=1e-5,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=5000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # 6. 评估
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    score = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n" + "-"*40)
    print(f"🏆 模型评估 (针对残差):")
    print(f"   R^2 Score : {score:.5f}")
    print(f"   RMSE      : {rmse:.5f} mm")
    print("-" * 40)

    # 7. 保存模型
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler_x, SCALER_X_FILE)
    joblib.dump(scaler_y, SCALER_Y_FILE)

    # 8. 生成验证点
    print(f"\n📝 生成验证文件: {FILE_VALIDATION_OUTPUT}")
    test_targets = generate_random_test_points(VALIDATION_COUNT)
    
    # 预测流程:
    # 1. 计算特征
    # 2. 预测残差
    # 3. 补偿指令 = 目标 - 预测残差 (注意：这里不需要减 Offset，因为 Step 7 会自动处理 Offset)
    test_features = calculate_physics_features(test_targets)
    inputs_scaled = scaler_x.transform(test_features)
    
    pred_residuals = scaler_y.inverse_transform(model.predict(inputs_scaled))
    
    # 补偿策略:
    # 我们希望 Robot 走到 Target (法兰中心)
    # 既然误差是 Residuals，那我们就反向修补 Residuals
    compensated_cmds = test_targets - pred_residuals 

    df_out = pd.DataFrame({
        'orig_x': test_targets[:, 0], 'orig_y': test_targets[:, 1], 'orig_z': test_targets[:, 2],
        'comp_x': compensated_cmds[:, 0], 'comp_y': compensated_cmds[:, 1], 'comp_z': compensated_cmds[:, 2]
    }).round(4)
    df_out.to_csv(FILE_VALIDATION_OUTPUT, index=False)
    print("✅ 优化完成。此模型已学会'非线性残差'。")

if __name__ == "__main__":
    main()