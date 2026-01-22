import numpy as np
import matplotlib.pyplot as plt
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
ROBOT_PARAMS = [100, 250, 35, 23.4]
# =======================================

def calculate_physics_features(xyz_points):
    """ 复用 Step 5 的特征计算逻辑 """
    arm = kinematics.arm(l=ROBOT_PARAMS)
    features = []
    for point in xyz_points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        arm.inverse_kinematics(tip_x_y_z=[x, y, z])
        m1, m2, m3 = arm.theta
        features.append([x, y, z, r, np.cos(theta), np.sin(theta), m1, m2, m3])
    return np.array(features)

def main():
    print("="*60)
    print("   Step 5 Debug: 神经网络'脑图'可视化")
    print("   (检查模型是否学到了连续的物理规律)")
    print("="*60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 模型文件不存在，请先运行 Step 5。")
        return

    # 1. 加载模型
    model = joblib.load(MODEL_FILE)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)

    # 2. 生成网格测试点 (Z = -220mm 切片)
    Z_FIXED = -220
    grid_size = 20
    x = np.linspace(-150, 150, grid_size)
    y = np.linspace(-150, 150, grid_size)
    X, Y = np.meshgrid(x, y)
    
    test_points = []
    valid_indices = [] # 记录在工作空间圆内的点
    
    for i in range(grid_size):
        for j in range(grid_size):
            px, py = X[i, j], Y[i, j]
            # 只看半径 140mm 以内的点
            if px**2 + py**2 <= 140**2:
                test_points.append([px, py, Z_FIXED])
                valid_indices.append((i, j))

    test_points = np.array(test_points)
    
    # 3. 预测误差
    print(f"🔍 正在扫描空间点: {len(test_points)} 个...")
    features = calculate_physics_features(test_points)
    inputs_scaled = scaler_x.transform(features)
    pred_residuals_scaled = model.predict(inputs_scaled)
    pred_residuals = scaler_y.inverse_transform(pred_residuals_scaled)

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 提取预测出的误差向量 (放大 10 倍以便观察)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M = np.zeros_like(X) # 误差模长 (颜色)

    for k, (i, j) in enumerate(valid_indices):
        err_x, err_y, _ = pred_residuals[k]
        U[i, j] = err_x
        V[i, j] = err_y
        M[i, j] = np.sqrt(err_x**2 + err_y**2)

    # 画圆盘边界
    circle = plt.Circle((0, 0), 140, color='gray', fill=False, linestyle='--')
    ax.add_patch(circle)

    # 画矢量场 (Quiver Plot)
    # 箭头方向 = 模型认为该处的误差方向
    # 箭头颜色 = 误差大小
    q = ax.quiver(X, Y, U, V, M, cmap='jet', scale=20, width=0.003)
    
    plt.colorbar(q, label='Predicted Error Magnitude (mm)')
    plt.title(f'Neural Network Learned Error Field (Z={Z_FIXED}mm)\nSmooth Arrows = Physics; Random Arrows = Overfitting')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True, alpha=0.3)
    
    print("\n📊 绘图完成！")
    print("👉 请观察弹出的图像：")
    print("   1. 【好】如果箭头像水流一样平滑变化，说明模型学到了物理形变。")
    print("   2. 【坏】如果箭头杂乱无章、方向随机突变，说明模型过拟合（虚假收敛）。")
    
    plt.show()

if __name__ == "__main__":
    main()