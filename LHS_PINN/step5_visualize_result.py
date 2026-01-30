import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Delta_Torch import DeltaKinematics
from train_pinn import DeltaPINN
import os

# 解决 OMP 报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 配置 =================
DATA_FILE = "training_data_pinn.csv"
MODEL_FILE = "pinn_model.pth"
# =======================================

def main():
    print("📊 开始模型效果评估...")
    
    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print("❌ 找不到数据文件")
        return
    df = pd.read_csv(DATA_FILE)
    X = torch.tensor(df[['theta_1', 'theta_2', 'theta_3']].values, dtype=torch.float32)
    y_true = df[['meas_x', 'meas_y', 'meas_z']].values
    
    # 2. 加载模型
    # 需要先初始化物理层，因为模型结构依赖它
    physics = DeltaKinematics()
    model = DeltaPINN(physics)
    
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        print(f"✅ 已加载模型: {MODEL_FILE}")
    else:
        print("❌ 找不到模型文件")
        return

    model.eval()
    
    # 3. 预测
    with torch.no_grad():
        # final_pos, phys_pos, nn_correction
        y_pred, y_phys, y_corr = model(X)
        y_pred = y_pred.numpy()
        y_phys = y_phys.numpy()
        y_corr = y_corr.numpy()
        
    # 4. 计算误差
    diff = y_pred - y_true
    error_dist = np.linalg.norm(diff, axis=1)
    
    mae = np.mean(error_dist)
    rmse = np.sqrt(np.mean(error_dist**2))
    max_err = np.max(error_dist)
    
    print("="*40)
    print(f"🏆 最终评估结果 (N={len(df)})")
    print(f"   平均误差 (MAE):  {mae:.4f} mm")
    print(f"   均方根误差(RMSE): {rmse:.4f} mm")
    print(f"   最大误差 (Max):  {max_err:.4f} mm")
    print("="*40)
    
    # 5. 可视化分析
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 3D 轨迹对比
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # 只画前500个点避免卡顿
    ax1.scatter(y_true[:500,0], y_true[:500,1], y_true[:500,2], c='b', s=1, label='True (Meas)')
    ax1.scatter(y_pred[:500,0], y_pred[:500,1], y_pred[:500,2], c='r', s=1, label='Pred (PINN)')
    ax1.set_title("Trajectory Comparison (Subset)")
    ax1.legend()
    
    # 子图2: 误差直方图
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(error_dist, bins=50, color='orange', alpha=0.7)
    ax2.set_title("Error Distribution (mm)")
    ax2.set_xlabel("Error (mm)")
    
    # 子图3: XY平面误差矢量图 (关键！)
    # 这能帮我们要出误差是不是有规律（比如旋转）
    ax3 = fig.add_subplot(2, 2, 3)
    # 画误差向量：起点是真实位置，终点是预测位置
    # 放大误差方便观察 (*10)
    scale = 10
    q = ax3.quiver(
        y_true[::10, 0], y_true[::10, 1], # X, Y
        diff[::10, 0], diff[::10, 1],     # dX, dY
        angles='xy', scale_units='xy', scale=0.1, color='r', alpha=0.5
    )
    ax3.set_title(f"Error Vectors XY (Arrow Length x{scale})")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    
    # 子图4: NN修正量分布
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(y_corr[:200], marker='.')
    ax4.set_title("NN Correction Terms (First 200 pts)")
    ax4.legend(['dX', 'dY', 'dZ'])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()