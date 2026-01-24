import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================= 配置区域 =================
# 1. 之前生成的指令文件
FILE_COMMANDS = "pidl_verification_commands.csv"

# 2. 实测数据文件 (你必须要手动把 Tracker 数据存成这个名字)
# 格式: 必须包含 x, y, z (或者 meas_x 等)
FILE_MEASURED = "pidl_measured_data.csv"

# 3. 绘图配置
GLOBAL_R_LIMIT = 100.0 # 画图时的圆圈半径
# ===========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase6_inference import PIDL_Inference_Engine
    import Delta_3 as nominal_kinematics
except ImportError:
    print("❌ 无法导入必要模块，请检查路径。")
    sys.exit(1)

def analyze_results():
    print("="*60)
    print("   Phase 7 (Step 3): Diagnosis & Visualization (HD)")
    print("="*60)
    
    # 1. 检查文件
    if not os.path.exists(FILE_COMMANDS):
        print(f"❌ 找不到指令文件: {FILE_COMMANDS}")
        return
    if not os.path.exists(FILE_MEASURED):
        print(f"❌ 找不到实测文件: {FILE_MEASURED}")
        print("   请确保你已经把 Tracker 的数据保存为 'pidl_measured_data.csv'")
        return
        
    df_cmds = pd.read_csv(FILE_COMMANDS)
    df_meas = pd.read_csv(FILE_MEASURED)
    
    # 对齐数据长度
    n_points = min(len(df_cmds), len(df_meas))
    df_cmds = df_cmds.iloc[:n_points]
    df_meas = df_meas.iloc[:n_points]
    
    # 2. 提取数据
    targets = df_cmds[['target_x', 'target_y', 'target_z']].values
    
    # 兼容列名
    if 'x.1' in df_meas.columns:
        measured = df_meas[['x.1', 'y.1', 'z.1']].values
    elif 'meas_x' in df_meas.columns:
        measured = df_meas[['meas_x', 'meas_y', 'meas_z']].values
    else:
        measured = df_meas.iloc[:, -3:].values
        
    # 3. 计算误差
    # (A) 补偿后误差 (PIDL Error) = Target - Measured
    errors_post = np.linalg.norm(targets - measured, axis=1)
    
    # (B) 模拟补偿前误差 (Nominal Error)
    print("🔄 正在计算 '补偿前' 模拟误差 (用于对比)...")
    
    # 加载 PIDL 模型作为"真实物理世界的代理"
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    engine = PIDL_Inference_Engine(model_path)
    arm_nominal = nominal_kinematics.arm(l=[100, 250, 35, 23.4])
    
    errors_pre = []
    for i, target in enumerate(targets):
        # 1. 用名义参数算逆解
        if arm_nominal.inverse_kinematics(target):
            theta_nominal_rad = np.radians(arm_nominal.theta)
            # 2. 扔进 PIDL 模型看它实际会跑到哪
            real_pos_sim = engine.forward_predict(theta_nominal_rad).numpy()[0]
            errors_pre.append(np.linalg.norm(target - real_pos_sim))
        else:
            errors_pre.append(np.nan)
            
    errors_pre = np.array(errors_pre)
    
    # 4. 绘图准备
    r_vals = np.sqrt(targets[:,0]**2 + targets[:,1]**2)
    data = np.column_stack((targets[:,0], targets[:,1], r_vals, errors_pre, errors_post))
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    
    # 统计
    rmse_pre = np.sqrt(np.mean(data[:,3]**2))
    rmse_post = np.sqrt(np.mean(data[:,4]**2))
    max_pre = np.max(data[:,3])
    max_post = np.max(data[:,4])
    
    GLOBAL_VMIN = 0
    GLOBAL_VMAX = max(max_pre, max_post)
    GLOBAL_YLIM = [0, GLOBAL_VMAX * 1.15]
    
    print(f"\n📊 效果统计 (基于 {len(data)} 个点):")
    print(f"   补偿前 (Nominal): RMSE={rmse_pre:.2f}mm, Max={max_pre:.2f}mm")
    print(f"   补偿后 (PIDL)   : RMSE={rmse_post:.2f}mm, Max={max_post:.2f}mm")
    
    # 5. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def plot_scatter(ax, x, y, c, title, xlabel, ylabel):
        sc = ax.scatter(x, y, c=c, cmap='jet', alpha=0.7, s=20, vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
        return sc

    # Before
    sc1 = plot_scatter(axes[0,0], data[:,2], data[:,3], data[:,3], 
                       f"BEFORE: Error vs Radius\nMax: {max_pre:.2f}mm", "Radius", "Error")
    axes[0,0].set_ylim(GLOBAL_YLIM)
    plt.colorbar(sc1, ax=axes[0,0], label='Error (mm)')
    
    sc2 = plot_scatter(axes[0,1], data[:,0], data[:,1], data[:,3], 
                       f"BEFORE: Spatial Map\nRMSE: {rmse_pre:.2f}mm", "X", "Y")
    axes[0,1].axis('equal')
    plt.colorbar(sc2, ax=axes[0,1], label='Error (mm)')

    # After
    sc3 = plot_scatter(axes[1,0], data[:,2], data[:,4], data[:,4], 
                       f"AFTER (PIDL): Error vs Radius\nMax: {max_post:.2f}mm", "Radius", "Error")
    axes[1,0].set_ylim(GLOBAL_YLIM)
    plt.colorbar(sc3, ax=axes[1,0], label='Error (mm)')

    sc4 = plot_scatter(axes[1,1], data[:,0], data[:,1], data[:,4], 
                       f"AFTER (PIDL): Spatial Map\nRMSE: {rmse_post:.2f}mm", "X", "Y")
    axes[1,1].axis('equal')
    plt.colorbar(sc4, ax=axes[1,1], label='Error (mm)')
    
    plt.tight_layout()
    save_filename = "phase7_final_diagnosis_HD.png"
    plt.savefig(save_filename, dpi=300)
    print(f"\n💾 图片已保存: {save_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_results()