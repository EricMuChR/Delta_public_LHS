import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

# ================= 配置区域 =================
# 1. 之前的指令文件 (含 Target)
FILE_COMMANDS = "pidl_verification_commands.csv"

# 2. 实测原始数据 (由激光跟踪仪直接导出，未转换)
FILE_MEASURED_RAW = "pidl_measured_raw.csv"

# 3. 坐标转换文件
FILE_MATRIX_VAL = "matrix_val.npz"         # 由 phase7_realign_tracker.py 生成
FILE_OFFSET_JSON = "../LHS_Sampling/tool_offset.json" # 工具偏置

# 4. 绘图配置
GLOBAL_R_LIMIT = 100.0 
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
    print("   Phase 7 (Step 3): Diagnosis & Visualization (With Re-Alignment)")
    print("="*60)
    
    # 1. 检查文件完整性
    files_to_check = [FILE_COMMANDS, FILE_MEASURED_RAW, FILE_MATRIX_VAL, FILE_OFFSET_JSON]
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"❌ 缺少文件: {f}")
            if f == FILE_MATRIX_VAL:
                print("   -> 请先运行 phase7_realign_tracker.py 生成新矩阵。")
            return

    # 2. 加载数据与矩阵
    print("🔄 正在加载数据与坐标变换矩阵...")
    df_cmds = pd.read_csv(FILE_COMMANDS)
    df_raw = pd.read_csv(FILE_MEASURED_RAW)
    
    matrix_data = np.load(FILE_MATRIX_VAL)
    R_inv, t = matrix_data['R'], matrix_data['t']
    
    with open(FILE_OFFSET_JSON, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])
    print(f"   Tool Offset loaded: {tool_offset}")

    # 3. 数据对齐与坐标转换
    n_points = min(len(df_cmds), len(df_raw))
    df_cmds = df_cmds.iloc[:n_points]
    df_raw = df_raw.iloc[:n_points]
    
    targets = df_cmds[['target_x', 'target_y', 'target_z']].values
    
    # 读取原始测量数据 (兼容 x/y/z 或 x.1/y.1/z.1 或 meas_x...)
    if 'x.1' in df_raw.columns:
        pts_raw = df_raw[['x.1', 'y.1', 'z.1']].values
    elif 'meas_x' in df_raw.columns:
        pts_raw = df_raw[['meas_x', 'meas_y', 'meas_z']].values
    else:
        # 默认取最后三列，或者是前三列，取决于你的导出格式
        # 建议检查这里，通常跟踪仪导出的是 x,y,z
        if 'x' in df_raw.columns:
            pts_raw = df_raw[['x', 'y', 'z']].values
        else:
            pts_raw = df_raw.iloc[:, 0:3].values

    # === 核心：坐标系转换 (Tracker -> Robot) ===
    # 公式: P_robot = R_inv * (P_tracker - t)
    print("⚡ 执行坐标系转换 (Tracker -> Robot Base)...")
    pts_robot_base = np.dot(R_inv, (pts_raw - t).T).T
    
    # 减去工具偏置，得到法兰中心坐标 (与 Target 对比)
    measured = pts_robot_base - tool_offset

    # 4. 计算误差
    # (A) 补偿后误差 (PIDL)
    errors_post = np.linalg.norm(targets - measured, axis=1)
    
    # (B) 模拟补偿前误差 (Nominal)
    print("🔄 正在计算 '补偿前' 模拟误差...")
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    engine = PIDL_Inference_Engine(model_path)
    arm_nominal = nominal_kinematics.arm(l=[100, 250, 35, 23.4])
    
    errors_pre = []
    for i, target in enumerate(targets):
        if arm_nominal.inverse_kinematics(target):
            theta_rad = np.radians(arm_nominal.theta)
            # 预测名义控制下的实际位置
            real_pos_sim = engine.forward_predict(theta_rad).numpy()[0]
            errors_pre.append(np.linalg.norm(target - real_pos_sim))
        else:
            errors_pre.append(np.nan)
    errors_pre = np.array(errors_pre)
    
    # 5. 统计与绘图
    r_vals = np.sqrt(targets[:,0]**2 + targets[:,1]**2)
    data = np.column_stack((targets[:,0], targets[:,1], r_vals, errors_pre, errors_post))
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    
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
    
    # 绘图逻辑
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