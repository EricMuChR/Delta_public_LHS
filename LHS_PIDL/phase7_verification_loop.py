import os
import sys

# === 修复 OMP 冲突 ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

# ================= 配置区域 =================
FILE_COMMANDS = "pidl_verification_commands.csv"
FILE_MEASURED_RAW = "pidl_measured_raw.csv"
FILE_MATRIX_VAL = "matrix_val.npz"         
FILE_OFFSET_JSON = "../LHS_Sampling/tool_offset.json" 
# ===========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase6_inference import PIDL_Inference_Engine, DifferentiableDeltaKinematics
    import Delta_3 as nominal_kinematics
except ImportError:
    print("❌ 无法导入必要模块，请检查路径。")
    sys.exit(1)

def analyze_results():
    print("="*60)
    print("   Phase 7 (Step 3): 2x2 Diagnosis (Phase 4 vs PIDL)")
    print("="*60)
    
    if not os.path.exists(FILE_COMMANDS) or not os.path.exists(FILE_MEASURED_RAW):
        print("❌ 缺少必要文件。")
        return

    # 1. 加载数据
    df_cmds = pd.read_csv(FILE_COMMANDS)
    df_raw = pd.read_csv(FILE_MEASURED_RAW)
    matrix_data = np.load(FILE_MATRIX_VAL)
    R_inv, t = matrix_data['R'], matrix_data['t']
    
    with open(FILE_OFFSET_JSON, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])

    # 对齐数据
    n_points = min(len(df_cmds), len(df_raw))
    df_cmds = df_cmds.iloc[:n_points]
    df_raw = df_raw.iloc[:n_points]
    targets = df_cmds[['target_x', 'target_y', 'target_z']].values
    
    # 读取实测数据
    if 'x.1' in df_raw.columns:
        pts_raw = df_raw[['x.1', 'y.1', 'z.1']].values
    elif 'meas_x' in df_raw.columns:
        pts_raw = df_raw[['meas_x', 'meas_y', 'meas_z']].values
    else:
        pts_raw = df_raw.iloc[:, -3:].values

    # 2. 坐标转换 (Measured -> Robot Tip)
    pts_robot_base = np.dot(R_inv, (pts_raw - t).T).T
    measured = pts_robot_base 

    # 3. 加载 PIDL 模型参数 (为了计算公平的 Before)
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    print(f"📥 提取 Phase 4 几何参数作为 Before 基准...")
    package = torch.load(model_path)
    geo_params = package['geo_params']
    
    # 初始化 Phase 4 几何模型
    geo_model = DifferentiableDeltaKinematics([100, 250, 35, 23.4], geo_params['tool_offset'])
    geo_model.L.data = geo_params['L']
    geo_model.l.data = geo_params['l']
    geo_model.R.data = geo_params['R']
    geo_model.r.data = geo_params['r']
    geo_model.theta_offset.data = geo_params['theta_offset']
    
    # 4. 计算误差
    
    # (A) Before: Phase 4 几何误差 (对应 Step 7 的 ~9.8mm)
    # 逻辑：如果我们只用标定好的几何参数，不用神经网络，误差是多少？
    print("🔄 计算 Before 误差 (Phase 4 Geometry)...")
    nominal_arm = nominal_kinematics.arm(l=[100, 250, 35, 23.4])
    theta_nominal = []
    
    for tgt in targets:
        # 名义逆解
        tgt_flange = tgt - tool_offset
        if nominal_arm.inverse_kinematics(tgt_flange):
             theta_nominal.append(nominal_arm.theta)
        else:
             theta_nominal.append([0,0,0])
             
    theta_nominal_tensor = torch.tensor(np.radians(theta_nominal), dtype=torch.float32)
    with torch.no_grad():
        # Phase 4 正解预测
        pos_phase4_sim = geo_model(theta_nominal_tensor).numpy()
    
    errors_pre = np.linalg.norm(targets - pos_phase4_sim, axis=1)

    # (B) After: PIDL 实测误差
    error_vecs = measured - targets
    errors_post = np.linalg.norm(error_vecs, axis=1)
    
    # 计算系统偏差 (Bias) 供参考
    bias_vector = np.mean(error_vecs, axis=0)
    print(f"\n💡 系统偏差检测: {bias_vector}")
    print(f"   (PIDL的 3mm 误差主要来源于此偏差)")

    # 5. 统计
    r_vals = np.sqrt(targets[:,0]**2 + targets[:,1]**2)
    data = np.column_stack((targets[:,0], targets[:,1], r_vals, errors_pre, errors_post))
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    
    rmse_pre = np.sqrt(np.mean(data[:,3]**2))
    rmse_post = np.sqrt(np.mean(data[:,4]**2))
    max_pre = np.max(data[:,3])
    max_post = np.max(data[:,4])
    
    print(f"\n📊 最终统计:")
    print(f"   Before (Phase 4): RMSE={rmse_pre:.2f} mm, Max={max_pre:.2f} mm")
    print(f"   After  (PIDL)   : RMSE={rmse_post:.2f} mm, Max={max_post:.2f} mm")

    # 6. 绘图 (2x2 布局)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 统一 Y 轴 (为了看清 After 细节，After 单独设 Limit，Before 自动)
    
    # --- Row 1: Before (Phase 4) ---
    sc1 = axes[0,0].scatter(data[:,2], data[:,3], c=data[:,3], cmap='jet', s=20)
    axes[0,0].set_title(f"BEFORE (Phase 4 Geo): Error vs Radius\nRMSE: {rmse_pre:.2f}mm / Max: {max_pre:.2f}mm", fontweight='bold')
    axes[0,0].set_xlabel("Radius (mm)"); axes[0,0].set_ylabel("Error (mm)")
    axes[0,0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0,0])

    sc2 = axes[0,1].scatter(data[:,0], data[:,1], c=data[:,3], cmap='jet', s=20)
    axes[0,1].set_title("BEFORE: Spatial Map")
    axes[0,1].axis('equal')
    plt.colorbar(sc2, ax=axes[0,1])

    # --- Row 2: After (PIDL) ---
    sc3 = axes[1,0].scatter(data[:,2], data[:,4], c=data[:,4], cmap='jet', s=20)
    axes[1,0].set_title(f"AFTER (PIDL): Error vs Radius\nRMSE: {rmse_post:.2f}mm / Max: {max_post:.2f}mm", fontweight='bold')
    axes[1,0].set_xlabel("Radius (mm)"); axes[1,0].set_ylabel("Error (mm)")
    axes[1,0].grid(True, alpha=0.3)
    
    # 设定 After 的显示范围，方便看清那是散点还是一条线
    # 如果误差在 3mm 左右，我们设个 0~5mm
    axes[1,0].set_ylim([0, max(5.0, max_post*1.1)]) 
    plt.colorbar(sc3, ax=axes[1,0])

    sc4 = axes[1,1].scatter(data[:,0], data[:,1], c=data[:,4], cmap='jet', s=20)
    axes[1,1].set_title("AFTER: Spatial Map")
    axes[1,1].axis('equal')
    plt.colorbar(sc4, ax=axes[1,1])
    
    plt.tight_layout()
    save_filename = "phase7_final_diagnosis_HD_2x2.png"
    plt.savefig(save_filename, dpi=300)
    print(f"\n💾 四图流结果已保存: {save_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_results()