import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

# ================= 配置 =================
FILE_TARGETS = "points_with_compensation.csv"
FILE_MEASURED_RAW = "validation_500.csv" 
FILE_MATRIX_VAL = "matrix_val.npz"
FILE_OFFSET_JSON = "tool_offset.json"
MATCH_THRESHOLD = 20.0 
# =======================================

def main():
    print("="*60)
    print("   Step 7 Diagnosis: 补偿前后效果对比 (全统一标尺)")
    print("="*60)

    # 1. 加载配置
    if not os.path.exists(FILE_OFFSET_JSON) or not os.path.exists(FILE_MATRIX_VAL):
        print("❌ 缺少必要配置文件。")
        return
        
    with open(FILE_OFFSET_JSON, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])
    
    matrix_data = np.load(FILE_MATRIX_VAL)
    R_inv, t = matrix_data['R'], matrix_data['t']

    # 2. 加载数据
    df_targets = pd.read_csv(FILE_TARGETS)
    if 'orig_x' in df_targets.columns:
        pts_target = df_targets[['orig_x', 'orig_y', 'orig_z']].values
        pts_comp_cmd = df_targets[['comp_x', 'comp_y', 'comp_z']].values
    else:
        print("❌ 目标文件格式不对，缺少 orig_x/comp_x 列。")
        return

    if not os.path.exists(FILE_MEASURED_RAW):
        print(f"❌ 找不到实测数据: {FILE_MEASURED_RAW}")
        return
    df_meas = pd.read_csv(FILE_MEASURED_RAW)
    try:
        pts_meas_raw = df_meas[['x', 'y', 'z']].values
    except KeyError:
        pts_meas_raw = df_meas.iloc[:, -3:].values

    # 3. 坐标转换
    pts_meas_robot = np.dot(R_inv, (pts_meas_raw - t).T).T
    pts_meas_flange = pts_meas_robot - tool_offset

    # 4. 双指针匹配
    plot_data = [] 
    cmd_cursor = 0
    meas_cursor = 0
    
    while meas_cursor < len(pts_meas_flange) and cmd_cursor < len(pts_target):
        p_meas = pts_meas_flange[meas_cursor]
        p_orig = pts_target[cmd_cursor]
        p_comp_cmd = pts_comp_cmd[cmd_cursor]
        
        dist_post = np.linalg.norm(p_meas - p_orig)
        
        if dist_post < MATCH_THRESHOLD:
            err_vec_pre = p_orig - p_comp_cmd
            dist_pre = np.linalg.norm(err_vec_pre)
            
            r = np.sqrt(p_orig[0]**2 + p_orig[1]**2)
            plot_data.append([p_orig[0], p_orig[1], r, dist_pre, dist_post])
            meas_cursor += 1
            cmd_cursor += 1
        else:
            if meas_cursor + 1 < len(pts_meas_flange):
                dist_next = np.linalg.norm(pts_meas_flange[meas_cursor+1] - p_orig)
                if dist_next < dist_post:
                    meas_cursor += 1
                    continue
            cmd_cursor += 1

    if not plot_data:
        print("❌ 没有有效数据点。")
        return
        
    data = np.array(plot_data) 
    
    # 5. 统计与标尺计算
    rmse_pre = np.sqrt(np.mean(data[:,3]**2))
    rmse_post = np.sqrt(np.mean(data[:,4]**2))
    max_pre = np.max(data[:,3])
    max_post = np.max(data[:,4])
    
    # === 关键：统一所有标尺 ===
    # 1. 颜色标尺 (Colorbar)
    GLOBAL_VMIN = 0
    GLOBAL_VMAX = max(max_pre, max_post)
    
    # 2. Y轴刻度标尺 (Y-Limit)
    # 取最大误差的 1.1 倍，让最高点不顶格
    GLOBAL_YLIM = [0, GLOBAL_VMAX * 1.1]

    print(f"\n📊 效果统计 (基于 {len(data)} 个点):")
    print(f"   补偿前: RMSE={rmse_pre:.2f}mm, Max={max_pre:.2f}mm")
    print(f"   补偿后: RMSE={rmse_post:.2f}mm, Max={max_post:.2f}mm")
    print(f"   📏 统一 Y 轴范围: 0 ~ {GLOBAL_YLIM[1]:.2f} mm")

    # 6. 绘图 (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    def plot_scatter(ax, x, y, c, title, xlabel, ylabel, vmin, vmax, vline=None, ylim=None):
        sc = ax.scatter(x, y, c=c, cmap='jet', alpha=0.7, s=20, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if vline:
            ax.axvline(vline, color='r', linestyle='--', label='Safe Zone (110mm)')
            ax.legend()
        # 强制设置 Y 轴范围
        if ylim:
            ax.set_ylim(ylim)
        return sc

    # --- 第一排：补偿前 (Before) ---
    # 左图：误差 vs 半径 (设置 ylim)
    sc1 = plot_scatter(axes[0,0], data[:,2], data[:,3], data[:,3], 
                       f"BEFORE: Error vs Radius", 
                       "Radius (mm)", "Error (mm)", 
                       GLOBAL_VMIN, GLOBAL_VMAX, vline=110, ylim=GLOBAL_YLIM)
    plt.colorbar(sc1, ax=axes[0,0], label='Error (mm)')
    
    # 右图：空间分布
    sc2 = plot_scatter(axes[0,1], data[:,0], data[:,1], data[:,3], 
                       f"BEFORE: Spatial Distribution\nRMSE: {rmse_pre:.2f} mm", 
                       "X (mm)", "Y (mm)",
                       GLOBAL_VMIN, GLOBAL_VMAX)
    circle = plt.Circle((0, 0), 125, color='gray', fill=False, linestyle='--')
    axes[0,1].add_patch(circle)
    axes[0,1].axis('equal')
    plt.colorbar(sc2, ax=axes[0,1], label='Error (mm)')

    # --- 第二排：补偿后 (After) ---
    # 左图：误差 vs 半径 (设置同样的 ylim) -> 你会看到点都趴在下面
    sc3 = plot_scatter(axes[1,0], data[:,2], data[:,4], data[:,4], 
                       f"AFTER: Error vs Radius", 
                       "Radius (mm)", "Error (mm)",
                       GLOBAL_VMIN, GLOBAL_VMAX, vline=110, ylim=GLOBAL_YLIM)
    plt.colorbar(sc3, ax=axes[1,0], label='Error (mm)')

    # 右图：空间分布
    sc4 = plot_scatter(axes[1,1], data[:,0], data[:,1], data[:,4], 
                       f"AFTER: Spatial Distribution\nRMSE: {rmse_post:.2f} mm", 
                       "X (mm)", "Y (mm)",
                       GLOBAL_VMIN, GLOBAL_VMAX)
    circle = plt.Circle((0, 0), 125, color='gray', fill=False, linestyle='--')
    axes[1,1].add_patch(circle)
    axes[1,1].axis('equal')
    plt.colorbar(sc4, ax=axes[1,1], label='Error (mm)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()