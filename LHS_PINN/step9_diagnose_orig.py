import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys

# ================= 配置 =================
FILE_TARGETS = "points_with_compensation_cmp_origin.csv" # 包含 orig_x, orig_y, orig_z
FILE_MEASURED_RAW = "original_500.csv"        # 跟踪仪导出的原始数据
FILE_MATRIX_VAL = "matrix_val.npz"            # Step 5.5 生成
FILE_OFFSET_JSON = "tool_offset.json"         # Step 4 生成

# 画图配置
# GLOBAL_VMAX 不再硬编码，改为动态计算
GLOBAL_VMIN = 0.0
# =======================================

def plot_scatter(ax, x, y, c, title, xlabel, ylabel, vmin, vmax, vline=None, ylim=None):
    # edgecolors='none' 去掉点的描边，让颜色更纯粹
    sc = ax.scatter(x, y, c=c, cmap='jet', s=20, vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    if vline: ax.axvline(vline, color='k', linestyle='--', alpha=0.5)
    if ylim: ax.set_ylim(ylim)
    return sc

def main():
    print("="*60)
    print("   Step 9 Diagnosis: 原始数据误差报告")
    print("="*60)

    # 1. 检查文件
    for f in [FILE_TARGETS, FILE_MEASURED_RAW, FILE_MATRIX_VAL, FILE_OFFSET_JSON]:
        if not os.path.exists(f):
            print(f"❌ 缺少文件: {f}")
            return

    # 2. 加载参数
    with open(FILE_OFFSET_JSON, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])
    print(f"✅ Tool Offset: {tool_offset}")
    
    mat = np.load(FILE_MATRIX_VAL)
    R_inv, t = mat['R'], mat['t']

    # 3. 处理测量数据
    df_meas = pd.read_csv(FILE_MEASURED_RAW)
    df_meas.columns = [c.strip().lower() for c in df_meas.columns]
    
    if 'x' in df_meas.columns and 'y' in df_meas.columns and 'z' in df_meas.columns:
        pts_raw = df_meas[['x', 'y', 'z']].values
    else:
        print("⚠️ 未检测到标准列名(x,y,z)，尝试读取第 2,3,4 列作为坐标...")
        pts_raw = df_meas.iloc[:, 1:4].values 

    # 坐标变换: Raw -> Robot -> Flange
    pts_robot = np.dot(R_inv, (pts_raw - t).T).T
    # 扣除偏置
    pts_meas = pts_robot - tool_offset 
    
    # 4. 加载目标数据
    df_tgt = pd.read_csv(FILE_TARGETS)
    pts_tgt = df_tgt[['orig_x', 'orig_y', 'orig_z']].values
    
    # 5. 强制顺序对齐 (One-to-One Sequential Matching)
    n_tgt = len(pts_tgt)
    n_meas = len(pts_meas)
    n_min = min(n_tgt, n_meas)
    
    if n_tgt != n_meas:
        print(f"⚠️ 警告: 点数不一致! Target: {n_tgt}, Measured: {n_meas}")
        print(f"   -> 将只使用前 {n_min} 个点进行对比。")
    
    pts_tgt = pts_tgt[:n_min]
    pts_meas = pts_meas[:n_min]
    
    # 计算误差
    diff = pts_meas - pts_tgt
    errors = np.linalg.norm(diff, axis=1)
    
    # 构造数据矩阵: [tx, ty, tz, mx, my, mz, err]
    data = np.column_stack((pts_tgt, pts_meas, errors))
    
    print(f"✅ 数据对齐完成: {len(data)} 点")
    if len(data) == 0: return

    # 6. 统计指标
    rmse = np.sqrt(np.mean(errors**2))
    max_err = np.max(errors)
    idx_max = np.argmax(errors)
    
    print("-" * 40)
    print(f"🏆 RMSE: {rmse:.4f} mm")
    print(f"📉 MAX : {max_err:.4f} mm")
    print("-" * 40)

    # 7. 动态设定 VMAX (关键修改)
    # 让最红的颜色对应最大误差，或者稍微大一点点(取整)以便好看
    # 这里直接用 max_err，保证最差点最红
    # DYNAMIC_VMAX = max_err 
    DYNAMIC_VMAX = 10.0 
    print(f"🎨 动态色标范围: 0.0 ~ {DYNAMIC_VMAX:.3f} mm")

    # 8. 画图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 半径计算
    radii = np.linalg.norm(data[:, 0:2], axis=1)
    
    # 左图: Error vs Radius
    title_1 = f"Original Error: Error vs Radius\nMax: {max_err:.3f} mm (R={radii[idx_max]:.1f})"
    sc1 = plot_scatter(axes[0], radii, errors, errors, 
                       title_1, "Radius (mm)", "Error (mm)",
                       GLOBAL_VMIN, DYNAMIC_VMAX, vline=100,
                       ylim=(0, 10))
    plt.colorbar(sc1, ax=axes[0], label='Error (mm)')
    
    # 右图: Spatial Distribution
    title_2 = f"Spatial Distribution (XY)\nRMSE: {rmse:.4f} mm"
    sc2 = plot_scatter(axes[1], data[:,0], data[:,1], errors,
                       title_2, "X (mm)", "Y (mm)",
                       GLOBAL_VMIN, DYNAMIC_VMAX)
    
    # 虚线圆圈 (R=100)
    circle = plt.Circle((0, 0), 100, color='gray', fill=False, linestyle='--')
    axes[1].add_patch(circle)
    axes[1].axis('equal')
    plt.colorbar(sc2, ax=axes[1], label='Error (mm)')
    
    plt.tight_layout()

    # ================== 保存图片代码 ==================
    save_file = "Original_Report.png"
    # dpi=300 代表高分辨率 (打印级清晰度)
    # bbox_inches='tight' 用于自动裁剪周围多余白边，防止文字被切掉
    plt.savefig(save_file, dpi=1000, bbox_inches='tight')
    print(f"💾 高清图片已保存至: {save_file}")
    # ======================================================

    plt.show()

if __name__ == "__main__":
    main()