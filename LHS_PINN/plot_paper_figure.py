import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import json

# ================= 全局格式与字体配置 =================
# 设置全局字体为 Times New Roman，基础字号 18
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# ================= 配置常量 =================
FILE_TGT_ORIG = "points_with_compensation_cmp_origin.csv"
FILE_MEAS_ORIG = "original_500.csv"
FILE_TGT_PINN = "points_with_compensation.csv"
FILE_MEAS_PINN = "validation_500.csv"
FILE_MATRIX_VAL = "matrix_val.npz"
FILE_OFFSET_JSON = "tool_offset.json"

# 统一色标范围
GLOBAL_VMIN = 0.0
GLOBAL_VMAX = 10.0

# 根据用户图片提取的 "Coffee Roast" 色系代码
# 从浅黄(低误差) 平滑过渡到 深棕(高误差)
COFFEE_COLORS = [
    "#FFF7BC", 
    "#FEE391", 
    "#FEC44F", 
    "#FE9929", 
    "#EC7014", 
    "#CC4C02", 
    "#8C2D04"
]
# 生成平滑连续的渐变色带
COFFEE_CMAP = mcolors.LinearSegmentedColormap.from_list("coffee_roast", COFFEE_COLORS)

def load_and_compute_error(target_file, meas_raw_file, matrix_file, offset_file):
    """提取公共的数据加载与误差计算逻辑"""
    with open(offset_file, 'r') as f:
        tool_offset = np.array(json.load(f)["tool_offset"])
    
    mat = np.load(matrix_file)
    R_inv, t = mat['R'], mat['t']

    df_meas = pd.read_csv(meas_raw_file)
    df_meas.columns = [c.strip().lower() for c in df_meas.columns]
    
    if 'x' in df_meas.columns and 'y' in df_meas.columns and 'z' in df_meas.columns:
        pts_raw = df_meas[['x', 'y', 'z']].values
    else:
        pts_raw = df_meas.iloc[:, 1:4].values 

    pts_robot = np.dot(R_inv, (pts_raw - t).T).T
    pts_meas = pts_robot - tool_offset 
    
    df_tgt = pd.read_csv(target_file)
    pts_tgt = df_tgt[['orig_x', 'orig_y', 'orig_z']].values
    
    n_min = min(len(pts_tgt), len(pts_meas))
    pts_tgt = pts_tgt[:n_min]
    pts_meas = pts_meas[:n_min]
    
    errors = np.linalg.norm(pts_meas - pts_tgt, axis=1)
    return pts_tgt, errors

def plot_scatter(ax, x, y, c, title, xlabel, ylabel, vline=None, is_spatial=False):
    """标准化的子图绘制函数（移除内部色标）"""
    sc = ax.scatter(x, y, c=c, cmap=COFFEE_CMAP, s=25, 
                    vmin=GLOBAL_VMIN, vmax=GLOBAL_VMAX, 
                    alpha=0.9, edgecolors='gold', linewidths=0.4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 去除网格线
    ax.grid(False)
    
    if vline: 
        ax.axvline(vline, color='k', linestyle='--', alpha=0.5)
    
    if is_spatial:
        # 工作空间边界虚线并设置等比例
        circle = plt.Circle((0, 0), 100, color='gray', fill=False, linestyle='--')
        ax.add_patch(circle)
        ax.axis('equal')
        ax.set_xlim(-110, 110)
        ax.set_ylim(-110, 110)
        
    # 注意：此处不再调用 plt.colorbar()，而是将散点图对象(sc)返回
    # 供后续生成全局统一个色标使用
    return sc

def main():
    print("="*60)
    print("   生成论文最终版 2x3 误差对比联合大图 (Coffee Roast 配色)")
    print("="*60)

    # 1. 检查文件是否齐全
    files_needed = [FILE_TGT_ORIG, FILE_MEAS_ORIG, FILE_TGT_PINN, FILE_MEAS_PINN, FILE_MATRIX_VAL, FILE_OFFSET_JSON]
    for f in files_needed:
        if not os.path.exists(f):
            print(f"❌ 缺少文件: {f}，请确保所有数据文件都在当前目录。")
            return

    # 2. 计算 Original 和 PINN 的误差数据
    print("正在处理 Original 数据...")
    pts_orig, err_orig = load_and_compute_error(FILE_TGT_ORIG, FILE_MEAS_ORIG, FILE_MATRIX_VAL, FILE_OFFSET_JSON)
    rmse_orig = np.sqrt(np.mean(err_orig**2))
    max_orig = np.max(err_orig)
    
    print("正在处理 PINN Compensated 数据...")
    pts_pinn, err_pinn = load_and_compute_error(FILE_TGT_PINN, FILE_MEAS_PINN, FILE_MATRIX_VAL, FILE_OFFSET_JSON)
    rmse_pinn = np.sqrt(np.mean(err_pinn**2))
    max_pinn = np.max(err_pinn)

    radii_orig = np.linalg.norm(pts_orig[:, 0:2], axis=1)
    radii_pinn = np.linalg.norm(pts_pinn[:, 0:2], axis=1)
    
    z_orig = pts_orig[:, 2]
    z_pinn = pts_pinn[:, 2]

    # 3. 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # ------------------ 第一行：Original ------------------
    plot_scatter(axes[0, 0], pts_orig[:,0], pts_orig[:,1], err_orig,
                 f"Original: Spatial (XY)\nRMSE: {rmse_orig:.3f} mm", 
                 "X (mm)", "Y (mm)", is_spatial=True)
    
    plot_scatter(axes[0, 1], radii_orig, err_orig, err_orig,
                 f"Original: Error vs Radius\nMax: {max_orig:.3f} mm", 
                 "Radius (mm)", "Error (mm)", vline=100)
    axes[0, 1].set_ylim(0, 11) 
    
    sc_ref = plot_scatter(axes[0, 2], z_orig, err_orig, err_orig,
                          "Original: Error vs Z-axis", 
                          "Z (mm)", "Error (mm)")
    axes[0, 2].set_ylim(0, 11)

    # ------------------ 第二行：PINN Compensated ------------------
    plot_scatter(axes[1, 0], pts_pinn[:,0], pts_pinn[:,1], err_pinn,
                 f"PINN: Spatial (XY)\nRMSE: {rmse_pinn:.3f} mm", 
                 "X (mm)", "Y (mm)", is_spatial=True)
    
    plot_scatter(axes[1, 1], radii_pinn, err_pinn, err_pinn,
                 f"PINN: Error vs Radius\nMax: {max_pinn:.3f} mm", 
                 "Radius (mm)", "Error (mm)", vline=100)
    axes[1, 1].set_ylim(0, 11)
    
    plot_scatter(axes[1, 2], z_pinn, err_pinn, err_pinn,
                 "PINN: Error vs Z-axis", 
                 "Z (mm)", "Error (mm)")
    axes[1, 2].set_ylim(0, 11)

    # 4. 全局排版与添加全局 Colorbar
    # 调整子图间的间距，右侧留出空间给全局色标
    fig.subplots_adjust(right=0.92, wspace=0.25, hspace=0.35)
    
    # 在右侧添加一个跨越两行的 Colorbar 坐标轴 [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7]) 
    
    # 利用上面任意一个子图的返回值 (sc_ref) 生成全局色标
    cbar = fig.colorbar(sc_ref, cax=cbar_ax)
    cbar.set_label('Error (mm)', fontdict={'family': 'Times New Roman', 'size': 18})

    # 5. 保存高质量图片
    save_file = "Paper_Figure_Error_Comparison_Coffee.png"
    plt.savefig(save_file, dpi=1000, bbox_inches='tight')
    print(f"✅ 高清论文用图已生成并保存至: {save_file}")

    plt.show()

if __name__ == "__main__":
    main()