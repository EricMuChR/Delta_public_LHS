import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ================= 配置 =================
FILE_MEAS = "repeatability_meas.csv" 
NUM_POINTS = 12       
CYCLES = 10           
# =======================================

def main():
    print("="*60)
    print("   Step 3 V2: ISO 9283 深度误差分析 (可视化增强版)")
    print("="*60)

    if not os.path.exists(FILE_MEAS):
        print(f"❌ 找不到数据文件: {FILE_MEAS}")
        return

    # 1. 读取数据
    df = pd.read_csv(FILE_MEAS)
    try:
        data = df[['x', 'y', 'z']].values
    except KeyError:
        data = df.iloc[:, -3:].values

    # 验证行数
    expected_rows = NUM_POINTS * CYCLES
    if len(data) != expected_rows:
        print(f"⚠️ 数据行数 ({len(data)}) 与预期 ({expected_rows}) 不符，将自动截断或对其。")
        data = data[:expected_rows]

    # 2. 数据重组与计算
    # 结构: [Point_Index, Cycle_Index, XYZ]
    groups = []
    
    # 用于存储所有偏差数据以便绘图
    # [Point_ID, Cycle_ID, Dev_X, Dev_Y, Dev_Z, Dist_to_Center]
    deviation_records = [] 
    
    print("-" * 65)
    print(f"{'Point':<6} | {'RP (mm)':<8} | {'Mean X':<8} | {'Mean Y':<8} | {'Mean Z':<8}")
    print("-" * 65)

    rp_values = []
    
    # 按点位分组计算
    for i in range(NUM_POINTS):
        # 提取第 i 个点的所有 10 次测量 (索引: i, i+12, i+24...)
        indices = [j * NUM_POINTS + i for j in range(CYCLES)]
        cluster = data[indices]
        
        # 计算质心
        centroid = np.mean(cluster, axis=0)
        
        # 计算偏差向量
        deviations = cluster - centroid
        
        # 计算距离 (l_j)
        dists = np.linalg.norm(deviations, axis=1)
        
        # ISO 9283 RP
        l_bar = np.mean(dists)
        s_l = np.std(dists, ddof=1)
        rp = l_bar + 3 * s_l
        rp_values.append(rp)
        
        print(f"P{i+1:02d}   | {rp:<8.4f} | {centroid[0]:<8.2f} | {centroid[1]:<8.2f} | {centroid[2]:<8.2f}")
        
        # 记录详细偏差数据用于绘图
        for cycle_idx, dev in enumerate(deviations):
            deviation_records.append([i+1, cycle_idx+1, dev[0], dev[1], dev[2], dists[cycle_idx]])

    deviation_df = pd.DataFrame(deviation_records, columns=['Point', 'Cycle', 'Dev_X', 'Dev_Y', 'Dev_Z', 'Dist'])
    
    avg_rp = np.mean(rp_values)
    max_rp = np.max(rp_values)
    
    print("-" * 65)
    print(f"🏆 统计结果: 平均 RP = {avg_rp:.4f} mm | 最差 RP = {max_rp:.4f} mm")
    print("-" * 65)

    # ================= 绘图部分 (4合1仪表盘) =================
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Repeatability Analysis (ISO 9283)\nAvg RP: {avg_rp:.4f}mm | Max RP: {max_rp:.4f}mm", fontsize=16, fontweight='bold')

    # --- 图 1: 3D 归一化散布云 (所有点的抖动叠加) ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # 颜色按 Point 区分
    sc1 = ax1.scatter(deviation_df['Dev_X'], deviation_df['Dev_Y'], deviation_df['Dev_Z'], 
                      c=deviation_df['Point'], cmap='tab20', s=40, alpha=0.8)
    ax1.set_title("1. Normalized 3D Jitter Cloud (All Points)", fontsize=12)
    ax1.set_xlabel("X Dev (mm)")
    ax1.set_ylabel("Y Dev (mm)")
    ax1.set_zlabel("Z Dev (mm)")
    # 强制等比例，看清球形还是椭球形
    max_range = np.array([deviation_df['Dev_X'].max()-deviation_df['Dev_X'].min(), 
                          deviation_df['Dev_Y'].max()-deviation_df['Dev_Y'].min(), 
                          deviation_df['Dev_Z'].max()-deviation_df['Dev_Z'].min()]).max() / 2.0
    mid_x = (deviation_df['Dev_X'].max()+deviation_df['Dev_X'].min()) * 0.5
    mid_y = (deviation_df['Dev_Y'].max()+deviation_df['Dev_Y'].min()) * 0.5
    mid_z = (deviation_df['Dev_Z'].max()+deviation_df['Dev_Z'].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- 图 2: 单点误差散点图 (寻找坏点) ---
    ax2 = fig.add_subplot(2, 2, 2)
    # X轴: P01-P12, Y轴: 距离中心偏差
    for pt in range(1, NUM_POINTS + 1):
        subset = deviation_df[deviation_df['Point'] == pt]
        ax2.scatter([pt]*len(subset), subset['Dist'], alpha=0.6, c='blue', s=30)
        # 画出该点的 RP 线
        ax2.plot([pt-0.3, pt+0.3], [rp_values[pt-1], rp_values[pt-1]], 'r-', alpha=0.5)
    
    ax2.set_xticks(range(1, NUM_POINTS + 1))
    ax2.set_xlabel("Test Point ID (P01 - P12)")
    ax2.set_ylabel("Distance from Centroid (mm)")
    ax2.set_title("2. Error Magnitude per Position (Red Line = ISO RP)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # --- 图 3: XYZ 轴向偏差箱线图 (诊断轴系刚性) ---
    ax3 = fig.add_subplot(2, 2, 3)
    box_data = [deviation_df['Dev_X'], deviation_df['Dev_Y'], deviation_df['Dev_Z']]
    ax3.boxplot(box_data, labels=['X Axis', 'Y Axis', 'Z Axis'], patch_artist=True)
    ax3.set_title("3. Vibration Distribution by Axis", fontsize=12)
    ax3.set_ylabel("Deviation (mm)")
    ax3.grid(True, axis='y', alpha=0.3)
    # 如果 Z 轴特别长，说明 Z 方向刚性差

    # --- 图 4: 循环漂移趋势 (Cycle Drift) ---
    ax4 = fig.add_subplot(2, 2, 4)
    cycle_means = deviation_df.groupby('Cycle')['Dist'].mean()
    ax4.plot(cycle_means.index, cycle_means.values, 'o-', linewidth=2, color='green')
    ax4.set_xlabel("Cycle Number (1-10)")
    ax4.set_ylabel("Mean Error (mm)")
    ax4.set_title("4. Thermal Drift / Warm-up Trend", fontsize=12)
    ax4.set_xticks(range(1, CYCLES + 1))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()