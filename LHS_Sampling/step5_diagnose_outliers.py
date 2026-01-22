import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================= 配置 =================
FILE_CMD = "lhs_final.csv"       
FILE_MEAS = "tracker_final.csv"
# 与之前保持一致的清洗阈值 (mm)
THRESHOLD_MM = 10.0
# =======================================

def main():
    print("="*60)
    print("   Step 5 Diagnosis: 离群点(Outliers) 深度侦查")
    print("="*60)

    if not os.path.exists(FILE_CMD):
        print("❌ 找不到输入文件。")
        return

    # 1. 加载数据
    df_cmd = pd.read_csv(FILE_CMD)
    df_meas = pd.read_csv(FILE_MEAS)
    
    # 使用索引作为时间轴
    indices = np.arange(len(df_cmd))
    
    X_cmd = df_cmd[['x', 'y', 'z']].values
    Y_meas = df_meas[['x', 'y', 'z']].values
    
    # 2. 计算残差 (去偏后)
    Error_Raw = Y_meas - X_cmd
    Offset = np.mean(Error_Raw, axis=0)
    Error_Baseline = Error_Raw - Offset
    Norms = np.linalg.norm(Error_Baseline, axis=1)
    
    # 3. 标记离群点
    is_outlier = Norms > THRESHOLD_MM
    n_outliers = np.sum(is_outlier)
    
    print(f"📊 诊断对象: {n_outliers} 个异常点 (占比 {n_outliers/len(df_cmd)*100:.2f}%)")
    print(f"   判定标准: 去偏误差 > {THRESHOLD_MM} mm")

    if n_outliers == 0:
        print("🎉 没有发现异常点，无需诊断。")
        return

    # 4. 绘图分析
    fig = plt.figure(figsize=(18, 10))
    
    # --- 子图 1: 时序分析 (Time Series) ---
    # 检查是否是从某时刻开始集中报错 (暗示同步错位)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(indices[~is_outlier], Norms[~is_outlier], s=5, c='blue', alpha=0.3, label='Normal')
    ax1.scatter(indices[is_outlier], Norms[is_outlier], s=10, c='red', alpha=0.8, label='Outlier')
    ax1.axhline(y=THRESHOLD_MM, color='green', linestyle='--', label='Threshold')
    ax1.set_title("1. Temporal Distribution (Is it a sync shift?)")
    ax1.set_xlabel("Sample Index (Time Sequence)")
    ax1.set_ylabel("Error Norm (mm)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- 子图 2: 空间分布 XY (Spatial XY) ---
    # 检查是否集中在边缘 (暗示坐标系或机械限制)
    ax2 = fig.add_subplot(2, 2, 2)
    # 画工作空间圆
    circle = plt.Circle((0, 0), 125, color='gray', fill=False, linestyle='--')
    ax2.add_patch(circle)
    ax2.scatter(X_cmd[~is_outlier, 0], X_cmd[~is_outlier, 1], s=5, c='blue', alpha=0.1)
    ax2.scatter(X_cmd[is_outlier, 0], X_cmd[is_outlier, 1], s=20, c='red', marker='x', label='Outlier')
    ax2.set_title("2. Spatial Distribution (XY Plane)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.axis('equal')
    ax2.legend()
    
    # --- 子图 3: 空间分布 Z (Spatial Z) ---
    # 检查是否集中在顶部或底部
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(X_cmd[~is_outlier, 2], Norms[~is_outlier], s=5, c='blue', alpha=0.3)
    ax3.scatter(X_cmd[is_outlier, 2], Norms[is_outlier], s=10, c='red', alpha=0.8)
    ax3.set_title("3. Error vs Z Height")
    ax3.set_xlabel("Z (mm)")
    ax3.set_ylabel("Error Norm (mm)")
    ax3.grid(True)

    # --- 子图 4: 错位测试 (Shift Test) ---
    # 尝试把数据错位 -5 到 +5 格，看看误差是否骤降
    # 如果错位后误差变小，说明 100% 是同步问题
    ax4 = fig.add_subplot(2, 2, 4)
    shifts = range(-5, 6)
    mean_errs = []
    
    # 取中间一段数据做测试，避免边缘溢出
    cut = 100
    sub_meas = Y_meas[cut:-cut]
    
    for s in shifts:
        # Cmd 移动 s 格
        sub_cmd = X_cmd[cut+s : -cut+s]
        # 简单对齐中心
        err = sub_meas - sub_cmd
        off = np.mean(err, axis=0)
        res = np.linalg.norm(err - off, axis=1)
        mean_errs.append(np.mean(res))
        
    ax4.plot(shifts, mean_errs, 'o-', color='purple')
    ax4.set_title("4. Index Shift Check (The lower the better)")
    ax4.set_xlabel("Shift Amount")
    ax4.set_ylabel("Mean Residual Error (mm)")
    ax4.grid(True)
    
    # 找到最佳 Shift
    best_shift = shifts[np.argmin(mean_errs)]
    print("\n🕵️ 自动诊断结果:")
    if best_shift != 0:
        print(f"⚠️ 警告: 极大概率发生数据错位！")
        print(f"   建议 Command 数据相对于 Measure 数据移动 {best_shift} 格。")
        print(f"   这会把平均误差从 {mean_errs[5]:.2f}mm 降低到 {min(mean_errs):.2f}mm。")
    else:
        print(f"✅ 数据同步看起来正常 (最佳 Shift = 0)。")
        print(f"   请重点观察图 1 和图 2，判断是随机飞点还是边缘失效。")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()