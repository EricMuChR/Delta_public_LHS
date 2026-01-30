import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
INPUT_FILE = "training_data_pinn.csv"
OUTPUT_FILE = "training_data_pinn_cleaned.csv"
THRESHOLD_MM = 15.0  # 第一次过滤阈值: 大于15mm的点被视为坏点(飞点/奇异点)
AUTO_CENTER = True   # 是否自动去除残留的平均偏差(推荐True)
# ===========================================

def analyze_error(df, label="原始数据"):
    """ 计算并打印误差统计信息 """
    # 提取坐标
    cmd = df[['cmd_x', 'cmd_y', 'cmd_z']].values
    meas = df[['meas_x', 'meas_y', 'meas_z']].values
    
    # 计算点对点距离 (欧氏距离)
    diff = meas - cmd
    dist = np.linalg.norm(diff, axis=1)
    
    mae = np.mean(dist)
    rmse = np.sqrt(np.mean(dist**2))
    max_err = np.max(dist)
    
    print(f"📊 [{label}] 样本数: {len(df)}")
    print(f"   - MAE (平均误差): {mae:.4f} mm")
    print(f"   - RMSE (均方根):  {rmse:.4f} mm")
    print(f"   - MAX (最大误差): {max_err:.4f} mm")
    
    return dist, diff

def main():
    print("="*60)
    print("🧹 PINN 训练数据清洗工具")
    print("="*60)
    
    # 1. 读取数据
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    # 2. 原始数据分析
    dist, diff = analyze_error(df, "清洗前")
    
    # 3. 第一轮过滤: 剔除离群点 (Outliers)
    # 逻辑: 任何误差超过阈值的点，直接扔掉
    valid_mask = dist < THRESHOLD_MM
    n_dropped = len(df) - np.sum(valid_mask)
    
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    print("-" * 40)
    print(f"✂️ 剔除误差 > {THRESHOLD_MM}mm 的坏点: {n_dropped} 个")
    
    if len(df_clean) == 0:
        print("❌ 警告: 所有数据都被剔除了！请检查阈值或数据质量。")
        return

    # 4. (可选) 自动归零 / 去除残留偏差 (Re-centering)
    # Step 4 自动算出的 Offset 可能还有 0.x mm 的误差，这里把它彻底修平
    if AUTO_CENTER:
        cmd_c = df_clean[['cmd_x', 'cmd_y', 'cmd_z']].values
        meas_c = df_clean[['meas_x', 'meas_y', 'meas_z']].values
        
        # 计算残留偏差 (Bias)
        bias = np.mean(meas_c - cmd_c, axis=1) # axis=0 是对列求均值，axis=1 是行... 等等，应该是 axis=0
        bias = np.mean(meas_c - cmd_c, axis=0)
        
        print(f"⚖️ 检测到残留偏差 (Bias): [{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}]")
        print("   -> 正在执行二次归零...")
        
        # 修正测量值
        df_clean['meas_x'] -= bias[0]
        df_clean['meas_y'] -= bias[1]
        df_clean['meas_z'] -= bias[2]
        
        # 5. 第二轮分析 (清洗后)
        analyze_error(df_clean, "清洗 & 归零后")
        
    else:
        analyze_error(df_clean, "清洗后")

    # 6. 保存
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print("="*60)
    print(f"💾 已保存清洗后的数据至: {OUTPUT_FILE}")
    print("🚀 下一步: 请修改 train_pinn.py 中的 DATA_FILE 为此新文件名，然后重新训练。")

if __name__ == "__main__":
    main()