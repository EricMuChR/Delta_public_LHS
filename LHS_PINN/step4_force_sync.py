import pandas as pd
import numpy as np
import scipy.spatial.transform as transform

# ================= 配置 =================
FILE_CMD = "data_angles_collected.csv"      # 指令数据 (11362)
FILE_MEAS = "aligned_measurement_motor.csv" # 测量数据 (11366)
OUTPUT_FILE = "training_data_pinn.csv"

# 这是一个宽松的对齐脚本，不做距离剔除，只做序列对齐
# =======================================

def main():
    print("="*60)
    print("   Step 4: 强制序列对齐 (Force Sync Sequence)")
    print("="*60)

    # 1. 读取数据
    try:
        df_cmd = pd.read_csv(FILE_CMD)
        df_meas = pd.read_csv(FILE_MEAS)
    except FileNotFoundError:
        print("❌ 找不到输入文件，请检查文件名。")
        return

    n_cmd = len(df_cmd)
    n_meas = len(df_meas)
    print(f"📖 指令点数: {n_cmd} | 测量点数: {n_meas}")

    # 提取坐标用于计算最佳偏移 (取前1000个点快速计算)
    # 注意：cmd_x, cmd_y, cmd_z
    cmd_xyz = df_cmd[['cmd_x', 'cmd_y', 'cmd_z']].values
    meas_xyz = df_meas[['x', 'y', 'z']].values

    # 2. 滑动窗口寻找最佳 Index Offset
    # 假设测量数据可能比指令数据 多出 或 少了 几个点
    # 我们尝试 shift 从 -20 到 +20
    best_shift = 0
    min_error = float('inf')
    
    # 搜索范围
    search_range = range(-20, 21)
    
    print("🔍 正在计算最佳序列偏移 (Shift)...")
    
    for shift in search_range:
        # 截取重叠部分
        # shift > 0: meas 滞后 (meas[shift:] vs cmd[:-shift])
        # shift < 0: meas 超前 (meas[:shift] vs cmd[-shift:])
        
        c_start = max(0, -shift)
        c_end = min(n_cmd, n_meas - shift)
        
        m_start = max(0, shift)
        m_end = min(n_meas, n_cmd + shift)
        
        if c_end - c_start < 100: continue # 重叠太少
        
        valid_cmd = cmd_xyz[c_start:c_end]
        valid_meas = meas_xyz[m_start:m_end]
        
        # 计算平均距离 (简单的欧氏距离)
        # 这里我们只取前 500 个点做采样对比，速度快
        sample_size = min(len(valid_cmd), 500)
        diff = valid_cmd[:sample_size] - valid_meas[:sample_size]
        # 去掉均值(Offset)后再算误差，防止因为固定平移导致匹配失败
        diff_centered = diff - np.mean(diff, axis=0)
        
        mse = np.mean(np.sum(diff_centered**2, axis=1))
        
        if mse < min_error:
            min_error = mse
            best_shift = shift

    print(f"✅ 锁定最佳偏移: Meas Shift = {best_shift}")
    print(f"   (最小均方误差: {min_error:.2f})")

    # 3. 根据最佳 Shift 对齐数据
    c_start = max(0, -best_shift)
    c_end = min(n_cmd, n_meas - best_shift)
    m_start = max(0, best_shift)
    m_end = min(n_meas, n_cmd + best_shift)

    df_cmd_aligned = df_cmd.iloc[c_start:c_end].reset_index(drop=True)
    df_meas_aligned = df_meas.iloc[m_start:m_end].reset_index(drop=True)

    # 4. 计算并去除全局 Offset (平均误差)
    # PINN 训练需要去除这个工具坐标系造成的固定偏差
    xyz_cmd = df_cmd_aligned[['cmd_x', 'cmd_y', 'cmd_z']].values
    xyz_meas = df_meas_aligned[['x', 'y', 'z']].values
    
    offset_vec = np.mean(xyz_meas - xyz_cmd, axis=0)
    print("="*40)
    print(f"🤖 全局 Tool Offset: [{offset_vec[0]:.4f}, {offset_vec[1]:.4f}, {offset_vec[2]:.4f}]")
    print("   (将自动从测量数据中减去此 Offset)")
    print("="*40)

    # 5. 合并数据
    # Meas - Offset
    xyz_meas_corrected = xyz_meas - offset_vec
    
    # 还要计算一下最终的残差分布，看看是不是真的对齐了
    final_diff = xyz_meas_corrected - xyz_cmd
    final_mse = np.sqrt(np.mean(np.sum(final_diff**2, axis=1)))
    print(f"📉 修正 Offset 后的平均定位误差: {final_mse:.4f} mm")
    
    if final_mse > 10.0:
        print("⚠️ 警告: 平均误差依然很大！可能存在坐标系旋转或镜像问题。")
        print("   请检查 X/Y 轴是否反了。")
    
    # 构造最终 DataFrame
    df_out = pd.DataFrame()
    df_out['Index'] = df_cmd_aligned.index
    df_out['cmd_x'] = df_cmd_aligned['cmd_x']
    df_out['cmd_y'] = df_cmd_aligned['cmd_y']
    df_out['cmd_z'] = df_cmd_aligned['cmd_z']
    
    # 角度
    if 'theta_1' in df_cmd_aligned.columns:
        df_out['theta_1'] = df_cmd_aligned['theta_1']
        df_out['theta_2'] = df_cmd_aligned['theta_2']
        df_out['theta_3'] = df_cmd_aligned['theta_3']
    else:
        # 兼容旧格式列名
        df_out['theta_1'] = df_cmd_aligned['motor_1']
        df_out['theta_2'] = df_cmd_aligned['motor_2']
        df_out['theta_3'] = df_cmd_aligned['motor_3']

    # 测量值 (已去 Offset)
    df_out['meas_x'] = xyz_meas_corrected[:, 0]
    df_out['meas_y'] = xyz_meas_corrected[:, 1]
    df_out['meas_z'] = xyz_meas_corrected[:, 2]

    # 6. 保存
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 数据已保存至: {OUTPUT_FILE} ({len(df_out)} 条)")
    print("✨ 你现在可以直接运行 train_pinn.py 了。")

if __name__ == "__main__":
    main()