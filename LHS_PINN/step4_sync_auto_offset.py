import pandas as pd
import numpy as np
import os
import json

# ================= 配置 =================
# 输入 1: 包含关节角度的采集指令数据 (由 step2 生成)
FILE_CMD_WITH_ANGLES = "data_angles_collected.csv" 
# 如果上面文件不存在，回退使用 lhs_points.csv (但不推荐，因为会丢失角度)
FILE_CMD_BACKUP = "lhs_points.csv"

# 输入 2: 经过 Step 3 转换后的测量数据 (基座坐标系)
# 注意: 你需要先运行 step_pre_process 把 tracker_raw 转成这个
# 或者如果你还没转，这里暂时先读取 tracker_raw.csv (假设已手动对齐)
# 也就是 Step 3 标定后的输出
FILE_MEAS_ALIGNED = "aligned_measurement_motor.csv" # 暂定名，需确认你是否有这个文件

# 输出
FILE_TRAIN_FINAL = "training_data_pinn.csv"  # 最终喂给 PINN 的数据
FILE_OFFSET_JSON = "tool_offset.json" 

# 参数
MATCH_THRESHOLD = 20.0  # 匹配允许的最大距离 (mm)
SEARCH_WINDOW = 20      # 向后搜索多少个点 (应对过度采集/抖动)
# =======================================

def main():
    print("="*60)
    print("   Step 4 (PINN版): 数据同步 & 去除抖动点")
    print("="*60)

    # 1. 读取指令数据 (优先读取带角度的)
    if os.path.exists(FILE_CMD_WITH_ANGLES):
        print(f"📖 读取指令数据: {FILE_CMD_WITH_ANGLES}")
        df_cmd = pd.read_csv(FILE_CMD_WITH_ANGLES)
        # 确保列名正确
        # Step 2 生成的列通常是: Index, Cmd_X, Cmd_Y, Cmd_Z, Theta1, Theta2, Theta3
        # 我们重命名一下以防万一
        col_map = {
            'Cmd_X': 'cmd_x', 'Cmd_Y': 'cmd_y', 'Cmd_Z': 'cmd_z',
            'Theta1': 'theta_1', 'Theta2': 'theta_2', 'Theta3': 'theta_3'
        }
        df_cmd.rename(columns=col_map, inplace=True)
    elif os.path.exists(FILE_CMD_BACKUP):
        print(f"⚠️ 警告: 找不到带角度的文件，回退使用 {FILE_CMD_BACKUP} (将无法训练 PINN 物理部分)")
        df_cmd = pd.read_csv(FILE_CMD_BACKUP)
        df_cmd.columns = ['cmd_x', 'cmd_y', 'cmd_z'] # 强制重命名
    else:
        print("❌ 找不到指令数据文件。")
        return

    # 2. 读取测量数据
    # 这里假设你已经把激光跟踪仪的数据转到了机器人基座坐标系
    # 如果还没有，请确保你的 aligned_measurement_motor.csv 是存在的
    if not os.path.exists(FILE_MEAS_ALIGNED):
        print(f"❌ 找不到测量数据: {FILE_MEAS_ALIGNED}")
        print("💡 提示: 你是否运行了 Step 3 的转换步骤？或者暂时想用 tracker_raw_data.csv？")
        return

    df_meas = pd.read_csv(FILE_MEAS_ALIGNED)
    # 假设列名是 x, y, z
    meas_cols = ['x', 'y', 'z']
    if not all(c in df_meas.columns for c in meas_cols):
        # 尝试适配 tracker_raw 的列名
        if 'Meas_X' in df_meas.columns:
            df_meas.rename(columns={'Meas_X':'x', 'Meas_Y':'y', 'Meas_Z':'z'}, inplace=True)
        else:
            print(f"❌ 测量文件列名无法识别，需要: {meas_cols}")
            return
            
    pts_cmd = df_cmd[['cmd_x', 'cmd_y', 'cmd_z']].values
    pts_meas = df_meas[['x', 'y', 'z']].values

    print(f"📊 数据概览: 指令点 {len(pts_cmd)} 个 | 测量点 {len(pts_meas)} 个")
    if len(pts_meas) > len(pts_cmd):
        print(f"   -> 检测到测量点多于指令点 (Diff: +{len(pts_meas)-len(pts_cmd)})，将启动窗口搜索模式。")

    # 3. 粗略对齐 (重心对齐) - 为了计算距离
    centroid_cmd = np.mean(pts_cmd, axis=0)
    centroid_meas = np.mean(pts_meas, axis=0)
    robust_offset = centroid_meas - centroid_cmd
    pts_meas_shifted = pts_meas - robust_offset
    
    # 4. 智能窗口匹配 (Window Search)
    valid_cmd_indices = []
    valid_meas_indices = []
    
    meas_cursor = 0 # 测量数据的游标
    
    print("🔍 开始同步匹配...")
    
    for i in range(len(pts_cmd)):
        current_cmd_pt = pts_cmd[i]
        
        # 在窗口范围内搜索最佳匹配
        # 范围: [当前游标, 当前游标 + SEARCH_WINDOW]
        window_end = min(meas_cursor + SEARCH_WINDOW, len(pts_meas))
        
        best_dist = MATCH_THRESHOLD
        best_idx = -1
        
        # 遍历窗口内的所有点，找距离最近的
        for j in range(meas_cursor, window_end):
            dist = np.linalg.norm(pts_meas_shifted[j] - current_cmd_pt)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        # 判定是否找到匹配
        if best_idx != -1:
            valid_cmd_indices.append(i)
            valid_meas_indices.append(best_idx)
            
            # 关键: 更新游标到最佳匹配点的下一位
            # 这样就跳过了 best_idx 之前的所有“抖动点”
            meas_cursor = best_idx + 1
        else:
            # 如果窗口内都没找到，说明这个点可能漏采了，或者误差太大了
            # 游标不动，给下一个指令点机会
            pass

    matched_count = len(valid_cmd_indices)
    print(f"✅ 匹配完成。最终有效点对: {matched_count} / {len(pts_cmd)}")

    if matched_count < len(pts_cmd) * 0.8:
        print("⚠️ 警告: 匹配率低于 80%，请检查 MATCH_THRESHOLD 或数据质量。")

    # 5. 精确计算 Tool Offset
    p_c_final = pts_cmd[valid_cmd_indices]
    p_m_final = pts_meas[valid_meas_indices]
    
    final_offset_vec = np.mean(p_m_final - p_c_final, axis=0)
    print("\n" + "="*45)
    print(f"🤖 [PINN] Tool Offset 更新:")
    print(f"   向量: {np.round(final_offset_vec, 4)}")
    print("="*45)
    
    # 保存 Offset
    with open(FILE_OFFSET_JSON, 'w') as f:
        json.dump({"tool_offset": list(final_offset_vec)}, f)
    
    # 6. 合并数据并保存
    # 我们要把 指令(含角度) 和 测量(x,y,z) 合并在一行
    df_cmd_final = df_cmd.iloc[valid_cmd_indices].reset_index(drop=True)
    df_meas_final = df_meas.iloc[valid_meas_indices][['x', 'y', 'z']].reset_index(drop=True)
    
    # 重命名测量列为 meas_x, meas_y, meas_z
    df_meas_final.columns = ['meas_x', 'meas_y', 'meas_z']
    
    # 横向合并
    df_out = pd.concat([df_cmd_final, df_meas_final], axis=1)
    
    df_out.to_csv(FILE_TRAIN_FINAL, index=False)
    print(f"💾 最终训练数据已保存至: {FILE_TRAIN_FINAL}")
    print(f"   (包含列: {list(df_out.columns)})")

if __name__ == "__main__":
    main()