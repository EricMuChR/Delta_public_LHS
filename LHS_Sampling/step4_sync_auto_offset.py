import pandas as pd
import numpy as np
import os
import json # 新增

# ================= 配置 =================
FILE_CMD_ORIGINAL = "lhs_points.csv"
FILE_MEAS_ALIGNED = "aligned_measurement_motor.csv" 

FILE_CMD_FINAL = "lhs_final.csv"
FILE_MEAS_FINAL = "tracker_final.csv"

# 新增：Offset 保存路径
FILE_OFFSET_JSON = "tool_offset.json" 

MATCH_THRESHOLD = 35.0 
# =======================================

def main():
    print("="*60)
    print("   Step 4: 数据同步 & Offset 自动提取")
    print("="*60)

    # 1. 读取数据
    if not os.path.exists(FILE_CMD_ORIGINAL) or not os.path.exists(FILE_MEAS_ALIGNED):
        print("❌ 找不到输入文件。")
        return

    df_cmd = pd.read_csv(FILE_CMD_ORIGINAL)
    pts_cmd = df_cmd[['x', 'y', 'z']].values

    df_meas = pd.read_csv(FILE_MEAS_ALIGNED)
    pts_meas = df_meas[['x', 'y', 'z']].values

    # 2. 重心对齐 (Pre-alignment)
    centroid_cmd = np.mean(pts_cmd, axis=0)
    centroid_meas = np.mean(pts_meas, axis=0)
    robust_offset = centroid_meas - centroid_cmd
    pts_meas_shifted = pts_meas - robust_offset
    
    # 3. 智能匹配 (Auto Sync)
    print("🔍 开始匹配同步...")
    valid_cmd_indices = []
    valid_meas_indices = []
    cmd_cursor = 0
    meas_cursor = 0
    
    while meas_cursor < len(pts_meas) and cmd_cursor < len(pts_cmd):
        p_meas = pts_meas_shifted[meas_cursor]
        p_cmd = pts_cmd[cmd_cursor]
        dist = np.linalg.norm(p_meas - p_cmd)
        
        if dist < MATCH_THRESHOLD:
            valid_meas_indices.append(meas_cursor)
            valid_cmd_indices.append(cmd_cursor)
            meas_cursor += 1
            cmd_cursor += 1
        else:
            if meas_cursor + 1 < len(pts_meas):
                dist_next = np.linalg.norm(pts_meas_shifted[meas_cursor+1] - p_cmd)
                if dist_next < dist:
                    meas_cursor += 1
                    continue
            cmd_cursor += 1

    matched_count = len(valid_cmd_indices)
    print(f"✅ 匹配完成。有效点对: {matched_count}")

    if matched_count < 100:
        print("❌ 匹配数量太少，请检查数据。")
        return

    # 4. 精确 Offset 计算
    p_c_matched = pts_cmd[valid_cmd_indices]
    p_m_matched = pts_meas[valid_meas_indices]
    
    final_offset = np.mean(p_m_matched - p_c_matched, axis=0)
    offset_list = list(np.round(final_offset, 4))
    
    print("\n" + "="*45)
    print(f"🤖 [重要] Tool Offset 已计算:")
    print(f"   向量: {offset_list}")
    print("="*45)
    
    # 5. 保存 Offset 到 JSON (关键新增)
    with open(FILE_OFFSET_JSON, 'w') as f:
        json.dump({"tool_offset": list(final_offset)}, f)
    print(f"💾 Offset 已自动保存至: {FILE_OFFSET_JSON} (Step 7 将使用此文件)")
    
    # 6. 保存最终文件
    df_cmd.iloc[valid_cmd_indices].to_csv(FILE_CMD_FINAL, index=False)
    df_meas.iloc[valid_meas_indices].to_csv(FILE_MEAS_FINAL, index=False)
    print(f"📂 训练数据已就绪。")

if __name__ == "__main__":
    main()