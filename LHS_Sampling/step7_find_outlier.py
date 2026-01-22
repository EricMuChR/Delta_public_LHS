import pandas as pd
import numpy as np
import os
import json

# ================= 配置 =================
FILE_TARGETS = "points_with_compensation.csv"
FILE_MEASURED_RAW = "validation_500.csv" 
FILE_CLEAN_OUTPUT = "validation_500_clean.csv" # 输出的新文件名

FILE_MATRIX_VAL = "matrix_val.npz"
FILE_OFFSET_JSON = "tool_offset.json"
MATCH_THRESHOLD = 20.0 
# =======================================

def main():
    print("="*60)
    print("   Step 7 Extra: 离群点抓捕与剔除")
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
    else:
        pts_target = df_targets[['x', 'y', 'z']].values

    df_meas = pd.read_csv(FILE_MEASURED_RAW)
    try:
        pts_meas_raw = df_meas[['x', 'y', 'z']].values
    except KeyError:
        pts_meas_raw = df_meas.iloc[:, -3:].values

    # 3. 坐标转换
    pts_meas_robot = np.dot(R_inv, (pts_meas_raw - t).T).T
    pts_meas_flange = pts_meas_robot - tool_offset

    # 4. 带索引的双指针匹配
    # 我们需要记录下每个误差对应的"原始行号"
    errors_with_index = [] # 存 [error, meas_original_index, target_index]
    
    cmd_cursor = 0
    meas_cursor = 0
    
    while meas_cursor < len(pts_meas_flange) and cmd_cursor < len(pts_target):
        p_meas = pts_meas_flange[meas_cursor]
        p_cmd = pts_target[cmd_cursor]
        dist = np.linalg.norm(p_meas - p_cmd)
        
        if dist < MATCH_THRESHOLD:
            # 记录误差和它在原始文件中的行号 (meas_cursor)
            errors_with_index.append({
                "error": dist,
                "meas_idx": meas_cursor,
                "cmd_idx": cmd_cursor,
                "coord": pts_meas_raw[meas_cursor]
            })
            meas_cursor += 1
            cmd_cursor += 1
        else:
            if meas_cursor + 1 < len(pts_meas_flange):
                dist_next = np.linalg.norm(pts_meas_flange[meas_cursor+1] - p_cmd)
                if dist_next < dist:
                    meas_cursor += 1
                    continue
            cmd_cursor += 1

    if not errors_with_index:
        print("❌ 没有找到有效匹配点。")
        return

    # 5. 抓捕嫌疑人 (按误差从大到小排序)
    sorted_errors = sorted(errors_with_index, key=lambda x: x["error"], reverse=True)
    top_1 = sorted_errors[0]
    
    print(f"\n🕵️ 抓捕结果：最大误差嫌疑人")
    print("-" * 30)
    print(f"ERROR     : {top_1['error']:.4f} mm")
    print(f"原始行号  : 第 {top_1['meas_idx']} 行 (Excel中显示为第 {top_1['meas_idx']+2} 行)")
    print(f"原始坐标  : {top_1['coord']}")
    print("-" * 30)
    
    # 顺便看看 Top 5
    print("\n🔍 误差 Top 5 名单:")
    for i, item in enumerate(sorted_errors[:5]):
        print(f"#{i+1}: Error={item['error']:.4f}mm | Row={item['meas_idx']}")

    # 6. 自动剔除并保存
    print("\n" + "="*60)
    confirm = input(f"❓ 是否删除误差最大的点 (Row {top_1['meas_idx']}) 并生成新文件? (y/n): ").strip().lower()
    
    if confirm == 'y':
        # 剔除最大误差的那一行
        # 注意：如果有多个非常大的离群点，你可能想多删几个。
        # 这里演示只删最大的一个。
        bad_index = top_1['meas_idx']
        
        # Drop
        df_clean = df_meas.drop(index=bad_index)
        
        # Save
        df_clean.to_csv(FILE_CLEAN_OUTPUT, index=False)
        print(f"\n✅ 已生成清洗后的文件: {FILE_CLEAN_OUTPUT}")
        print(f"👉 下一步: 请修改 Step 7 代码，将输入文件改为 '{FILE_CLEAN_OUTPUT}'，然后重新算分！")
    else:
        print("操作取消。")

if __name__ == "__main__":
    main()