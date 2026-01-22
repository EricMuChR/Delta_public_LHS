import pandas as pd
import numpy as np
import os
import json
import sys

# ================= 配置区域 =================
FILE_TARGETS = "points_with_compensation.csv"
FILE_MEASURED_RAW = "validation_500.csv"
FILE_REPORT = "validation_absolute_report.csv"

# 优先寻找的新矩阵 (Step 5.5 生成)
FILE_MATRIX_VAL = "matrix_val.npz"
# 备用的旧矩阵 (Step 3 生成)
FILE_MATRIX_TRAIN = "step3_matrix.npz"

# Offset 文件 (Step 4 生成)
FILE_OFFSET_JSON = "tool_offset.json"

# 匹配阈值 (mm)
MATCH_THRESHOLD = 20.0 
# ===========================================

def load_matrix_and_offset():
    """ 智能加载矩阵和 Offset """
    # 1. 加载 Offset
    if not os.path.exists(FILE_OFFSET_JSON):
        print(f"❌ 致命错误: 找不到 {FILE_OFFSET_JSON}。请检查 Step 4 是否运行。")
        sys.exit(1)
    
    with open(FILE_OFFSET_JSON, 'r') as f:
        data = json.load(f)
        tool_offset = np.array(data["tool_offset"])
    print(f"✅ 已加载 Tool Offset: {tool_offset}")

    # 2. 加载 Matrix
    matrix_file = None
    if os.path.exists(FILE_MATRIX_VAL):
        print(f"✅ 检测到新矩阵 (验证专用): {FILE_MATRIX_VAL}")
        matrix_file = FILE_MATRIX_VAL
    elif os.path.exists(FILE_MATRIX_TRAIN):
        print(f"⚠️ 警告: 未找到验证矩阵，回退使用训练矩阵: {FILE_MATRIX_TRAIN}")
        print("   (如果跟踪仪移动过，结果将完全错误！)")
        matrix_file = FILE_MATRIX_TRAIN
    else:
        print("❌ 找不到任何坐标变换矩阵 (.npz)。")
        sys.exit(1)
        
    matrix_data = np.load(matrix_file)
    return matrix_data['R'], matrix_data['t'], tool_offset

def main():
    print("="*60)
    print("   Step 7: 绝对定位精度验证 (双指针同步版)")
    print("="*60)
    
    # 1. 加载环境参数
    R_inv, t, tool_offset = load_matrix_and_offset()
    
    # 2. 加载目标点 (Target)
    if not os.path.exists(FILE_TARGETS):
        print(f"❌ 找不到目标文件: {FILE_TARGETS}")
        return
    df_targets = pd.read_csv(FILE_TARGETS)
    
    # 注意：验证时对比的是“原始意图(orig)”和“实际到达”
    if 'orig_x' in df_targets.columns:
        pts_target = df_targets[['orig_x', 'orig_y', 'orig_z']].values
    else:
        print("⚠️ 找不到 orig_x 列，尝试使用 x/y/z 作为目标。")
        pts_target = df_targets[['x', 'y', 'z']].values

    # 3. 加载实测数据 (Measured)
    if not os.path.exists(FILE_MEASURED_RAW):
        print(f"❌ 找不到实测数据: {FILE_MEASURED_RAW}")
        return
    df_meas = pd.read_csv(FILE_MEASURED_RAW)
    try:
        pts_meas_raw = df_meas[['x', 'y', 'z']].values
    except KeyError:
        pts_meas_raw = df_meas.iloc[:, -3:].values # 兼容无表头情况
    
    print(f"📊 数据加载: Target={len(pts_target)}, Measured={len(pts_meas_raw)}")

    # 4. 坐标转换 (Tracker -> Robot)
    print("🔄 应用坐标变换...")
    # P_robot = R_inv @ (P_tracker - t)
    pts_meas_robot = np.dot(R_inv, (pts_meas_raw - t).T).T
    
    # 5. 扣除 Tool Offset
    print("🛠️ 扣除工具偏置...")
    # 这里直接应用我们已知且固定的 tool_offset
    pts_meas_flange = pts_meas_robot - tool_offset

    # 6. 双指针数据同步 (Step 4 Logic)
    print("🔍 开始双指针匹配同步...")
    
    valid_errors = []
    valid_pairs_count = 0
    
    cmd_cursor = 0
    meas_cursor = 0
    
    # 双指针循环
    while meas_cursor < len(pts_meas_flange) and cmd_cursor < len(pts_target):
        p_meas = pts_meas_flange[meas_cursor]
        p_cmd = pts_target[cmd_cursor]
        
        # 计算当前对的距离
        dist = np.linalg.norm(p_meas - p_cmd)
        
        if dist < MATCH_THRESHOLD:
            # === 匹配成功 ===
            valid_errors.append(dist)
            valid_pairs_count += 1
            meas_cursor += 1
            cmd_cursor += 1
        else:
            # === 匹配失败，判断是谁的问题 ===
            # 检查下一个测量点是否离当前指令更近？
            # 如果是，说明当前测量点是多余的/噪音，应该跳过测量点
            if meas_cursor + 1 < len(pts_meas_flange):
                dist_next = np.linalg.norm(pts_meas_flange[meas_cursor+1] - p_cmd)
                if dist_next < dist:
                    # 下一个测量点更好，说明当前测量点无效
                    meas_cursor += 1
                    continue
            
            # 否则，说明当前指令对应的测量点丢失了（漏采），跳过当前指令
            cmd_cursor += 1

    print(f"✅ 匹配完成。有效点对: {valid_pairs_count} / {len(pts_target)}")

    if valid_pairs_count < 10:
        print("❌ 匹配数量过少，请检查 MATCH_THRESHOLD 或坐标系矩阵是否正确。")
        return

    # 7. 报告生成
    errors = np.array(valid_errors)
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    print("-" * 50)
    print(f"🏆 最终验证结果 (RMSE):")
    print(f"   RMSE     : {rmse:.4f} mm")
    print(f"   Mean Err : {mean_err:.4f} mm")
    print(f"   Max Error: {max_err:.4f} mm")
    print("-" * 50)
    
    # 保存 CSV 报告
    with open(FILE_REPORT, "w") as f:
        f.write(f"metric,value,unit\nrmse,{rmse:.4f},mm\nmax,{max_err:.4f},mm\nmean,{mean_err:.4f},mm\n")
    print(f"📄 报告已保存: {FILE_REPORT}")

if __name__ == "__main__":
    main()