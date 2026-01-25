import pandas as pd
import numpy as np
import os
import sys
import time
from scipy.stats import qmc # 导入LHS库

# ================= 配置 =================
OUTPUT_COMMANDS_CSV = "pidl_verification_commands.csv" 
VALIDATION_COUNT = 500 
MAX_RADIUS_GEN = 100.0  
Z_RANGE = [-240, -190]

# ⚠️ 机器人物理限位 (必须与驱动代码一致)
# 稍微留一点余量 (比如限位是 -42，我们设 -41)
JOINT_LIMIT_MIN = -41.0
JOINT_LIMIT_MAX = 89.0
# =======================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase6_inference import PIDL_Inference_Engine
except ImportError:
    print("❌ 无法导入 PIDL 引擎，请检查 phase6_inference.py 是否存在")
    sys.exit(1)

def generate_lhs_candidates(count, r_max, z_min, z_max):
    """ 生成一批候选点 (可能会多生成一些，用于筛选) """
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=count)
    points = []
    for s in sample:
        u_r, u_theta, u_z = s
        r = r_max * np.sqrt(u_r)
        theta = u_theta * 2 * np.pi
        z = z_min + u_z * (z_max - z_min)
        points.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(points)

def is_safe_angles(angles_deg):
    """ 检查角度是否在安全范围内 """
    for theta in angles_deg:
        if theta < JOINT_LIMIT_MIN or theta > JOINT_LIMIT_MAX:
            return False
    return True

def main():
    print("="*60)
    print(f"   Phase 7 (Step 1): Generate Safe PIDL Points (LHS + Safety Check)")
    print("="*60)

    # 1. 加载模型
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    if not os.path.exists(model_path):
        print("❌ 没找到 PIDL 模型文件。")
        return

    engine = PIDL_Inference_Engine(model_path)

    # 2. 循环生成直到凑齐 VALIDATION_COUNT 个安全点
    valid_commands = []
    attempts = 0
    max_attempts = 10  # 防止死循环
    
    needed = VALIDATION_COUNT
    
    print(f"🎯 目标: 生成 {needed} 个在关节限位 [{JOINT_LIMIT_MIN}, {JOINT_LIMIT_MAX}] 内的点...")

    while len(valid_commands) < VALIDATION_COUNT and attempts < max_attempts:
        attempts += 1
        # 每次多生成 50% 的候选点，用来剔除坏点
        batch_size = int(needed * 1.5) + 10
        candidates = generate_lhs_candidates(batch_size, MAX_RADIUS_GEN, Z_RANGE[0], Z_RANGE[1])
        
        print(f"   [Batch {attempts}] 生成 {len(candidates)} 个候选点进行逆解筛选...")
        
        for target in candidates:
            if len(valid_commands) >= VALIDATION_COUNT:
                break
                
            # PIDL 逆解
            angles_deg, _, _ = engine.inverse_solve(target)
            
            # 安全检查
            if is_safe_angles(angles_deg):
                # 记录有效点: [target_x, target_y, target_z, t1, t2, t3]
                entry = list(target) + list(angles_deg)
                valid_commands.append(entry)
            else:
                # 默默丢弃，不打印，避免刷屏
                pass
                
        needed = VALIDATION_COUNT - len(valid_commands)
        print(f"   -> 当前有效点数: {len(valid_commands)}/{VALIDATION_COUNT}")

    if len(valid_commands) < VALIDATION_COUNT:
        print(f"⚠️ 警告: 尝试 {max_attempts} 次后仍未凑齐 500 个点。")
        print("   建议缩小生成半径 MAX_RADIUS_GEN 或调整 Z_RANGE。")
    
    # 3. 保存结果
    # 将 list 转为 DataFrame
    valid_commands = np.array(valid_commands)
    df_out = pd.DataFrame(valid_commands, columns=['target_x', 'target_y', 'target_z', 'theta1', 'theta2', 'theta3'])
    df_out = df_out.round(4)
    
    df_out.to_csv(OUTPUT_COMMANDS_CSV, index=False)
    print(f"\n💾 安全指令已保存至: {OUTPUT_COMMANDS_CSV}")
    print(f"📊 最终点数: {len(df_out)}")
    print("👉 这些点已全部通过 PIDL 逆解验证，且保证不撞限位。请去跑 phase7_robot_control.py 吧！")

if __name__ == "__main__":
    main()