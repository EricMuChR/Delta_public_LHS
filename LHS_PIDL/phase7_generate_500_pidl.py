import pandas as pd
import numpy as np
import os
import sys
import time
from scipy.stats import qmc # 导入LHS库

# ================= 配置 =================
OUTPUT_COMMANDS_CSV = "pidl_verification_commands.csv" # 输出给机器人的指令
VALIDATION_COUNT = 500 
MAX_RADIUS_GEN = 100.0  # 生成半径 R=100mm
Z_RANGE = [-240, -190]  # Z轴范围 [-240, -190]
# =======================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase6_inference import PIDL_Inference_Engine
except ImportError:
    print("❌ 无法导入 PIDL 引擎，请检查 phase6_inference.py 是否存在")
    sys.exit(1)

def generate_lhs_points(count, r_max, z_min, z_max):
    """ 使用拉丁超立方采样 (LHS) 生成均匀分布的圆柱空间测试点 """
    print(f"1. 正在使用 LHS 生成 {count} 个全空间覆盖点 (R <= {r_max}mm)...")
    
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=count) # 生成 [0, 1) 区间的均匀样本
    
    points = []
    for s in sample:
        u_r, u_theta, u_z = s
        # 映射 r: r = R * sqrt(u) 保证点在圆面上分布均匀
        r = r_max * np.sqrt(u_r)
        # 映射 theta: 0 ~ 2pi
        theta = u_theta * 2 * np.pi
        # 映射 z: z_min ~ z_max
        z = z_min + u_z * (z_max - z_min)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    
    return np.array(points)

def main():
    print("="*60)
    print(f"   Phase 7 (Step 1): Generate {VALIDATION_COUNT} PIDL Points (LHS)")
    print("="*60)

    # 1. 加载 PIDL 模型
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    if not os.path.exists(model_path):
        print("❌ 没找到 PIDL 模型文件，请先跑 Phase 5。")
        return

    engine = PIDL_Inference_Engine(model_path)

    # 2. 生成 LHS 采样点
    targets = generate_lhs_points(VALIDATION_COUNT, MAX_RADIUS_GEN, Z_RANGE[0], Z_RANGE[1])

    # 3. 使用 PIDL 引擎进行逆解 (Inverse Solve)
    print("2. 正在使用 PIDL 引擎计算精准电机角度...")
    motor_commands = []
    
    start_time = time.time()
    
    for i, target in enumerate(targets):
        # PIDL 核心：数值迭代求解最优角度
        angles_deg, _, _ = engine.inverse_solve(target)
        motor_commands.append(angles_deg)
        
        if (i+1) % 50 == 0:
            print(f"   已处理 {i+1}/{len(targets)} 个点...")

    total_time = time.time() - start_time
    print(f"✅ 计算完成! 平均耗时: {total_time/len(targets)*1000:.1f} ms/point")

    # 4. 保存结果
    df_out = pd.DataFrame({
        'target_x': targets[:, 0],
        'target_y': targets[:, 1],
        'target_z': targets[:, 2],
        'theta1': [cmd[0] for cmd in motor_commands],
        'theta2': [cmd[1] for cmd in motor_commands],
        'theta3': [cmd[2] for cmd in motor_commands]
    }).round(4)
    
    df_out.to_csv(OUTPUT_COMMANDS_CSV, index=False)
    print(f"\n💾 指令已保存至: {OUTPUT_COMMANDS_CSV}")
    print(f"📊 包含点数: {len(df_out)}")
    print("👉 下一步：请运行 phase7_robot_control.py 执行这些指令。")

if __name__ == "__main__":
    main()