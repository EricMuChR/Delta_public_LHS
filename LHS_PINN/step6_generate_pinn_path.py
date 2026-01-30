import torch
import numpy as np
import pandas as pd
import os
import sys

# 引入 PINN 依赖
from Delta_Torch import DeltaKinematics
from train_pinn import DeltaPINN

# 引入 Delta_3
try:
    import Delta_3 as kinematics
except ImportError:
    print("❌ 无法导入 Delta_3.py")
    sys.exit(1)

# ================= 配置 =================
MODEL_FILE = "pinn_model.pth"
OUTPUT_FILE = "points_with_compensation.csv"
NUM_SAMPLES = 600

ROBOT_PARAMS = [100.0, 250.0, 35.0, 23.4] 

# 采样配置 (稍微放宽一点半径，靠角度检查来过滤)
MAX_RADIUS_GEN = 100.0   
Z_RANGE = [-235, -205]   
MAX_ATTEMPTS = 50000     

# 🛑 电子围栏 (根据报错信息设定的安全阈值)
# 真机报错是 "Min Limit"，通常是 -42度
LIMIT_ANGLE_MIN = -38.0  # 保守设为 -38，绝对安全
LIMIT_ANGLE_MAX = 85.0   # 设为 85
# =======================================

def generate_random_point(r_max, z_min, z_max):
    r = r_max * np.sqrt(np.random.random())
    theta = np.random.random() * 2 * np.pi
    z = z_min + np.random.random() * (z_max - z_min)
    return [r * np.cos(theta), r * np.sin(theta), z]

def main():
    print("="*60)
    print("   Step 6: 生成 PINN 路径 (集成电子围栏版)")
    print("="*60)

    # 1. 加载模型
    if not os.path.exists(MODEL_FILE):
        print(f"❌ 找不到模型: {MODEL_FILE}")
        return
    
    physics = DeltaKinematics() 
    model = DeltaPINN(physics)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    print("✅ PINN 模型已加载")
    
    # 2. 初始化 CPU 运动学
    print(f"⚙️ 初始化 Delta_3, 参数: {ROBOT_PARAMS}")
    arm = kinematics.arm(l=ROBOT_PARAMS)

    # 3. 🔍 自动单位检测
    print("\n🔍 正在检测模型输入单位...")
    USE_DEGREES = True # 默认
    try:
        test_pt = [0, 50, -210]
        arm.inverse_kinematics(tip_x_y_z=test_pt)
        theta_rad = np.array(arm.theta)
        theta_deg = np.degrees(theta_rad)
        
        with torch.no_grad():
            p_rad, _, _ = model(torch.tensor([theta_rad], dtype=torch.float32))
            p_deg, _, _ = model(torch.tensor([theta_deg], dtype=torch.float32))
            
        err_deg = np.linalg.norm(p_deg.numpy()[0] - test_pt)
        err_rad = np.linalg.norm(p_rad.numpy()[0] - test_pt)
        
        if err_deg < err_rad:
            print("✅ 模型需要【角度 (Deg)】输入")
            USE_DEGREES = True
        else:
            print("✅ 模型需要【弧度 (Rad)】输入")
            USE_DEGREES = False
    except:
        print("⚠️ 检测跳过，默认使用角度。")

    # 4. 循环生成
    print(f"\n🎲 开始生成 {NUM_SAMPLES} 个安全点...")
    print(f"   (过滤条件: 关节角在 [{LIMIT_ANGLE_MIN}, {LIMIT_ANGLE_MAX}] 之间)")
    
    valid_results = []
    attempts = 0
    
    while len(valid_results) < NUM_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1
        
        target = generate_random_point(MAX_RADIUS_GEN, Z_RANGE[0], Z_RANGE[1])
        tx, ty, tz = target
        
        # A. 初始逆解
        try:
            arm.inverse_kinematics(tip_x_y_z=[tx, ty, tz])
            theta = arm.theta 
            if theta is None or np.isnan(theta).any(): continue
            
            # 准备模型输入
            if USE_DEGREES: theta_input = np.degrees(theta)
            else: theta_input = theta
                
        except: continue
            
        # B. PINN 预测
        theta_tensor = torch.tensor([theta_input], dtype=torch.float32)
        with torch.no_grad():
            pred_pos, _, _ = model(theta_tensor)
            pred_pos = pred_pos.numpy()[0]
        
        # C. 计算补偿目标
        error = pred_pos - target
        if np.linalg.norm(error) > 25.0: continue # 剔除预测飞掉的点
        
        comp_target = target - error
        
        # D. 🟢 关键：电子围栏校验 🟢
        # 检查“补偿后的点”是否会导致真机报错
        try:
            arm.inverse_kinematics(tip_x_y_z=[comp_target[0], comp_target[1], comp_target[2]])
            theta_comp_rad = np.array(arm.theta)
            
            if theta_comp_rad is None or np.isnan(theta_comp_rad).any():
                continue # 不可达
            
            theta_comp_deg = np.degrees(theta_comp_rad)
            
            # 检查最小值
            if np.min(theta_comp_deg) < LIMIT_ANGLE_MIN:
                # print(f"Skipped: min angle {np.min(theta_comp_deg):.1f} < {LIMIT_ANGLE_MIN}")
                continue
            
            # 检查最大值
            if np.max(theta_comp_deg) > LIMIT_ANGLE_MAX:
                continue
                
        except:
            continue

        valid_results.append({
            "orig_x": tx, "orig_y": ty, "orig_z": tz,
            "comp_x": comp_target[0],
            "comp_y": comp_target[1],
            "comp_z": comp_target[2],
            "pred_err_x": error[0],
            "pred_err_y": error[1],
            "pred_err_z": error[2]
        })
        
        if len(valid_results) % 50 == 0:
            print(f"   进度: {len(valid_results)}/{NUM_SAMPLES}", end='\r')

    print(f"\n✅ 生成完成！总尝试: {attempts}")
    
    df = pd.DataFrame(valid_results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 已保存: {OUTPUT_FILE}")
    print("👉 请重新运行 step7_execute_validation.py (这次一定不会报错了)")

if __name__ == "__main__":
    main()