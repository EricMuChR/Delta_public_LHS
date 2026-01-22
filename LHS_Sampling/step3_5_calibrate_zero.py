import numpy as np
import pandas as pd
import sys
import os
import json

# ================= 驱动导入 =================
# 获取当前脚本所在目录 (.../LHS_Sampling)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录 (.../Delta_public_LHS)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import DrDelta as robot
except ImportError:
    print("❌ 无法导入 DrDelta，请检查路径。")
    sys.exit(1)

# ================= 导入 Step 3 工具 =================
# 我们需要 fit_circle_3d 来重新计算真实的基座半径 R_base
try:
    from step3_motor_center_align import fit_circle_3d, get_circumcenter
except ImportError:
    print("❌ 无法导入 step3_motor_center_align.py。")
    print("   请确保它在同一目录下 (这是 Step 3 的核心脚本)。")
    sys.exit(1)

# ================= 核心物理参数 (请再次核对) =================
L_ARM_ACTIVE = 100.0   # 主动臂长 (L)
L_ARM_PASSIVE = 250.0  # 从动臂长 (l)
R_END_PLATFORM = 35.0  # 动平台半径 (r_end)

# ================= 文件路径配置 =================
FILE_ARCS = "motor_arcs.csv"        # Step 3 产物: 真实几何数据
FILE_MATRIX = "step3_matrix.npz"    # Step 3 产物: 坐标变换矩阵
FILE_OFFSET = "tool_offset.json"    # Step 4 产物: 工具偏置

# 默认 Offset (如果在 Step 4 还没生成文件时使用)
# 格式: [x, y, z] (法兰中心 -> 靶球中心)
DEFAULT_OFFSET = [-0.2697684, -1.2462669, -31.57363315] 
# ==========================================================

def main():
    print("="*60)
    print("   Step 3.5: 零点几何标定 (Target Guidance)")
    print("   目标：利用几何对称性，恢复机器人的正交坐标系")
    print("="*60)

    # 1. 检查必要文件
    if not os.path.exists(FILE_ARCS) or not os.path.exists(FILE_MATRIX):
        print("❌ 缺少 Step 3 的生成文件 (motor_arcs.csv 或 step3_matrix.npz)")
        print("   -> 请先运行 Step 3 完成基础几何解算。")
        return

    # 2. 读取 Offset (关键步骤)
    # 我们优先读取 Step 4 生成的 json，如果没有则使用默认值
    if os.path.exists(FILE_OFFSET):
        try:
            with open(FILE_OFFSET, 'r') as f:
                offset_data = json.load(f)
                v_offset = np.array(offset_data["tool_offset"])
            print(f"✅ 已加载工具偏置文件: {FILE_OFFSET}")
        except Exception as e:
            print(f"⚠️  Offset文件读取失败 ({e})，使用默认值。")
            v_offset = np.array(DEFAULT_OFFSET)
    else:
        print(f"⚠️  未找到 {FILE_OFFSET}，使用默认历史值 (可行)。")
        v_offset = np.array(DEFAULT_OFFSET)
    
    print(f"   当前使用的 Offset: {v_offset}")

    # 3. 读取矩阵与计算真实 R_base
    print("\n🔄 正在解析 Step 3 数据...")
    
    # 3.1 读取矩阵
    matrix_data = np.load(FILE_MATRIX)
    R_inv = matrix_data['R'] # Tracker -> Robot
    t_vec = matrix_data['t'] 
    # 计算逆矩阵: Robot -> Tracker
    R_fwd = R_inv.T 

    # 3.2 重新计算 R_base (静平台真实半径)
    # 这一步是为了保证 Z 轴高度计算使用的是真实的物理尺寸，而不是理论值
    df_arcs = pd.read_csv(FILE_ARCS)
    centers = []
    col_id = 'motor_id' if 'motor_id' in df_arcs.columns else 'motor'
    try:
        for mid in [1, 2, 3]:
            pts = df_arcs[df_arcs[col_id] == mid][['x','y','z']].values
            if len(pts) < 3: raise ValueError(f"电机{mid}数据不足")
            c, _ = fit_circle_3d(pts)
            centers.append(c)
        
        origin = get_circumcenter(centers[0], centers[1], centers[2])
        R_base_real = np.mean([np.linalg.norm(c - origin) for c in centers])
        print(f"   实测静平台半径 (R_base): {R_base_real:.4f} mm")
    except Exception as e:
        print(f"❌ 计算 R_base 失败: {e}")
        return

    # 4. 计算理论零点 (Robot Coordinate Frame)
    # 当三个主动臂水平 (0度) 时：
    # 水平投影距离 D = (R_base + L) - r_end
    d_proj = (R_base_real + L_ARM_ACTIVE) - R_END_PLATFORM
    
    # 几何合法性检查
    if L_ARM_PASSIVE**2 < d_proj**2:
        print("❌ 致命错误：几何参数不合法！(L_passive 太短，构不成三角形)")
        print(f"   D_proj={d_proj:.2f}, L_passive={L_ARM_PASSIVE}")
        return
    
    # 垂直高度 Z = -sqrt(l^2 - D^2)
    z_drop = np.sqrt(L_ARM_PASSIVE**2 - d_proj**2)
    z_home_robot = -z_drop # 向下为负
    
    # 4.1 法兰中心理论位置
    p_flange_zero = np.array([0.0, 0.0, z_home_robot])
    
    # 4.2 靶球中心理论位置 (叠加 Offset)
    p_smr_zero = p_flange_zero + v_offset
    
    print("-" * 40)
    print(f"📐 机器人内部理论零位:")
    print(f"   法兰高度 (Z_Flange): {z_home_robot:.3f} mm")
    print(f"   靶球高度 (Z_SMR)   : {p_smr_zero[2]:.3f} mm (含Offset)")

    # 5. 转换到激光跟踪仪坐标系 (Tracker Frame)
    # P_tracker = R_fwd * P_robot + t
    p_tracker_target = np.dot(R_fwd, p_smr_zero) + t_vec
    
    print("\n" + "#"*60)
    print("🎯  请将机器人【靶球中心】移动到以下【激光跟踪仪坐标】")
    print("#"*60)
    print(f"   X : {p_tracker_target[0]:.4f}")
    print(f"   Y : {p_tracker_target[1]:.4f}")
    print(f"   Z : {p_tracker_target[2]:.4f}")
    print("#"*60)
    
    # 6. 人工引导环节 (DRO模式)
    print("\n👉 操作指南 (MANUAL JOGGING):")
    print("1. 请查看激光跟踪仪软件 (SA / PolyWorks) 的实时读数窗口。")
    print("2. 手动控制机器人 XYZ，使读数逼近上述目标值。")
    print("   - 优先对齐 X 和 Y (保证居中)")
    print("   - 其次对齐 Z (保证高度)")
    print("   - 推荐误差控制在 0.1mm 以内")
    print("3. 对齐后，观察三个主动臂是否目测水平 (双重验证)。")
    
    print("\n⚠️  警告：按下回车后将强制重置零点，请确保位置准确！")
    input("⌨️  位置已确认？按 [Enter] 键执行设零...")

    # 7. 连接并设零
    print("\n正在连接机器人...")
    try:
        # 使用你之前的参数初始化
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[L_ARM_ACTIVE, L_ARM_PASSIVE, R_END_PLATFORM, 23.4]
        )
        
        print("⚡ 正在写入零点 (set_zero_pose)...")
        # 调用 set_zero_pose，该函数会重置所有关节角度为 0
        if ro.set_zero_pose():
            print("\n✅ 零点设置成功！(Zero Pose Set)")
            print("="*60)
            print("🧨 【必须执行的操作】 🧨")
            print("1. 你的机器人坐标系已改变，旧数据已失效。")
            print("2. 请删除旧文件: 'lhs_final.csv' 和 'tracker_final.csv'。")
            print("3. 立即运行 Step 4，重新采集 10,000 个数据点。")
            print("="*60)
        else:
            print("\n❌ 设零失败 (Driver returned False)。")
            print("   请检查机器人连接或驱动状态。")
            
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")

if __name__ == "__main__":
    main()