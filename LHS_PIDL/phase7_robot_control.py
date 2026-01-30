import pandas as pd
import time
import sys
import os

# ================= 驱动导入 =================
# 假设 DrDelta 在上一级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import DrDelta as robot
except ImportError:
    print("❌ 无法导入 DrDelta 库，请检查路径。")
    sys.exit(1)
# ==========================================

# ================= 配置 =================
# 读取由 Step 1 生成的指令文件
FILE_COMMANDS = "pidl_verification_commands.csv"
# =======================================

def main():
    print("="*60)
    print("   Phase 7 (Step 2): Robot Execution (PIDL Joint Control)")
    print("="*60)

    # 1. 读取指令表
    if not os.path.exists(FILE_COMMANDS):
        print(f"❌ 找不到指令文件: {FILE_COMMANDS}")
        print("   -> 请先运行 'phase7_generate_500_pidl.py'。")
        return
    
    df = pd.read_csv(FILE_COMMANDS)
    print(f"✅ 已加载 {len(df)} 个 PIDL 验证点。")
    
    if not all(col in df.columns for col in ['theta1', 'theta2', 'theta3']):
        print("❌ CSV 格式错误，缺少 theta 列。")
        return

    # 2. 初始化机器人
    print("\n🔌 正在连接机器人...")
    try:
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[100, 250, 35, 23.4]
        )
        # 先回个安全位置
        print("   移动到安全起始点 [0, 0, -210]...")
        ro.set_position(tip_x_y_z=[0, 0, -210], speed=20, acceleration=20)
        time.sleep(3) 
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print("\n✅ 机器人就绪。")
    input("👉 按 Enter 键开始执行 500 点验证流程 (请确保激光跟踪仪已就绪)...")

    # 3. 执行循环
    start_time = time.time()
    
    for i, row in df.iterrows():
        # 取出 PIDL 计算出的“黄金角度”
        angles = [row['theta1'], row['theta2'], row['theta3']]
        target_pos = [row['target_x'], row['target_y'], row['target_z']]
        
        print(f"[{i+1}/{len(df)}] Target: {target_pos}")
        print(f"   -> PIDL Joints: {angles}")
        
        # === 核心: 直接发送关节角度 ===
        ro.set_joints(angle_list=angles, speed=500, acceleration=200) # 速度可调
        ro.position_done()
        
        # 等待运动到位 + 稳定 (根据你的Tracker采集速度调整)
        time.sleep(2.5) 
        
        print("   📡 采集窗口 (Tracker Recording...)")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"🎉 验证执行完毕！耗时: {total_time/60:.1f} min")
    print("="*60)
    print("👉 请保存激光跟踪仪数据为 'pidl_measured_raw.csv' (在 LHS_PIDL 目录下)。")
    print("👉 然后运行 'phase7_verification_loop.py' 进行画图分析。")

if __name__ == "__main__":
    main()