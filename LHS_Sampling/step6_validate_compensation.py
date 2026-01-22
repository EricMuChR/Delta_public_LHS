import pandas as pd
import time
import sys
import os

# ================= 驱动导入 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import DrDelta as robot
# ==========================================

# ================= 配置 =================
# 只读取算法仓库生成的补偿文件
FILE_COMP_POINTS = "points_with_compensation.csv" 
# =======================================

def check_matrix_validity():
    """ 验证前的矩阵时效性检查 """
    print("\n" + "="*60)
    print("   ⚠️  验证前环境检查 (Pre-Flight Check)")
    print("="*60)
    
    if not os.path.exists("step3_matrix.npz"):
        print("❌ 错误：找不到 'step3_matrix.npz'。")
        print("   -> 请先运行 Step 3 生成坐标变换矩阵，否则无法进行绝对精度验证。")
        sys.exit(1)
        
    # 读取文件修改时间
    mtime = os.path.getmtime("step3_matrix.npz")
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    
    print(f"当前使用的坐标变换矩阵生成于: 【{time_str}】")
    print("-" * 40)
    print("请确认：在该时间点之后，激光跟踪仪 或 机器人底座 是否移动过？")
    print(" [1] 没动过，矩阵有效 -> 继续执行")
    print(" [2] 动过/不确定 -> 退出并重新标定")
    print("-" * 40)
    
    choice = input("请输入选择 (1/2): ").strip()
    if choice != '1':
        print("\n🚫 停止执行。")
        print("操作建议：")
        print("1. 让机器人画圆采集 'motor_arcs.csv'。")
        print("2. 运行 'step3_motor_center_align.py' 更新矩阵。")
        print("3. 回来重新运行本程序。")
        sys.exit(0)
    print("✅ 确认矩阵有效，准备启动机器人...\n")

def main():
    # 1. 启动检查
    check_matrix_validity()
    
    print("="*60)
    print("   Step 6: 补偿验证 (离线数据驱动版)")
    print("="*60)

    # 2. 读取补偿指令表
    if not os.path.exists(FILE_COMP_POINTS):
        print(f"❌ 找不到文件: {FILE_COMP_POINTS}")
        print("请在算法仓库运行 export_compensated_points.py，并将生成的 csv 复制到这里。")
        return
    
    df = pd.read_csv(FILE_COMP_POINTS)
    print(f"✅ 已加载 {len(df)} 个补偿点。")

    # 3. 初始化机器人
    print("\n正在连接机器人...")
    try:
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[100, 250, 35, 23.4]
        )
        ro.set_position(tip_x_y_z=[0, 0, -210], speed=10, acceleration=10)
        ro.position_done()
    except Exception as e:
        print(f"连接失败: {e}")
        return

    input("\n按 Enter 开始执行验证流程...")

    # 4. 执行循环
    for i, row in df.iterrows():
        # 取出补偿后的坐标 (这就是机器人实际上要执行的指令)
        target_comp = [row['comp_x'], row['comp_y'], row['comp_z']]
        
        # 取出原始坐标 (仅用于显示)
        target_orig = [row['orig_x'], row['orig_y'], row['orig_z']]
        
        print(f"[{i+1}/{len(df)}]")
        print(f"  原始意图: {target_orig}")
        print(f"  实际执行: {target_comp}")
        
        # 发送补偿后的指令
        ro.set_position(tip_x_y_z=target_comp, speed=500, acceleration=300)
        ro.position_done()
        
        print("  ⏳ 等待采集...")
        time.sleep(2)  # 给激光跟踪仪留出时间采集数据
        
    print("\n🎉 验证执行完毕！")
    print("请导出激光跟踪仪数据为 'validation_500.csv' 并运行 Step 7。")

if __name__ == "__main__":
    main()