import pandas as pd
import time
import sys
import os
import csv
import DrDelta as robot # 确保 DrDelta 在同级目录下

# ================= 配置 =================
POINTS_FILE = "lhs_points.csv"
OUTPUT_FILE = "data_angles_collected.csv"
# =======================================

def load_points(filename=POINTS_FILE):
    if not os.path.exists(filename):
        print(f"错误: 找不到 {filename}。")
        sys.exit(1)
    return pd.read_csv(filename).values.tolist()

def init_robot():
    print("正在初始化机器人...")
    try:
        # 使用你原本的参数初始化
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[100, 250, 35, 23.4]
        )
        # 初始回零/安全位置
        ro.set_position(tip_x_y_z=[0, 0, -210], speed=10, acceleration=10)
        ro.position_done()
        print("机器人就绪。")
        return ro
    except Exception as e:
        print(f"连接失败: {e}")
        return None

def move_and_record(ro, idx, target, csv_writer, use_pre_point):
    x, y, z = target
    
    # 1. 运动控制逻辑
    if use_pre_point:
        pre_z = min(z + 5, -185) # 简单限位保护
        ro.set_position(tip_x_y_z=[x, y, pre_z], speed=500, acceleration=300)
        ro.position_done()
        # 下压
        ro.set_position(tip_x_y_z=target, speed=250, acceleration=300)
    else:
        ro.set_position(tip_x_y_z=target, speed=500, acceleration=300)
    
    # 等待物理稳定
    ro.position_done()
    time.sleep(0.5) 

    # 2. 【关键修改】读取关节角度 (PINN 输入)
    # 尝试读取多次以确保数据有效
    angles = None
    for retry in range(3):
        angles = ro.get_current_model_angles() # 调用我们在 DrDelta 中新增的函数
        if angles: break
        time.sleep(0.1)
    
    if angles is None:
        print(f"⚠️ 第 {idx} 点角度读取失败！")
        angles = [0.0, 0.0, 0.0] # 填充占位符，避免程序崩溃
    else:
        # 打印部分信息确认运行正常
        print(f"   -> 角度: [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]")

    # 3. 写入数据文件
    # 格式: [Index, Cmd_X, Cmd_Y, Cmd_Z, Theta1, Theta2, Theta3]
    csv_writer.writerow([idx, x, y, z, angles[0], angles[1], angles[2]])
    
    return angles

def main():
    points = load_points()
    total = len(points)
    print(f"已加载 {total} 个采样点。")

    # 交互设置
    mode_in = input("运动策略 [1]直接 [2]竖直下压 (默认2): ").strip()
    use_pre = (mode_in != '1')
    
    start_idx_str = input(f"起始点序号 (0-{total-1}, 默认0): ").strip()
    start_idx = int(start_idx_str) if start_idx_str.isdigit() else 0

    ro = init_robot()
    if not ro: return

    # 初始化 CSV 文件
    # 如果是断点续传（start_idx > 0），则使用追加模式 'a'，否则覆盖 'w'
    file_mode = 'a' if start_idx > 0 else 'w'
    
    print(f"\n准备开始采集。数据将保存至: {OUTPUT_FILE}")
    print("注意：请同时操作激光跟踪仪进行测量（如果跟踪仪未与此脚本联动）。")
    input("按回车键开始...")

    with open(OUTPUT_FILE, file_mode, newline='') as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入表头
        if start_idx == 0:
            writer.writerow(['Index', 'Cmd_X', 'Cmd_Y', 'Cmd_Z', 'Theta1', 'Theta2', 'Theta3'])

        try:
            for i, target in enumerate(points[start_idx:], start=start_idx):
                print(f"\n[Progress] {i}/{total-1} -> Target: {target}")
                
                # 执行运动并记录角度
                move_and_record(ro, i, target, writer, use_pre)
                
                # 强制刷新缓冲区，防止程序中断丢失数据
                f.flush() 
                
                # 激光跟踪仪测量时间窗口
                print("⏳ 等待测量...")
                time.sleep(1.85)

            print("\n🎉 采集完成！")
            
        except KeyboardInterrupt:
            print("\n🛑 用户手动中止。数据已保存。")

if __name__ == "__main__":
    main()