import pandas as pd
import time
import sys
import os

# ================= 跨文件夹导入驱动 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ==================================================

import DrDelta as robot

def load_points(filename="lhs_points.csv"):
    if not os.path.exists(filename):
        print(f"错误: 找不到 {filename}，请先运行 step1。")
        sys.exit(1)
    return pd.read_csv(filename).values.tolist()

def init_robot():
    print("正在初始化机器人...")
    try:
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[100, 250, 35, 23.4]
        )
        ro.set_position(tip_x_y_z=[0, 0, -210], speed=10, acceleration=10)
        ro.position_done()
        print("机器人就绪。")
        return ro
    except Exception as e:
        print(f"连接失败: {e}")
        return None

def move_to_point(ro, target, use_pre_point, speed=500, acc=300):
    x, y, z = target
    
    if use_pre_point:
        # 预备点策略: 上方5mm
        pre_z = z + 5
        if pre_z > -185: pre_z = -185 # 简单保护
        pre_point = [x, y, pre_z]
        
        print(f"  >> 1. 前往预备点: {pre_point}")
        ro.set_position(tip_x_y_z=pre_point, speed=speed, acceleration=acc)
        ro.position_done()
        
        print(f"  >> 2. 竖直下压至: {target}")
        ro.set_position(tip_x_y_z=target, speed=speed/2, acceleration=acc)
        ro.position_done()
    else:
        print(f"  >> 直接前往: {target}")
        ro.set_position(tip_x_y_z=target, speed=speed, acceleration=acc)
        ro.position_done()

def main():
    points = load_points()
    total = len(points)
    print(f"已加载 {total} 个采样点。")

    # --- 选择 1: 运动策略 ---
    print("-" * 40)
    print("请选择【运动策略】:")
    print(" [1] 直接到达")
    print(" [2] 竖直下压 (推荐)")
    mode_in = input("输入 1 或 2: ").strip()
    use_pre = (mode_in == '2')

    # --- 选择 2: 执行模式 ---
    print("-" * 40)
    print("请选择【执行模式】:")
    print(" [1] 手动步进 (调试用)")
    print(" [2] 自动连续 (批量采集)")
    exec_in = input("输入 1 或 2: ").strip()
    is_auto = (exec_in == '2')
    
    # --- 初始化机器人 ---
    ro = init_robot()
    if not ro: return

    # ================= 自动模式逻辑 (含断点续测) =================
    if is_auto:
        # === 选择起始点 ===
        print("-" * 40)
        print(f"共 {total} 个点 (序号 0 - {total-1})")
        start_idx_str = input(f"请输入【起始点序号】(直接回车默认从 0 开始): ").strip()
        
        start_idx = 0
        if start_idx_str.isdigit():
            start_idx = int(start_idx_str)
            if start_idx < 0 or start_idx >= total:
                print("⚠️ 输入序号超出范围，已重置为 0")
                start_idx = 0
        
        print(f"\n=== 启动自动采集 (从第 {start_idx} 点开始，共 {total} 点) ===")
        print("⚠️  注意：如需中途停止，请按 Ctrl+C 强制结束程序\n")
        
        try:
            # 使用切片 points[start_idx:] 从指定位置开始遍历
            # enumerate(..., start=start_idx) 确保打印的序号是正确的
            for i, target in enumerate(points[start_idx:], start=start_idx):
                
                print(f"\n[自动] 正在执行 -> 第 {i}/{total-1} 点 (Excel行号 {i+2})")
                print(f"       目标坐标: {target}")
                
                move_to_point(ro, target, use_pre, speed=500, acc=300)
                
                print(f"✅ 第 {i} 点采集完成。")
                
                # 延时给激光跟踪仪留出时间 (秒)
                time.sleep(1.7) 
                
            print("\n🎉 所有采样点自动执行完毕！")
            
        except KeyboardInterrupt:
            print("\n🛑 用户手动中止自动运行。")
            print(f"提示: 下次你可以输入序号 {i} 来继续采集。")
    
    # ================= 手动模式逻辑 =================
    else:
        curr_idx = -1
        print("\n=== 进入手动控制模式 ===")
        print("控制: [n]下一个 / [b]上一个 / [q]退出")
        
        while True:
            status = f"当前: {curr_idx}/{total-1}" if curr_idx >= 0 else "当前: 未开始"
            print(f"{status} | 等待指令...")
            
            cmd = input("指令 > ").lower().strip()
            next_idx = curr_idx
            
            if cmd == 'n' or cmd == '':
                if curr_idx < total - 1: next_idx += 1
                else: print("到底了！"); continue
            elif cmd == 'b':
                if curr_idx > 0: next_idx -= 1
                else: print("到头了！"); continue
            elif cmd == 'q':
                break
            else: continue
                
            if next_idx != curr_idx:
                curr_idx = next_idx
                target = points[curr_idx]
                print(f"执行 -> 第 {curr_idx} 点 {target}")
                # 手动模式也用安全速度
                move_to_point(ro, target, use_pre, speed=500, acc=300) 
                print("✅ 到位")

if __name__ == "__main__":
    main()