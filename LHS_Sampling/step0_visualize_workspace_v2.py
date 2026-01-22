import time
import sys
import os
import math
import msvcrt  # Windows 专用键盘监听

# ================= 跨文件夹导入驱动 =================
# 获取当前脚本所在目录的上一级目录 (即 Delta_public)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# ==========================================================

import DrDelta as robot 

# ==========================================
# 1. 定义工作空间参数
# ==========================================
# --- 形状 A: 长方体 (Box) ---
BOX_X_MIN, BOX_X_MAX = -100, 100
BOX_Y_MIN, BOX_Y_MAX = -100, 100

# --- 形状 B: 圆柱体 (Cylinder) ---
CYL_RADIUS = 125  # 半径 125mm 

# --- 通用 Z 轴范围 ---
# 注意: Z_MIN 是底部(更深)，Z_MAX 是顶部(更浅)
Z_MIN, Z_MAX = -240, -190  

SAFE_HOME = [0, 0, -200]
CHECK_SPEED = 200
CHECK_ACC = 100

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def interpolate_path(start, end, step_size=10):
    """切分路径，确保长距离移动时的安全"""
    dist = get_dist(start, end)
    if dist < step_size:
        return [end]
    
    steps = int(dist / step_size)
    points = []
    vector = [(end[0]-start[0])/dist, (end[1]-start[1])/dist, (end[2]-start[2])/dist]
    
    for i in range(1, steps + 1):
        new_p = [
            start[0] + vector[0] * step_size * i,
            start[1] + vector[1] * step_size * i,
            start[2] + vector[2] * step_size * i
        ]
        points.append(new_p)
    if points[-1] != end:
        points.append(end)
    return points

def check_emergency_stop():
    if msvcrt.kbhit():
        msvcrt.getch()
        return True
    return False

def emergency_retreat(ro):
    print("\n" + "!"*40)
    print("   【急停触发】 正在紧急回撤至安全点...")
    print("!"*40)
    try:
        ro.set_position(tip_x_y_z=SAFE_HOME, speed=20, acceleration=20)
        ro.position_done()
        print("已回撤。程序终止。")
    except Exception as e:
        print(f"回撤失败: {e}")
    sys.exit(0)

def safe_move_segment(ro, current_pos, target_pos):
    waypoints = interpolate_path(current_pos, target_pos, step_size=10)
    for pt in waypoints:
        if check_emergency_stop():
            emergency_retreat(ro)
        ro.set_position(tip_x_y_z=pt, speed=CHECK_SPEED, acceleration=CHECK_ACC)
        ro.position_done()
    return target_pos

def main():
    print("="*60)
    print("   工作空间可视化巡检 (支持 圆柱/长方体)")
    print("="*60)
    
    # === 形状选择 ===
    print("请选择工作空间形状:")
    print(" [1] 长方体 (Box) - 检查 8 个角")
    print(" [2] 圆柱体 (Cylinder) - 检查上下底面的圆周")
    choice = input("输入 1 或 2: ").strip()
    is_cylinder = (choice == '2')

    key_points = []

    if is_cylinder:
        print(f"模式: 圆柱体 | 半径: {CYL_RADIUS} | Z: [{Z_MIN}, {Z_MAX}]")
        
        # --- 1. 顶部圆周 (Z_MAX) ---
        # 1.1 先去顶部圆心
        key_points.append([0, 0, Z_MAX])
        
        # 1.2 画顶部圆周 (0度到360度，每10度一个点)
        for deg in range(0, 370, 10):
            rad = math.radians(deg)
            x = CYL_RADIUS * math.cos(rad)
            y = CYL_RADIUS * math.sin(rad)
            key_points.append([x, y, Z_MAX])
            
        # --- 2. 底部圆周 (Z_MIN) ---
        # 2.1 先回到底部圆心安全过渡
        key_points.append([0, 0, Z_MIN])
        
        # 2.2 画底部圆周
        for deg in range(0, 370, 10):
            rad = math.radians(deg)
            x = CYL_RADIUS * math.cos(rad)
            y = CYL_RADIUS * math.sin(rad)
            key_points.append([x, y, Z_MIN])
            
    else:
        # 长方体模式
        print(f"模式: 长方体 | X: [{BOX_X_MIN},{BOX_X_MAX}] Y: [{BOX_Y_MIN},{BOX_Y_MAX}]")
        key_points = [
            [BOX_X_MIN, BOX_Y_MIN, Z_MIN], [BOX_X_MAX, BOX_Y_MIN, Z_MIN], 
            [BOX_X_MAX, BOX_Y_MAX, Z_MIN], [BOX_X_MIN, BOX_Y_MAX, Z_MIN], 
            [BOX_X_MIN, BOX_Y_MIN, Z_MIN], # 闭合底面
            [BOX_X_MIN, BOX_Y_MIN, Z_MAX], # 上升到顶面
            [BOX_X_MAX, BOX_Y_MIN, Z_MAX], [BOX_X_MAX, BOX_Y_MAX, Z_MAX], 
            [BOX_X_MIN, BOX_Y_MAX, Z_MAX], [BOX_X_MIN, BOX_Y_MIN, Z_MAX] # 闭合顶面
        ]

    print(f"急停操作: 按下【空格键】立即回撤。")
    input("请确认安全，按 [Enter] 开始巡检...")

    print("正在初始化机器人...")
    try:
        ro = robot.robot(
            MAX_list_temp=[90, 90, 90], 
            MIN_list_temp=[-42, -42, -42], 
            L_temp=[100, 250, 35, 23.4]
        )
        ro.set_position(tip_x_y_z=SAFE_HOME, speed=10, acceleration=10)
        ro.position_done()
        current_pos = SAFE_HOME
    except Exception as e:
        print(f"初始化失败 (请检查是否已连接CAN卡): {e}")
        return

    print("\n开始巡检...")
    total_pts = len(key_points)
    
    for i, target in enumerate(key_points):
        # 简单打印进度
        print(f"目标 [{i+1}/{total_pts}]: {[round(x,1) for x in target]}")
        
        current_pos = safe_move_segment(ro, current_pos, target)
        
        # 到达每个关键点后稍微停顿，方便观察
        # 如果是画圆模式，停顿时间可以短一点，看起来流畅些
        sleep_time = 0.05 if is_cylinder else 0.5
        time.sleep(sleep_time)

    print("\n巡检完成，回安全点。")
    safe_move_segment(ro, current_pos, SAFE_HOME)
    print("测试通过。")

if __name__ == "__main__":
    main()