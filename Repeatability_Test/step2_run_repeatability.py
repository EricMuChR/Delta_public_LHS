import pandas as pd
import time
import sys
import os
import msvcrt  # Windows 下用于检测键盘输入

# ================= 导入驱动 =================
# 获取当前脚本所在目录 (.../Delta_public_LHS/Repeatability_Test)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录 (.../Delta_public_LHS)
parent_dir = os.path.dirname(current_dir)

# 将其加入系统路径
sys.path.append(parent_dir)

try:
    import DrDelta as robot
    print(f"✅ 成功导入 DrDelta 驱动")
except ImportError:
    print("❌ 无法导入 DrDelta。")
    print(f"   请确认 DrDelta.py 是否位于: {parent_dir}")
    sys.exit(1)
# =======================================================

# ================= 配置 =================
FILE_TARGETS = "repeatability_targets.csv"
CYCLES = 10     # 循环 10 次
SPEED = 200     # 运动速度
ACCEL = 100     # 加速度
# =======================================

def wait_for_key():
    """ 
    等待用户按键交互 
    返回: 'next', 'back', 'retry', 'quit'
    """
    print("   👉 [Enter] 下一点 | [B] 上一点 | [R] 重试当前点 | [Q] 退出")
    
    # 清空之前的按键缓存
    while msvcrt.kbhit():
        msvcrt.getch()
        
    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Windows下 Enter 键通常是 \r
            if key == b'\r': 
                return 'next'
            elif key.lower() == b'b':
                return 'back'
            elif key.lower() == b'r':
                return 'retry'
            elif key.lower() == b'q':
                return 'quit'
            time.sleep(0.05)

def main():
    print("="*60)
    print(f"   Step 2: 重复性采集 (12点 x 10次 = 120点)")
    print("="*60)

    # 1. 检查文件
    if not os.path.exists(FILE_TARGETS):
        print("❌ 找不到点位文件，请先运行 Step 1。")
        return

    # 2. 读取点位
    df = pd.read_csv(FILE_TARGETS)
    points = df[['x', 'y', 'z']].values
    names = df['PointName'].values

    # 3. 连接机器人
    try:
        print("🤖 正在连接机器人...")
        ro = robot.robot() 
        
        print("   移动到初始位置...")
        ro.set_position([0, 0, -200], speed=200, acceleration=200)
        ro.position_done()
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    print(f"\n✅ 准备就绪！")
    print(f"⚠️  重要提示：如果使用 [B] 键返回，请务必在激光跟踪仪软件中【删除】那个错误采集的点，")
    print(f"    否则后续的数据对应关系会错乱！")
    input("⌨️  按 Enter 开始第一轮循环...")

    # 4. 开始循环采集
    for cycle in range(1, CYCLES + 1):
        print(f"\n🔄 --- Cycle {cycle} / {CYCLES} ---")
        
        i = 0
        while i < len(points):
            pt = points[i]
            name = names[i]
            
            # 全局进度索引 (1~120)
            global_count = (cycle - 1) * len(points) + i + 1
            
            print(f"\n📍 [{global_count}/120] 目标 {name} (Cycle {cycle}): {pt}")
            
            # 运动指令
            try:
                ro.set_position(pt, speed=SPEED, acceleration=ACCEL)
                ro.position_done()
            except Exception as e:
                print(f"⚠️ 运动指令发送失败: {e}")
            
            # 停稳等待
            time.sleep(0.5) 
            print("   ✅ 到达。请触发采集。")
            
            # 等待用户操作
            action = wait_for_key()
            
            if action == 'next':
                i += 1
            
            elif action == 'back':
                if i > 0:
                    i -= 1
                    print(f"   🔙 正在返回上一个点 ({names[i]})...")
                    print("   ⚠️  请记得在跟踪仪中删除刚才多采的数据！")
                else:
                    print("   🚫 已经是本轮第一个点了，无法返回上一轮。")
            
            elif action == 'retry':
                print("   ⚠️  收到重试指令，保持当前点不动...")
                # i 不自增，下一次循环继续跑这个点
            
            elif action == 'quit':
                print("👋 用户终止程序。")
                return

    print("\n🎉 采集全部完成！")
    print("👉 请导出激光跟踪仪数据为 'repeatability_meas.csv'，然后运行 Step 3。")

if __name__ == "__main__":
    main()