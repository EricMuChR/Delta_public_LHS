import pandas as pd
import numpy as np  # <--- 补上了这行
import time
import sys
import os
import DrDelta as robot 

# ================= 配置 =================
INPUT_FILE = "points_with_compensation.csv"
# =======================================

def main():
    print("="*60)
    print("   Step 7: 执行验证运动 (请配合激光跟踪仪)")
    print("="*60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}")
        return
        
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ 加载 {len(df)} 个补偿点")
    
    try:
        print("🤖 连接机器人...")
        ro = robot.robot()
        # 先回一个安全点
        ro.set_position([0,0,-240], speed=50) 
        ro.position_done()
        print("✅ 初始化成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return
        
    input("🚀 按 Enter 键开始执行运动... (请确保跟踪仪已开始记录)")
    
    for i, row in df.iterrows():
        # 读取数据
        target = [row['comp_x'], row['comp_y'], row['comp_z']]
        orig = [row['orig_x'], row['orig_y'], row['orig_z']]
        
        # 打印进度 (这里用到 np.round)
        print(f"[{i+1}/{len(df)}] 目标: {np.round(orig,1)} | 实际发送: {np.round(target,1)}")
        
        # 发送指令
        # ro.set_position(target, speed=500, acceleration=300) 
        ro.set_position(orig, speed=500, acceleration=300) 
        ro.position_done() 
        
        # 稍微停顿，等待机构稳定，方便跟踪仪捕捉
        # time.sleep(0.1) 
        time.sleep(2) 
        
    print("✅ 执行完毕！")
    print("👉 请从跟踪仪导出数据为 'validation_500.csv' (需包含 X, Y, Z 列)。")
    print("   然后运行 step8_diagnose_pinn.py 生成最终报告。")

if __name__ == "__main__":
    main()