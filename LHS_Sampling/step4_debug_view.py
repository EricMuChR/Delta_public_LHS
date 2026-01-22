import pandas as pd
import numpy as np
import os

# ================= 配置 =================
FILE_CMD = "lhs_points.csv"
FILE_MEAS = "aligned_measurement_motor.csv" # Step 3 的产出
# =======================================

def main():
    print("="*60)
    print("   Step 4 诊断工具: 数据透视")
    print("="*60)

    if not os.path.exists(FILE_CMD) or not os.path.exists(FILE_MEAS):
        print("❌ 找不到文件。")
        return

    df_cmd = pd.read_csv(FILE_CMD)
    pts_cmd = df_cmd[['x', 'y', 'z']].values

    df_meas = pd.read_csv(FILE_MEAS)
    pts_meas = df_meas[['x', 'y', 'z']].values

    print(f"1. 数据量对比:")
    print(f"   指令点 (CMD) : {len(pts_cmd)} 行")
    print(f"   实测点 (MEAS): {len(pts_meas)} 行")
    print("-" * 40)

    print(f"2. 坐标范围对比 (Range Check):")
    print(f"   CMD  X: [{pts_cmd[:,0].min():.1f}, {pts_cmd[:,0].max():.1f}]")
    print(f"   MEAS X: [{pts_meas[:,0].min():.1f}, {pts_meas[:,0].max():.1f}]")
    print("   --------------------------------")
    print(f"   CMD  Y: [{pts_cmd[:,1].min():.1f}, {pts_cmd[:,1].max():.1f}]")
    print(f"   MEAS Y: [{pts_meas[:,1].min():.1f}, {pts_meas[:,1].max():.1f}]")
    print("   --------------------------------")
    print(f"   CMD  Z: [{pts_cmd[:,2].min():.1f}, {pts_cmd[:,2].max():.1f}]")
    print(f"   MEAS Z: [{pts_meas[:,2].min():.1f}, {pts_meas[:,2].max():.1f}]")
    print("-" * 40)

    print(f"3. 前 5 点数值对比 (Head Check):")
    print(f"{'Index':<5} | {'CMD (x, y, z)':<30} | {'MEAS (x, y, z)':<30}")
    for i in range(5):
        c = pts_cmd[i]
        m = pts_meas[i]
        print(f"{i:<5} | {c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}   | {m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f}")
    
    print("-" * 40)
    print("4. 诊断建议:")
    
    z_cmd_mean = np.mean(pts_cmd[:,2])
    z_meas_mean = np.mean(pts_meas[:,2])
    
    if z_cmd_mean < 0 and z_meas_mean > 0:
        print("🔴 严重问题: Z轴反向！")
        print("   指令Z是负的(向下)，实测Z是正的(向上)。")
        print("   -> 请重新运行 Step 3，并修改代码中的 z_axis 方向 (z_axis = -z_axis)。")
    elif abs(z_cmd_mean - z_meas_mean) > 100:
        print("🟠 偏移过大: Z轴偏差超过 100mm。")
        print("   这可能是因为 Measure 数据开头包含了一段'回零'或'静止'数据。")
        print("   -> 建议使用'增强版 Step 4'进行对齐。")
    else:
        print("🟢 Z轴方向看起来一致。")
        print("   问题可能出在数据开头的时间同步上。")

if __name__ == "__main__":
    main()