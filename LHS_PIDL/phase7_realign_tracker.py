import sys
import os
import json

# ================= 配置 =================
# 1. 重新采集的电机画圆数据 (你必须手动采集这个文件!)
NEW_ARCS_FILE = "motor_arcs_val.csv" 

# 2. 输出: 新的验证用矩阵
NEW_MATRIX_FILE = "matrix_val.npz"

# 3. 输出: 验证用的 R 参数 (仅作记录，不覆盖训练用的)
NEW_PARAMS_FILE = "robot_base_params_val.json"
# =======================================

# 导入隔壁文件夹的 step3
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sampling_dir = os.path.join(parent_dir, "LHS_Sampling")
sys.path.append(sampling_dir)

try:
    from step3_motor_center_align import run_alignment_process
except ImportError:
    print(f"❌ 错误: 无法在 {sampling_dir} 找到 step3_motor_center_align.py")
    sys.exit(1)

def main():
    print("="*60)
    print("   Phase 7 (Extra): Tracker Re-Alignment")
    print("   ⚠️  用于激光跟踪仪移动后的坐标系重构")
    print("="*60)
    
    print("请确认:")
    print("1. 激光跟踪仪已移动到新位置。")
    print(f"2. 机器人已重新执行主动臂画圆，并采集数据保存为: {NEW_ARCS_FILE}")
    print("-" * 40)
    
    if not os.path.exists(NEW_ARCS_FILE):
        print(f"❌ 未找到 {NEW_ARCS_FILE}。")
        print("   -> 请先让机器人控制三个主动臂分别画圆，并将 Tracker 数据保存为此文件名。")
        return

    confirm = input("确认生成新矩阵? (y/n): ").strip().lower()
    if confirm != 'y': return

    # 调用 Step 3 的逻辑
    # 注意：我们不需要转换旧的测量数据，所以最后两个参数传 None
    success = run_alignment_process(
        file_arcs=NEW_ARCS_FILE,
        file_matrix_out=NEW_MATRIX_FILE,
        file_params_out=NEW_PARAMS_FILE, 
        file_raw_meas=None,
        file_aligned_meas=None
    )
    
    if success:
        print("\n🎉 新坐标系矩阵建立成功！")
        print(f"💾 矩阵文件: {NEW_MATRIX_FILE}")
        print("-" * 40)
        print("👉 下一步: 运行 phase7_robot_control.py 跑 500 个点")
        print("👉 最后: 运行 phase7_verification_loop.py (它会自动加载这个矩阵进行数据转换)")

if __name__ == "__main__":
    main()