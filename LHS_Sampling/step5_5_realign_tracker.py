import sys
import os
# 导入 Step 3 的处理函数
try:
    from step3_motor_center_align import run_alignment_process
except ImportError:
    print("❌ 错误: 找不到 step3_motor_center_align.py，请确保它在同一目录下。")
    sys.exit(1)

# ================= 配置 =================
# 你必须重新采集一次电机画圆数据，保存为这个名字
NEW_ARCS_FILE = "motor_arcs_val.csv" 
# 输出新的验证用矩阵
NEW_MATRIX_FILE = "matrix_val.npz"
# =======================================

def main():
    print("="*60)
    print("   Step 5.5: 验证阶段坐标系重构 (Re-Alignment)")
    print("   ⚠️  适用场景: 在训练后移动了激光跟踪仪")
    print("="*60)
    
    print("请确认:")
    print("1. 激光跟踪仪已移动到新位置。")
    print(f"2. 机器人已重新执行画圆动作，并采集数据保存为: {NEW_ARCS_FILE}")
    print("-" * 40)
    
    if not os.path.exists(NEW_ARCS_FILE):
        print(f"❌ 未找到 {NEW_ARCS_FILE}。请先让机器人画圆并导出数据。")
        return

    confirm = input("确认继续生成新矩阵? (y/n): ").strip().lower()
    if confirm != 'y': return

    # 调用 Step 3 的逻辑，但不处理测量点，只生成矩阵
    success = run_alignment_process(
        file_arcs=NEW_ARCS_FILE,
        file_matrix_out=NEW_MATRIX_FILE,
        file_raw_meas=None,     # 不需要转换旧数据
        file_aligned_meas=None
    )
    
    if success:
        print("\n🎉 新坐标系矩阵建立成功！")
        print(f"👉 下一步: 运行 Step 6 (机器人验证跑点)")
        print(f"👉 最后: 运行 Step 7 (它会自动加载 {NEW_MATRIX_FILE})")

if __name__ == "__main__":
    main()