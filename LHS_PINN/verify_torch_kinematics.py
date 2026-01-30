import torch
import numpy as np
import Delta_3 as delta_cpu
from Delta_Torch import DeltaKinematics

def test_consistency():
    print("="*60)
    print("   Delta 模型验证: CPU (Standard) vs PyTorch (Parametric)")
    print("="*60)

    # 1. 定义几何参数
    R_BASE = 35.0
    R_MOBILE = 23.4
    L1 = 100.0
    L2 = 250.0
    
    # 显式定义 Z Offset
    Z_OFFSET_VAL = 0

    # 2. 初始化 CPU 模型
    params_cpu = [L1, L2, R_BASE, R_MOBILE]
    robot_cpu = delta_cpu.arm(l=params_cpu)
    print(f"🤖 CPU 模型 (基准): {params_cpu}")

    # 3. 初始化 PyTorch 模型
    robot_torch = DeltaKinematics()
    
    # 显式更新 geometry 并传入 z_off
    robot_torch.update_geometry(
        l1=L1, l2=L2, 
        r_base=R_BASE, r_mobile=R_MOBILE, 
        z_off=Z_OFFSET_VAL
    )
    
    # Debug 打印：确认 Offset 真的写进去了
    print(f"🔥 PyTorch 模型 (待测):")
    print(f"   R={robot_torch.r_base.item():.1f}, r={robot_torch.r_mobile.item():.1f}")
    print(f"   Z_Offset={robot_torch.z_offset.item():.4f} (Should be {Z_OFFSET_VAL})")

    # 4. 生成测试数据
    batch_size = 10
    angles_np = np.random.uniform(-30, 80, (batch_size, 3))
    inputs_tensor = torch.tensor(angles_np, dtype=torch.float32)
    
    # 5. 执行对比
    print("-" * 75)
    print(f"{'Index':<6} {'Input (Deg)':<30} {'Diff (mm)':<12} {'Status'}")
    print("-" * 75)

    max_fk_error = 0.0
    valid_pos = [] # 收集 CPU 的正确结果供后面残差检查用

    for i in range(batch_size):
        # --- CPU 计算 ---
        angle_list = angles_np[i].tolist()
        robot_cpu.tip_x_y_z = [0, 0, -200]
        success = robot_cpu.forward_kinematics_position(angle_list)
        pos_cpu = np.array(robot_cpu.tip_x_y_z)
        valid_pos.append(pos_cpu)

        # --- PyTorch 计算 ---
        with torch.no_grad():
            pos_torch = robot_torch(inputs_tensor[i:i+1]).squeeze(0).numpy()

        # --- 比较 ---
        diff = np.linalg.norm(pos_cpu - pos_torch)
        if diff > max_fk_error: max_fk_error = diff
        
        status = "✅ PASS" if diff < 1e-3 else "❌ FAIL"
        print(f"{i:<6} {str(np.round(angle_list,1)):<30} {diff:.6f}       {status}")

    # 6. 物理残差检查
    print("-" * 75)
    valid_pos_tensor = torch.tensor(np.array(valid_pos), dtype=torch.float32)
    
    with torch.no_grad():
        # 把 CPU 算出的“含Offset的坐标”喂回去
        # compute_physics_residue 会自动先减去 z_offset，再验证几何约束
        residues = robot_torch.compute_physics_residue(inputs_tensor, valid_pos_tensor)
        mean_residue = residues.mean().item()

    print(f"📉 对 CPU 数据的平均物理残差: {mean_residue:.4f}")
    
    print("\n" + "="*60)
    print(f"🏆 FK 最大定位误差: {max_fk_error:.6f} mm")
    
    if max_fk_error < 1e-3 and mean_residue < 100.0:
        print("✨ 验证完美通过！两个模型已完全对齐。")
        print("   准备进入训练阶段。")
    else:
        print("⚠️ 仍有偏差，请检查 update_geometry 是否重置了 Offset。")

if __name__ == "__main__":
    test_consistency()