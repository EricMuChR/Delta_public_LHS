import torch
import numpy as np
import Delta_3 as delta_cpu
from Delta_Torch import DeltaKinematics

def debug():
    print("🔍 开始极简诊断...")
    
    # 1. 设定几何参数
    L1, L2, R, r = 100.0, 250.0, 35.0, 23.4
    
    # 2. 初始化 Torch 模型
    model = DeltaKinematics()
    model.update_geometry(l1=L1, l2=L2, r_base=R, r_mobile=r, z_off=-16.3)
    
    # 3. 初始化 CPU 模型
    cpu_net = delta_cpu.arm(l=[L1, L2, R, r])
    
    # 4. 设定一个简单的测试角度 (全0度，水平)
    # 输入角度: [0, 0, 0]
    angles = np.array([0.0, 0.0, 0.0])
    
    # --- CPU 计算 ---
    cpu_net.forward_kinematics_position(angles.tolist())
    pos_cpu = np.array(cpu_net.tip_x_y_z)
    print(f"\n[CPU Result] Angles: {angles}")
    print(f"Pos: {pos_cpu}")
    
    # --- Torch 计算 Residue ---
    # 构造 Tensor
    theta_tensor = torch.tensor([angles], dtype=torch.float32) # [1, 3]
    pos_tensor = torch.tensor([pos_cpu], dtype=torch.float32)  # [1, 3]
    
    # 强制调用 debug 打印
    print("\n[Torch Residue Check]")
    # 获取肘部位置
    elbows = model._get_elbow_positions(theta_tensor)
    print(f"Elbows (Torch):\n{elbows[0].detach().numpy()}")
    
    # 手动算距离
    diff = elbows[0].detach().numpy() - pos_cpu
    dist_sq = np.sum(diff**2, axis=1)
    target_sq = L2**2
    
    print(f"\nTarget L2^2: {target_sq}")
    print(f"Actual Dist^2: {dist_sq}")
    print(f"Diff: {np.abs(dist_sq - target_sq)}")
    
    # 调用函数
    res = model.compute_physics_residue(theta_tensor, pos_tensor)
    print(f"\nFunction Output: {res.item()}")

if __name__ == "__main__":
    debug()