import torch
import torch.nn as nn
import numpy as np

class DifferentiableDeltaKinematics(nn.Module):
    def __init__(self, nominal_params, tool_offset):
        """
        可微分的 Delta 机器人正运动学层。
        包含: [L, l, R, r] + [Theta Offsets]
        
        Args:
            nominal_params (list): [L, l, R, r]
            tool_offset (list/tensor): [x, y, z]
        """
        super().__init__()
        
        L_val, l_val, R_val, r_val = nominal_params
        
        # 1. 几何参数 (Parameter)
        self.L = nn.Parameter(torch.tensor([L_val]*3, dtype=torch.float32)) # 主动臂
        self.l = nn.Parameter(torch.tensor([l_val]*3, dtype=torch.float32)) # 从动臂
        self.R = nn.Parameter(torch.tensor(R_val, dtype=torch.float32))     # 静平台
        self.r = nn.Parameter(torch.tensor(r_val, dtype=torch.float32))     # 动平台
        
        # 2. 新增: 关节零点偏置 (Joint Zero Offsets)
        # 初始值为 0，允许正负微调
        self.theta_offset = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        
        # 3. 工具偏置
        self.register_buffer('tool_offset', torch.tensor(tool_offset, dtype=torch.float32))

        # 初始全锁
        self.lock_all_params()

    def lock_all_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def unlock_geometric_params(self):
        self.L.requires_grad = True
        self.l.requires_grad = True
        self.r.requires_grad = True
        # R 建议保持锁死
        self.theta_offset.requires_grad = True # 解锁零点

    def forward(self, theta_input):
        """
        Args:
            theta_input (Tensor): (Batch, 3) 原始指令角度
        """
        # === 核心修改: 施加零点偏置 ===
        # 真实角度 = 指令角度 + 零点误差
        # 注意广播机制: (Batch, 3) + (3,)
        theta = theta_input + self.theta_offset
        
        batch_size = theta.shape[0]
        
        # 以下是标准的 Delta 正解逻辑
        L = self.L
        l = self.l
        R = self.R
        r = self.r
        
        sqrt3 = np.sqrt(3)
        
        s1, s2, s3 = torch.sin(theta[:, 0]), torch.sin(theta[:, 1]), torch.sin(theta[:, 2])
        c1, c2, c3 = torch.cos(theta[:, 0]), torch.cos(theta[:, 1]), torch.cos(theta[:, 2])
        
        Dr = R - r 
        
        # Arm 1
        x1 = sqrt3/2 * (Dr + L[0] * c1)
        y1 = -0.5    * (Dr + L[0] * c1)
        z1 = -L[0] * s1
        
        # Arm 2
        x2 = -sqrt3/2 * (Dr + L[1] * c2)
        y2 = -0.5     * (Dr + L[1] * c2)
        z2 = -L[1] * s2
        
        # Arm 3
        x3 = torch.zeros_like(x1)
        y3 = Dr + L[2] * c3
        z3 = -L[2] * s3
        
        w1 = x1**2 + y1**2 + z1**2 - l[0]**2
        w2 = x2**2 + y2**2 + z2**2 - l[1]**2
        w3 = x3**2 + y3**2 + z3**2 - l[2]**2
        
        d_z13 = z1 - z3
        d_z12 = z1 - z2
        d_x12 = x1 - x2
        d_x13 = x1 - x3
        d_y12 = y1 - y2
        d_y13 = y1 - y3
        d_w12 = w1 - w2
        d_w13 = w1 - w3
        
        N1 = d_z13 * d_x12 - d_z12 * d_x13
        N2 = d_z13 * d_y12 - d_z12 * d_y13
        N3 = 0.5 * (d_z13 * d_w12 - d_z12 * d_w13)
        
        N4 = d_y13 * d_x12 - d_y12 * d_x13
        N5 = d_y13 * d_z12 - d_y12 * d_z13
        N6 = 0.5 * (d_y13 * d_w12 - d_y12 * d_w13)
        
        epsilon = 1e-6
        
        R1 = -N1 / (N2 + epsilon)
        R2 = N3 / (N2 + epsilon)
        R3 = -N4 / (N5 + epsilon)
        R4 = N6 / (N5 + epsilon)
        
        A = 1 + R1**2 + R3**2
        B = 2 * (R1*R2 + R3*R4 - R1*y1 - R3*z1 - x1)
        C = R2**2 + R4**2 + w1 - 2*(R2*y1 + R4*z1)
        
        delta = B**2 - 4*A*C
        delta_safe = torch.nn.functional.relu(delta)
        
        x_sol_1 = (-B + torch.sqrt(delta_safe)) / (2*A + epsilon)
        z_sol_1 = R3 * x_sol_1 + R4
        
        x_sol_2 = (-B - torch.sqrt(delta_safe)) / (2*A + epsilon)
        z_sol_2 = R3 * x_sol_2 + R4
        
        # 优先选 Z < 0 的解
        mask = (z_sol_1 < 0).float()
        x_sol = mask * x_sol_1 + (1 - mask) * x_sol_2
        
        y_sol = R1 * x_sol + R2
        z_sol = R3 * x_sol + R4
        
        flange_pos = torch.stack([x_sol, y_sol, z_sol], dim=1)
        tool_pos = flange_pos + self.tool_offset
        
        return tool_pos

# --- 验证代码 (Test Suite) ---
if __name__ == "__main__":
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    try:
        import Delta_3 as nominal
    except ImportError:
        print("❌ 无法导入 Delta_3.py")
        sys.exit(1)
    
    print("="*60)
    print("   Phase 2 Verification: Kinematics & Zero-Offset")
    print("="*60)

    # 1. 基础一致性测试 (Identity Check)
    # 确保 theta_offset = 0 时，结果和 Delta_3.py 一致
    test_angles_deg = [10.0, 20.0, -15.0] 
    test_angles_rad = torch.tensor(np.radians([test_angles_deg]), dtype=torch.float32)
    
    params = [100, 250, 35, 23.4] 
    offset = [0, 0, 0] # 暂时忽略工具偏置以便对比纯运动学
    
    # 初始化 PyTorch 模型
    model = DifferentiableDeltaKinematics(params, offset)
    model.eval()
    
    with torch.no_grad():
        pos_torch = model(test_angles_rad).numpy()[0]
    
    # 初始化 Numpy 模型
    arm = nominal.arm(l=params)
    arm.forward_kinematics_position(test_angles_deg)
    pos_numpy = np.array(arm.tip_x_y_z)
    
    print(f"\n[Test 1] Baseline Check (Offset=0):")
    print(f"   Angles (deg): {test_angles_deg}")
    print(f"   Torch Output: {pos_torch}")
    print(f"   Numpy Output: {pos_numpy}")
    
    diff = np.linalg.norm(pos_torch - pos_numpy)
    print(f"   Difference  : {diff:.6f} mm")
    
    if diff < 1e-4:
        print("   ✅ Baseline Passed.")
    else:
        print("   ❌ Baseline Failed.")

    # 2. 零点偏置功能测试 (Offset Functionality Check)
    # 我们手动给模型加一个零点偏置，看看输出变不变
    print(f"\n[Test 2] Zero-Offset Effect Check:")
    
    # 给电机1加 1 度偏置 (约 0.017 rad)
    offset_val_deg = 1.0
    offset_val_rad = np.radians(offset_val_deg)
    
    with torch.no_grad():
        # 修改模型参数
        model.theta_offset[0] = offset_val_rad # Motor 1 offset
        pos_offset = model(test_angles_rad).numpy()[0]
        
    print(f"   Injecting Offset: Motor 1 += {offset_val_deg} deg")
    print(f"   New Output      : {pos_offset}")
    print(f"   Original Output : {pos_torch}")
    
    shift = np.linalg.norm(pos_offset - pos_torch)
    print(f"   Output Shift    : {shift:.4f} mm")
    
    if shift > 0.1:
        print("   ✅ Offset Logic Passed (Output changed as expected).")
    else:
        print("   ❌ Offset Logic Failed (Output didn't change!).")