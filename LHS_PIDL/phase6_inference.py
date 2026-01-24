import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json

# ================= 配置 =================
# 目标点 (你可以修改这里来测试不同位置)
TEST_TARGETS_MM = [
    [0, 0, -250],      # 中心点
    [50, 0, -240],     # X轴偏移
    [0, 50, -240],     # Y轴偏移
    [30, 30, -260],    # 象限点
]
# =======================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase2_diff_kinematics import DifferentiableDeltaKinematics
except ImportError:
    print("❌ 无法导入模块，请检查路径")
    sys.exit(1)

# === 定义同样的残差网络结构 (必须与 Phase 5 一致) ===
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

class PIDL_Inference_Engine:
    def __init__(self, model_path):
        print(f"📥 Loading PIDL Model: {model_path}")
        package = torch.load(model_path)
        
        # 1. 恢复几何参数
        geo_params = package['geo_params']
        # 注意: 这里的 nominal_params 初始值不重要，会被覆盖
        self.geo_model = DifferentiableDeltaKinematics([100, 250, 35, 23.4], geo_params['tool_offset'])
        
        self.geo_model.L.data = geo_params['L']
        self.geo_model.l.data = geo_params['l']
        self.geo_model.R.data = geo_params['R']
        self.geo_model.r.data = geo_params['r']
        self.geo_model.theta_offset.data = geo_params['theta_offset']
        
        self.geo_model.lock_all_params()
        self.geo_model.eval()
        
        # 2. 恢复神经网络
        self.nn_model = ResidualMLP()
        self.nn_model.load_state_dict(package['nn_state'])
        self.nn_model.eval()
        
        print("✅ Model Loaded Successfully.")

    def forward_predict(self, theta_rad):
        """
        PIDL 正运动学预测: Theta -> Pos
        Pos = Geo(Theta) + NN(Theta)
        """
        if not torch.is_tensor(theta_rad):
            theta_rad = torch.tensor(theta_rad, dtype=torch.float32)
        
        if theta_rad.dim() == 1:
            theta_rad = theta_rad.unsqueeze(0) # (1, 3)
            
        with torch.no_grad():
            pos_geo = self.geo_model(theta_rad)
            res_pred = self.nn_model(theta_rad)
            pos_final = pos_geo + res_pred
            
        return pos_final

    def inverse_solve(self, target_pos_mm, max_iter=50, tol=0.01):
        """
        PIDL 逆运动学求解 (核心功能): Target Pos -> Optimal Theta
        使用数值迭代法 (梯度下降/牛顿法思路) 寻找能让 PIDL 模型输出 target 的 theta。
        """
        target_tensor = torch.tensor(target_pos_mm, dtype=torch.float32).unsqueeze(0)
        
        # 1. 初始猜测: 使用纯几何模型的解析逆解 (作为起点)
        # 这里为了简化，我们直接从名义参数的解析解开始，或者粗暴地从0开始迭代
        # 更好的方法是调用 Delta_3.py 的 inverse_kinematics 得到初始 theta
        # 这里我们用 PyTorch 的梯度下降来“凑”出这个角
        
        # 初始猜测角度 (设为 0 度附近)
        theta_opt = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        
        # 优化器: 专门用来调整 theta 以匹配 target
        optimizer = torch.optim.LBFGS([theta_opt], lr=0.1, max_iter=10) # LBFGS 收敛极快
        
        loss_val = float('inf')
        
        # 迭代求解
        for i in range(max_iter):
            def closure():
                optimizer.zero_grad()
                
                # PIDL Forward: Geo + NN
                # 注意: 这里 Geo 模型和 NN 模型的参数是锁死的，只有 input theta 在变
                # 我们需要临时通过允许梯度的路径来计算
                
                # 手动前向传播以保留梯度图 (因为 self.forward_predict用了no_grad)
                # Geo Part
                theta_input = theta_opt + self.geo_model.theta_offset
                
                # ... (这里复用 Geo 逻辑，为简洁直接调 model) ...
                # 为了代码简洁，我们假定 geo_model 的 forward 允许梯度回传到 input
                # 我们需要临时解锁 forward_predict 里的 no_grad 限制
                
                # Hack: 直接调用子模块
                pos_geo = self.geo_model(theta_opt) # 这里的 theta_opt 是变量
                pos_nn = self.nn_model(theta_opt)
                pos_pred = pos_geo + pos_nn
                
                loss = torch.nn.MSELoss()(pos_pred, target_tensor)
                loss.backward()
                return loss

            loss_val = optimizer.step(closure).item()
            
            if loss_val < (tol**2): # 误差平方小于阈值
                break
                
        final_pos = self.forward_predict(theta_opt).numpy()[0]
        final_theta_deg = np.degrees(theta_opt.detach().numpy()[0])
        error = np.linalg.norm(final_pos - target_pos_mm)
        
        return final_theta_deg, final_pos, error

def main():
    print("="*60)
    print("   Phase 6: PIDL Inference & Inverse Solver")
    print("="*60)
    
    model_path = os.path.join(current_dir, "pidl_final_model.pth")
    if not os.path.exists(model_path):
        print("Please run Phase 5 first.")
        return

    engine = PIDL_Inference_Engine(model_path)
    
    print(f"\n🎯 Target Verification (Solver Tolerance: 0.01mm)")
    print(f"{'Target (mm)':<25} | {'PIDL Solved Angles (deg)':<30} | {'Pred Pos (mm)':<25} | {'Err'}")
    print("-" * 90)
    
    for target in TEST_TARGETS_MM:
        start_time = time.time()
        
        # 求解: 给定坐标 -> 算出电机角度
        angles, pred_pos, error = engine.inverse_solve(target)
        
        cost_time = (time.time() - start_time) * 1000 # ms
        
        target_str = f"[{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}]"
        angles_str = f"[{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]"
        pred_str = f"[{pred_pos[0]:.2f}, {pred_pos[1]:.2f}, {pred_pos[2]:.2f}]"
        
        print(f"{target_str:<25} | {angles_str:<30} | {pred_str:<25} | {error:.4f} mm")

    print("\n💡 Done! Use `engine.inverse_solve([x,y,z])` to control your robot.")

if __name__ == "__main__":
    main()