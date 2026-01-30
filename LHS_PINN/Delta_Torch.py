import torch
import torch.nn as nn
import numpy as np

class DeltaKinematics(nn.Module):
    def __init__(self, device='cpu'):
        super(DeltaKinematics, self).__init__()
        self.device = device
        
        # =========================================================
        # 1. 几何参数 (Learnable Geometric Parameters)
        # =========================================================
        self.l1 = nn.Parameter(torch.tensor(100.0)) 
        self.l2 = nn.Parameter(torch.tensor(250.0))
        self.r_base = nn.Parameter(torch.tensor(35.0))
        self.r_mobile = nn.Parameter(torch.tensor(23.4))
        
        # 2. 零点与安装误差
        self.offset_theta1 = nn.Parameter(torch.tensor(0.0))
        self.offset_theta2 = nn.Parameter(torch.tensor(0.0))
        self.offset_theta3 = nn.Parameter(torch.tensor(0.0))
        self.d_phi1 = nn.Parameter(torch.tensor(0.0))
        self.d_phi2 = nn.Parameter(torch.tensor(0.0))
        self.d_phi3 = nn.Parameter(torch.tensor(0.0))

        # =========================================================
        # 3. 末端 Z 轴偏置 (Global Z Offset)
        # =========================================================
        self.z_offset = nn.Parameter(torch.tensor(0.0)) 

        self.deg_to_rad = torch.pi / 180.0
        self.base_phi_deg = torch.tensor([-30.0, 210.0, 90.0], device=device)

    def _get_elbow_positions(self, theta_batch):
        """
        内部辅助函数：根据当前的角度和参数，计算三个主动臂肘部(Elbow)的坐标
        """
        batch_size = theta_batch.shape[0]
        theta_rad = torch.zeros_like(theta_batch)
        theta_rad[:, 0] = (theta_batch[:, 0] + self.offset_theta1) * self.deg_to_rad
        theta_rad[:, 1] = (theta_batch[:, 1] + self.offset_theta2) * self.deg_to_rad
        theta_rad[:, 2] = (theta_batch[:, 2] + self.offset_theta3) * self.deg_to_rad

        phi_rad = torch.zeros(3, device=self.device)
        phi_rad[0] = (self.base_phi_deg[0] + self.d_phi1) * self.deg_to_rad
        phi_rad[1] = (self.base_phi_deg[1] + self.d_phi2) * self.deg_to_rad
        phi_rad[2] = (self.base_phi_deg[2] + self.d_phi3) * self.deg_to_rad

        L1 = self.l1
        R = self.r_base
        r = self.r_mobile
        Rr = R - r 

        elbows = torch.zeros((batch_size, 3, 3), device=self.device)
        for i in range(3):
            r_proj = Rr + L1 * torch.cos(theta_rad[:, i])
            elbows[:, i, 0] = r_proj * torch.cos(phi_rad[i])
            elbows[:, i, 1] = r_proj * torch.sin(phi_rad[i])
            elbows[:, i, 2] = -L1 * torch.sin(theta_rad[:, i])
            
        return elbows

    def forward(self, theta_batch):
        """
        正运动学: 输出坐标包含 z_offset
        """
        elbows = self._get_elbow_positions(theta_batch)
        p1 = elbows[:, 0, :]
        p2 = elbows[:, 1, :]
        p3 = elbows[:, 2, :]
        L2 = self.l2
        
        # 三球交汇逻辑 (简化的向量法)
        ex = p2 - p1
        d = torch.norm(ex, dim=1, keepdim=True)
        ex = ex / (d + 1e-6)
        
        i = torch.sum(ex * (p3 - p1), dim=1, keepdim=True)
        ey = (p3 - p1) - i * ex
        ey = ey / (torch.norm(ey, dim=1, keepdim=True) + 1e-6)
        
        ez = torch.cross(ex, ey, dim=1)
        j = torch.sum(ey * (p3 - p1), dim=1, keepdim=True)
        
        x = (d / 2) # 对于对称结构，x=d/2
        y = (L2**2 - L2**2 + i**2 + j**2) / (2*j) - (i/j)*x 
        
        # 修正：之前推导的通用公式里，对于等边三角形布局
        # i^2 + j^2 是 p1到p3距离的平方
        # 这里直接用更通用的代数解法可能更稳，但先沿用之前的逻辑
        # 只要 elbows 是对的，这个解析解就是对的
        
        z_sq = L2**2 - x**2 - y**2
        z_geom = torch.sqrt(torch.relu(z_sq)) # 取正值，因为 ez 是向下的
        
        # 计算纯几何中心
        pos_geom = p1 + x * ex + y * ey + z_geom * ez
        
        # [关键修改] 加上 offset
        # Pos_Final = Pos_Geom + Z_Offset
        # 注意: pos_geom 是一个向量，z_offset 加在 Z 轴上
        pos_final = pos_geom.clone()
        pos_final[:, 2] = pos_final[:, 2] + self.z_offset
        
        return pos_final

    def compute_physics_residue(self, theta_batch, pos_batch):
        """
        计算物理残差
        注意：我们需要把 pos_batch (包含 offset) 还原回 pos_geom，
        才能去和 elbows (纯几何) 比较距离。
        """
        # 1. 还原纯几何位置 (Remove Offset)
        pos_geom = pos_batch.clone()
        pos_geom[:, 2] = pos_geom[:, 2] - self.z_offset
        
        # 2. 获取 Elbows
        elbows = self._get_elbow_positions(theta_batch)
        
        # 3. 计算距离 (Elbows <-> Pos_Geom)
        diff = elbows - pos_geom.unsqueeze(1)
        dist_sq = torch.sum(diff**2, dim=2)
        
        L2_sq = self.l2 ** 2
        residue = torch.abs(dist_sq - L2_sq)
        
        return torch.sum(residue, dim=1)

    def update_geometry(self, l1, l2, r_base, r_mobile, z_off=None):
        with torch.no_grad():
            self.l1.fill_(l1)
            self.l2.fill_(l2)
            self.r_base.fill_(r_base)
            self.r_mobile.fill_(r_mobile)
            if z_off is not None:
                self.z_offset.fill_(z_off)
            else:
                # 自动估算: 如果没给，就保持默认 -16.3
                pass