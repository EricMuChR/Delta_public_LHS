import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from Delta_Torch import DeltaKinematics
import os
import json

# ================= 配置 =================
DATA_FILE = "training_data_pinn_cleaned.csv"
MODEL_SAVE_PATH = "pinn_model.pth"
LOSS_PLOT_PATH = "pinn_loss_curve.png"

# 超参数
EPOCHS = 3000          # 训练轮数
LR = 0.005             # 学习率
BATCH_SIZE = 64        # 批次大小
PHYSICS_WEIGHT = 0.5   # 物理Loss的权重 (lambda)
# =======================================

# 1. 定义混合网络 (Hybrid Network)
class DeltaPINN(nn.Module):
    def __init__(self, physics_layer):
        super(DeltaPINN, self).__init__()
        
        # A. 纯黑盒拟合部分 (Fitting Error)
        # 输入: 3个角度 -> 输出: 3个坐标修正量(或直接坐标)
        # 策略: 我们让 NN 预测 "残差修正量" (Delta_Pos)，然后叠加到 物理预测 (Phys_Pos) 上
        # 这样网络只需要学习 "物理模型搞不定的那部分误差"
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(), # Tanh 适合物理回归
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # Output: dx, dy, dz
        )
        
        # B. 物理层 (Physics Layer)
        # 这是一个包含可学习参数(L1, L2...)的层
        self.physics = physics_layer

    def forward(self, theta):
        # 1. 物理层预测 (基于当前学习到的几何参数)
        # 这相当于一个 "粗略但符合物理规律" 的预测
        phys_pos = self.physics(theta)
        
        # 2. 神经网络预测 (黑盒修正)
        # 它可以修补 摩擦力、齿轮间隙 等非几何误差
        nn_correction = self.net(theta)
        
        # 3. 最终预测
        final_pos = phys_pos + nn_correction
        
        return final_pos, phys_pos, nn_correction

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        print("   请先运行 step4_sync_auto_offset.py 生成训练数据。")
        return None, None
        
    df = pd.read_csv(DATA_FILE)
    print(f"📊 加载数据: {len(df)} 条样本")
    
    # 提取输入 (theta) 和 标签 (meas_x, meas_y, meas_z)
    # 确保列名匹配 step4 的输出
    X = df[['theta_1', 'theta_2', 'theta_3']].values
    y = df[['meas_x', 'meas_y', 'meas_z']].values
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    print("🚀 启动 PINN 训练...")
    
    # 1. 准备数据
    X_train, y_train = load_data()
    if X_train is None: return
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. 初始化物理内核
    physics_layer = DeltaKinematics()
    # 确保 Z_Offset 归零 (我们刚才验证过的)
    physics_layer.update_geometry(100.0, 250.0, 35.0, 23.4, z_off=0.0)
    
    # 3. 初始化 PINN
    model = DeltaPINN(physics_layer)
    
    # 优化器: 同时优化 NN权重 和 物理参数(L1, L2...)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Loss 函数
    criterion_data = nn.MSELoss() # 监督学习 Loss
    
    history = {'total': [], 'data': [], 'phys': []}
    
    # 4. 训练循环
    print(f"   配置: Epochs={EPOCHS} | Lambda={PHYSICS_WEIGHT} | Batch={BATCH_SIZE}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0
        ep_data = 0
        ep_phys = 0
        
        for theta_batch, target_pos in dataloader:
            optimizer.zero_grad()
            
            # --- Forward ---
            # pred_pos: 最终预测 (用于和标签对比)
            # phys_pos: 纯物理预测 (用于计算物理约束)
            pred_pos, phys_pos, _ = model(theta_batch)
            
            # --- Loss 1: Data Loss (Ground Truth) ---
            # 预测值必须接近真实测量值
            loss_data = criterion_data(pred_pos, target_pos)
            
            # --- Loss 2: Physics Loss (Geometric Constraint) ---
            # 这里的核心思想：
            # 我们希望 'phys_pos' (物理层的输出) 自身要满足几何约束 (L2长度守恒)
            # 同时，物理层里的参数 (L1, R...) 会为了满足这个约束而被优化
            # 注意: 我们也可以对 pred_pos 施加物理约束，但这会限制 NN 的修正能力
            # 这里我们选择约束物理层，让物理层尽可能“真实”
            residue = physics_layer.compute_physics_residue(theta_batch, phys_pos)
            loss_phys = torch.mean(residue)
            
            # --- Total Loss ---
            loss = loss_data + PHYSICS_WEIGHT * loss_phys
            
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            ep_data += loss_data.item()
            ep_phys += loss_phys.item()
        
        # 记录
        avg_loss = ep_loss / len(dataloader)
        history['total'].append(avg_loss)
        history['data'].append(ep_data / len(dataloader))
        history['phys'].append(ep_phys / len(dataloader))
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.4f} (Data: {history['data'][-1]:.4f} + Phys: {history['phys'][-1]:.4f})")
            # 打印一下当前的物理参数学习情况
            print(f"    [Params] L1: {model.physics.l1.item():.2f} | R: {model.physics.r_base.item():.2f}")

    # 5. 保存与可视化
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")
    
    # 导出学习到的物理参数
    learned_params = {
        "l1": model.physics.l1.item(),
        "l2": model.physics.l2.item(),
        "r_base": model.physics.r_base.item(),
        "r_mobile": model.physics.r_mobile.item(),
        "d_phi": [
            model.physics.d_phi1.item(),
            model.physics.d_phi2.item(),
            model.physics.d_phi3.item()
        ],
        "theta_offset": [
            model.physics.offset_theta1.item(),
            model.physics.offset_theta2.item(),
            model.physics.offset_theta3.item()
        ]
    }
    with open("learned_physics_params.json", "w") as f:
        json.dump(learned_params, f, indent=4)
    print("📝 学习到的物理参数已导出。")

    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(history['data'], label='Data Loss (MSE)')
    plt.plot(history['phys'], label='Physics Loss (Residue)')
    plt.yscale('log')
    plt.legend()
    plt.title("PINN Training Process")
    plt.grid(True)
    plt.savefig(LOSS_PLOT_PATH)
    print("📈 Loss 曲线已保存。")

if __name__ == "__main__":
    main()