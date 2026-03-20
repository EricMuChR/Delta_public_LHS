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
import torch.optim.lr_scheduler as lr_scheduler
import json

# ================= 配置 =================
DATA_FILE = "training_data_pinn_cleaned.csv"
MODEL_SAVE_PATH = "pinn_model.pth"
LOSS_CSV_PATH = "pinn_loss_history.csv"        
DUAL_LOSS_PLOT_PATH = "pinn_dual_loss_curve.png" 

# 超参数
EPOCHS = 500           
LR = 0.005             
BATCH_SIZE = 64        
PHYSICS_WEIGHT = 0.5   
# =======================================

# 1. 定义混合网络 (Hybrid Network)
class DeltaPINN(nn.Module):
    def __init__(self, physics_layer):
        super(DeltaPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(), 
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 3) 
        )
        self.physics = physics_layer

    def forward(self, theta):
        phys_pos = self.physics(theta)
        nn_correction = self.net(theta)
        final_pos = phys_pos + nn_correction
        return final_pos, phys_pos, nn_correction

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        return None, None
        
    df = pd.read_csv(DATA_FILE)
    print(f"📊 加载数据: {len(df)} 条样本")
    
    X = df[['theta_1', 'theta_2', 'theta_3']].values
    y = df[['meas_x', 'meas_y', 'meas_z']].values
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ================= 新增：指数移动平均 (EMA) 平滑函数 =================
def smooth_curve(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ================= 功能 1：训练并保存数据 =================
def train_and_save():
    print("🚀 启动 PINN 训练 (仅 500 轮)...")
    
    X_train, y_train = load_data()
    if X_train is None: return
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    physics_layer = DeltaKinematics()
    physics_layer.update_geometry(100.0, 250.0, 35.0, 23.4, z_off=0.0)
    
    model = DeltaPINN(physics_layer)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_data = nn.MSELoss() 
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    history = {'total': [], 'data': [], 'phys': []}
    
    print(f"   配置: Epochs={EPOCHS} | Lambda={PHYSICS_WEIGHT} | Batch={BATCH_SIZE}")
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0
        ep_data = 0
        ep_phys = 0
        
        for theta_batch, target_pos in dataloader:
            optimizer.zero_grad()
            
            pred_pos, phys_pos, _ = model(theta_batch)
            loss_data = criterion_data(pred_pos, target_pos)
            
            residue = physics_layer.compute_physics_residue(theta_batch, pred_pos)
            loss_phys = torch.mean(residue)
            
            loss = loss_data + PHYSICS_WEIGHT * loss_phys
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            ep_loss += loss.item()
            ep_data += loss_data.item()
            ep_phys += loss_phys.item()
        
        avg_loss = ep_loss / len(dataloader)
        history['total'].append(avg_loss)
        history['data'].append(ep_data / len(dataloader))
        history['phys'].append(ep_phys / len(dataloader))
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1:04d} | Loss: {avg_loss:.4f} (Data: {history['data'][-1]:.4f} + Phys: {history['phys'][-1]:.4f})")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 模型已保存: {MODEL_SAVE_PATH}")
    
    df_history = pd.DataFrame(history)
    df_history.index.name = 'epoch'
    df_history.to_csv(LOSS_CSV_PATH)
    print(f"💾 Loss 历史数据已保存至: {LOSS_CSV_PATH}")

# ================= 功能 2：读取数据并绘制双 Y 轴图 (带平滑与字体设置) =================
def plot_dual_axis():
    print(f"📈 准备绘制双 Y 轴 Loss 曲线 (带 EMA 平滑, Times New Roman, 18号字)...")
    if not os.path.exists(LOSS_CSV_PATH):
        print(f"❌ 找不到 Loss 数据文件: {LOSS_CSV_PATH}。请先执行功能 1 进行训练。")
        return
        
    df = pd.read_csv(LOSS_CSV_PATH)
    epochs = df['epoch']
    data_loss_raw = df['data'].values
    phys_loss_raw = df['phys'].values
    
    # ---------------- 全局字体与字号设置 ----------------
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    # ---------------------------------------------------
    
    # 应用 EMA 平滑
    SMOOTH_WEIGHT = 0.85
    data_loss_smooth = smooth_curve(data_loss_raw, weight=SMOOTH_WEIGHT)
    phys_loss_smooth = smooth_curve(phys_loss_raw, weight=SMOOTH_WEIGHT)
    
    # 因为字号变大了，适当调大一点画布，防止文字互相挤压或被裁减
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # ---------------- 绘制左侧 Y 轴 (Data Loss) ----------------
    color1 = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Data Loss (MSE)', color=color1, fontweight='bold')
    
    ax1.plot(epochs, data_loss_raw, color=color1, alpha=0.2)
    line1, = ax1.plot(epochs, data_loss_smooth, color=color1, linewidth=2.5, label='Data Loss (Smoothed)')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log') 
    ax1.grid(False) 
    
    # ---------------- 创建共享 X 轴的右侧 Y 轴 (Physics Loss) ----------------
    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('Physics Loss (Residue)', color=color2, fontweight='bold')
    
    ax2.plot(epochs, phys_loss_raw, color=color2, alpha=0.2)
    line2, = ax2.plot(epochs, phys_loss_smooth, color=color2, linewidth=2.5, label='Physics Loss (Smoothed)')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log') 
    ax2.grid(False) 
    
    # ---------------- 合并图例 ----------------
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    # 图例字体也会自动继承 18 号字
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title(f"PINN Training Process (Dual Y-Axis, EMA Smoothing={SMOOTH_WEIGHT})")
    fig.tight_layout() 
    
    plt.savefig(DUAL_LOSS_PLOT_PATH, dpi=1200)
    print(f"✅ 带有平滑效果和新字体的双 Y 轴图表已保存至: {DUAL_LOSS_PLOT_PATH}")
    plt.show()

# ================= 主程序入口 =================
def main():
    print("=" * 50)
    print("请选择要执行的功能:")
    print("  [1] 重新训练模型 (500 轮) 并将 Loss 数据导出为 CSV")
    print("  [2] 读取现有的 CSV 数据，并绘制带平滑效果的双 Y 轴 Loss 图表")
    print("=" * 50)
    
    choice = input("请输入 1 或 2: ").strip()
    
    if choice == '1':
        train_and_save()
    elif choice == '2':
        plot_dual_axis()
    else:
        print("❌ 输入无效，请输入 1 或 2。程序退出。")

if __name__ == "__main__":
    main()