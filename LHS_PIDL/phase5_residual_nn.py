import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from LHS_PIDL.phase1_data_loader import DeltaPhysicsDataset
    from LHS_PIDL.phase2_diff_kinematics import DifferentiableDeltaKinematics
except ImportError:
    try:
        from phase1_data_loader import DeltaPhysicsDataset
        from phase2_diff_kinematics import DifferentiableDeltaKinematics
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)

# === 定义残差网络 ===
class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: 3个关节角 + 3个名义位置(作为提示特征)
        # 输出: 3个坐标修正量 (dx, dy, dz)
        self.net = nn.Sequential(
            nn.Linear(3, 128), # 仅输入角度 theta
            nn.BatchNorm1d(128),
            nn.Tanh(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            
            nn.Linear(128, 3) # 输出 dx, dy, dz
        )
        
        # 初始化最后一层为接近0，确保初始状态下 NN 不捣乱
        nn.init.uniform_(self.net[-1].weight, -0.001, 0.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # 输出被 Tanh 限制在 -1~1 之间? 不，这里没加 Tanh 输出层
        # 我们希望 NN 能输出 mm 级别的修正
        return self.net(x)

def run_residual_training():
    print("="*60)
    print("   Phase 5: Residual Neural Network Training")
    print("="*60)
    
    # 1. 加载数据
    base_path = os.path.join(parent_dir, "LHS_Sampling")
    csv_path = os.path.join(base_path, "training_data_merged.csv")
    offset_path = os.path.join(base_path, "tool_offset.json")
    
    dataset = DeltaPhysicsDataset(csv_path, offset_path)
    
    # 2. 加载 Phase 4 标定好的参数
    calib_file = os.path.join(current_dir, "calibrated_params.pth")
    if not os.path.exists(calib_file):
        print("❌ 找不到 calibrated_params.pth！请先运行 Phase 4。")
        return

    print(f"📥 Loading calibrated parameters from: {calib_file}")
    calib_data = torch.load(calib_file)
    
    # 3. 初始化物理模型 (并赋值)
    # 注意：这里初始化时用什么参数不重要，因为马上会被 load_state_dict 覆盖
    # 但为了保险，还是传名义值
    model_geo = DifferentiableDeltaKinematics([100, 250, 35, 23.4], dataset.tool_offset)
    
    # 手动赋值参数
    model_geo.L.data = calib_data['L']
    model_geo.l.data = calib_data['l']
    model_geo.R.data = calib_data['R']
    model_geo.r.data = calib_data['r']
    model_geo.theta_offset.data = calib_data['theta_offset']
    
    # 锁死物理模型 (它现在是我们的基准，不许动了)
    model_geo.lock_all_params()
    model_geo.eval() # 设为评估模式
    
    # 4. 准备残差数据 (Pre-calculate Residuals)
    # 这一步很关键：我们先算出 Phase 4 模型目前的误差，作为 NN 的训练目标
    print("\n🧮 Calculating Geometric Residuals...")
    with torch.no_grad():
        # 预测位置 (物理模型)
        pred_geo = model_geo(dataset.inputs)
        # 残差 = 真实值 - 物理预测
        residuals = dataset.targets - pred_geo
        
        initial_rmse = residuals.norm(dim=1).mean().item()
        print(f"📉 Baseline RMSE (Phase 4): {initial_rmse:.4f} mm")
    
    # 创建新的数据集: Input=Theta, Target=Residuals
    res_dataset = TensorDataset(dataset.inputs, residuals)
    # 划分训练/验证集 (80/20)
    train_size = int(0.8 * len(res_dataset))
    val_size = len(res_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(res_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # 5. 初始化神经网络
    model_nn = ResidualMLP()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    print("\n🚀 Starting NN Training...")
    epochs = 300
    train_loss_history = []
    val_loss_history = []
    
    best_val_rmse = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # --- Training ---
        model_nn.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred_res = model_nn(batch_x)
            loss = nn.MSELoss()(pred_res, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        
        train_rmse = np.sqrt(epoch_loss / len(train_ds))
        train_loss_history.append(train_rmse)
        
        # --- Validation ---
        model_nn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred_res = model_nn(batch_x)
                loss = nn.MSELoss()(pred_res, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_rmse = np.sqrt(val_loss / len(val_ds))
        val_loss_history.append(val_rmse)
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model_nn.state_dict()
        
        scheduler.step(val_rmse)
        
        if (epoch+1) % 20 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

    # 6. 最终评估
    print("\n✅ NN Training Complete!")
    
    # 加载最佳权重
    model_nn.load_state_dict(best_model_state)
    
    # 在全集上评估最终效果
    model_nn.eval()
    with torch.no_grad():
        # 混合预测: 物理 + NN
        pred_geo = model_geo(dataset.inputs)
        pred_nn = model_nn(dataset.inputs)
        final_pred = pred_geo + pred_nn
        
        final_errors = (final_pred - dataset.targets).norm(dim=1)
        final_rmse = final_errors.mean().item()
        max_error = final_errors.max().item()
    
    print("-" * 40)
    print(f"📉 Final Results (Hybrid PIDL):")
    print(f"   Baseline (Phase 4): {initial_rmse:.4f} mm")
    print(f"   Final    (Phase 5): {final_rmse:.4f} mm")
    print(f"   Max Error         : {max_error:.4f} mm")
    print("-" * 40)
    
    # 保存最终混合模型
    # 我们保存两个部分: 几何参数字典 + NN 权重字典
    final_model_package = {
        'geo_params': calib_data,
        'nn_state': best_model_state
    }
    save_path = os.path.join(current_dir, "pidl_final_model.pth")
    torch.save(final_model_package, save_path)
    print(f"💾 Final PIDL model saved to: {save_path}")
    
    # 绘图
    plt.figure()
    plt.plot(train_loss_history, label='Train RMSE')
    plt.plot(val_loss_history, label='Val RMSE')
    plt.title("Phase 5: Residual NN Training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (mm)")
    plt.legend()
    plt.savefig(os.path.join(current_dir, "phase5_loss.png"))

if __name__ == "__main__":
    run_residual_training()