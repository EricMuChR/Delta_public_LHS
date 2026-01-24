import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import copy # 用于备份最佳模型

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

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🌱 Random Seed Set to: {seed}")

def run_static_calibration():
    print("="*60)
    print("   Phase 4 (Final v4): Best-Model Saving & Tighter Constraints")
    print("="*60)
    
    # 1. 设置种子
    set_seed(42)

    # 2. 加载数据
    base_path = os.path.join(parent_dir, "LHS_Sampling")
    csv_path = os.path.join(base_path, "training_data_merged.csv")
    offset_path = os.path.join(base_path, "tool_offset.json")
    
    dataset = DeltaPhysicsDataset(csv_path, offset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    
    # 3. 初始化
    nominal_params = [100.0, 250.0, dataset.R_base, 23.4]
    model = DifferentiableDeltaKinematics(nominal_params, dataset.tool_offset)
    
    # 4. 策略: 锁半径，训零点
    model.lock_all_params()
    model.L.requires_grad = True       
    model.l.requires_grad = True       
    model.theta_offset.requires_grad = True 
    model.R.requires_grad = False
    model.r.requires_grad = False
    
    print("\n🔒 Strategy: Save Best Model + Tight Constraints (±2mm)")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    epochs = 300 
    loss_history = []
    
    # === 关键变量: 记录最佳状态 ===
    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0
    
    with torch.no_grad():
        initial_pred = model(dataset.inputs)
        initial_error = (initial_pred - dataset.targets).norm(dim=1).mean().item()
        print(f"\n📉 Pre-Calibration RMSE: {initial_error:.4f} mm")

    print("\n🚀 Starting Calibration...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = nn.MSELoss()(preds, batch_targets)
            loss.backward()
            optimizer.step()
            
            # === Safety Clamp (更严格的限制) ===
            with torch.no_grad():
                # 既然制造精度可信，我们只允许微调 ±2.0mm
                # 防止模型为了代偿零点误差而把杆长改得面目全非
                model.L.data.clamp_(100.0 - 2.0, 100.0 + 2.0)
                model.l.data.clamp_(250.0 - 2.0, 250.0 + 2.0)
                # 零点限制 ±0.1 rad (~5.7度)
                model.theta_offset.data.clamp_(-0.1, 0.1)
            
            epoch_loss += loss.item() * batch_inputs.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        rmse = np.sqrt(avg_loss)
        loss_history.append(rmse)
        
        # === 关键逻辑: 保存最佳模型 ===
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            # 深拷贝当前的参数状态
            best_model_state = copy.deepcopy(model.state_dict())
        
        scheduler.step(rmse)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch+1) % 20 == 0:
            # 打印当前 RMSE 和 历史最佳 RMSE
            print(f"   Epoch {epoch+1}/{epochs} | Curr: {rmse:.4f} mm | Best: {best_rmse:.4f} (Ep {best_epoch+1})")

    print("\n✅ Calibration Complete!")
    
    # === 恢复最佳参数 ===
    print(f"\n🏆 Restoring Best Model from Epoch {best_epoch+1} (RMSE: {best_rmse:.4f} mm)...")
    model.load_state_dict(best_model_state)
    
    print("\n📋 Final Calibrated Parameters (Best Snapshot):")
    
    def fmt_param(name, nominal, calibrated_tensor):
        vals = calibrated_tensor.detach().numpy()
        if vals.size == 1:
            diff = vals - nominal
            print(f"   {name:<12}: {vals:.4f} (Delta: {diff:+.4f})")
        else:
            print(f"   {name:<12}:")
            for i, v in enumerate(vals):
                diff = v - nominal
                print(f"      Arm {i+1} : {v:.4f} (Delta: {diff:+.4f})")

    fmt_param("L (mm)", 100.0, model.L)
    fmt_param("l (mm)", 250.0, model.l)
    fmt_param("R (mm)", dataset.R_base, model.R)
    fmt_param("r (mm)", 23.4, model.r)
    
    offsets_deg = np.degrees(model.theta_offset.detach().numpy())
    print(f"   Theta Offs (deg):")
    for i, v in enumerate(offsets_deg):
        print(f"      Motor {i+1}: {v:+.4f} deg")
    
    with torch.no_grad():
        final_errors = (model(dataset.inputs) - dataset.targets).norm(dim=1)
        final_rmse = final_errors.mean().item()
        max_error = final_errors.max().item()
        
    print("-" * 40)
    print(f"📉 Improvement: {initial_error:.4f} mm -> {final_rmse:.4f} mm")
    print(f"⚠️ Max Error  : {max_error:.4f} mm")
    print("-" * 40)
    
    calibrated_params = {
        'L': model.L.data.clone(),
        'l': model.l.data.clone(),
        'R': model.R.data.clone(),
        'r': model.r.data.clone(),
        'theta_offset': model.theta_offset.data.clone(),
        'tool_offset': model.tool_offset.data.clone()
    }
    save_path = os.path.join(current_dir, "calibrated_params.pth")
    torch.save(calibrated_params, save_path)
    print(f"💾 Saved to: {save_path}")

    plt.figure()
    plt.plot(loss_history)
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.title(f"Phase 4: Best RMSE {best_rmse:.4f} at Epoch {best_epoch+1}")
    plt.legend()
    plt.savefig(os.path.join(current_dir, "phase4_loss.png"))

if __name__ == "__main__":
    run_static_calibration()