import os
# === 关键修复: 解决 PyTorch 和 NumPy 的 OpenMP 冲突 ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 导入前两个阶段的模块
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

def analyze_sensitivity():
    print("="*60)
    print("   Phase 3: Geometric Parameter Sensitivity Analysis")
    print("="*60)
    
    # 1. 加载数据
    base_path = os.path.join(parent_dir, "LHS_Sampling")
    csv_path = os.path.join(base_path, "training_data_merged.csv")
    offset_path = os.path.join(base_path, "tool_offset.json")
    
    dataset = DeltaPhysicsDataset(csv_path, offset_path)
    
    # 随机采样 100 个点
    indices = torch.randperm(len(dataset))[:100]
    sample_thetas = dataset.inputs[indices] 
    
    # 2. 初始化可微分模型
    # [L, l, R, r]
    nominal_params = [100.0, 250.0, 35.0, 23.4]
    model = DifferentiableDeltaKinematics(nominal_params, dataset.tool_offset)
    
    # === 强制解锁所有参数进行分析 ===
    model.L.requires_grad = True
    model.l.requires_grad = True
    model.r.requires_grad = True
    model.R.requires_grad = True 
    
    # 3. 计算 Jacobian
    print("\n🧮 Computing Gradients (Jacobian)...")
    
    param_names = [
        'L_1', 'L_2', 'L_3',  # 主动臂长
        'l_1', 'l_2', 'l_3',  # 从动臂长
        'R_base',             # 静平台半径
        'r_plat'              # 动平台半径
    ]
    
    gradients = []
    
    for i in range(len(sample_thetas)):
        theta = sample_thetas[i:i+1] # (1, 3)
        
        pos = model(theta) # (1, 3)
        target_scalar = pos.norm() 
        
        model.zero_grad()
        target_scalar.backward()
        
        grads = [
            model.L.grad[0].item(), model.L.grad[1].item(), model.L.grad[2].item(),
            model.l.grad[0].item(), model.l.grad[1].item(), model.l.grad[2].item(),
            model.R.grad.item(), 
            model.r.grad.item()
        ]
        gradients.append(grads)
        
    gradients = np.array(gradients) # (100, 8)
    
    # 4. 分析结果
    sensitivity_score = np.mean(np.abs(gradients), axis=0)
    
    print("\n📊 Parameter Sensitivity Scores (Avg Gradient Magnitude):")
    print(f"{'Parameter':<10} | {'Impact Score':<15} | {'Rank'}")
    print("-" * 40)
    
    sorted_indices = np.argsort(sensitivity_score)[::-1]
    
    for rank, idx in enumerate(sorted_indices):
        print(f"{param_names[idx]:<10} | {sensitivity_score[idx]:.6f}        | {rank+1}")
        
    # 5. 相关性分析
    print("\n🔗 Coupling Analysis (Correlation Matrix):")
    corr_matrix = np.corrcoef(gradients.T)
    
    # 打印矩阵
    header = "      " + " ".join([f"{n:>6}" for n in param_names])
    print(header)
    for i in range(len(param_names)):
        row_str = f"{param_names[i]:>6} "
        for j in range(len(param_names)):
            val = corr_matrix[i, j]
            # 关注强相关 (绝对值 > 0.95)
            if abs(val) > 0.95 and i != j:
                marker = " [!!] " # 醒目标记
            else:
                marker = f"{val:5.2f} "
            row_str += marker
        print(row_str)

    # 6. 保存图片
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=param_names, yticklabels=param_names, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Parameter Coupling Correlation Matrix")
        save_path = os.path.join(current_dir, "phase3_correlation.png")
        plt.savefig(save_path)
        print(f"\n🖼️  Correlation heatmap saved to: {save_path}")
    except Exception as e:
        print(f"⚠️ Visualization skipped: {e}")

if __name__ == "__main__":
    analyze_sensitivity()