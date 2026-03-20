import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
LOSS_CSV_PATH = "pinn_loss_history.csv"        # 您之前保存的 loss 数据文件
SINGLE_AXIS_PLOT_PATH = "pinn_single_axis_loss_curve.png" 

LAMBDA_WEIGHT = 50   # 物理损失的放大权重
SMOOTH_WEIGHT = 0.85 # EMA 平滑权重
# =======================================

# ================= 指数移动平均 (EMA) 平滑函数 =================
def smooth_curve(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ================= 绘制单 Y 轴图表 =================
def plot_single_axis():
    print("📈 准备绘制单 Y 轴 Loss 曲线 (带 EMA 平滑, Times New Roman, 18号字)...")
    if not os.path.exists(LOSS_CSV_PATH):
        print(f"❌ 找不到 Loss 数据文件: {LOSS_CSV_PATH}。请确保该文件在当前目录下。")
        return
        
    df = pd.read_csv(LOSS_CSV_PATH)
    epochs = df['epoch']
    
    # 获取 Data Loss 原始数据
    data_loss_raw = df['data'].values
    # 核心修改：获取 Physics Loss 原始数据并直接乘上 lambda 权重
    phys_loss_raw = df['phys'].values * LAMBDA_WEIGHT
    
    # ---------------- 全局字体与字号设置 ----------------
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    # ---------------------------------------------------
    
    # 应用 EMA 平滑
    data_loss_smooth = smooth_curve(data_loss_raw, weight=SMOOTH_WEIGHT)
    phys_loss_smooth = smooth_curve(phys_loss_raw, weight=SMOOTH_WEIGHT)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ---------------- 绘制统一 Y 轴的数据 ----------------
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss Value', fontweight='bold')
    
    # 1. 绘制 Data Loss (含浅色原始波动背景和深色平滑线)
    ax.plot(epochs, data_loss_raw, color=color1, alpha=0.2)
    ax.plot(epochs, data_loss_smooth, color=color1, linewidth=2.5, label='Data Loss')
    
    # 2. 绘制 加权后的 Physics Loss
    ax.plot(epochs, phys_loss_raw, color=color2, alpha=0.2)
    # 使用 LaTeX 语法渲染漂亮的数学公式图例
    ax.plot(epochs, phys_loss_smooth, color=color2, linewidth=2.5, label=r'Weighted Physics Loss ($\lambda \mathcal{L}_{phys}$)')
    
    # 统一使用对数坐标
    ax.set_yscale('log') 
    ax.grid(False) 
    
    # ---------------- 显示图例并保存 ----------------
    ax.legend(loc='upper right')
    
    plt.title(f"PINN Training Process (Single Y-Axis, $\lambda$={LAMBDA_WEIGHT})")
    fig.tight_layout() 
    
    plt.savefig(SINGLE_AXIS_PLOT_PATH, dpi=1200)
    print(f"✅ 单 Y 轴图表已保存至: {SINGLE_AXIS_PLOT_PATH}")
    plt.show()

if __name__ == "__main__":
    plot_single_axis()