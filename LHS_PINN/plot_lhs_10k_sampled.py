import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_top_tier_workspace_times_new_roman():
    # ==========================================
    # 1. 数据加载与降采样 (Data Loading & Subsampling)
    # ==========================================
    try:
        # 读取一万点的合并数据集
        df_all = pd.read_csv('training_data_pinn_cleaned.csv')
        
        # 随机降采样到 2500 个点，保持分布特性，避免视觉拥挤
        N_samples = 2500
        df_sampled = df_all.sample(n=N_samples, random_state=42)
        
        # 提取物理坐标 XYZ
        X = df_sampled.iloc[:, 1]
        Y = df_sampled.iloc[:, 2]
        Z = df_sampled.iloc[:, 3]
        
        # 提取电机角度
        theta1 = df_sampled.iloc[:, 4]
        theta2 = df_sampled.iloc[:, 5]
        theta3 = df_sampled.iloc[:, 6]
        
    except Exception as e:
        print(f"数据读取失败，请检查文件: {e}")
        return

    # ==========================================
    # 2. 全局字体与绘图设置 (Times New Roman, Size 18)
    # ==========================================
    # 强制所有字体使用 Times New Roman，字号统一为 18
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14  # 刻度数字稍微小一点，防止互相挤压
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.unicode_minus'] = False
    
    # 稍微加宽图片尺寸，给 18 号大字体留出足够的排版空间
    fig = plt.figure(figsize=(16, 7), dpi=300)

    # ==========================================
    # 3. 左子图：关节空间的 LHS 均匀分布
    # ==========================================
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 绘制气泡感散点
    ax1.scatter(theta1, theta2, theta3, 
                c='#94A89A', s=30, alpha=0.85, 
                edgecolors='white', linewidth=0.4)
    
    # 强制限制三个电机角度坐标轴的显示范围在正负 40 之间
    ax1.set_xlim([-40, 40])
    ax1.set_ylim([-40, 40])
    ax1.set_zlim([-40, 40])
    
    ax1.set_title(f'(a) LHS Joint Space (N={N_samples})', pad=20, fontweight='bold')
    ax1.set_xlabel(r'$\theta_1$ [deg]', labelpad=15)
    ax1.set_ylabel(r'$\theta_2$ [deg]', labelpad=15)
    ax1.set_zlabel(r'$\theta_3$ [deg]', labelpad=15)

    # ==========================================
    # 4. 右子图：笛卡尔空间的纯物理落点 (移除灰色圆柱)
    # ==========================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 绘制实际物理落点
    ax2.scatter(X, Y, Z, 
                c='#D8C06A', s=30, alpha=0.85, 
                edgecolors='white', linewidth=0.4)
    
    ax2.set_title('(b) Cartesian Workspace', pad=20, fontweight='bold')
    ax2.set_xlabel('X [mm]', labelpad=15)
    ax2.set_ylabel('Y [mm]', labelpad=15)
    ax2.set_zlabel('Z [mm]', labelpad=15)

    # ==========================================
    # 5. 视角与网格的高级优化
    # ==========================================
    for ax in [ax1, ax2]:
        # 统一设置一个优雅的等轴测视角
        ax.view_init(elev=25, azim=45)
            
        # 让 3D 坐标轴的背板变为完全透明（纯白背景）
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # 设置浅灰色的高级虚线网格
        ax.xaxis._axinfo["grid"].update({"color": "#D3D3D3", "linestyle": "--"})
        ax.yaxis._axinfo["grid"].update({"color": "#D3D3D3", "linestyle": "--"})
        ax.zaxis._axinfo["grid"].update({"color": "#D3D3D3", "linestyle": "--"})
        
        # 调整坐标轴数字的间距，防止字号变大后紧贴坐标轴
        ax.tick_params(axis='x', pad=5)
        ax.tick_params(axis='y', pad=5)
        ax.tick_params(axis='z', pad=5)

    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig('LHS_Workspace_TNR.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('LHS_Workspace_TNR.png', format='png', dpi=300, bbox_inches='tight')
    print("绘图完成！已生成纯净版的 LHS_Workspace_TNR.pdf")
    
    plt.show()

if __name__ == '__main__':
    plot_top_tier_workspace_times_new_roman()