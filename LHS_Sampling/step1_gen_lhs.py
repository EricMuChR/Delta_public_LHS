import numpy as np
import pandas as pd
from scipy.stats import qmc 
import math

def generate_lhs_samples(density_mm=6, output_file="lhs_points.csv"):
    # ================= 配置区域 =================
    # 通用 Z 轴范围
    z_bounds = [-240, -190]
    
    # 形状参数
    BOX_X = [-100, 100]
    BOX_Y = [-100, 100]
    CYL_RADIUS = 125  # 半径 125mm
    # ===========================================

    print("-" * 40)
    print("请选择采样空间形状:")
    print(" [1] 长方体 (Box) - 传统模式")
    print(" [2] 圆柱体 (Cylinder) - Delta 推荐")
    choice = input("输入 1 或 2 (默认2): ").strip()
    is_cylinder = (choice != '1') # 默认圆柱

    # === 1. 计算体积和点数 ===
    if is_cylinder:
        # 圆柱体积 = PI * r^2 * h
        volume = math.pi * (CYL_RADIUS**2) * (z_bounds[1] - z_bounds[0])
        print(f"模式: 圆柱体 (R={CYL_RADIUS}, Z={z_bounds})")
    else:
        # 长方体体积
        volume = (BOX_X[1]-BOX_X[0]) * (BOX_Y[1]-BOX_Y[0]) * (z_bounds[1]-z_bounds[0])
        print(f"模式: 长方体 (X={BOX_X}, Y={BOX_Y}, Z={z_bounds})")

    unit_vol = density_mm ** 3
    n_samples = int(volume / unit_vol)
    
    # 限制最大最小点数
    n_samples = max(10, min(n_samples, 20000))
    print(f"空间密度: {density_mm}mm/点 -> 计划生成 {n_samples} 个点")

    # === 2. LHS 采样 (在 0-1 的单位超立方体内生成) ===
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(n=n_samples) # 得到 N x 3 的矩阵，每列都在 [0,1]

    # === 3. 坐标映射 ===
    final_points = []
    
    if is_cylinder:
        # col 0 -> Z 轴 (线性)
        # col 1 -> 角度 Theta (0 到 2pi)
        # col 2 -> 半径 r (开根号分布，确保面积均匀)
        
        for row in sample:
            u_z, u_theta, u_r = row
            
            # Z: 线性映射
            z = z_bounds[0] + u_z * (z_bounds[1] - z_bounds[0])
            
            # Theta: 0 ~ 2*pi
            theta = u_theta * 2 * math.pi
            
            # R: R_max * sqrt(u_r)
            r = CYL_RADIUS * math.sqrt(u_r)
            
            # 极坐标转直角坐标
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            
            final_points.append([x, y, z])
            
    else:
        # 长方体: 直接线性缩放 XYZ
        l_bounds = np.array([BOX_X[0], BOX_Y[0], z_bounds[0]])
        u_bounds = np.array([BOX_X[1], BOX_Y[1], z_bounds[1]])
        final_points = qmc.scale(sample, l_bounds, u_bounds)

    # === 4. 保存 ===
    df = pd.DataFrame(final_points, columns=['x', 'y', 'z'])
    df = df.round(3)
    df.to_csv(output_file, index=False)
    
    print(f"采样完成！文件已保存至: {output_file}")
    if is_cylinder:
        print("提示: 圆柱模式下，点在 x-y 平面上呈圆形分布。")
    print("预览:\n", df.head())

if __name__ == "__main__":
    generate_lhs_samples()