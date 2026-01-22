import pandas as pd
import numpy as np
from scipy.stats import qmc 

# ================= 配置 =================
OUTPUT_FILE = "repeatability_targets.csv"
NUM_POINTS = 12       # 测试点数量
R_MAX = 105.0         # 最大半径 
R_MIN = 5.0          # 最小半径 (避开绝对中心)
Z_RANGE = [-240, -210] # 高度范围
SEED = 42             # 固定种子，保证结果可复现
# =======================================

def main():
    print("="*60)
    print(f"   Step 1: 生成重复性测试点 (LHS 采样, N={NUM_POINTS})")
    print("="*60)

    # 1. LHS 采样
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    sample = sampler.random(n=NUM_POINTS)
    
    # 2. 映射到圆柱空间
    # sqrt(u) 保证在圆盘上均匀分布
    r_normalized = np.sqrt(sample[:, 0]) 
    radii = r_normalized * (R_MAX - R_MIN) + R_MIN
    thetas = sample[:, 1] * 2 * np.pi
    zs = sample[:, 2] * (Z_RANGE[1] - Z_RANGE[0]) + Z_RANGE[0]
    
    # 3. 转为 Cartesian 坐标
    points = []
    for r, theta, z in zip(radii, thetas, zs):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
        
    # 4. 保存
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df = df.round(3)
    df.insert(0, 'PointName', [f"P{i+1:02d}" for i in range(len(points))])
    
    print(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()