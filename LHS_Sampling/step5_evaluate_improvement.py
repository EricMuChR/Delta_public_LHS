import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================= 导入运动学库 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import Delta_3 as kinematics
except ImportError:
    print("❌ 无法导入 Delta_3.py")
    sys.exit(1)

# ================= 配置 =================
FILE_CMD = "lhs_final.csv"       
FILE_MEAS = "tracker_final.csv"  
MODEL_FILE = "delta_error_model.pkl"
SCALER_X_FILE = "scaler_x.pkl"
SCALER_Y_FILE = "scaler_y.pkl"
ROBOT_PARAMS = [100, 250, 35, 23.4] 

# 清洗阈值 (建议与训练时保持一致，例如 10mm)
OUTLIER_THRESHOLD_MM = 10.0
# =======================================

def calculate_physics_features(xyz_points):
    """ 复用 Step 5 的特征计算 """
    arm = kinematics.arm(l=ROBOT_PARAMS)
    features = []
    for point in xyz_points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        arm.inverse_kinematics(tip_x_y_z=[x, y, z])
        m1, m2, m3 = arm.theta
        features.append([x, y, z, r, np.cos(theta), np.sin(theta), m1, m2, m3])
    return np.array(features)

def calc_metrics(errors, name):
    """ 辅助函数：计算指标 """
    norms = np.linalg.norm(errors, axis=1)
    mae = np.mean(norms)
    rmse = np.sqrt(np.mean(norms**2))
    max_e = np.max(norms)
    return mae, rmse, max_e

def main():
    print("="*60)
    print("   Step 5 Evaluation (Pro): 智能净值评估")
    print("="*60)

    if not os.path.exists(MODEL_FILE):
        print("❌ 模型未找到。")
        return

    # 1. 加载数据
    print("1. 加载数据...")
    df_cmd = pd.read_csv(FILE_CMD)
    df_meas = pd.read_csv(FILE_MEAS)
    X_raw = df_cmd[['x', 'y', 'z']].values
    Y_meas = df_meas[['x', 'y', 'z']].values
    
    # 2. 基础计算
    Error_Raw = Y_meas - X_raw
    Offset = np.mean(Error_Raw, axis=0)
    Error_Baseline = Error_Raw - Offset # 去除 Offset 后的残差
    
    # === 3. 智能清洗 (关键步骤) ===
    # 我们根据"仅去除Offset后的残差"来判断是否为飞点
    # 逻辑：如果一个点连 Offset 都解释不了，且偏差极大(>10mm)，那它就是测量事故
    baseline_norms = np.linalg.norm(Error_Baseline, axis=1)
    mask = baseline_norms < OUTLIER_THRESHOLD_MM
    
    n_total = len(X_raw)
    n_clean = np.sum(mask)
    n_dropped = n_total - n_clean
    
    print(f"\n🧹 数据清洗报告:")
    print(f"   总样本数 : {n_total}")
    print(f"   有效样本 : {n_clean}")
    print(f"   剔除飞点 : {n_dropped} (占比 {n_dropped/n_total*100:.2f}%)")
    print("-" * 60)

    # 4. 神经网络预测 (只预测干净数据，或者全量预测后筛选)
    print("2. 神经网络介入...")
    model = joblib.load(MODEL_FILE)
    scaler_x = joblib.load(SCALER_X_FILE)
    scaler_y = joblib.load(SCALER_Y_FILE)
    
    X_features = calculate_physics_features(X_raw)
    X_scaled = scaler_x.transform(X_features)
    Pred_Residuals_Scaled = model.predict(X_scaled)
    Pred_Residuals = scaler_y.inverse_transform(Pred_Residuals_Scaled)
    
    Error_Final = Error_Baseline - Pred_Residuals
    
    # 5. 分组评估
    # Group A: 全量数据 (All)
    mae_all, rmse_all, max_all = calc_metrics(Error_Final, "全量")
    
    # Group B: 净值数据 (Clean)
    mae_clean, rmse_clean, max_clean = calc_metrics(Error_Final[mask], "清洗后")
    
    # 基准对比 (Clean Baseline)
    mae_base_clean, rmse_base_clean, max_base_clean = calc_metrics(Error_Baseline[mask], "仅Offset(清洗后)")

    print(f"\n📊 最终性能对比 (基于 {n_clean} 个有效点)")
    print(f"{'方案':<20} | {'MAE (平均)':<10} | {'RMSE (均方根)':<10} | {'Max (最大)':<10}")
    print("-" * 65)
    print(f"{'1. 仅做 Offset':<20} | {mae_base_clean:<10.4f} | {rmse_base_clean:<10.4f} | {max_base_clean:<10.4f}")
    print(f"{'2. NN 误差补偿':<20} | {mae_clean:<10.4f} | {rmse_clean:<10.4f} | {max_clean:<10.4f}")
    print("="*65)
    
    # 计算提升
    imp_rmse = (rmse_base_clean - rmse_clean) / rmse_base_clean * 100
    print(f"\n🚀 真实性能提升 (剔除测量飞点后):")
    print(f"   模型将误差(RMSE) 进一步降低了: {imp_rmse:.2f}%")
    
    if n_dropped > 0:
        print(f"\n⚠️  注：全量数据(含飞点)的 RMSE 为 {rmse_all:.4f} mm。")
        print(f"    差异来源：那 {n_dropped} 个飞点是测量系统的问题，不是机器人的问题。")

if __name__ == "__main__":
    main()