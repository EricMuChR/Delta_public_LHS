import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 跨文件夹导入 Delta_3.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    import Delta_3 as nominal_kinematics
except ImportError:
    print("❌ 无法导入 Delta_3.py，请确保 LHS_PIDL 位于 Delta_public_LHS 目录下")
    sys.exit(1)

# 名义参数 [L, l, R, r]
ROBOT_NOMINAL_PARAMS = [100, 250, 35, 23.4]

class DeltaPhysicsDataset(Dataset):
    def __init__(self, csv_file, offset_file, base_radius_override=None):
        """
        Args:
            csv_file: 训练数据
            offset_file: 工具偏置
        """
        self.df = pd.read_csv(csv_file)
        
        # 1. 加载工具偏置
        with open(offset_file, 'r') as f:
            offset_data = json.load(f)
            self.tool_offset = np.array(offset_data['tool_offset'], dtype=np.float32)
        print(f"✅ Loaded Tool Offset: {self.tool_offset}")

        # 2. 确定静平台半径 (R_base) - 智能加载逻辑
        self.R_base = ROBOT_NOMINAL_PARAMS[2] # 默认 35.0
        
        # 尝试寻找 Step 3 生成的 robot_base_params.json
        # 假设它位于 csv_file (LHS_Sampling) 同级目录下
        sampling_dir = os.path.dirname(csv_file)
        params_json_path = os.path.join(sampling_dir, "robot_base_params.json")
        
        if base_radius_override:
            self.R_base = float(base_radius_override)
            print(f"ℹ️  [Manual Override] Using R_base: {self.R_base:.4f} mm")
            
        elif os.path.exists(params_json_path):
            try:
                with open(params_json_path, 'r') as f:
                    p_data = json.load(f)
                    real_R = float(p_data["R_base"])
                    self.R_base = real_R
                print(f"✅ [Auto-Load] Found Step 3 Calibration!")
                print(f"   Loaded Real R_base: {self.R_base:.4f} mm (Nominal: {ROBOT_NOMINAL_PARAMS[2]})")
            except Exception as e:
                print(f"⚠️ Failed to read {params_json_path}: {e}")
                print(f"   Fallback to Nominal R: {self.R_base}")
        else:
            print(f"ℹ️  [Nominal] No Step 3 params found. Using Nominal R: {self.R_base}")

        # 3. 初始化标准运动学模型 (用于逆解)
        # 注意: 逆解通常使用"名义参数"来反算角度，因为控制器里就是这么写的
        # 这里的 L, l, r 还是用名义值，R 用我们刚决定的 R_base
        # 这样能最大程度还原控制器当时发出的指令角度
        ik_params = list(ROBOT_NOMINAL_PARAMS)
        ik_params[2] = self.R_base # 更新 R
        
        self.nominal_arm = nominal_kinematics.arm(l=ik_params)
        
        # 4. 解析列名
        print("🔄 Converting Command Coordinates to Joint Angles...")
        cols_cmd = ['x', 'y', 'z']
        
        if 'x.1' in self.df.columns:
            cols_meas = ['x.1', 'y.1', 'z.1']
        elif 'meas_x' in self.df.columns:
            cols_meas = ['meas_x', 'meas_y', 'meas_z']
        else:
            cols_meas = self.df.columns[-3:].tolist()
            
        print(f"   Inputs (Cmd) Cols: {cols_cmd}")
        print(f"   Targets (Meas) Cols: {cols_meas}")

        self.inputs = []
        self.targets = []
        
        valid_count = 0
        total_count = len(self.df)

        for idx, row in self.df.iterrows():
            cmd_pos = row[cols_cmd].values.astype(float)
            real_pos = row[cols_meas].values.astype(np.float32)
            
            if self.nominal_arm.inverse_kinematics(tip_x_y_z=cmd_pos):
                theta = np.array(self.nominal_arm.theta, dtype=np.float32)
                self.inputs.append(theta)
                self.targets.append(real_pos)
                valid_count += 1
            else:
                pass

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

        print(f"✅ Dataset Ready. Valid samples: {valid_count}/{total_count} "
              f"({valid_count/total_count*100:.1f}%)")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

if __name__ == "__main__":
    base_path = os.path.join(parent_dir, "LHS_Sampling")
    csv_path = os.path.join(base_path, "training_data_merged.csv")
    offset_path = os.path.join(base_path, "tool_offset.json")
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: {csv_path} not found.")
        sys.exit(0)

    dataset = DeltaPhysicsDataset(csv_path, offset_path)
    
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    print("\n--- Data Sample Check ---")
    for thetas, reals in loader:
        print(f"Theta (rad): \n{thetas}")
        print(f"Real Pos (mm): \n{reals}")
        break