import pandas as pd
import numpy as np
import os
import yaml 

FILE_CMD = "lhs_final.csv"
FILE_MEAS = "tracker_final.csv"
OUTPUT_DATA_FILE = "training_data_merged.csv"
OUTPUT_CONFIG_FILE = "config_generated.yaml"

# ⚠️ 务必确认这里的物理参数
ROBOT_NOMINAL = {
    "fixed_platform_radius": 35.0,    # R
    "moving_platform_radius": 23.4,   # r
    "active_arm_length": 100.0,       # l1 (主动臂)
    "passive_arm_length": 250.0,      # l2 (从动臂)
    "moving_platform_hinge_spacing": 49.0 # 球铰间距
}

# ⚠️ 务必填入 Step 4 算出的值
TOOL_OFFSET = [-0.2698, -1.2463, -31.5736] 

def main():
    # ... (合并数据逻辑同前) ...
    df_cmd = pd.read_csv(FILE_CMD)
    df_meas = pd.read_csv(FILE_MEAS)
    df_merged = pd.concat([df_cmd, df_meas], axis=1)
    df_merged.to_csv(OUTPUT_DATA_FILE, index=False)

    # ... (生成 Config 逻辑) ...
    config = {
        "run_settings": { "mode": "identify", "device": "cpu" }, # 确保是 identify
        "paths": {
            "identification_data": os.path.abspath(OUTPUT_DATA_FILE), 
            "identified_params_output": "results/identified_params.npy",
            # ... 其他路径保持默认 ...
        },
        "robot_parameters": {
            **ROBOT_NOMINAL, # 解包上面的参数
            "tool_offset": TOOL_OFFSET 
        },
        "stage_one_settings": {
            "global_search": { "population_size": 96, "generations": 700, "param_lower_bound": -25.0, "param_upper_bound": 25.0 },
            "local_finetune": {
                "adam": { "learning_rate": 0.001, "max_iterations": 100000, "convergence_grad_norm": 1e-5 },
                "lbfgs": { "max_iterations": 500, "convergence_gtol": 1e-9 }
            }
        }
        # ... (Stage 2 设置可忽略) ...
    }
    
    with open(OUTPUT_CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    print("Step 5 完成。请搬运文件。")

if __name__ == "__main__":
    main()