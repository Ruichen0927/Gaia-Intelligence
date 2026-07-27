# src/tools/lithology_tool.py
"""岩性解释工具：使用多矿物模型求解岩性组分。"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import Dict

from scipy.optimize import nnls
from tqdm import tqdm

from common.config_loader import ensure_dirs


def _get_params(params: Dict) -> Dict:
    defaults = {
        "den_col": "DEN",
        "cnl_col": "CNL",
        "ac_col": "AC",
        "vsh_col": "vsh_final",
        "phie_col": "phie_final",
        "vsan_col": "vsan_auto",
        "vlim_col": "vlim_auto",
        "vdol_col": "vdol_auto",
        "carb_auto_col": "carb_auto",
        "mineral_matrix": {
            "sandstone": [2.65, -0.04, 182],
            "limestone": [2.71, 0.0, 156],
            "dolomite": [2.87, 0.02, 143]
        },
        "shale_params": [2.55, 0.35, 320],
        "fluid_params": [1.0, 1.0, 620],
        "output_subfolder": "interpreted_wells"
    }
    return {**defaults, **params}


def solve_lithology(row: pd.Series, params: Dict) -> pd.Series:
    p = _get_params(params)
    required_cols = [p["den_col"], p["cnl_col"], p["ac_col"], p["vsh_col"], p["phie_col"]]
    if row[required_cols].isnull().any():
        return pd.Series([np.nan, np.nan, np.nan], index=[p["vsan_col"], p["vlim_col"], p["vdol_col"]])

    A = np.array([
        p["mineral_matrix"]["sandstone"],
        p["mineral_matrix"]["limestone"],
        p["mineral_matrix"]["dolomite"]
    ]).T

    b = np.array([
        row[p["den_col"]] - row[p["vsh_col"]] * p["shale_params"][0] - row[p["phie_col"]] * p["fluid_params"][0],
        (row[p["cnl_col"]] / 100.0) - row[p["vsh_col"]] * p["shale_params"][1] - row[p["phie_col"]] * p["fluid_params"][1],
        row[p["ac_col"]] - row[p["vsh_col"]] * p["shale_params"][2] - row[p["phie_col"]] * p["fluid_params"][2]
    ])

    matrix_volume_sum = 1.0 - row[p["vsh_col"]] - row[p["phie_col"]]
    if matrix_volume_sum < 0:
        matrix_volume_sum = 0

    A_constrained = np.vstack([A, [1, 1, 1]])
    b_constrained = np.append(b, matrix_volume_sum)

    try:
        solution, _ = nnls(A_constrained, b_constrained)
        vsan, vlim, vdol = solution
        total_solved_vol = solution.sum()
        if total_solved_vol > 1e-6:
            factor = matrix_volume_sum / total_solved_vol
            vsan, vlim, vdol = vsan * factor, vlim * factor, vdol * factor
        return pd.Series([vsan, vlim, vdol], index=[p["vsan_col"], p["vlim_col"], p["vdol_col"]])
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan], index=[p["vsan_col"], p["vlim_col"], p["vdol_col"]])


def run_calculation(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: lithology_run_calculation =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    output_subdir = os.path.join(output_directory, params["output_subfolder"])
    ensure_dirs(output_subdir)

    print("--- 步骤1: 加载所有井的饱和度解释成果 ---")
    all_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not all_files:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")

    well_data_dict = {}
    print(f"找到 {len(all_files)} 个文件，开始加载...")
    for f in all_files:
        well_name = os.path.basename(f).replace('_saturation_interpreted.csv', '')
        df = pd.read_csv(f)
        well_data_dict[well_name] = df

    tqdm.pandas()
    exported_files = []
    print("\n--- 步骤2: 开始批量计算所有井的岩性组分 ---")
    for well_name, well_df in well_data_dict.items():
        print(f"\n--- 正在处理井: {well_name} ---")
        df_litho = well_df.copy()
        print("  - 正在应用多矿物模型求解岩性组分...")
        tqdm.pandas(desc="  - 计算进度")
        litho_results = df_litho.progress_apply(lambda row: solve_lithology(row, params), axis=1)
        df_interpreted = pd.concat([df_litho, litho_results], axis=1)
        df_interpreted[params["carb_auto_col"]] = df_interpreted[params["vlim_col"]] + df_interpreted[params["vdol_col"]]
        print("  - 岩性组分计算完成。")

        output_path = os.path.join(output_subdir, f"{well_name}_lithology_interpreted.csv")
        df_interpreted.to_csv(output_path, index=False, encoding='utf-8-sig')
        exported_files.append(output_path)

    print(f"\n--- 批量计算完成，所有单井成果已保存到: {os.path.abspath(output_subdir)} ---")
    print("\n所有岩性组分计算流程成功完成！")

    return {
        "success": True,
        "output_files": exported_files,
        "message": f"Lithology calculation completed for {len(well_data_dict)} wells."
    }
