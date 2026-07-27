# src/tools/saturation_tool.py
"""饱和度解释工具：包含策略构建与批量解释执行两个原子能力。"""

import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import statsmodels.api as sm

from common.config_loader import ensure_dirs, extract_json_from_response
from llm.caller import load_agent


def _get_params(params: Dict) -> Dict:
    defaults = {
        "layer_col": "LAYER",
        "depth_col": "DEPTH",
        "phie_final_col": "phie_final",
        "vsh_col": "vsh_final",
        "rt_col": "RT",
        "rxo_col": "RXO",
        "den_col": "DEN",
        "sw_true_col": "SW",
        "sw_final_col": "sw_final",
        "sxo_final_col": "sxo_final",
        "so_final_col": "so_final",
        "movable_oil_col": "so_movable",
        "rock_type_col": "rock_type",
        "pickett_plot_quantile": 0.15,
        "min_points_for_pickett": 50,
        "shale_model_vsh_threshold": 0.15,
        "global_fallback_params": {"a": 1.0, "m": 2.0, "n": 2.0, "rw": 0.1, "rsh": 10.0},
        "param_sanity_ranges": {
            "a": [0.6, 1.2],
            "m": [1.5, 2.8],
            "n": [1.5, 2.5],
            "rw": [0.02, 5.0],
            "rsh": [1.0, 50.0]
        },
        "fluid_density": 1.0,
        "rock_type_rules": {
            "RT5_Cemented_Sand": [0.0, 0.3, 0.0, 1.0, 0.08, 999.0],
            "RT1_High_Quality_Sand": [0.0, 0.15, 0.12, 1.0, -999.0, 0.08],
            "RT2_Medium_Quality_Sand": [0.0, 0.25, 0.08, 0.12, -999.0, 0.08],
            "RT4_Tight_Sand": [0.0, 0.25, 0.0, 0.08, -999.0, 0.08],
            "RT3_Shaly_Sand": [0.15, 0.5, 0.0, 1.0, -999.0, 999.0]
        },
        "default_rock_type": "Undefined",
        "strategy_filename": "saturation_strategy_agent.json",
        "output_subfolder": "interpreted_wells"
    }
    return {**defaults, **params}


def auto_rock_typer(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    df_copy = df.copy()
    phie = df_copy[p["phie_final_col"]].fillna(0)
    rho_ma_app = (df_copy[p["den_col"]].fillna(2.65) - phie * p["fluid_density"]) / (1 - phie + 1e-6)
    rho_ma_offset = rho_ma_app - 2.65
    df_copy[p["rock_type_col"]] = p["default_rock_type"]
    for rt_name, rules in p["rock_type_rules"].items():
        vsh_min, vsh_max, phie_min, phie_max, offset_min, offset_max = rules
        mask = (
            (df_copy[p["vsh_col"]].fillna(1.0) >= vsh_min) &
            (df_copy[p["vsh_col"]].fillna(1.0) < vsh_max) &
            (phie >= phie_min) &
            (phie < phie_max) &
            (rho_ma_offset.fillna(0) >= offset_min) &
            (rho_ma_offset.fillna(0) < offset_max)
        )
        df_copy.loc[mask, p["rock_type_col"]] = rt_name
    return df_copy


def get_final_params_from_agent(agent, calibration_report: Dict, group_stats: Dict) -> Dict:
    report_str = json.dumps(calibration_report, indent=2)
    stats_str = json.dumps({k: round(v, 4) if isinstance(v, (int, float)) and pd.notna(v) else v for k, v in group_stats.items()}, indent=2)
    prompt = f"Python助手提交的Pickett图自动标定报告如下:\n{report_str}\n\n该岩石类型的宏观统计特征如下:\n{stats_str}\n\n请审核此报告并确定最终的岩电参数。"
    try:
        raw_response = agent.generate_response(user_input=prompt)
        return json.loads(extract_json_from_response(raw_response))
    except Exception as e:
        print(f"    - Agent决策失败: {e}")
    return None


def build_strategy(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: saturation_build_strategy =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    agent_config = paths["saturation_advisor_agent_config"]
    strategy_path = os.path.join(output_directory, params["strategy_filename"])
    ensure_dirs(output_directory)

    print("--- 步骤1: 加载所有井的上一阶段解释成果 ---")
    all_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not all_files:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    df_list = [pd.read_csv(f) for f in all_files]
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"  - 数据加载完成，共合并 {len(master_df)} 条数据点。")

    print("\n--- 步骤2: 自动岩石物理分类 ---")
    master_df_classified = auto_rock_typer(master_df, params)

    agent = load_agent(agent_config, project_root)
    print("\n--- 步骤3: 初始化Agent成功 ---")

    detailed_strategies = {}
    print("\n--- 步骤4: 为每个“地层-岩石类型”组合进行标定并请求Agent决策 ---")
    for (layer, rt), group_df in master_df_classified.groupby([params["layer_col"], params["rock_type_col"]]):
        if rt == params["default_rock_type"] or len(group_df) < 20:
            continue
        group_name = f"{layer}_{rt}"
        print(f"  - 正在处理组合: {group_name} ({len(group_df)}个点)")

        calibration_report = {
            "status": "failed",
            "reason": "Not enough data for Pickett plot",
            "pickett_plot_calibration": None,
            "rsh_median": None
        }
        shale_zone_df = group_df[(group_df[params["vsh_col"]] > params["shale_model_vsh_threshold"] + 0.2) & (group_df[params["rt_col"]] > 0)]
        rsh_median = shale_zone_df[params["rt_col"]].median()
        if pd.notna(rsh_median):
            calibration_report["rsh_median"] = round(rsh_median, 4)

        valid_df = group_df.dropna(subset=[params["rt_col"], params["phie_final_col"]])
        valid_df = valid_df[(valid_df[params["phie_final_col"]] > 0.001) & (valid_df[params["rt_col"]] > 0)]
        if len(valid_df) >= params["min_points_for_pickett"]:
            y, x = np.log10(valid_df[params["rt_col"]]), sm.add_constant(np.log10(valid_df[params["phie_final_col"]]))
            try:
                model = sm.QuantReg(y, x).fit(q=params["pickett_plot_quantile"])
                intercept, slope = model.params
                m, rw = -slope, (10 ** intercept) / params["global_fallback_params"]['a']
                calibration_report["status"] = "success"
                calibration_report["reason"] = f"Successfully fitted with {len(valid_df)} data points."
                calibration_report["pickett_plot_calibration"] = {
                    "a": params["global_fallback_params"]['a'],
                    "m": round(m, 4),
                    "n": params["global_fallback_params"]['n'],
                    "rw": round(rw, 4)
                }
            except Exception as e:
                calibration_report["status"] = "failed"
                calibration_report["reason"] = f"QuantReg calculation error: {e}"

        group_stats = {
            "phie_avg": group_df[params["phie_final_col"]].mean(),
            "vsh_avg": group_df[params["vsh_col"]].mean()
        }
        final_params = get_final_params_from_agent(agent, calibration_report, group_stats)
        if final_params:
            detailed_strategies[group_name] = final_params
            print("    -> Agent决策完成。")
        else:
            print("    -> Agent决策失败，该组合将使用全局后备参数。")

    final_strategy = {
        "project_info": {
            "description": "Agent-driven saturation strategy (V2 - Pickett-reviewed).",
            "generation_date": datetime.now().strftime("%Y-%m-%d")
        },
        "detailed_strategies": detailed_strategies,
        "global_fallback_params": params["global_fallback_params"]
    }
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump(final_strategy, f, indent=2, ensure_ascii=False)
    print(f"\n--- 成功！饱和度解释策略文件已保存到: {strategy_path} ---")

    return {
        "success": True,
        "output_files": [strategy_path],
        "message": f"Saturation strategy built for {len(detailed_strategies)} layer-rocktype groups."
    }


def solve_simandoux(rt, phie, vsh, rw, rsh, a, m, n):
    sw = 0.5
    for _ in range(20):
        sw_old = sw
        term1 = (a * rw) / (phie ** m)
        term2 = (vsh * sw) / rsh
        sw_inv_n = ((1 / rt) - term2) * term1
        if sw_inv_n <= 0:
            return 1.0
        sw = (1 / sw_inv_n) ** (1 / n)
        if abs(sw - sw_old) < 0.001:
            return np.clip(sw, 0.05, 1.0)
    return np.clip(sw, 0.05, 1.0)


def calculate_saturation_for_resistivity(df: pd.DataFrame, strategy: Dict, res_col: str, params: Dict) -> pd.Series:
    p = _get_params(params)
    results = pd.Series(np.nan, index=df.index)
    global_fallback = strategy.get("global_fallback_params", p["global_fallback_params"])
    for index, row in df.iterrows():
        layer = str(row.get(params["layer_col"], 'Undefined'))
        rt_label = row.get(params["rock_type_col"], p["default_rock_type"])
        phie, vsh, res = row.get(params["phie_final_col"]), row.get(params["vsh_col"]), row.get(res_col)
        if pd.isna(phie) or phie <= 0 or pd.isna(res) or res <= 0:
            continue
        strategy_key = f"{layer}_{rt_label}"
        sparams = strategy.get("detailed_strategies", {}).get(strategy_key, global_fallback)
        rw = sparams.get('rw', global_fallback['rw'])
        rsh = sparams.get('rsh', global_fallback['rsh'])
        a = sparams.get('a', global_fallback['a'])
        m = sparams.get('m', global_fallback['m'])
        n = sparams.get('n', global_fallback['n'])
        if pd.isna(vsh) or vsh < params["shale_model_vsh_threshold"]:
            term = (a * rw) / ((phie ** m) * res)
            sw_val = term ** (1 / n)
        elif rsh is not None and pd.notna(rsh):
            sw_val = solve_simandoux(res, phie, vsh, rw, rsh, a, m, n)
        else:
            term = (a * rw) / ((phie ** m) * res)
            sw_val = term ** (1 / n)
        results.loc[index] = sw_val
    return results.clip(0.05, 1.0)


def run_calculation(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: saturation_run_calculation =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    strategy_path = os.path.join(output_directory, params["strategy_filename"])
    output_subdir = os.path.join(output_directory, params["output_subfolder"])
    ensure_dirs(output_subdir)

    print("--- 步骤1: 加载单井数据与饱和度解释策略 ---")
    well_data_dict = {
        os.path.basename(f).replace('_permeability_interpreted.csv', ''): pd.read_csv(f)
        for f in glob.glob(os.path.join(input_directory, '*_permeability_interpreted.csv'))
    }
    if not well_data_dict:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    print(f"找到 {len(well_data_dict)} 个待解释文件...")

    with open(strategy_path, 'r', encoding='utf-8') as f:
        strategy = json.load(f)
    print(f"成功加载饱和度解释策略文件: {strategy_path}")

    all_interpreted_dfs = []
    exported_files = []
    print("\n--- 步骤2: 开始批量解释所有井 ---")
    for well_name, well_df in well_data_dict.items():
        print(f"  - 正在解释井: {well_name}")
        interpreted_df = _run_saturation_interpretation(well_df, strategy, params)
        output_path = os.path.join(output_subdir, f"{well_name}_saturation_interpreted.csv")
        interpreted_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        exported_files.append(output_path)
        all_interpreted_dfs.append(interpreted_df)

    print(f"\n--- 批量解释完成，所有单井成果已保存到: {output_subdir} ---")
    if all_interpreted_dfs:
        final_df_all_wells = pd.concat(all_interpreted_dfs, ignore_index=True)
        _validate_and_report(final_df_all_wells, params)

    return {
        "success": True,
        "output_files": exported_files,
        "message": f"Saturation calculation completed for {len(well_data_dict)} wells."
    }


def _run_saturation_interpretation(df: pd.DataFrame, strategy: Dict, params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    df_interpreted = df.copy()
    for col in [p["phie_final_col"], p["vsh_col"], p["rt_col"], p["rxo_col"], p["den_col"], p["layer_col"]]:
        if col not in df_interpreted.columns:
            df_interpreted[col] = np.nan if col != p["layer_col"] else "Undefined"
    df_interpreted = auto_rock_typer(df_interpreted, params)
    print("  - 正在计算含水饱和度 (Sw)...")
    df_interpreted[p["sw_final_col"]] = calculate_saturation_for_resistivity(df_interpreted, strategy, p["rt_col"], params)
    print("  - 正在计算冲洗带含水饱和度 (Sxo)...")
    df_interpreted[p["sxo_final_col"]] = calculate_saturation_for_resistivity(df_interpreted, strategy, p["rxo_col"], params)
    print("  - 正在计算含油饱和度与可动油...")
    df_interpreted[p["so_final_col"]] = 1 - df_interpreted[p["sw_final_col"]]
    df_interpreted[p["movable_oil_col"]] = (
        df_interpreted[p["sxo_final_col"]] - df_interpreted[p["sw_final_col"]]
    ) * (1 - df_interpreted[p["vsh_col"]].fillna(0)) * df_interpreted[p["phie_final_col"]].fillna(0)
    df_interpreted[p["movable_oil_col"]] = df_interpreted[p["movable_oil_col"]].clip(lower=0)
    return df_interpreted


def _validate_and_report(df_all_wells: pd.DataFrame, params: Dict):
    print("\n--- 步骤3: 最终精度评估与报告 ---")
    sw_true_col, sw_final_col = params["sw_true_col"], params["sw_final_col"]
    df_copy = df_all_wells.copy()
    if sw_true_col not in df_copy.columns or df_copy[sw_true_col].isnull().all():
        print("警告: 未找到饱和度真值列 (SW)，跳过精度评估。")
        return
    df_copy[sw_true_col] = pd.to_numeric(df_copy[sw_true_col], errors='coerce')
    is_percent = df_copy[sw_true_col].dropna().quantile(0.95, interpolation='lower') > 1.5 if not df_copy[sw_true_col].dropna().empty else False
    if is_percent:
        df_copy.loc[(df_copy[sw_true_col] < 0) | (df_copy[sw_true_col] > 150), sw_true_col] = np.nan
    else:
        df_copy.loc[(df_copy[sw_true_col] < 0) | (df_copy[sw_true_col] > 1.5), sw_true_col] = np.nan
    df_copy['sw_true_frac'] = df_copy[sw_true_col] / 100.0 if is_percent else df_copy[sw_true_col]
    df_eval = df_copy.dropna(subset=['sw_true_frac', sw_final_col])
    if df_eval.empty:
        print("警告: 数据集中没有足够的、可用于评估饱和度的点。")
        return
    mae = (df_eval[sw_final_col] - df_eval['sw_true_frac']).abs().mean()
    print(f"\n--- 含水饱和度(Sw)全局评估报告 ---\n平均绝对误差 (MAE): {mae*100:.2f}%")
