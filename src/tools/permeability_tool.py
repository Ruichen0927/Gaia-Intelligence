# src/tools/permeability_tool.py
"""渗透率解释工具：包含策略构建与批量解释执行两个原子能力。"""

import pandas as pd
import numpy as np
import json
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from sklearn.metrics.pairwise import cosine_similarity

from common.config_loader import ensure_dirs, extract_json_from_response
from llm.caller import load_agent


def _get_params(params: Dict) -> Dict:
    defaults = {
        "layer_col": "LAYER",
        "depth_col": "DEPTH",
        "gr_col": "GR",
        "den_col": "DEN",
        "ac_col": "AC",
        "th_col": "TH",
        "k_col": "K",
        "u_col": "U",
        "cgr_col": "CGR",
        "phie_final_col": "phie_final",
        "vsh_col": "vsh_final",
        "perm_col": "PERM",
        "k_final_col": "k_final",
        "perm_model_used_col": "permeability_model_used",
        "rock_type_col": "rock_type",
        "shale_vsh_threshold": 0.6,
        "shale_perm_background_value": 0.001,
        "fluid_density": 1.0,
        "rock_type_rules": {
            "RT5_Cemented_Sand": [0.0, 0.3, 0.0, 1.0, 0.08, 999.0],
            "RT1_High_Quality_Sand": [0.0, 0.15, 0.12, 1.0, -999.0, 0.08],
            "RT2_Medium_Quality_Sand": [0.0, 0.25, 0.08, 0.12, -999.0, 0.08],
            "RT4_Tight_Sand": [0.0, 0.25, 0.0, 0.08, -999.0, 0.08],
            "RT3_Shaly_Sand": [0.15, 0.50, 0.0, 1.0, -999.0, 999.0]
        },
        "default_rock_type": "Undefined",
        "use_dynamic_few_shots": False,
        "few_shot_suffix": "_with_fewshot",
        "zero_shot_suffix": "_zero_shot",
        "strategy_filename": "permeability_strategy_agent.json",
        "output_subfolder": "interpreted_wells"
    }
    merged = {**defaults, **params}
    return merged


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


def get_model_selection(agent, group_stats: Dict, group_name: str) -> Dict:
    stats_str = json.dumps({k: round(v, 4) if isinstance(v, (int, float)) and pd.notna(v) else v for k, v in group_stats.items()}, indent=2)
    prompt = f"地层-岩石类型组合 '{group_name}' 的统计特征如下:\n{stats_str}\n请选择最合适的模型。"
    try:
        raw_response = agent.generate_response(user_input=prompt)
        return json.loads(extract_json_from_response(raw_response))
    except Exception as e:
        print(f"    - 模型选择Agent失败: {e}")
    return None


def get_parameter_estimation(agent, prompt: str, few_shot_examples: list) -> Dict:
    try:
        raw_response = agent.generate_response(user_input=prompt, few_shot_examples=few_shot_examples)
        json_str = extract_json_from_response(raw_response)
        return json.loads(json_str.replace("'", '"'))
    except Exception as e:
        print(f"    - 参数估算Agent失败: {e}")
    return None


def build_strategy(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: permeability_build_strategy =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    model_selector_agent_config = paths["model_selector_agent_config"]
    parameter_estimator_agent_config = paths["parameter_estimator_agent_config"]
    strategy_path = os.path.join(output_directory, params["strategy_filename"])
    ensure_dirs(output_directory)

    print("--- 步骤1: 加载所有井的孔隙度解释成果 ---")
    all_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not all_files:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    df_list = [pd.read_csv(f) for f in all_files]
    master_df = pd.concat(df_list, ignore_index=True)
    print(f"  - 成功加载并合并了 {len(all_files)} 口井的数据。")

    print("\n--- 步骤2: 对所有数据进行自动岩石物理分类 ---")
    master_df_classified = auto_rock_typer(master_df, params)
    print("  - 岩石分类完成。")

    cookbook = []
    cookbook_vectors = {}
    cookbook_names = []
    cookbook_matrix = np.array([])

    if params["use_dynamic_few_shots"]:
        print("\n--- 步骤3: [模式: Few-Shot] 从原始优化策略生成动态教科书 ---")
        optimized_strategy_path = os.path.join(output_directory, "permeability_interpretation_strategy.json")
        try:
            with open(optimized_strategy_path, 'r', encoding='utf-8') as f:
                optimized_strategy = json.load(f)
        except FileNotFoundError:
            print(f"  - 警告: 未找到原始优化策略文件 '{optimized_strategy_path}'。无法生成教科书，将以Zero-Shot模式运行。")
            optimized_strategy = None

        if optimized_strategy:
            feature_vectors_map = {}
            for name, model_data in optimized_strategy.get("formation_rocktype_strategies", {}).items():
                parts = name.split('_', 1)
                if len(parts) != 2:
                    continue
                layer, rt = parts
                group_df = master_df_classified[(master_df_classified[params["layer_col"]] == layer) & (master_df_classified[params["rock_type_col"]] == rt)]
                if group_df.empty:
                    continue
                group_stats = {
                    "phie_avg": group_df[params["phie_final_col"]].mean(),
                    "vsh_avg": group_df[params["vsh_col"]].mean(),
                    "gr_avg": group_df[params["gr_col"]].mean()
                }
                params_data = model_data.get("coeffs", model_data)
                assistant_params = {
                    "a": params_data.get("a"),
                    "b": params_data.get("b"),
                    "c1_phie": params_data.get("c1_phie"),
                    "c2_vsh": params_data.get("c2_vsh"),
                    "c3_gr": params_data.get("c3_gr"),
                    "c4_ac": params_data.get("c4_ac"),
                    "c5_intercept": params_data.get("c5_intercept")
                }
                feature_vector = [v for v in group_stats.values() if pd.notna(v)]
                if not feature_vector:
                    continue
                feature_vectors_map[name] = feature_vector
                cookbook.append({
                    "name": name,
                    "feature_vector": feature_vector,
                    "user_prompt": f"岩石类型的统计特征如下:\n{json.dumps(group_stats, indent=2)}\n请为 '{model_data['model_type']}' 模型估算一套经验系数。",
                    "assistant_response": json.dumps(assistant_params)
                })
            if cookbook:
                cookbook_path = os.path.join(output_directory, "permeability_few_shot_cookbook.json")
                with open(cookbook_path, 'w', encoding='utf-8') as f:
                    json.dump(cookbook, f, indent=2, ensure_ascii=False)
                print(f"  - 成功生成并保存了包含 {len(cookbook)} 个真实范例的Few-Shot教科书。")
                cookbook_vectors = {item['name']: item['feature_vector'] for item in cookbook}
                cookbook_names = list(cookbook_vectors.keys())
                cookbook_matrix = np.array([cookbook_vectors[name] for name in cookbook_names])
    else:
        print("\n--- 步骤3: [模式: Zero-Shot] 已跳过Few-Shot教科书生成。 ---")

    model_selector_agent = load_agent(model_selector_agent_config, project_root)
    param_estimator_agent = load_agent(parameter_estimator_agent_config, project_root)
    print("\n--- 步骤4: 初始化所有Agent成功 ---")

    strategies = {}
    print("\n--- 步骤5: 为每个组合请求决策 ---")
    for (layer, rt), group_df in master_df_classified.groupby([params["layer_col"], params["rock_type_col"]]):
        if rt == params["default_rock_type"] or len(group_df) < 20:
            continue
        group_name = f"{layer}_{rt}"
        print(f"  - 正在处理组合: {group_name} ({len(group_df)}个点)")
        group_stats = {
            "phie_avg": group_df[params["phie_final_col"]].mean(),
            "vsh_avg": group_df[params["vsh_col"]].mean(),
            "gr_avg": group_df[params["gr_col"]].mean()
        }
        current_vector = np.array([v for v in group_stats.values() if pd.notna(v)])

        model_choice = get_model_selection(model_selector_agent, group_stats, group_name)
        if not model_choice or "selected_model" not in model_choice:
            print("    -> 模型选择失败，跳过。")
            continue
        selected_model_type = model_choice["selected_model"]
        print(f"    - 模型选择Agent推荐: {selected_model_type}")

        dynamic_few_shots = []
        if params["use_dynamic_few_shots"] and len(cookbook_matrix) > 0 and len(current_vector) == cookbook_matrix.shape[1]:
            similarity = cosine_similarity(current_vector.reshape(1, -1), cookbook_matrix)[0]
            top_k_indices = np.argsort(similarity)[-3:][::-1]
            for idx in top_k_indices:
                example_name = cookbook_names[idx]
                example = next(item for item in cookbook if item["name"] == example_name)
                dynamic_few_shots.append({"role": "user", "content": example["user_prompt"]})
                dynamic_few_shots.append({"role": "assistant", "content": example["assistant_response"]})

        param_prompt = f"岩石类型的统计特征如下:\n{json.dumps(group_stats, indent=2)}\n请为 '{selected_model_type}' 模型估算一套经验系数。"
        parameters = get_parameter_estimation(param_estimator_agent, param_prompt, dynamic_few_shots)
        if not parameters:
            print("    -> 参数估算失败，跳过。")
            continue
        print("    - 参数估算Agent成功返回系数。")
        strategies[group_name] = {
            "model_to_use": {"model_type": selected_model_type, "parameters": parameters},
            "reasoning": model_choice.get("reasoning", "N/A")
        }

    final_strategy_file = {
        "project_info": {
            "description": "Agent-driven permeability strategy.",
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "few_shot_enabled": params["use_dynamic_few_shots"]
        },
        "permeability_models": strategies
    }
    with open(strategy_path, 'w', encoding='utf-8') as f:
        json.dump(final_strategy_file, f, indent=2, ensure_ascii=False)
    print(f"\n--- 成功！最终策略文件已保存到: {strategy_path} ---")

    return {
        "success": True,
        "output_files": [strategy_path],
        "message": f"Permeability strategy built for {len(strategies)} layer-rocktype groups."
    }


def run_calculation(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: permeability_run_calculation =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    strategy_path = os.path.join(output_directory, params["strategy_filename"])
    output_subdir = os.path.join(output_directory, params["output_subfolder"])
    ensure_dirs(output_subdir)

    print("---步骤1: 加载单井数据与Agent渗透率策略---")
    well_data_dict = {
        os.path.basename(f).replace('_porosity_interpreted.csv', ''): pd.read_csv(f)
        for f in glob.glob(os.path.join(input_directory, '*_porosity_interpreted.csv'))
    }
    if not well_data_dict:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    print(f"  - 找到 {len(well_data_dict)} 口井待解释。")

    with open(strategy_path, 'r', encoding='utf-8') as f:
        strategy = json.load(f)
    print(f"  - 成功加载Agent渗透率策略文件: {strategy_path}")
    few_shot_enabled = strategy.get("project_info", {}).get("few_shot_enabled", False)
    print(f"  - 策略生成模式: {'Few-Shot' if few_shot_enabled else 'Zero-Shot'}")

    all_interpreted_dfs = []
    exported_files = []
    print("\n--- 步骤2: 开始批量解释所有井 ---")
    for well_name, well_df in well_data_dict.items():
        print(f"  - 正在解释井: {well_name}")
        interpreted_df = _run_permeability_calculation(well_df, strategy, params)
        output_path = os.path.join(output_subdir, f"{well_name}_permeability_interpreted.csv")
        interpreted_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        exported_files.append(output_path)
        all_interpreted_dfs.append(interpreted_df)

    print(f"\n--- 批量解释完成，所有单井成果已保存到: {output_subdir} ---")
    if all_interpreted_dfs:
        final_df_all_wells = pd.concat(all_interpreted_dfs, ignore_index=True)
        _validate_and_report(final_df_all_wells, few_shot_enabled, params, output_directory)

    return {
        "success": True,
        "output_files": exported_files,
        "message": f"Permeability calculation completed for {len(well_data_dict)} wells."
    }


def _run_permeability_calculation(df: pd.DataFrame, strategy: Dict, params: Dict) -> pd.DataFrame:
    df_interpreted = df.copy()
    cols = [params["gr_col"], params["ac_col"], params["vsh_col"], params["phie_final_col"],
            params["den_col"], params["layer_col"], params["th_col"], params["k_col"], params["u_col"]]
    for col in cols:
        if col not in df_interpreted.columns:
            df_interpreted[col] = np.nan if col != params["layer_col"] else 'Undefined'

    df_interpreted = auto_rock_typer(df_interpreted, params)
    df_interpreted[params["k_final_col"]] = np.nan
    df_interpreted[params["perm_model_used_col"]] = "N/A"

    is_shale = df_interpreted[params["vsh_col"]] > params["shale_vsh_threshold"]
    df_interpreted.loc[is_shale, params["k_final_col"]] = params["shale_perm_background_value"]
    df_interpreted.loc[is_shale, params["perm_model_used_col"]] = "Shale Zone"

    mask_to_calculate = df_interpreted[params["perm_model_used_col"]] == "N/A"
    for index, row in df_interpreted[mask_to_calculate].iterrows():
        layer = str(row.get(params["layer_col"], 'Undefined'))
        rt = row.get(params["rock_type_col"], params["default_rock_type"])
        phie = row.get(params["phie_final_col"])
        if pd.isna(phie) or phie <= 0:
            df_interpreted.loc[index, params["k_final_col"]] = params["shale_perm_background_value"]
            df_interpreted.loc[index, params["perm_model_used_col"]] = "Zero Porosity"
            continue

        strategy_key = f"{layer}_{rt}"
        model_info = strategy.get("permeability_models", {}).get(strategy_key)
        if not model_info:
            df_interpreted.loc[index, params["perm_model_used_col"]] = f"No_Model_For_{rt}"
            continue

        model_type = model_info['model_to_use']['model_type']
        model_params = model_info['model_to_use']['parameters']
        k_val = np.nan
        try:
            if model_type == "PowerLaw":
                k_val = model_params.get("a", 1.0) * (phie ** model_params.get("b", 1.0))
            elif model_type == "MultiLinear_V2":
                log_k = (
                    model_params.get("c5_intercept", 0.0) +
                    model_params.get("c1_phie", 0.0) * phie +
                    model_params.get("c2_vsh", 0.0) * row.get(params["vsh_col"], 0.0) +
                    model_params.get("c3_gr", 0.0) * row.get(params["gr_col"], 0.0) +
                    model_params.get("c4_ac", 0.0) * row.get(params["ac_col"], 0.0)
                )
                k_val = 10 ** log_k
        except (TypeError, KeyError, ValueError):
            df_interpreted.loc[index, params["perm_model_used_col"]] = f"Param_Error_{model_type}"
            continue

        df_interpreted.loc[index, params["k_final_col"]] = k_val
        df_interpreted.loc[index, params["perm_model_used_col"]] = f"Agent_{model_type}"

    df_interpreted[params["k_final_col"]] = df_interpreted[params["k_final_col"]].clip(lower=0.0001).fillna(params["shale_perm_background_value"])
    return df_interpreted


def _validate_and_report(df_all_wells: pd.DataFrame, few_shot_enabled: bool, params: Dict, output_directory: str):
    print("\n---步骤3: 最终精度评估与报告---")
    perm_col, k_final_col = params["perm_col"], params["k_final_col"]
    df_all_wells[perm_col] = pd.to_numeric(df_all_wells[perm_col], errors='coerce')
    df_all_wells[k_final_col] = pd.to_numeric(df_all_wells[k_final_col], errors='coerce')

    mask = ((df_all_wells[perm_col] > 0) & (df_all_wells[k_final_col] > 0)) & ((df_all_wells[perm_col] < 2000) & (df_all_wells[k_final_col] < 2000))
    df_eval = df_all_wells[mask].copy()
    if df_eval.empty:
        print("警告: 数据集中没有足够的、可用于评估渗透率的点。")
        return

    perm_true = df_eval[perm_col]
    perm_pred_raw = df_eval[k_final_col]
    y_min = np.min(perm_true)
    y_max = np.max(perm_true)
    y_range = y_max - y_min
    print(f"  - 目标变量范围: {y_min:.4f} 到 {y_max:.4f} (范围: {y_range:.4f})")

    perm_pred = np.clip(perm_pred_raw, y_min, y_max)
    truncated_count = np.sum((perm_pred_raw < y_min) | (perm_pred_raw > y_max))
    if truncated_count > 0:
        print(f"  - 注意: 评估前对预测值进行了截断处理，共有 {truncated_count} 个点超出范围")

    log_perm_true = np.log10(perm_true)
    log_perm_pred = np.log10(perm_pred)
    rmse_log = np.sqrt(np.mean((log_perm_pred - log_perm_true) ** 2))
    mae_log = np.mean(abs(log_perm_pred - log_perm_true))

    if log_perm_pred.std() < 1e-9:
        r_squared_log = 0.0
    else:
        correlation_matrix = np.corrcoef(log_perm_pred, log_perm_true)
        if np.isnan(correlation_matrix).any():
            r_squared_log = 0.0
        else:
            r_squared_log = correlation_matrix[0, 1] ** 2

    mae_original = np.mean(np.abs(perm_pred - perm_true))
    mae_relative = mae_original / y_range if y_range > 0 else np.nan
    relative_error_percent = (mae_original * 100) / np.mean(perm_true) if np.mean(perm_true) > 0 else np.nan
    accuracy = (1 - mae_relative) * 100 if y_range > 0 and not np.isnan(mae_relative) else np.nan

    print("\n---渗透率(K)全局评估报告 (vs. 工程师解释真值)---")
    print(f"  - 目标变量范围: {y_min:.4f} 到 {y_max:.4f} (范围: {y_range:.4f})")
    print("\n  [对数域评估指标]")
    print(f"  - 对数均方根误差 (Log RMSE): {rmse_log:.4f}")
    print(f"  - 对数域决定系数 (Log R²): {r_squared_log:.4f}")
    print(f"  - 对数平均绝对误差 (Log MAE): {mae_log:.4f}")
    print("\n  [原始域评估指标]")
    print(f"  - 平均绝对误差 (MAE): {mae_original:.4f}")
    print(f"  - 相对平均绝对误差 (MAE/y_range): {mae_relative:.4f} ({mae_relative*100:.2f}%)")
    print(f"  - 相对误差百分比: {relative_error_percent:.2f}%")
    print(f"  - 准确率: {accuracy:.2f}%")

    if mae_relative <= 0.1:
        accuracy_level = "优秀 (相对误差 ≤ 10%)"
    elif mae_relative <= 0.2:
        accuracy_level = "良好 (相对误差 ≤ 20%)"
    elif mae_relative <= 0.3:
        accuracy_level = "一般 (相对误差 ≤ 30%)"
    else:
        accuracy_level = "需要改进 (相对误差 > 30%)"
    print(f"\n  [精度评估]\n  - 模型精度等级: {accuracy_level}")

    suffix = params["few_shot_suffix"] if few_shot_enabled else params["zero_shot_suffix"]
    report_filename = f'permeability_evaluation_report{suffix}.txt'
    report_path = os.path.join(output_directory, report_filename)
    report_lines = [
        "---渗透率(K)全局评估报告 (vs. 工程师解释真值)---",
        f"  - 目标变量范围: {y_min:.4f} 到 {y_max:.4f} (范围: {y_range:.4f})",
        "\n  [对数域评估指标]",
        f"  - 对数均方根误差 (Log RMSE): {rmse_log:.4f}",
        f"  - 对数域决定系数 (Log R²): {r_squared_log:.4f}",
        f"  - 对数平均绝对误差 (Log MAE): {mae_log:.4f}",
        "\n  [原始域评估指标]",
        f"  - 平均绝对误差 (MAE): {mae_original:.4f}",
        f"  - 相对平均绝对误差 (MAE/y_range): {mae_relative:.4f} ({mae_relative*100:.2f}%)",
        f"  - 相对误差百分比: {relative_error_percent:.2f}%",
        f"  - 准确率: {accuracy:.2f}%",
        f"\n  [精度评估]\n  - 模型精度等级: {accuracy_level}"
    ]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\n评估总结报告已成功保存至: {os.path.abspath(report_path)}")
