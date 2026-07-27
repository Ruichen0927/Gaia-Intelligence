# src/tools/porosity_tool.py
"""孔隙度解释工具：包含知识库构建与批量解释执行两个原子能力。"""

import pandas as pd
import numpy as np
import json
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from common.config_loader import ensure_dirs, extract_json_from_response
from llm.caller import load_agent


def _get_params(params: Dict) -> Dict:
    defaults = {
        "depth_col": "DEPTH",
        "layer_col": "LAYER",
        "gr_col": "GR",
        "den_col": "DEN",
        "cnl_col": "CNL",
        "ac_col": "AC",
        "pe_col": "PE",
        "cal_col": "CAL",
        "bit_col": "BIT",
        "vsh_col": "vsh_final",
        "por_col": "POR",
        "phie_final_col": "phie_final",
        "feature_curves_for_clustering": ["GR", "DEN", "CNL", "AC", "PE", "vsh_final"],
        "n_clusters": 10,
        "segmentation_curves": ["PE", "GR", "DEN"],
        "segmentation_threshold_std": 0.5,
        "min_segment_length": 5,
        "bone_params": {
            "Sandstone": {"rho_ma": 2.65, "delta_t_ma": 182, "nphi_ma": -0.04},
            "Limestone": {"rho_ma": 2.71, "delta_t_ma": 156, "nphi_ma": 0.00},
            "Dolomite": {"rho_ma": 2.87, "delta_t_ma": 143, "nphi_ma": 0.02},
            "Shale": {"rho_ma": 2.68, "delta_t_ma": 280, "nphi_ma": 0.35}
        },
        "fluid_params": {"rho_f": 1.0, "delta_t_f": 620, "nphi_f": 1.0},
        "wellbore_reference": {"bit_unit": "mm", "bad_hole_threshold_inch": 1.5},
        "gas_effect_reference_threshold": 0.08,
        "knowledgebase_filename": "porosity_knowledgebase.json",
        "output_subfolder": "interpreted_wells"
    }
    merged = {**defaults, **params}
    return merged


def intelligent_segmentation_for_df(df: pd.DataFrame, params: Dict) -> List[tuple]:
    p = _get_params(params)
    df_copy = df.copy().reset_index(drop=True)
    print(f"  - 正在处理 {len(df_copy)} 个深度点...")
    breakpoints = {0, len(df_copy) - 1}
    for curve in p["segmentation_curves"]:
        if curve in df_copy.columns:
            series = df_copy[curve].dropna()
            if len(series) < p["min_segment_length"] * 2:
                continue
            rolling_mean = series.rolling(window=max(5, p["min_segment_length"]), center=True, min_periods=1).mean()
            mean_diff = rolling_mean.diff().abs()
            threshold = series.std() * p["segmentation_threshold_std"]
            if threshold > 0:
                potential_bps = series.index[mean_diff > threshold]
                breakpoints.update(potential_bps)
    sorted_bps = sorted(list(breakpoints))
    segments = []
    start_idx = sorted_bps[0]
    for end_idx in sorted_bps[1:]:
        if (end_idx - start_idx) >= p["min_segment_length"]:
            segments.append((start_idx, end_idx - 1))
            start_idx = end_idx
    if (len(df_copy) - 1 - start_idx) >= p["min_segment_length"]:
        segments.append((start_idx, len(df_copy) - 1))
    return segments


def create_segment_briefing(df_segment: pd.DataFrame, params: Dict) -> Dict:
    p = _get_params(params)
    feature_curves = p["feature_curves_for_clustering"]
    all_feature_keys = [f"{curve}_avg" for curve in feature_curves] + ["caliper_enlargement_inch", "phi_d_n_diff_avg"]
    briefing = {key: np.nan for key in all_feature_keys}
    for curve in feature_curves:
        if curve in df_segment.columns and df_segment[curve].notna().any():
            briefing[f"{curve}_avg"] = df_segment[curve].mean()

    cal_col, bit_col = p["cal_col"], p["bit_col"]
    if cal_col in df_segment.columns and bit_col in df_segment.columns and df_segment[bit_col].notna().any():
        bit_cm = df_segment[bit_col] / 10.0 if p["wellbore_reference"]["bit_unit"] == "mm" else df_segment[bit_col]
        enlargement_cm = (df_segment[cal_col] - bit_cm).mean()
        briefing["caliper_enlargement_inch"] = enlargement_cm / 2.54

    den_col, cnl_col = p["den_col"], p["cnl_col"]
    if den_col in df_segment.columns and cnl_col in df_segment.columns:
        rho_ma_ref = p["bone_params"]["Sandstone"]["rho_ma"]
        nphi_ma_ref = p["bone_params"]["Sandstone"]["nphi_ma"]
        rho_f = p["fluid_params"]["rho_f"]
        nphi_f = p["fluid_params"]["nphi_f"]
        phi_d_ref = (rho_ma_ref - df_segment[den_col]) / (rho_ma_ref - rho_f)
        phi_n_ref = (df_segment[cnl_col] / 100.0 - nphi_ma_ref) / (nphi_f - nphi_ma_ref)
        briefing["phi_d_n_diff_avg"] = (phi_d_ref - phi_n_ref).mean()
    return briefing


def get_diagnosis_from_agent(agent, briefing: Dict) -> Dict:
    briefing_str = json.dumps({k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in briefing.items() if pd.notna(v)}, indent=2)
    prompt = f"请根据以下测井模式的统计情报，做出诊断:\n{briefing_str}"
    try:
        raw_response = agent.generate_response(user_input=prompt)
        return json.loads(extract_json_from_response(raw_response))
    except Exception as e:
        print(f"    - 诊断Agent失败: {e}")
    return None


def get_strategy_from_agent(agent, briefing: Dict, diagnosis: Dict, few_shot_examples: list) -> Dict:
    briefing_str = json.dumps({k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in briefing.items() if pd.notna(v)}, indent=2)
    diagnosis_str = json.dumps(diagnosis, indent=2)
    prompt = f"统计情报:\n{briefing_str}\n\n诊断结论:\n{diagnosis_str}\n\n请基于以上信息，选择模型并对骨架参数进行微调。"
    try:
        raw_response = agent.generate_response(user_input=prompt, few_shot_examples=few_shot_examples)
        return json.loads(extract_json_from_response(raw_response))
    except Exception as e:
        print(f"    - 参数Agent失败: {e}")
    return None


def build_knowledgebase(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: porosity_build_knowledgebase =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    strategy_file_vsh = paths.get("strategy_file_vsh")
    diagnosis_agent_config = paths["diagnosis_agent_config"]
    parameter_agent_config = paths["parameter_agent_config"]
    kb_filename = params["knowledgebase_filename"]
    kb_path = os.path.join(output_directory, kb_filename)
    ensure_dirs(output_directory)

    start_time = time.time()
    print("--- 步骤1: 加载所有井的泥质解释成果 ---")
    all_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not all_files:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    df_list = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    print("  - 正在执行增强的数据清洗...")
    combined_df.replace([-9999, -999.25, -9999.0, -999.2500], np.nan, inplace=True)
    geologic_limits = {
        params["gr_col"]: (0, 1000),
        params["den_col"]: (1.0, 3.5),
        params["cnl_col"]: (-15, 100),
        params["ac_col"]: (100, 1000),
        params["pe_col"]: (0, 10)
    }
    for col, (min_val, max_val) in geologic_limits.items():
        if col in combined_df.columns:
            original_count = len(combined_df)
            combined_df = combined_df[(combined_df[col].isnull()) | (combined_df[col].between(min_val, max_val))]
            cleaned_count = len(combined_df)
            if original_count > cleaned_count:
                print(f"    - 清理 {col}: 移除了 {original_count - cleaned_count} 个超出范围的点。")
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"  - 清洗完成，剩余 {len(combined_df)} 个有效深度点。")

    print("\n--- 步骤2: 对所有数据进行智能分段 ---")
    segments = intelligent_segmentation_for_df(combined_df, params)
    print(f"  - 共划分出 {len(segments)} 个计算单元。")

    print("\n--- 步骤3: 提取特征并进行K-Means聚类 ---")
    features = [create_segment_briefing(combined_df.iloc[start:end + 1], params) for start, end in segments]
    feature_df = pd.DataFrame(features).fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    kmeans = KMeans(n_clusters=params["n_clusters"], random_state=42, n_init='auto')
    feature_df['cluster'] = kmeans.fit_predict(scaled_features)
    print(f"  - 成功聚类为 {params['n_clusters']} 个测井模式。")

    print("\n--- 步骤4: 为每个测井模式请求Agent制定标准策略 ---")
    knowledgebase = {
        "metadata": {
            "description": "Porosity interpretation knowledgebase.",
            "creation_date": pd.Timestamp.now().isoformat(),
            "n_clusters": params["n_clusters"],
            "feature_columns": feature_df.columns[:-1].tolist()
        },
        "cluster_centers": pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=feature_df.columns[:-1]
        ).to_dict(orient='index'),
        "strategies": {}
    }

    few_shot_examples = []
    diagnosis_agent = load_agent(diagnosis_agent_config, project_root)
    parameter_agent = load_agent(parameter_agent_config, project_root)
    print("  - 所有Agent初始化成功。")

    for i in range(params["n_clusters"]):
        print(f"\n  - 正在为 Cluster {i} 请求策略...")
        cluster_center_briefing = knowledgebase['cluster_centers'][i]
        print("    - (1/2) 请求诊断...")
        diagnosis = get_diagnosis_from_agent(diagnosis_agent, cluster_center_briefing)
        if not diagnosis:
            continue
        print(f"    -> 诊断完成: {diagnosis.get('dominant_lithology')}, {diagnosis.get('wellbore_condition')}, {diagnosis.get('special_effects')}")
        print("    - (2/2) 请求参数微调...")
        strategy = get_strategy_from_agent(parameter_agent, cluster_center_briefing, diagnosis, few_shot_examples)
        if not strategy:
            continue
        full_strategy = {"diagnosis": diagnosis, "strategy": strategy, "reasoning": diagnosis.get('reasoning', '')}
        knowledgebase['strategies'][f'Cluster_{i}'] = full_strategy
        print(f"    -> 成功获取完整策略, 推荐模型: {strategy.get('porosity_model_to_apply', 'N/A')}")

    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(knowledgebase, f, indent=2, ensure_ascii=False)

    end_time = time.time()
    print(f"\n--- 成功！孔隙度解释知识库已生成并保存到: {kb_path} ---")
    print(f"--- 总耗时: {end_time - start_time:.2f} 秒 ---")

    return {
        "success": True,
        "output_files": [kb_path],
        "message": f"Porosity knowledgebase built with {params['n_clusters']} clusters."
    }


def find_closest_cluster(segment_briefing: Dict, cluster_centers_df: pd.DataFrame, scaler: StandardScaler) -> str:
    briefing_series = pd.Series(segment_briefing)
    briefing_df_aligned = briefing_series.reindex(cluster_centers_df.columns).fillna(0).to_frame().T
    scaled_briefing = scaler.transform(briefing_df_aligned)
    scaled_centers = scaler.transform(cluster_centers_df)
    distances = cdist(scaled_briefing, scaled_centers)[0]
    closest_cluster_index = np.argmin(distances)
    return f"Cluster_{closest_cluster_index}"


def apply_strategy_to_segment(df_segment: pd.DataFrame, strategy: Dict, vsh_params: Dict, params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    df_copy = df_segment.copy()
    model = strategy['strategy']['porosity_model_to_apply']
    bone = strategy['strategy']['bone_parameters_to_use']
    rho_ma, delta_t_ma, nphi_ma = bone['rho_ma'], bone['delta_t_ma'], bone['nphi_ma']
    rho_f = p["fluid_params"]["rho_f"]
    delta_t_f = p["fluid_params"]["delta_t_f"]
    nphi_f = p["fluid_params"]["nphi_f"]
    rho_sh = vsh_params.get('rho_sh', 2.68)
    delta_t_sh = vsh_params.get('delta_t_sh', 280)
    nphi_sh = vsh_params.get('nphi_sh', 35.0)

    den_col, ac_col, cnl_col, vsh_col = p["den_col"], p["ac_col"], p["cnl_col"], p["vsh_col"]
    phi_d_corr = (rho_ma - df_copy[den_col]) / (rho_ma - rho_f) - df_copy[vsh_col] * (rho_ma - rho_sh) / (rho_ma - rho_f)
    phi_s_corr = (df_copy[ac_col] - delta_t_ma) / (delta_t_f - delta_t_ma) - df_copy[vsh_col] * (delta_t_sh - delta_t_ma) / (delta_t_f - delta_t_ma)
    phi_n_corr = (df_copy[cnl_col] / 100.0 - nphi_ma) / (nphi_f - nphi_ma) - df_copy[vsh_col] * (nphi_sh / 100.0 - nphi_ma) / (nphi_f - nphi_ma)
    phi_gas_corr = np.sqrt((phi_d_corr.clip(0) ** 2 + phi_n_corr.clip(0) ** 2) / 2)

    if model == 'Density':
        phi_total = phi_d_corr
    elif model == 'Sonic':
        phi_total = phi_s_corr
    elif model == 'Gas_Correction':
        phi_total = phi_gas_corr
    else:
        phi_total = phi_d_corr

    df_copy['lithology_kb'] = strategy['diagnosis']['dominant_lithology']
    df_copy['porosity_model_kb'] = model
    df_copy[p["phie_final_col"]] = (phi_total * (1 - df_copy[vsh_col])).clip(0, 0.5)
    return df_copy


def run_calculation(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: porosity_run_calculation =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    strategy_file_vsh = paths["strategy_file_vsh"]
    kb_path = os.path.join(output_directory, params["knowledgebase_filename"])
    output_subdir = os.path.join(output_directory, params["output_subfolder"])
    ensure_dirs(output_subdir)

    print("--- 步骤1: 加载数据与AI知识库 ---")
    well_data_dict = {
        os.path.basename(f).replace('_shale_interpreted.csv', ''): pd.read_csv(f)
        for f in glob.glob(os.path.join(input_directory, '*_shale_interpreted.csv'))
    }
    if not well_data_dict:
        raise FileNotFoundError(f"在 '{input_directory}' 中未找到输入文件。")
    print(f"  - 找到 {len(well_data_dict)} 口井待解释。")

    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledgebase = json.load(f)
    print(f"  - 成功加载孔隙度知识库: {kb_path}")

    with open(strategy_file_vsh, 'r', encoding='utf-8') as f:
        strategy_vsh = json.load(f)
    print("  - 成功加载泥岩参数策略。")

    feature_columns = knowledgebase['metadata']['feature_columns']
    cluster_centers_df = pd.DataFrame(knowledgebase['cluster_centers']).T[feature_columns]
    scaler = StandardScaler().fit(cluster_centers_df)
    strategies_map = knowledgebase['strategies']
    vsh_params_map = {
        layer: p.get('final_optimized_params', {})
        for layer, p in strategy_vsh.get('formation_strategies', {}).items()
    }

    print("\n--- 步骤2: 开始批量解释所有井 ---")
    all_interpreted_dfs = []
    exported_files = []
    for well_name, df in well_data_dict.items():
        print(f"  - 正在解释井: {well_name}")
        df.replace([-9999, -999.25], np.nan, inplace=True)
        if df.empty or len(df) < params["min_segment_length"]:
            continue

        segments_indices = intelligent_segmentation_for_df(df, params)
        processed_segments = []
        for start, end in segments_indices:
            df_segment = df.iloc[start:end + 1]
            if df_segment.empty:
                continue
            briefing = create_segment_briefing(df_segment, params)
            closest_cluster = find_closest_cluster(briefing, cluster_centers_df, scaler)
            if closest_cluster in strategies_map:
                strategy = strategies_map[closest_cluster]
                dominant_layer = df_segment[params["layer_col"]].mode()[0] if not df_segment[params["layer_col"]].empty and df_segment[params["layer_col"]].notna().any() else 'default'
                vsh_params = vsh_params_map.get(str(dominant_layer), {})
                processed_segment = apply_strategy_to_segment(df_segment, strategy, vsh_params, params)
                processed_segments.append(processed_segment)

        if processed_segments:
            interpreted_df = pd.concat(processed_segments).reindex(df.index)
            output_path = os.path.join(output_subdir, f"{well_name}_porosity_interpreted.csv")
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
        "message": f"Porosity calculation completed for {len(well_data_dict)} wells."
    }


def _validate_and_report(final_df_all_wells: pd.DataFrame, params: Dict):
    print("\n--- 步骤3: 孔隙度最终精度评估与报告 ---")
    por_col, phie_col = params["por_col"], params["phie_final_col"]
    if por_col in final_df_all_wells.columns and final_df_all_wells[por_col].notna().any():
        por_series = pd.to_numeric(final_df_all_wells[por_col], errors='coerce')
        por_series[(por_series < 0) | (por_series > 100)] = np.nan
        is_percent = por_series.dropna().quantile(0.95, interpolation='lower') > 1.0
        if is_percent:
            final_df_all_wells['por_frac'] = por_series / 100.0
        else:
            final_df_all_wells['por_frac'] = por_series

        final_df_all_wells['por_abs_diff'] = (final_df_all_wells[phie_col] - final_df_all_wells['por_frac']).abs()
        final_df_all_wells['por_rel_error_pct'] = final_df_all_wells.apply(
            lambda row: (row['por_abs_diff'] / row['por_frac']) * 100 if row['por_frac'] != 0 else 0,
            axis=1
        )
        y_min = final_df_all_wells['por_frac'].min()
        y_max = final_df_all_wells['por_frac'].max()
        y_range = y_max - y_min
        mae = final_df_all_wells['por_abs_diff'].mean() / y_range if y_range > 0 else np.nan
        mean_rel_error_pct = final_df_all_wells['por_rel_error_pct'].mean()
        print(f"  - 全局平均绝对误差 (MAE): {mae:.4f}")
        print(f"  - 全局平均相对误差率: {mean_rel_error_pct:.2f}%")
    else:
        print("  - 未找到孔隙度真值列，跳过评估。")
