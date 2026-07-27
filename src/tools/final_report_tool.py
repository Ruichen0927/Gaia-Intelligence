# src/tools/final_report_tool.py
"""最终报告工具：包含决策规则学习与最终交付生成两个原子能力。"""

import pandas as pd
import numpy as np
import glob
import os
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common.config_loader import ensure_dirs, extract_json_from_response
from llm.caller import load_agent


def _get_params(params: Dict) -> Dict:
    defaults = {
        "layer_col": "LAYER",
        "rock_type_col": "rock_type",
        "depth_col": "DEPTH",
        "result_col": "RESULT",
        "sw_col": "sw_final",
        "phie_col": "phie_final",
        "k_col": "k_final",
        "vsh_col": "vsh_final",
        "result_auto_col": "RESULT_AUTO",
        "final_conclusion_col": "FINAL_CONCLUSION",
        "rules_filename": "decision_rules.json",
        "smoothing_window": 3,
        "min_layer_thickness": 0.5,
        "temp_pointwise_subfolder": "temp_pointwise",
        "final_deliverable_subfolder": "final_deliverables"
    }
    return {**defaults, **params}


def learn_rules(df: pd.DataFrame, allowed_labels: List[str], params: Dict) -> Dict:
    p = _get_params(params)
    rules = {}
    df_learn = df.dropna(subset=[p["layer_col"], p["rock_type_col"], p["result_col"], p["sw_col"], p["phie_col"], p["k_col"]])
    df_learn = df_learn[df_learn[p["result_col"]].isin(allowed_labels)]

    for (layer, rt, result), group in df_learn.groupby([p["layer_col"], p["rock_type_col"], p["result_col"]]):
        stats = {
            "sw_p25": group[p["sw_col"]].quantile(0.25),
            "sw_p75": group[p["sw_col"]].quantile(0.75),
            "phie_p25": group[p["phie_col"]].quantile(0.25),
            "phie_p75": group[p["phie_col"]].quantile(0.75),
            "k_p25": group[p["k_col"]].quantile(0.25),
            "k_p75": group[p["k_col"]].quantile(0.75),
            "count": len(group)
        }
        key = f"{layer}_{rt}"
        if key not in rules:
            rules[key] = {}
        rules[key][result] = {k: round(v, 4) if pd.notna(v) else None for k, v in stats.items()}
    return rules


def apply_rules_vectorized(df: pd.DataFrame, rules: Dict, params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    df_processed = df.copy()
    all_unique_conclusions = set()
    for ruleset in rules.values():
        all_unique_conclusions.update(ruleset.keys())
    scores_df = pd.DataFrame(0, index=df_processed.index, columns=list(all_unique_conclusions))

    for (layer, rt), group_df_orig in df_processed.groupby([p["layer_col"], p["rock_type_col"]]):
        rule_key = f"{layer}_{rt}"
        ruleset = rules.get(rule_key)
        if not ruleset:
            continue
        for conclusion, stats in ruleset.items():
            if stats and stats.get('count', 0) >= 5:
                mask_sw = group_df_orig[p["sw_col"]].between(stats.get('sw_p25', -np.inf), stats.get('sw_p75', np.inf))
                mask_phie = group_df_orig[p["phie_col"]].between(stats.get('phie_p25', -np.inf), stats.get('phie_p75', np.inf))
                mask_k = group_df_orig[p["k_col"]].between(stats.get('k_p25', -np.inf), stats.get('k_p75', np.inf))
                scores_for_this_conclusion = mask_sw.astype(int) + mask_phie.astype(int) + mask_k.astype(int)
                scores_df.loc[group_df_orig.index, conclusion] = scores_for_this_conclusion

    df_processed[p["result_auto_col"]] = scores_df.idxmax(axis=1)
    max_scores = scores_df.max(axis=1)
    is_tie = (scores_df.eq(max_scores, axis=0)).sum(axis=1) > 1
    is_low_score = max_scores < 2
    needs_expert_review_mask = is_tie | is_low_score | (max_scores == 0)
    df_processed.loc[needs_expert_review_mask, p["result_auto_col"]] = "待会诊"
    df_processed.loc[pd.isna(df_processed[p["result_auto_col"]]), p["result_auto_col"]] = "干层"
    return df_processed


def learn_rules_tool(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: final_report_learn_rules =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    rules_path = os.path.join(output_directory, params["rules_filename"])
    ensure_dirs(output_directory)

    print("--- 步骤1: 加载所有井的岩性解释成果 ---")
    all_files = glob.glob(os.path.join(input_directory, '*.csv'))
    if not all_files:
        raise FileNotFoundError(f"输入目录 '{input_directory}' 中未找到文件。")
    master_df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)

    allowed_labels = master_df[params["result_col"]].dropna().unique().tolist()
    allowed_labels = [label for label in allowed_labels if not pd.Series([label]).str.contains("未解释|none|I类|II类|III类", case=False, na=False).any()]
    print(f"  - 识别到的原始有效结论标签（Agent白名单）：{sorted(allowed_labels)}")

    print("\n--- 步骤2: 从工程师结论中学习决策规则 ---")
    decision_rules = learn_rules(master_df, allowed_labels, params)
    with open(rules_path, 'w', encoding='utf-8') as f:
        json.dump(decision_rules, f, indent=2, ensure_ascii=False)
    print(f"  - 成功！决策规则已学习并保存到: {rules_path} ---")

    print("\n--- 步骤3: 应用规则生成逐点初步解释结论 ---")
    pointwise_df_prelim = apply_rules_vectorized(master_df, decision_rules, params)
    output_temp_dir = os.path.join(output_directory, params["temp_pointwise_subfolder"])
    ensure_dirs(output_temp_dir)
    prelim_path = os.path.join(output_temp_dir, "all_wells_prelim_report.csv")
    pointwise_df_prelim.to_csv(prelim_path, index=False)
    print(f"  - 初步解释结论已保存到: {os.path.abspath(output_temp_dir)} ---")

    return {
        "success": True,
        "output_files": [rules_path, prelim_path],
        "message": f"Decision rules learned and preliminary report generated with {len(allowed_labels)} labels."
    }


def hybrid_conclusion_engine_refine(df: pd.DataFrame, rules: Dict, agent, few_shot_examples: list, allowed_labels: List[str], params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    df_report = df.copy()
    needs_expert_review_mask = df_report[p["result_auto_col"]] == "待会诊"
    review_df = df_report[needs_expert_review_mask].copy()
    print(f"    - (1/2) Agent正在对疑难点进行批量会诊... ({len(review_df)}个点)")

    if not review_df.empty:
        feature_cols = [p["phie_col"], p["vsh_col"], p["sw_col"], p["k_col"]]
        features = review_df[feature_cols].fillna(0)
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        n_clusters_to_use = min(20, len(review_df))
        if n_clusters_to_use > 1:
            kmeans = KMeans(n_clusters=n_clusters_to_use, random_state=42, n_init='auto').fit(scaled_features)
            review_df['cluster'] = kmeans.labels_
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        else:
            review_df['cluster'] = range(len(review_df))
            cluster_centers = features.values
            n_clusters_to_use = len(review_df)

        cluster_conclusions = {}
        for i in tqdm(range(n_clusters_to_use), desc="    - Agent会诊进度"):
            center_features = cluster_centers[i]
            point_data = dict(zip(feature_cols, center_features))
            prompt_text = "总工程师，请为以下一类典型的疑难数据点，给出一个统一的最终解释结论。\n"
            for key, value in point_data.items():
                prompt_text += f"- 平均 {key}: {value:.4f}\n"
            prompt_text += f"\n你的结论必须从以下标签列表中选择: {', '.join(allowed_labels)}\n"
            try:
                result = json.loads(extract_json_from_response(agent.generate_response(prompt_text, few_shot_examples)))
                cluster_conclusions[i] = result.get("conclusion", "干层")
            except Exception as e:
                print(f"      - Agent会诊失败: {e}")
                cluster_conclusions[i] = "干层"
            print(f"      - 疑难模式簇 {i} (phie={point_data[p['phie_col']]:.2f}, sw={point_data[p['sw_col']]:.2f}, k={point_data[p['k_col']]:.2f}) -> 结论: {cluster_conclusions[i]}")

        review_df[p["result_auto_col"]] = review_df['cluster'].map(cluster_conclusions)
        df_report.loc[needs_expert_review_mask, p["result_auto_col"]] = review_df[p["result_auto_col"]]
    else:
        print("    - 无疑难点需要专家会诊。")

    print("    - (2/2) 逐点解释结论已生成。")
    return df_report


def consolidate_layers(df: pd.DataFrame, well_name: str, params: Dict) -> pd.DataFrame:
    p = _get_params(params)
    print(f"    - 正在对井 '{well_name}' 进行层段合并和优化...")
    df_processed = df.copy()
    if p["result_auto_col"] not in df_processed.columns or df_processed[p["result_auto_col"]].isnull().all():
        print(f"      - 警告: 井 '{well_name}' 缺少 '{p['result_auto_col']}' 列或全为空，无法进行层段合并。")
        df_processed[p["final_conclusion_col"]] = "无法合并"
        return df_processed

    conclusion_change = df_processed[p["result_auto_col"]] != df_processed[p["result_auto_col"]].shift()
    df_processed['layer_id'] = conclusion_change.cumsum()

    final_layers = []
    for _, group in df_processed.groupby('layer_id'):
        if group.empty:
            continue
        top, bottom = group[p["depth_col"]].min(), group[p["depth_col"]].max()
        thickness = bottom - top
        conclusion = group[p["result_auto_col"]].mode()[0]
        if thickness < p["min_layer_thickness"]:
            if final_layers and final_layers[-1]['conclusion'] == conclusion:
                final_layers[-1]['bottom'] = bottom
            else:
                final_layers.append({'top': top, 'bottom': bottom, 'conclusion': conclusion})
        else:
            final_layers.append({'top': top, 'bottom': bottom, 'conclusion': conclusion})

    df_processed[p["final_conclusion_col"]] = "非储层"
    for layer in final_layers:
        mask = (df_processed[p["depth_col"]] >= layer['top']) & (df_processed[p["depth_col"]] <= layer['bottom'])
        df_processed.loc[mask, p["final_conclusion_col"]] = layer['conclusion']
    return df_processed


def generate_deliverable(config: Dict, project_root: Path) -> Dict:
    print("\n===== 启动工具: final_report_generate_deliverable =====")
    paths = config["paths"]
    params = _get_params(config.get("parameters", {}))

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    agent_config = paths["final_conclusion_agent_config"]
    rules_path = os.path.join(output_directory, params["rules_filename"])
    prelim_path = os.path.join(output_directory, params["temp_pointwise_subfolder"], "all_wells_prelim_report.csv")
    output_deliverable_dir = os.path.join(output_directory, params["final_deliverable_subfolder"])
    ensure_dirs(output_deliverable_dir)

    print("--- 步骤1: 加载初步解释数据和原始标签 ---")
    if not os.path.exists(prelim_path):
        raise FileNotFoundError(f"未找到初步解释数据文件: '{prelim_path}'。请先运行 final_report_learn_rules。")
    master_prelim_df = pd.read_csv(prelim_path, low_memory=False)
    well_data_dict = {}
    for well_name in master_prelim_df['LOG_ID_FROM_FILENAME'].unique():
        well_data_dict[well_name] = master_prelim_df[master_prelim_df['LOG_ID_FROM_FILENAME'] == well_name].copy()

    with open(rules_path, 'r', encoding='utf-8') as f:
        decision_rules_temp = json.load(f)
    allowed_labels = sorted(list(set(conclusion for ruleset in decision_rules_temp.values() for conclusion in ruleset.keys())))
    print(f"  - 成功加载 {len(well_data_dict)} 口井的初步解释数据。")
    print(f"  - 识别到的原始有效结论标签（Agent白名单）：{allowed_labels}")

    with open(agent_config, 'r', encoding='utf-8') as f:
        agent_config_data = json.load(f)
    allowed_labels_str = ', '.join([f"'{label}'" for label in allowed_labels])
    agent_config_data['system_message'] = agent_config_data['system_message'].replace("[ALLOWED_LABELS_PLACEHOLDER]", allowed_labels_str)

    temp_config_path = os.path.join(output_directory, "temp_agent_config.json")
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(agent_config_data, f, ensure_ascii=False, indent=2)

    agent = load_agent(temp_config_path, project_root)
    print("\n--- 步骤2: 初始化总工程师Agent成功 ---")

    few_shot_examples = [
        {
            "role": "user",
            "content": "总工程师，请为以下这一类典型的疑难数据点给出一个统一的最终解释结论。\n- 平均 phie_final: 0.1800\n- 平均 vsh_final: 0.0500\n- 平均 sw_final: 0.3500\n- 平均 k_final: 250.5000\n\n你的结论必须从以下标签列表中选择: [ALLOWED_LABELS_PLACEHOLDER]".replace("[ALLOWED_LABELS_PLACEHOLDER]", allowed_labels_str)
        },
        {
            "role": "assistant",
            "content": json.dumps({"conclusion": "油层", "reasoning": "物性极好(高孔高渗)，含水饱和度较低，是典型的优质油层。"})
        },
        {
            "role": "user",
            "content": "总工程师，请为以下这一类典型的疑难数据点给出一个统一的最终解释结论。\n- 平均 phie_final: 0.1000\n- 平均 vsh_final: 0.4500\n- 平均 sw_final: 0.6500\n- 平均 k_final: 5.2000\n\n你的结论必须从以下标签列表中选择: [ALLOWED_LABELS_PLACEHOLDER]".replace("[ALLOWED_LABELS_PLACEHOLDER]", allowed_labels_str)
        },
        {
            "role": "assistant",
            "content": json.dumps({"conclusion": "差油层", "reasoning": "泥质含量高，渗透率低，虽然含油但产能有限，划为差油层。"})
        }
    ]

    with open(rules_path, 'r', encoding='utf-8') as f:
        decision_rules = json.load(f)

    print("\n--- 步骤3: 应用混合智能引擎并生成最终交付成果 ---")
    for well_name in well_data_dict:
        well_data_dict[well_name]['original_index'] = well_data_dict[well_name].index

    master_df_all_wells = pd.concat(well_data_dict.values(), ignore_index=True)
    pointwise_df_processed = hybrid_conclusion_engine_refine(
        master_df_all_wells, decision_rules, agent, few_shot_examples, allowed_labels, params
    )

    all_final_dfs = []
    exported_files = []
    print("\n  - 正在将处理结果按井拆分...")
    for well_name, well_df_orig in well_data_dict.items():
        original_indices = well_df_orig.index
        well_processed_df = pointwise_df_processed[pointwise_df_processed['original_index'].isin(original_indices)].copy()
        well_processed_df.set_index('original_index', inplace=True)
        if 'original_index' in well_processed_df.columns:
            well_processed_df.drop(columns=['original_index'], inplace=True)
        final_df = consolidate_layers(well_processed_df, well_name, params)
        deliverable_path = os.path.join(output_deliverable_dir, f"{well_name}_final_deliverable.csv")
        final_df.to_csv(deliverable_path, index=False)
        exported_files.append(deliverable_path)
        all_final_dfs.append(final_df)

    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    print(f"\n--- 成功！最终交付成果已保存到: {os.path.abspath(output_deliverable_dir)} ---")

    print("\n--- 步骤4: 最终成层结论评估 ---")
    all_reports_df = pd.concat(all_final_dfs, ignore_index=True)
    df_eval = all_reports_df.dropna(subset=[params["result_col"], params["final_conclusion_col"]])
    df_eval = df_eval[~df_eval[params["result_col"]].str.contains("未解释|none", case=False, na=False)]
    if not df_eval.empty:
        labels_in_eval = list(set(df_eval[params["result_col"]].unique()) | set(df_eval[params["final_conclusion_col"]].unique()))
        labels_for_report = [label for label in allowed_labels if label in labels_in_eval]
        print(f"\n--- 最终总体准确率 ---: {accuracy_score(df_eval[params['result_col']], df_eval[params['final_conclusion_col']]):.2%}")
        print("\n--- 最终分类详细评估报告 ---")
        print(classification_report(df_eval[params["result_col"]], df_eval[params["final_conclusion_col"]], labels=labels_for_report, zero_division=0))

    print("\n🎉🎉🎉 恭喜！所有自动化解释流程已全部成功完成！ 🎉🎉🎉")
    return {
        "success": True,
        "output_files": exported_files,
        "message": f"Final deliverables generated for {len(well_data_dict)} wells."
    }
