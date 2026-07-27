"""自然语言指令解析器：将用户输入映射为可执行动作。"""

import re
from typing import List, Dict, Any


# 功能组工具列表
DATA_TOOLS = [
    "data_load_well_logs", "data_clean_curves", "data_detect_outliers",
    "data_impute_missing", "data_split_by_formation", "data_merge_well_data",
    "data_qc_report"
]
FEATURE_TOOLS = [
    "feature_compute_derived_curves", "feature_normalize_curves", "feature_window_features"
]
VIZ_TOOLS = ["viz_crossplot", "viz_histogram", "viz_pickett_plot"]
STAT_TOOLS = ["stat_summary", "stat_correlation"]
ML_TOOLS = [
    "ml_train_regressor", "ml_predict", "ml_evaluate",
    "ml_train_classifier", "ml_predict_classifier", "ml_evaluate_classifier",
    "ml_cluster_data", "ml_save_load_model"
]
GEO_TOOLS = [
    "shale_build_strategy", "shale_run_interpretation",
    "porosity_build_knowledgebase", "porosity_run_calculation",
    "permeability_build_strategy", "permeability_run_calculation",
    "saturation_build_strategy", "saturation_run_calculation",
    "lithology_run_calculation",
    "final_report_learn_rules", "final_report_generate_deliverable"
]

# 工具别名 -> pipeline step / MCP tool
TOOL_ALIASES = {
    "shale": ["shale_build_strategy", "shale_run_interpretation"],
    "泥质": ["shale_build_strategy", "shale_run_interpretation"],
    "泥质含量": ["shale_build_strategy", "shale_run_interpretation"],
    "porosity": ["porosity_build_knowledgebase", "porosity_run_calculation"],
    "孔隙": ["porosity_build_knowledgebase", "porosity_run_calculation"],
    "孔隙度": ["porosity_build_knowledgebase", "porosity_run_calculation"],
    "permeability": ["permeability_build_strategy", "permeability_run_calculation"],
    "渗透": ["permeability_build_strategy", "permeability_run_calculation"],
    "渗透率": ["permeability_build_strategy", "permeability_run_calculation"],
    "saturation": ["saturation_build_strategy", "saturation_run_calculation"],
    "饱和": ["saturation_build_strategy", "saturation_run_calculation"],
    "饱和度": ["saturation_build_strategy", "saturation_run_calculation"],
    "lithology": ["lithology_run_calculation"],
    "岩性": ["lithology_run_calculation"],
    "final": ["final_report_learn_rules", "final_report_generate_deliverable"],
    "report": ["final_report_learn_rules", "final_report_generate_deliverable"],
    "报告": ["final_report_learn_rules", "final_report_generate_deliverable"],
    "最终报告": ["final_report_learn_rules", "final_report_generate_deliverable"],
    # 新增细粒度工具别名
    "数据": DATA_TOOLS,
    "data": DATA_TOOLS,
    "清洗": ["data_clean_curves", "data_impute_missing"],
    "特征": FEATURE_TOOLS,
    "feature": FEATURE_TOOLS,
    "可视化": VIZ_TOOLS,
    "viz": VIZ_TOOLS,
    "交会图": ["viz_crossplot"],
    "统计": STAT_TOOLS,
    "stat": STAT_TOOLS,
    "机器学习": ML_TOOLS,
    "ml": ML_TOOLS,
    "聚类": ["ml_cluster_data"],
    "回归": ["ml_train_regressor", "ml_predict", "ml_evaluate"],
    "分类": ["ml_train_classifier", "ml_predict_classifier", "ml_evaluate_classifier"],
}

# 阶段到流程法停止点的映射（包含该阶段之前的所有步骤）
PHASE_TO_STOP = {
    "shale": "shale_run_interpretation",
    "泥质": "shale_run_interpretation",
    "porosity": "porosity_run_calculation",
    "孔隙": "porosity_run_calculation",
    "permeability": "permeability_run_calculation",
    "渗透": "permeability_run_calculation",
    "saturation": "saturation_run_calculation",
    "饱和": "saturation_run_calculation",
    "lithology": "lithology_run_calculation",
    "岩性": "lithology_run_calculation",
    "final": "final_report_generate_deliverable",
    "report": "final_report_generate_deliverable",
    "报告": "final_report_generate_deliverable",
}


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def parse_command(method: str, text: str) -> Dict[str, Any]:
    """解析自然语言指令，返回动作描述。

    返回示例：
        {"mode": "flow", "action": "运行全流程", "stop_at": None}
        {"mode": "mcp", "action": "调用泥质含量与孔隙度工具", "tools": [...]}
    """
    text = text.lower().strip()
    all_keywords = ["全部", "全流程", "所有", "运行", "run all", "full pipeline", "全部流程"]

    if method == "流程法":
        if _contains_any(text, all_keywords):
            return {"mode": "flow", "action": "运行完整流程", "stop_at": None}
        for phase, stop in PHASE_TO_STOP.items():
            if phase in text:
                return {"mode": "flow", "action": f"运行至 {phase} 阶段", "stop_at": stop}
        return {"mode": "flow", "action": "运行完整流程", "stop_at": None}

    # MCP 方法
    if _contains_any(text, all_keywords):
        return {
            "mode": "mcp",
            "action": "顺序调用全部地质解释工具",
            "tools": GEO_TOOLS
        }

    selected_tools = []
    for alias, tools in TOOL_ALIASES.items():
        if alias in text:
            for t in tools:
                if t not in selected_tools:
                    selected_tools.append(t)

    if not selected_tools:
        return {"mode": "mcp", "action": "未识别到具体工具，默认顺序调用全部地质解释工具", "tools": GEO_TOOLS}

    return {"mode": "mcp", "action": f"调用工具: {', '.join(selected_tools)}", "tools": selected_tools}
