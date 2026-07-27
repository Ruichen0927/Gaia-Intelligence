"""自然语言指令解析器：将用户输入映射为可执行动作。

支持两种模式：
1. 流程法：基于关键词映射到 pipeline 停止点。
2. MCP 方法：默认使用 LLM Agent 智能选工具，失败时回退到硬编码别名表。
"""

import re
from pathlib import Path
from typing import List, Dict, Any


# 功能组工具列表（仅作为 LLM 失败时的 fallback）
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

# 工具别名 -> pipeline step / MCP tool（fallback 用）
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


def _keyword_based_mcp_parse(text: str) -> Dict[str, Any]:
    """基于硬编码别名的 MCP 解析（fallback）。"""
    text = text.lower().strip()
    all_keywords = ["全部", "全流程", "所有", "运行", "run all", "full pipeline", "全部流程"]

    if _contains_any(text, all_keywords):
        return {
            "mode": "mcp",
            "action": "顺序调用全部地质解释工具",
            "tools": GEO_TOOLS,
            "llm_used": False,
            "reasoning": "关键词匹配：全部/全流程",
        }

    selected_tools = []
    for alias, tools in TOOL_ALIASES.items():
        if alias in text:
            for t in tools:
                if t not in selected_tools:
                    selected_tools.append(t)

    if not selected_tools:
        return {
            "mode": "mcp",
            "action": "未识别到具体工具，默认顺序调用全部地质解释工具",
            "tools": GEO_TOOLS,
            "llm_used": False,
            "reasoning": "未匹配到任何工具别名",
        }

    return {
        "mode": "mcp",
        "action": f"调用工具: {', '.join(selected_tools)}",
        "tools": selected_tools,
        "llm_used": False,
        "reasoning": f"关键词匹配：{', '.join(selected_tools)}",
    }


def _llm_based_mcp_parse(text: str, project_root: Path = None) -> Dict[str, Any]:
    """使用 LLM Agent 解析自然语言并返回 MCP 计划（可包含 overrides）。"""
    # 延迟导入，避免在仅使用关键词解析时也加载 LLM 相关依赖
    from methods.mcp_planner import plan_from_natural_language

    root = project_root
    if root is None:
        from common.config_loader import get_project_root
        root = get_project_root()

    result = plan_from_natural_language(text, root)

    if result.get("success") and result.get("plan"):
        plan = result["plan"]
        tool_names = [item["tool"] for item in plan]
        return {
            "mode": "mcp",
            "action": f"LLM 选择工具: {', '.join(tool_names)}",
            "tools": tool_names,
            "plan": plan,
            "llm_used": True,
            "reasoning": result.get("reasoning", ""),
            "raw": result.get("raw", ""),
            "llm_error": result.get("error"),
        }

    # LLM 未能生成可用计划，返回失败标记供上层 fallback
    return {
        "mode": "mcp",
        "action": "LLM 未能识别工具",
        "tools": [],
        "plan": [],
        "llm_used": True,
        "reasoning": result.get("reasoning", ""),
        "raw": result.get("raw", ""),
        "llm_error": result.get("error") or "未返回可用工具计划",
    }


def parse_command(
    method: str,
    text: str,
    use_llm: bool = True,
    project_root: Path = None,
) -> Dict[str, Any]:
    """解析自然语言指令，返回动作描述。

    参数：
        method: "流程法" 或 "MCP方法"。
        text: 用户自然语言输入。
        use_llm: MCP 方法下是否使用 LLM Agent 智能选工具；失败自动 fallback 到关键词解析。
        project_root: 项目根目录；None 时自动推断。

    返回示例：
        {"mode": "flow", "action": "运行完整流程", "stop_at": None}
        {"mode": "mcp", "action": "LLM 选择工具: gr_mean", "tools": [...], "plan": [...], "reasoning": "..."}
    """
    text = text.lower().strip()
    all_keywords = ["全部", "全流程", "所有", "run all", "full pipeline", "全部流程"]

    if method == "流程法":
        # 先匹配具体阶段，避免“运行到孔隙度”被“运行”误匹配为完整流程
        for phase, stop in PHASE_TO_STOP.items():
            if phase in text:
                return {"mode": "flow", "action": f"运行至 {phase} 阶段", "stop_at": stop}
        if _contains_any(text, all_keywords):
            return {"mode": "flow", "action": "运行完整流程", "stop_at": None}
        return {"mode": "flow", "action": "运行完整流程", "stop_at": None}

    # MCP 方法
    if use_llm:
        llm_result = _llm_based_mcp_parse(text, project_root)
        if llm_result.get("tools"):
            return llm_result
        # fallback 到关键词解析
        fallback = _keyword_based_mcp_parse(text)
        fallback["llm_used"] = True
        fallback["llm_error"] = llm_result.get("llm_error")
        fallback["reasoning"] = (
            f"LLM 选工具失败（{llm_result.get('llm_error')}），已回退到关键词匹配"
        )
        return fallback

    return _keyword_based_mcp_parse(text)


# 保持向后兼容：旧的二参数调用仍使用默认行为（LLM 开启）
def _legacy_parse_command(method: str, text: str) -> Dict[str, Any]:
    return parse_command(method, text, use_llm=True)
