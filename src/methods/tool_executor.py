"""工具执行器：在 Agent Flow 中实际调用已注册的工具。"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root, load_tool_config
from common.utils import extract_json_from_response
from methods import mcp_orchestrator


def _render_placeholders(obj: Any, context: Dict[str, Any]) -> Any:
    """递归替换对象中的 {key} 占位符。"""
    if isinstance(obj, dict):
        return {k: _render_placeholders(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render_placeholders(v, context) for v in obj]
    if isinstance(obj, str):
        try:
            return obj.format(**context)
        except (KeyError, ValueError):
            return obj
    return obj


def parse_tool_overrides_from_response(response: str) -> Dict[str, Any]:
    """从 Agent 响应中解析 JSON 配置覆盖。

    支持两种格式：
    1. {"tool_overrides": {"tool_name": {"paths": {...}, "parameters": {...}}}, "summary": "..."}
    2. {"tools": [{"tool": "tool_name", "overrides": {"paths": {...}, "parameters": {...}}}], "summary": "..."}

    解析失败返回空 dict。
    """
    json_str = extract_json_from_response(response)
    if not json_str:
        return {}
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return {}

        # 统一转换为 tool_overrides 字典格式
        tool_overrides = data.get("tool_overrides", {})
        if isinstance(tool_overrides, dict):
            return data

        tools_list = data.get("tools", [])
        if isinstance(tools_list, list):
            normalized = {}
            for item in tools_list:
                if isinstance(item, dict) and "tool" in item:
                    normalized[item["tool"]] = item.get("overrides", {})
            data["tool_overrides"] = normalized
            return data
    except json.JSONDecodeError:
        pass
    return {}


def execute_tool(
    tool_name: str,
    overrides: Dict[str, Any] = None,
    outputs: Dict[str, Any] = None,
    initial_input: str = "",
    project_root: Path = None,
) -> Dict[str, Any]:
    """执行单个工具。

    参数：
        tool_name: 注册在 mcp_config.json 中的工具名。
        overrides: 覆盖默认配置的字典（已渲染或未渲染）。
        outputs: 上游节点输出上下文，用于再次渲染 overrides 中的占位符。
        initial_input: 流程初始输入。
        project_root: 项目根目录。

    返回：
        {"tool_name": str, "success": bool, "result": Any, "error": str|None}
    """
    root = project_root or get_project_root()
    registry = mcp_orchestrator.get_tool_registry(root)

    if tool_name not in registry:
        return {"tool_name": tool_name, "success": False, "result": None, "error": f"工具未注册: {tool_name}"}

    try:
        # 对 overrides 中的占位符进行二次渲染
        context = dict(outputs or {})
        context["__input__"] = initial_input
        rendered_overrides = _render_placeholders(overrides or {}, context)

        result = mcp_orchestrator.run_tool(
            tool_name=tool_name,
            overrides=rendered_overrides or None,
            project_root=root,
        )
        return {"tool_name": tool_name, "success": True, "result": result, "error": None}
    except Exception as e:
        return {"tool_name": tool_name, "success": False, "result": None, "error": str(e)}


def execute_node_tools(
    tools: List[str],
    llm_response: str,
    outputs: Dict[str, Any],
    initial_input: str = "",
    project_root: Path = None,
) -> Dict[str, Any]:
    """顺序执行节点配置的工具。

    参数：
        tools: 工具名列表。
        llm_response: Agent 的输出，尝试解析其中 tool_overrides。
        outputs: 上游节点输出上下文。
        initial_input: 流程初始输入。
        project_root: 项目根目录。

    返回：
        {
          "success": bool,
          "tool_results": [{"tool_name": ..., "success": ..., "result": ..., "error": ...}],
          "llm_summary": str,
          "error": str|None
        }
    """
    root = project_root or get_project_root()
    parsed = parse_tool_overrides_from_response(llm_response)
    tool_overrides = parsed.get("tool_overrides", {}) if isinstance(parsed, dict) else {}
    llm_summary = parsed.get("summary", llm_response) if isinstance(parsed, dict) else llm_response

    tool_results = []
    overall_success = True

    for tool_name in tools:
        overrides = tool_overrides.get(tool_name, {}) if isinstance(tool_overrides, dict) else {}
        exec_res = execute_tool(tool_name, overrides, outputs, initial_input, root)
        tool_results.append(exec_res)
        if not exec_res.get("success", False):
            overall_success = False

    error = None
    if not overall_success:
        failed = [r["tool_name"] for r in tool_results if not r.get("success", False)]
        error = f"以下工具执行失败: {', '.join(failed)}"

    return {
        "success": overall_success,
        "tool_results": tool_results,
        "llm_summary": llm_summary,
        "error": error,
    }
