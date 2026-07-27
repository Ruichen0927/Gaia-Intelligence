"""工具注册表格式化：为 Agent 提供动态、可注入的工具列表。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root, load_json


def list_tools_for_agent(project_root: Path = None) -> List[Dict[str, Any]]:
    """加载全部已注册工具，返回结构化列表（供 Agent 选择使用）。

    返回列表元素：
        {
            "name": 工具唯一标识,
            "module": Python 模块名,
            "function": 入口函数名,
            "default_config": 默认配置文件相对路径,
            "description": 工具描述,
            "source": "builtin" | "user"
        }
    """
    root = project_root or get_project_root()
    cfg = load_json("ctl/mcp_config.json", root)
    tools = cfg.get("tools", {})
    result = []
    for name, info in tools.items():
        item = {
            "name": name,
            "module": info.get("module", ""),
            "function": info.get("function", "run"),
            "default_config": info.get("default_config", ""),
            "description": info.get("description", ""),
            "source": info.get("source", "builtin"),
        }
        result.append(item)
    return result


def format_tools_for_prompt(
    tools: Optional[List[Dict[str, Any]]] = None,
    project_root: Path = None,
    include_source: bool = True,
) -> str:
    """将工具列表格式化为 Agent 可读的文本描述。

    参数：
        tools: 工具列表；None 时从注册表自动加载。
        project_root: 项目根目录。
        include_source: 是否标注工具来源（builtin/user）。

    返回：
        供 LLM 阅读的文本，每工具占两行，包含名称、来源、描述、模块、函数、默认配置。
    """
    if tools is None:
        tools = list_tools_for_agent(project_root)

    if not tools:
        return "（当前没有已注册工具）"

    lines = []
    for t in tools:
        name = t.get("name", "unknown")
        desc = t.get("description", "") or "无描述"
        module = t.get("module", "")
        function = t.get("function", "run")
        default_config = t.get("default_config", "")
        source = t.get("source", "builtin")

        source_tag = f" ({source})" if include_source else ""
        lines.append(f"- {name}{source_tag}: {desc}")
        lines.append(
            f"  module: {module}, function: {function}, default_config: {default_config}"
        )

    return "\n".join(lines)


def get_tool_names(project_root: Path = None) -> List[str]:
    """返回所有已注册工具名列表。"""
    tools = list_tools_for_agent(project_root)
    return [t["name"] for t in tools]


def validate_plan_tools(plan: List[Dict[str, Any]], project_root: Path = None) -> List[Dict[str, Any]]:
    """校验计划中的工具是否都在注册表中，过滤非法项并保留合法项。

    返回：
        {"valid": [...], "invalid": [...]}
    """
    valid_names = set(get_tool_names(project_root))
    valid = []
    invalid = []
    for item in plan:
        if isinstance(item, dict) and item.get("tool") in valid_names:
            valid.append(item)
        else:
            invalid.append(item)
    return {"valid": valid, "invalid": invalid}


def load_default_tool_paths(tool_name: str, project_root: Path = None) -> Dict[str, str]:
    """加载 ctl/default_tool_paths.json 并替换 {tool_name} 占位符。

    参数：
        tool_name: 当前工具名，用于替换占位符。
        project_root: 项目根目录。

    返回：
        包含默认 input_file、input_directory、output_directory 等路径的字典。
        若配置文件不存在，返回最简默认 output_directory。
    """
    root = project_root or get_project_root()
    default_path = root / "ctl" / "default_tool_paths.json"

    defaults = {"output_directory": f"data/processed_data/{tool_name}"}
    if default_path.exists():
        try:
            cfg = load_json("ctl/default_tool_paths.json", root)
            paths = cfg.get("paths", {})
            defaults.update({k: v.format(tool_name=tool_name) for k, v in paths.items()})
        except Exception:
            pass

    return defaults
