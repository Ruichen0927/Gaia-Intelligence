"""MCP 本地调度器：根据 mcp_config.json 注册表调用单个或多个工具。"""

import sys
import importlib
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root, load_json, load_tool_config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """递归合并 override 到 base。"""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get_tool_registry(project_root: Path = None) -> Dict[str, Dict[str, Any]]:
    """加载 MCP 工具注册表。"""
    if project_root is None:
        project_root = get_project_root()
    cfg = load_json("ctl/mcp_config.json", project_root)
    return cfg.get("tools", {})


def run_tool(tool_name: str,
             config_path: str = None,
             overrides: Dict[str, Any] = None,
             project_root: Path = None) -> Dict[str, Any]:
    """调用指定 MCP 工具。

    参数：
        tool_name: 工具名称（在 mcp_config.json 中注册）。
        config_path: 工具配置文件相对路径；None 时使用注册表中的 default_config。
        overrides: 覆盖配置项的字典。
        project_root: 项目根目录。

    返回：
        工具执行结果字典。
    """
    if project_root is None:
        project_root = get_project_root()
    registry = get_tool_registry(project_root)
    if tool_name not in registry:
        raise ValueError(f"未知工具: {tool_name}。可用工具: {list(registry.keys())}")

    info = registry[tool_name]
    cfg_rel = config_path or info["default_config"]
    cfg = load_tool_config(cfg_rel, project_root)
    if overrides:
        _deep_merge(cfg, overrides)

    module_name = info['module'] if '.' in info['module'] else f"tools.{info['module']}"
    module = importlib.import_module(module_name)
    func = getattr(module, info['function'])
    return func(cfg, project_root)


def run_plan(plan: list, project_root: Path = None) -> Dict[str, Any]:
    """顺序执行 MCP 计划列表。

    plan 示例：
        [
          {"tool": "shale_build_strategy", "config": "ctl/tool_configs/shale_build_strategy.json"},
          {"tool": "shale_run_interpretation"}
        ]
    """
    if project_root is None:
        project_root = get_project_root()
    results = []
    overall_success = True
    for item in plan:
        tool_name = item["tool"]
        config_path = item.get("config")
        overrides = item.get("overrides")
        try:
            result = run_tool(tool_name, config_path, overrides, project_root)
            results.append({"tool": tool_name, "result": result})
            if not result.get("success", False):
                overall_success = False
                break
        except Exception as e:
            overall_success = False
            results.append({"tool": tool_name, "error": str(e)})
            break
    return {"success": overall_success, "steps": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia MCP 本地调度器")
    parser.add_argument("--tool", help="要调用的单个工具名称")
    parser.add_argument("--config", help="工具配置文件相对路径")
    parser.add_argument("--overrides", help="JSON 格式的配置覆盖", default="{}")
    parser.add_argument("--plan", help="包含多个工具调用的 JSON 计划文件相对路径")
    args = parser.parse_args()

    root = get_project_root()
    if args.tool:
        overrides = json.loads(args.overrides)
        result = run_tool(args.tool, args.config, overrides, root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.plan:
        plan = load_json(args.plan, root)
        result = run_plan(plan, root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
