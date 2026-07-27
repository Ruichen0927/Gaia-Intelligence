"""MCP 自然语言规划器：让 LLM Agent 根据可用工具生成调用计划。"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root, load_json
from common.tool_registry import format_tools_for_prompt, validate_plan_tools
from common.utils import extract_json_from_response
from llm.caller import load_agent


DEFAULT_AGENT_CONFIG = "ctl/agent_configs/tool_selector_agent_config.json"


def _load_selector_agent(project_root: Path = None):
    root = project_root or get_project_root()
    return load_agent(DEFAULT_AGENT_CONFIG, root)


def _build_planning_prompt(user_input: str, tools_text: str) -> str:
    return f"""【可用工具列表】

{tools_text}

【用户需求】
{user_input}

请根据可用工具列表和用户需求，生成工具调用计划。只输出严格 JSON，不要包含任何解释或 markdown 代码块。"""


def plan_from_natural_language(
    user_input: str,
    project_root: Path = None,
) -> Dict[str, Any]:
    """根据自然语言输入生成 MCP 工具调用计划。

    返回：
        {
            "success": bool,
            "plan": list[dict],
            "reasoning": str,
            "raw": str,
            "error": str|None
        }
    """
    root = project_root or get_project_root()

    if not user_input or not user_input.strip():
        return {
            "success": True,
            "plan": [],
            "reasoning": "用户输入为空",
            "raw": "",
            "error": None,
        }

    try:
        agent = _load_selector_agent(root)
        tools_text = format_tools_for_prompt(project_root=root, include_source=True)
        prompt = _build_planning_prompt(user_input.strip(), tools_text)
        raw = agent.generate_response(user_input=prompt)

        json_str = extract_json_from_response(raw)
        if not json_str or json_str.strip() == "":
            return {
                "success": False,
                "plan": [],
                "reasoning": "",
                "raw": raw,
                "error": "无法从 Agent 响应中提取 JSON",
            }

        data = json.loads(json_str)
        reasoning = data.get("reasoning", "")
        plan = data.get("tools", [])

        if not isinstance(plan, list):
            return {
                "success": False,
                "plan": [],
                "reasoning": reasoning,
                "raw": raw,
                "error": f"Agent 返回的 tools 不是列表，而是 {type(plan).__name__}",
            }

        validation = validate_plan_tools(plan, root)
        valid_plan = validation["valid"]
        invalid_plan = validation["invalid"]

        error = None
        if invalid_plan:
            error = f"以下工具未在注册表中找到，已过滤: {invalid_plan}"

        # 规范化 plan 项：确保包含 config 字段，若省略则使用注册表 default_config
        registry = load_json("ctl/mcp_config.json", root).get("tools", {})
        normalized_plan = []
        for item in valid_plan:
            tool_name = item.get("tool")
            info = registry.get(tool_name, {})
            normalized = {
                "tool": tool_name,
                "config": item.get("config") or info.get("default_config", ""),
                "overrides": item.get("overrides", {}),
            }
            normalized_plan.append(normalized)

        return {
            "success": True,
            "plan": normalized_plan,
            "reasoning": reasoning,
            "raw": raw,
            "error": error,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "plan": [],
            "reasoning": "",
            "raw": raw if "raw" in locals() else "",
            "error": f"JSON 解析失败: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "plan": [],
            "reasoning": "",
            "raw": raw if "raw" in locals() else "",
            "error": f"生成计划时发生错误: {e}",
        }


def execute_plan(
    plan: List[Dict[str, Any]],
    project_root: Path = None,
) -> Dict[str, Any]:
    """执行 MCP 计划。

    参数：
        plan: 工具调用计划列表，每项至少包含 tool、config、overrides。
        project_root: 项目根目录。

    返回：
        mcp_orchestrator.run_plan 的执行结果。
    """
    root = project_root or get_project_root()
    # 延迟导入，避免循环依赖
    from methods.mcp_orchestrator import run_plan
    return run_plan(plan, root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gaia MCP 自然语言规划器")
    parser.add_argument("--input", "-i", required=True, help="自然语言需求")
    parser.add_argument("--execute", "-e", action="store_true", help="是否立即执行生成的计划")
    args = parser.parse_args()

    root = get_project_root()
    result = plan_from_natural_language(args.input, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.execute and result.get("success") and result.get("plan"):
        print("\n--- 执行计划 ---")
        exec_result = execute_plan(result["plan"], root)
        print(json.dumps(exec_result, ensure_ascii=False, indent=2))
