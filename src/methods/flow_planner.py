"""多 Agent 流程一键规划：根据自然语言需求自动生成流程配置。"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root
from common.tool_registry import list_tools_for_agent, format_tools_for_prompt
from common.utils import extract_json_from_response
from dev_platform import agent_manager, kb_manager, skill_manager, flow_manager
from llm.caller import load_agent


DEFAULT_AGENT_CONFIG = "ctl/agent_configs/flow_planner_agent_config.json"


def _load_planner_agent(project_root: Path = None):
    root = project_root or get_project_root()
    return load_agent(DEFAULT_AGENT_CONFIG, root)


def _build_planning_prompt(requirement: str, project_root: Path) -> str:
    agents = agent_manager.list_agents(project_root)
    tools = list_tools_for_agent(project_root)
    docs = kb_manager.list_documents(project_root)
    skills = skill_manager.list_skills(project_root)

    agents_text = "\n".join(f"- {a['agent_id']} ({a['model_type']}): {a.get('description', '')}" for a in agents) or "（暂无 Agent）"
    tools_text = format_tools_for_prompt(tools, include_source=False) or "（暂无工具）"
    docs_text = "\n".join(f"- {d['name']}: {d.get('summary', '')}" for d in docs) or "（暂无知识库文档）"
    skills_text = "\n".join(f"- {s['name']} ({s.get('title', '')}): {s.get('description', '')}" for s in skills) or "（暂无 Skill）"

    return f"""【可用 Agent】
{agents_text}

【可用工具】
{tools_text}

【可用知识库文档】
{docs_text}

【可用 Skill】
{skills_text}

【用户需求】
{requirement}

请根据以上可用资源和用户需求，生成一个多 Agent 流程配置 JSON。只输出严格 JSON，不要包含任何解释或 markdown 代码块。"""


def plan_flow_from_natural_language(
    requirement: str,
    project_root: Path = None,
) -> Dict[str, Any]:
    """根据自然语言需求自动生成 Agent Flow 配置。

    返回：
        {"success": bool, "flow": dict, "raw": str, "error": str|None}
    """
    root = project_root or get_project_root()

    if not requirement or not requirement.strip():
        return {"success": False, "flow": None, "raw": "", "error": "需求描述不能为空"}

    try:
        agent = _load_planner_agent(root)
        prompt = _build_planning_prompt(requirement.strip(), root)
        raw = agent.generate_response(user_input=prompt)

        json_str = extract_json_from_response(raw)
        if not json_str:
            return {"success": False, "flow": None, "raw": raw, "error": "无法从 Agent 响应中提取 JSON"}

        data = json.loads(json_str)

        # 规整边字段：支持 source/target 或 from/to
        normalized_edges = []
        for edge in data.get("edges", []):
            src = edge.get("from") or edge.get("source")
            dst = edge.get("to") or edge.get("target")
            if src and dst:
                normalized_edges.append({"from": src, "to": dst})
        data["edges"] = normalized_edges

        # 补充 flow_id 字段
        if "flow_id" not in data or not data["flow_id"]:
            # 优先使用顶层 id，否则根据流程名生成安全的 flow_id，保留中文、英文、数字和下划线
            import re
            name = data.get("id") or data.get("name", "ai_flow")
            safe = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]", "_", name).strip("_")
            data["flow_id"] = safe or "ai_flow"

        # 校验
        validation = flow_manager.validate_flow(data, root)
        if not validation["valid"]:
            return {
                "success": False,
                "flow": data,
                "raw": raw,
                "error": "生成的流程校验未通过:\n" + "\n".join(validation["errors"]),
            }

        return {"success": True, "flow": data, "raw": raw, "error": None}

    except json.JSONDecodeError as e:
        return {"success": False, "flow": None, "raw": raw if "raw" in locals() else "", "error": f"JSON 解析失败: {e}"}
    except Exception as e:
        return {"success": False, "flow": None, "raw": raw if "raw" in locals() else "", "error": f"规划失败: {e}"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia 多 Agent 流程规划器")
    parser.add_argument("--requirement", "-r", required=True, help="自然语言需求描述")
    args = parser.parse_args()

    root = get_project_root()
    result = plan_flow_from_natural_language(args.requirement, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
