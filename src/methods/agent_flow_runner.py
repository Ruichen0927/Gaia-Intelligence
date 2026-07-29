"""多 Agent 流程执行器：按拓扑顺序执行 Agent 节点，支持数据传递。"""

import sys
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root
from dev_platform import flow_manager
from llm.caller import load_agent


def run_flow(flow_id: str, initial_input: str = "", project_root: Path = None) -> Dict[str, Any]:
    """执行指定多 Agent 流程。

    参数：
        flow_id: 流程 ID；"pipeline" 表示内置流程法。
        initial_input: 流程初始输入。
        project_root: 项目根目录。

    返回：
        {"success": bool, "outputs": dict, "steps": list, "final_output": str, "error": str|None}
    """
    root = project_root or get_project_root()

    try:
        cfg = flow_manager.load_flow(flow_id, root)
    except Exception as e:
        return {"success": False, "outputs": {}, "steps": [], "final_output": "", "error": f"加载流程失败: {e}"}

    validation = flow_manager.validate_flow(cfg, root)
    if not validation["valid"]:
        return {
            "success": False,
            "outputs": {},
            "steps": [],
            "final_output": "",
            "error": "流程校验失败:\n" + "\n".join(validation["errors"]),
        }

    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])

    if not nodes:
        return {"success": True, "outputs": {}, "steps": [], "final_output": "", "error": None}

    try:
        order = flow_manager.topological_sort(nodes, edges)
    except ValueError as e:
        return {"success": False, "outputs": {}, "steps": [], "final_output": "", "error": str(e)}

    node_map = {n["id"]: n for n in nodes}
    outputs = {}
    steps = []
    final_output = ""

    for nid in order:
        node = node_map[nid]
        agent_id = node.get("agent_id")
        output_key = node.get("output_key", nid)

        try:
            prompt = flow_manager.build_node_prompt(node, outputs, initial_input, root)
            agent = load_agent(f"ctl/agent_configs/{agent_id}.json", root)
            response = agent.generate_response(user_input=prompt)

            outputs[output_key] = response
            final_output = response
            steps.append({
                "node_id": nid,
                "agent_id": agent_id,
                "output_key": output_key,
                "success": True,
                "output": response,
            })
        except Exception as e:
            steps.append({
                "node_id": nid,
                "agent_id": agent_id,
                "output_key": output_key,
                "success": False,
                "error": str(e),
            })
            return {
                "success": False,
                "outputs": outputs,
                "steps": steps,
                "final_output": final_output,
                "error": f"节点 '{nid}' 执行失败: {e}",
            }

    return {
        "success": True,
        "outputs": outputs,
        "steps": steps,
        "final_output": final_output,
        "error": None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia 多 Agent 流程执行器")
    parser.add_argument("--flow", required=True, help="流程 ID")
    parser.add_argument("--input", default="", help="流程初始输入")
    args = parser.parse_args()

    root = get_project_root()
    result = run_flow(args.flow, args.input, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
