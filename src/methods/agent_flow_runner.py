"""多 Agent 流程执行器：按拓扑顺序执行 Agent 节点，支持数据传递。"""

import sys
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root
from dev_platform import agent_manager, flow_manager
from llm.caller import load_agent
from methods import tool_executor


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
            agent_config_path = agent_manager.resolve_agent_config_path(agent_id, root)
            agent = load_agent(str(agent_config_path), root)

            # 为该节点生成临时 system message，明确团队角色
            system_override = flow_manager.build_node_system_message(node, cfg, agent.system_prompt)
            llm_response = agent.generate_response(user_input=prompt, system_message_override=system_override)

            tools = node.get("tools", [])
            if tools:
                # 工具执行节点：先解析 LLM 的参数建议，再实际调用工具
                tool_run = tool_executor.execute_node_tools(
                    tools=tools,
                    llm_response=llm_response,
                    outputs=outputs,
                    initial_input=initial_input,
                    project_root=root,
                )
                # 组合 LLM 总结与工具结果作为节点输出
                output_lines = ["【Agent 分析与计划】", llm_response, "", "【工具执行结果】"]
                for tr in tool_run["tool_results"]:
                    status = "成功" if tr.get("success") else "失败"
                    output_lines.append(f"- {tr['tool_name']}: {status}")
                    if tr.get("result"):
                        output_lines.append(f"  结果: {json.dumps(tr['result'], ensure_ascii=False)[:500]}")
                    if tr.get("error"):
                        output_lines.append(f"  错误: {tr['error']}")
                node_output = "\n".join(output_lines)
                step_success = tool_run.get("success", False)
                step_error = tool_run.get("error")
            else:
                node_output = llm_response
                step_success = True
                step_error = None

            outputs[output_key] = node_output
            final_output = node_output
            steps.append({
                "node_id": nid,
                "agent_id": agent_id,
                "output_key": output_key,
                "success": step_success,
                "llm_response": llm_response,
                "tool_results": tool_run["tool_results"] if tools else [],
                "output": node_output,
                "error": step_error,
            })

            if not step_success:
                return {
                    "success": False,
                    "outputs": outputs,
                    "steps": steps,
                    "final_output": final_output,
                    "error": f"节点 '{nid}' 执行失败: {step_error}",
                }
        except Exception as e:
            import traceback
            steps.append({
                "node_id": nid,
                "agent_id": agent_id,
                "output_key": output_key,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
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
