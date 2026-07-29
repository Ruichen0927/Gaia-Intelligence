"""多 Agent 流程配置管理：保存、加载、列出、校验 Agent Flow。"""

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root, load_json, save_json
from common.tool_registry import list_tools_for_agent, format_tools_for_prompt
from dev_platform import agent_manager, kb_manager, skill_manager


FLOW_DIR = "ctl/agent_flows"
PIPELINE_FILE = "ctl/pipeline_config.json"
BUILTIN_FLOW_ID = "pipeline"


def _get_flow_dir(project_root: Path) -> Path:
    return project_root / FLOW_DIR


def _sanitize_flow_id(flow_id: str) -> str:
    """把任意字符串转为安全的 flow_id（文件名）。"""
    flow_id = flow_id.strip().replace(" ", "_")
    flow_id = re.sub(r"[^\w\-]", "", flow_id)
    return flow_id or "untitled_flow"


def _is_builtin(flow_id: str) -> bool:
    return flow_id == BUILTIN_FLOW_ID


def list_flows(project_root: Path = None) -> List[Dict[str, Any]]:
    """列出所有流程：内置流程法 + 自定义 Agent Flow。"""
    root = project_root or get_project_root()
    flows = []

    # 内置流程法
    pipeline_path = root / PIPELINE_FILE
    if pipeline_path.exists():
        try:
            pipeline = load_json(PIPELINE_FILE, root)
            flows.append({
                "flow_id": BUILTIN_FLOW_ID,
                "name": "内置流程法",
                "description": f"来自 {PIPELINE_FILE} 的流程法配置，共 {len(pipeline)} 个步骤",
                "builtin": True,
                "path": PIPELINE_FILE,
            })
        except Exception:
            pass

    # 自定义 Agent Flow
    flow_dir = _get_flow_dir(root)
    if flow_dir.exists():
        for flow_file in sorted(flow_dir.glob("*.json")):
            try:
                cfg = json.loads(flow_file.read_text(encoding="utf-8"))
                flows.append({
                    "flow_id": cfg.get("flow_id", flow_file.stem),
                    "name": cfg.get("name", flow_file.stem),
                    "description": cfg.get("description", ""),
                    "builtin": False,
                    "path": str(flow_file.relative_to(root)),
                })
            except Exception:
                continue

    return flows


def load_flow(flow_id: str, project_root: Path = None) -> Dict[str, Any]:
    """加载流程配置；内置流程法会包装为统一格式。"""
    root = project_root or get_project_root()

    if _is_builtin(flow_id):
        pipeline = load_json(PIPELINE_FILE, root)
        # 把 pipeline_config.json 的每个 step 映射为 Agent Flow 节点
        nodes = []
        for i, step in enumerate(pipeline):
            tool = step.get("tool", "")
            function = step.get("function", "")
            nodes.append({
                "id": f"step_{i}",
                "agent_id": "ToolDeveloperAgent",  # 默认使用工具开发 Agent 作为执行 Agent
                "role_description": f"流程法第 {i + 1} 步执行专家",
                "task_instruction": f"调用工具 `{tool}` 的函数 `{function}` 执行步骤 `{step.get('step', '')}`",
                "tools": [tool] if tool else [],
                "kb_docs": [],
                "skills": [],
                "input_template": f"请调用工具 `{tool}` 的函数 `{function}` 执行步骤 `{step.get('step', '')}`。",
                "output_key": step.get("step", f"output_{i}"),
                "output_parser": "text",
            })
        edges = []
        for i in range(len(nodes) - 1):
            edges.append({"from": nodes[i]["id"], "to": nodes[i + 1]["id"]})
        return {
            "flow_id": BUILTIN_FLOW_ID,
            "name": "内置流程法",
            "description": f"来自 {PIPELINE_FILE} 的流程法配置",
            "builtin": True,
            "input": {"description": "初始输入", "default_value": ""},
            "nodes": nodes,
            "edges": edges,
        }

    flow_path = _get_flow_dir(root) / f"{_sanitize_flow_id(flow_id)}.json"
    if not flow_path.exists():
        raise FileNotFoundError(f"流程不存在: {flow_path}")
    cfg = json.loads(flow_path.read_text(encoding="utf-8"))
    cfg["builtin"] = False
    return cfg


def save_flow(
    flow_id: str,
    name: str,
    description: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, str]],
    input_desc: str,
    input_default: str,
    project_root: Path = None,
) -> Dict[str, Any]:
    """保存自定义 Agent Flow。"""
    root = project_root or get_project_root()

    safe_id = _sanitize_flow_id(flow_id)
    if not safe_id:
        return {"success": False, "message": "流程 ID 无效", "file": ""}

    # 预校验引用资源和必填字段
    pre_cfg = {
        "flow_id": safe_id,
        "nodes": nodes,
        "edges": edges,
    }
    validation = validate_flow(pre_cfg, root)
    if not validation["valid"]:
        return {"success": False, "message": "保存失败:\n" + "\n".join(validation["errors"]), "file": ""}

    cfg = {
        "flow_id": safe_id,
        "name": name.strip() or safe_id,
        "description": description.strip(),
        "input": {
            "description": input_desc.strip(),
            "default_value": input_default.strip(),
        },
        "nodes": nodes,
        "edges": edges,
    }

    flow_dir = _get_flow_dir(root)
    flow_dir.mkdir(parents=True, exist_ok=True)
    flow_path = flow_dir / f"{safe_id}.json"
    flow_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "message": f"流程 '{safe_id}' 已保存到 {flow_path.relative_to(root)}",
        "file": str(flow_path.relative_to(root)),
    }


def delete_flow(flow_id: str, project_root: Path = None) -> Dict[str, Any]:
    """删除自定义流程；内置流程法不可删除。"""
    root = project_root or get_project_root()

    if _is_builtin(flow_id):
        return {"success": False, "message": "内置流程法不可删除"}

    flow_path = _get_flow_dir(root) / f"{_sanitize_flow_id(flow_id)}.json"
    if not flow_path.exists():
        return {"success": False, "message": f"流程 '{flow_id}' 不存在"}
    flow_path.unlink()
    return {"success": True, "message": f"流程 '{flow_id}' 已删除"}


def validate_flow(flow_cfg: Dict[str, Any], project_root: Path = None) -> Dict[str, Any]:
    """校验流程配置中的引用是否真实存在。

    返回：
        {"valid": bool, "errors": List[str]}
    """
    root = project_root or get_project_root()
    errors = []

    nodes = flow_cfg.get("nodes", [])
    node_ids = {n.get("id") for n in nodes}

    # 可用资源
    agents = {a["agent_id"] for a in agent_manager.list_agents(root)}
    tools = {t["name"] for t in list_tools_for_agent(root)}
    docs = {d["name"] for d in kb_manager.list_documents(root)}
    skills = {s["name"] for s in skill_manager.list_skills(root)}

    for node in nodes:
        nid = node.get("id")
        if not nid:
            errors.append("存在未设置 id 的节点")
            continue
        agent_id = node.get("agent_id")
        if agent_id not in agents:
            errors.append(f"节点 '{nid}' 引用了不存在的 Agent: {agent_id}")

        if not node.get("role_description") or not str(node.get("role_description")).strip():
            errors.append(f"节点 '{nid}' 的 role_description（角色描述）不能为空")
        if not node.get("task_instruction") or not str(node.get("task_instruction")).strip():
            errors.append(f"节点 '{nid}' 的 task_instruction（任务说明）不能为空")

        for t in node.get("tools", []):
            if t not in tools:
                errors.append(f"节点 '{nid}' 引用了不存在的工具: {t}")
        for d in node.get("kb_docs", []):
            if d not in docs:
                errors.append(f"节点 '{nid}' 引用了不存在的知识库文档: {d}")
        for s in node.get("skills", []):
            if s not in skills:
                errors.append(f"节点 '{nid}' 引用了不存在的 Skill: {s}")

    for edge in flow_cfg.get("edges", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src not in node_ids:
            errors.append(f"边引用了不存在的源节点: {src}")
        if dst not in node_ids:
            errors.append(f"边引用了不存在的目标节点: {dst}")
        if src == dst:
            errors.append(f"节点 '{src}' 存在自环边")

    return {"valid": len(errors) == 0, "errors": errors}


def topological_sort(nodes: List[Dict[str, Any]], edges: List[Dict[str, str]]) -> List[str]:
    """对流程节点进行拓扑排序；若存在环则抛出 ValueError。"""
    node_ids = [n["id"] for n in nodes]
    node_set = set(node_ids)
    adj = defaultdict(list)
    in_degree = {nid: 0 for nid in node_set}

    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in node_set and dst in node_set:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque([nid for nid in node_ids if in_degree[nid] == 0])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(node_ids):
        raise ValueError("流程中存在环，无法执行")

    return result


def build_node_prompt(
    node_cfg: Dict[str, Any],
    outputs: Dict[str, Any],
    initial_input: str,
    project_root: Path = None,
) -> str:
    """构建某个 Agent 节点的最终输入 prompt。

    包含：团队上下文、角色任务、工具描述、知识库片段、Skill 内容、input_template 渲染结果。
    """
    root = project_root or get_project_root()
    parts = []

    # 角色与任务
    role = node_cfg.get("role_description", "")
    task = node_cfg.get("task_instruction", "")
    if role or task:
        parts.append("【你的角色与任务】\n" + (f"角色：{role}\n" if role else "") + (f"任务：{task}" if task else ""))

    # 工具描述
    tools = node_cfg.get("tools", [])
    if tools:
        all_tools = list_tools_for_agent(root)
        selected = [t for t in all_tools if t["name"] in tools]
        parts.append("【你可使用的工具】\n" + format_tools_for_prompt(selected, include_source=False))

    # 知识库片段
    kb_docs = node_cfg.get("kb_docs", [])
    if kb_docs:
        kb_texts = []
        for doc_name in kb_docs:
            results = kb_manager.search_documents(doc_name, top_k=1, project_root=root)
            if results:
                kb_texts.append(f"- 来自《{doc_name}》：{results[0]['chunk']}")
        if kb_texts:
            parts.append("【参考知识库文档】\n" + "\n".join(kb_texts))

    # Skill 内容
    skills = node_cfg.get("skills", [])
    if skills:
        all_skills = skill_manager.load_all_skills(root)
        selected = [s for s in all_skills if s.get("name") in skills]
        skill_text = skill_manager.format_skills_for_prompt(selected)
        if skill_text:
            parts.append(skill_text)

    # 渲染 input_template
    template = node_cfg.get("input_template", "{__input__}")
    context = dict(outputs)
    context["__input__"] = initial_input
    rendered = template.format(**context)

    parts.append("【当前任务】\n" + rendered)
    parts.append("【输出要求】\n请直接给出你的分析结果或决策，不要调用任何外部函数。你的输出文本将被下游节点作为输入使用。")

    return "\n\n".join(parts)


def build_node_system_message(
    node_cfg: Dict[str, Any],
    flow_cfg: Dict[str, Any],
    base_system_message: str,
) -> str:
    """为节点生成临时 system message 覆盖。

    在原 Agent system_message 基础上，追加该节点在团队中的角色说明。
    """
    role = node_cfg.get("role_description", "")
    task = node_cfg.get("task_instruction", "")
    flow_name = flow_cfg.get("name", "多 Agent 流程")

    override_parts = [base_system_message]
    override_parts.append(f"\n\n你当前正在参与一个名为『{flow_name}』的多 Agent 协作流程。")
    if role:
        override_parts.append(f"你在该流程中的角色是：{role}")
    if task:
        override_parts.append(f"你的具体任务是：{task}")
    override_parts.append("请根据用户输入和上下文，完成你的任务并输出清晰、可直接被下游节点使用的文本结果。")

    return "\n".join(override_parts)
