"""多 Agent 流程画布：可视化编排 Agent 工作流。"""

import json
from pathlib import Path

import streamlit as st

try:
    from streamlit_agraph import Node, Edge, Config
except ImportError:
    Node = Edge = Config = None

from common.config_loader import get_project_root
from common.tool_registry import list_tools_for_agent
from dev_platform import agent_manager, flow_manager, kb_manager, skill_manager
from methods.flow_planner import plan_flow_from_natural_language


def _project_root() -> Path:
    return get_project_root()


def _init_session():
    keys = {
        "flow_id": "",
        "flow_name": "",
        "flow_description": "",
        "flow_input_desc": "用户初始输入",
        "flow_input_default": "",
        "flow_nodes": [],
        "flow_edges": [],
        "flow_edit_mode": False,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_flow_editor():
    for k in ["flow_id", "flow_name", "flow_description", "flow_input_desc", "flow_input_default", "flow_nodes", "flow_edges", "flow_edit_mode"]:
        if k in st.session_state:
            del st.session_state[k]
    _init_session()


def _load_flow_into_editor(flow_id: str):
    root = _project_root()
    cfg = flow_manager.load_flow(flow_id, root)
    st.session_state.flow_id = cfg.get("flow_id", flow_id)
    st.session_state.flow_name = cfg.get("name", "")
    st.session_state.flow_description = cfg.get("description", "")
    inp = cfg.get("input", {})
    st.session_state.flow_input_desc = inp.get("description", "用户初始输入")
    st.session_state.flow_input_default = inp.get("default_value", "")
    st.session_state.flow_nodes = cfg.get("nodes", [])
    st.session_state.flow_edges = cfg.get("edges", [])
    st.session_state.flow_edit_mode = True


def _build_flow_preview(nodes: list, edges: list, step_results: dict = None):
    """使用 streamlit-agraph 构建流程预览图。

    参数：
        step_results: 可选，节点执行结果 {node_id: {"success": bool}}，用于着色。
    """
    if Node is None or Edge is None or Config is None:
        st.error("请安装 streamlit-agraph: pip install streamlit-agraph")
        return None, None, None

    node_map = {n["id"]: n for n in nodes}
    graph_nodes = []
    graph_edges = []

    for i, node in enumerate(nodes):
        nid = node["id"]
        label = node.get("agent_id", nid)

        # 根据执行状态着色
        color = "#4C78A8"  # 默认蓝色
        if step_results:
            if nid in step_results:
                color = "#2E8B57" if step_results[nid].get("success") else "#E45756"  # 绿/红
            else:
                color = "#AAAAAA"  # 未执行灰色

        title_lines = [
            f"ID: {nid}",
            f"Agent: {node.get('agent_id', '')}",
            f"输出键: {node.get('output_key', '')}",
        ]
        if step_results and nid in step_results:
            status = "成功" if step_results[nid].get("success") else "失败"
            title_lines.append(f"状态: {status}")

        graph_nodes.append(Node(
            id=nid,
            label=label,
            title="\n".join(title_lines),
            color=color,
            shape="box",
            size=25,
            font={"color": "#ffffff", "size": 12},
            level=0,
        ))

    for edge in edges:
        src = edge.get("from")
        dst = edge.get("to")
        if src in node_map and dst in node_map:
            graph_edges.append(Edge(source=src, target=dst, color="#888888", arrows={"to": {"enabled": True}}))

    config = Config(
        width="100%",
        height=400,
        directed=True,
        physics=False,
        hierarchical=True,
        direction="LR",
        levelSeparation=200,
        nodeSpacing=100,
        treeSpacing=200,
    )
    return graph_nodes, graph_edges, config


def _tab_editor():
    st.subheader("🎨 流程编辑器")

    is_edit = st.session_state.get("flow_edit_mode", False)
    if is_edit:
        st.info(f"正在编辑流程：`{st.session_state.flow_id}`")

    # 元信息
    with st.expander("流程元信息", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            flow_id = st.text_input(
                "流程 ID *",
                value=st.session_state.flow_id,
                key="flow_id",
                help="唯一标识，将作为文件名",
                disabled=is_edit,
            )
            flow_name = st.text_input(
                "流程名称 *",
                value=st.session_state.flow_name,
                key="flow_name",
            )
        with col2:
            flow_description = st.text_area(
                "流程描述",
                value=st.session_state.flow_description,
                key="flow_description",
                height=80,
            )

        c1, c2 = st.columns(2)
        with c1:
            flow_input_desc = st.text_input(
                "初始输入说明",
                value=st.session_state.flow_input_desc,
                key="flow_input_desc",
            )
        with c2:
            flow_input_default = st.text_input(
                "初始输入默认值",
                value=st.session_state.flow_input_default,
                key="flow_input_default",
            )

    # AI 一键规划
    st.markdown("---")
    with st.expander("🤖 AI 一键规划", expanded=True):
        st.markdown("用自然语言描述你想搭建的流程，Agent 会自动生成节点、边和输入模板。")
        plan_requirement = st.text_area(
            "需求描述",
            height=80,
            placeholder="例如：先用 InterpretationAdvisorAgent 分析泥质含量并制定策略，再调用 shale_run_interpretation 工具执行计算。",
            key="flow_plan_requirement",
        )
        if st.button("🚀 生成流程", type="primary", key="btn_plan_flow"):
            if not plan_requirement.strip():
                st.warning("请先填写需求描述。")
            else:
                with st.spinner("Agent 正在规划流程，请稍候..."):
                    res = plan_flow_from_natural_language(plan_requirement.strip(), _project_root())
                if res["success"]:
                    flow = res["flow"]
                    st.session_state.flow_id = flow.get("flow_id", "ai_flow")
                    st.session_state.flow_name = flow.get("name", "AI 生成流程")
                    st.session_state.flow_description = flow.get("description", "")
                    inp = flow.get("input", {})
                    st.session_state.flow_input_desc = inp.get("description", "用户初始输入")
                    st.session_state.flow_input_default = inp.get("default_value", "")
                    st.session_state.flow_nodes = flow.get("nodes", [])
                    st.session_state.flow_edges = flow.get("edges", [])
                    st.session_state.flow_edit_mode = False
                    st.success("流程生成完成，请在下方检查并保存。")
                    st.rerun()
                else:
                    st.error(res.get("error", "生成失败"))
                    if res.get("raw"):
                        with st.expander("原始响应"):
                            st.text(res["raw"])

    # 节点管理
    st.markdown("---")
    st.subheader("🧩 节点管理")

    root = _project_root()
    agents = agent_manager.list_agents(root)
    agent_options = [a["agent_id"] for a in agents]
    tools = list_tools_for_agent(root)
    tool_options = [t["name"] for t in tools]
    docs = kb_manager.list_documents(root)
    doc_options = [d["name"] for d in docs]
    skills = skill_manager.list_skills(root)
    skill_options = [s["name"] for s in skills]

    # 添加节点
    with st.expander("➕ 添加节点", expanded=False):
        with st.form("add_node_form"):
            new_node_id = st.text_input("节点 ID *", help="例如 advisor、executor")
            new_agent_id = st.selectbox("选择 Agent *", agent_options) if agent_options else st.text_input("Agent ID *")
            new_tools = st.multiselect("可调用工具", tool_options)
            new_docs = st.multiselect("引用知识库文档", doc_options)
            new_skills = st.multiselect("引用 Skill", skill_options)
            new_template = st.text_area(
                "输入模板 *",
                value="{__input__}",
                height=100,
                help="可用占位符：{__input__} 和上游节点的 {output_key}",
            )
            new_output_key = st.text_input("输出键名 *", help="例如 strategy、result")

            submitted = st.form_submit_button("添加节点", type="primary")
            if submitted:
                if not new_node_id or not new_agent_id or not new_template or not new_output_key:
                    st.error("请填写所有必填项。")
                elif new_node_id in {n["id"] for n in st.session_state.flow_nodes}:
                    st.error(f"节点 ID '{new_node_id}' 已存在。")
                else:
                    st.session_state.flow_nodes.append({
                        "id": new_node_id.strip(),
                        "agent_id": new_agent_id.strip() if isinstance(new_agent_id, str) else new_agent_id,
                        "tools": new_tools,
                        "kb_docs": new_docs,
                        "skills": new_skills,
                        "input_template": new_template.strip(),
                        "output_key": new_output_key.strip(),
                        "output_parser": "text",
                    })
                    st.success(f"节点 '{new_node_id}' 已添加")
                    st.rerun()

    # 显示已有节点
    if st.session_state.flow_nodes:
        for i, node in enumerate(st.session_state.flow_nodes):
            with st.expander(f"节点: {node['id']} ({node['agent_id']}) -> {node['output_key']}"):
                st.json(node)
                if st.button("删除节点", key=f"del_node_{i}"):
                    # 同时删除相关边
                    st.session_state.flow_edges = [
                        e for e in st.session_state.flow_edges
                        if e.get("from") != node["id"] and e.get("to") != node["id"]
                    ]
                    st.session_state.flow_nodes.pop(i)
                    st.rerun()
    else:
        st.info("尚未添加节点。")

    # 边管理
    st.markdown("---")
    st.subheader("🔗 边管理（数据传递）")

    node_ids = [n["id"] for n in st.session_state.flow_nodes]
    with st.form("add_edge_form"):
        col_from, col_to = st.columns(2)
        with col_from:
            from_node = st.selectbox("源节点", node_ids, key="edge_from") if node_ids else st.text_input("源节点")
        with col_to:
            to_node = st.selectbox("目标节点", node_ids, key="edge_to") if node_ids else st.text_input("目标节点")

        if st.form_submit_button("添加边", type="primary"):
            if from_node and to_node:
                if from_node == to_node:
                    st.error("不能连接节点自身。")
                else:
                    st.session_state.flow_edges.append({"from": from_node, "to": to_node})
                    st.success(f"边 {from_node} -> {to_node} 已添加")
                    st.rerun()

    if st.session_state.flow_edges:
        for i, edge in enumerate(st.session_state.flow_edges):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"`{edge.get('from')}` → `{edge.get('to')}`")
            with c2:
                if st.button("删除", key=f"del_edge_{i}"):
                    st.session_state.flow_edges.pop(i)
                    st.rerun()

    # 可视化预览
    st.markdown("---")
    st.subheader("👁️ 流程预览")
    if st.session_state.flow_nodes:
        preview = _build_flow_preview(st.session_state.flow_nodes, st.session_state.flow_edges)
        if preview[0] is not None:
            from streamlit_agraph import agraph
            agraph(nodes=preview[0], edges=preview[1], config=preview[2])
    else:
        st.info("添加节点和边后将在此处显示预览。")

    # 保存流程
    st.markdown("---")
    c_save, c_clear = st.columns([1, 1])
    with c_save:
        if st.button("💾 保存流程", type="primary"):
            if not flow_id.strip() or not flow_name.strip():
                st.error("流程 ID 和名称不能为空。")
            elif not st.session_state.flow_nodes:
                st.error("至少需要添加一个节点。")
            else:
                res = flow_manager.save_flow(
                    flow_id=flow_id.strip(),
                    name=flow_name.strip(),
                    description=flow_description.strip(),
                    nodes=st.session_state.flow_nodes,
                    edges=st.session_state.flow_edges,
                    input_desc=flow_input_desc.strip(),
                    input_default=flow_input_default.strip(),
                    project_root=root,
                )
                if res["success"]:
                    st.success(res["message"])
                    _clear_flow_editor()
                    st.rerun()
                else:
                    st.error(res["message"])
    with c_clear:
        if st.button("🔄 清空编辑器"):
            _clear_flow_editor()
            st.rerun()


def _tab_list():
    st.subheader("📋 流程列表")

    root = _project_root()
    flows = flow_manager.list_flows(root)

    if not flows:
        st.info("暂无流程。")
        return

    for flow in flows:
        builtin = flow.get("builtin", False)
        badge = "🔵 内置" if builtin else "🟢 自定义"
        with st.expander(f"{badge} {flow['name']}  (`{flow['flow_id']}`)"):
            st.markdown(f"**描述**：{flow.get('description', '')}")
            st.markdown(f"**配置文件**：`{flow.get('path', '')}`")

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("✏️ 加载到编辑器", key=f"load_flow_{flow['flow_id']}"):
                    _load_flow_into_editor(flow["flow_id"])
                    st.rerun()
            with c2:
                if st.button("▶️ 运行", key=f"run_flow_{flow['flow_id']}"):
                    st.session_state.selected_run_flow = flow["flow_id"]
                    st.session_state.active_flow_tab = 2
                    st.rerun()
            with c3:
                if not builtin:
                    if st.button("🗑️ 删除", key=f"del_flow_{flow['flow_id']}"):
                        res = flow_manager.delete_flow(flow["flow_id"], root)
                        if res["success"]:
                            st.success(res["message"])
                            st.rerun()
                        else:
                            st.error(res["message"])


def _tab_run():
    st.subheader("▶️ 运行流程")

    root = _project_root()
    flows = flow_manager.list_flows(root)
    flow_options = {f["name"]: f["flow_id"] for f in flows}

    selected = st.session_state.get("selected_run_flow", "")
    default_name = ""
    if selected and selected in flow_options.values():
        default_name = [k for k, v in flow_options.items() if v == selected][0]

    chosen_name = st.selectbox("选择流程", list(flow_options.keys()), index=list(flow_options.values()).index(selected) if default_name else 0)
    flow_id = flow_options[chosen_name]

    # 加载默认值
    try:
        cfg = flow_manager.load_flow(flow_id, root)
        default_input = cfg.get("input", {}).get("default_value", "")
    except Exception:
        default_input = ""

    user_input = st.text_area("初始输入", value=default_input, height=100)

    if st.button("🚀 运行", type="primary"):
        with st.spinner("正在执行多 Agent 流程，请稍候..."):
            from methods.agent_flow_runner import run_flow
            result = run_flow(flow_id, user_input, root)

        if result["success"]:
            st.success("流程执行完成")
        else:
            st.error(f"流程执行失败: {result.get('error', '')}")

        # 构建执行状态用于可视化
        step_results = {step["node_id"]: step for step in result.get("steps", [])}

        st.markdown("---")
        st.subheader("🗺️ 执行步骤可视化")
        try:
            cfg = flow_manager.load_flow(flow_id, root)
            preview = _build_flow_preview(cfg.get("nodes", []), cfg.get("edges", []), step_results)
            if preview[0] is not None:
                from streamlit_agraph import agraph
                agraph(nodes=preview[0], edges=preview[1], config=preview[2])
                st.caption("🟩 绿色=成功  🟥 红色=失败  ⬜ 灰色=未执行")
        except Exception as e:
            st.warning(f"可视化生成失败: {e}")

        with st.expander("执行详情", expanded=True):
            for step in result.get("steps", []):
                if step.get("success"):
                    st.markdown(f"✅ **{step['node_id']}** ({step['agent_id']}) → `{step['output_key']}`")
                    st.text(step.get("output", "")[:500])
                else:
                    st.markdown(f"❌ **{step['node_id']}** ({step['agent_id']})")
                    st.error(step.get("error", ""))

        if result.get("final_output"):
            st.markdown("---")
            st.subheader("🏁 最终输出")
            st.markdown(result["final_output"])


def page_agent_flow_platform():
    st.title("🎨 Agent 流程画布")
    st.markdown("可视化编排多智能体流程：为每个 Agent 配置工具、知识库、Skill，并定义数据传递。")

    _init_session()

    tabs = st.tabs(["🎨 流程编辑", "📋 流程列表", "▶️ 运行流程"])
    with tabs[0]:
        _tab_editor()
    with tabs[1]:
        _tab_list()
    with tabs[2]:
        _tab_run()
