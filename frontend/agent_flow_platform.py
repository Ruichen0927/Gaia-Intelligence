"""多 Agent 流程画布：基于 streamlit-flow-component 的画布式工作流编排。"""

import json
from pathlib import Path

import streamlit as st

try:
    from streamlit_flow import streamlit_flow, StreamlitFlowNode, StreamlitFlowEdge, StreamlitFlowState
    HAS_FLOW_COMPONENT = True
except ImportError:
    StreamlitFlowNode = StreamlitFlowEdge = StreamlitFlowState = streamlit_flow = None
    HAS_FLOW_COMPONENT = False

from common.config_loader import get_project_root
from common.tool_registry import list_tools_for_agent
from dev_platform import agent_manager, flow_manager, kb_manager, skill_manager
from methods.flow_planner import plan_flow_from_natural_language


NODE_WIDTH = 220
NODE_HEIGHT = 100


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
        "selected_node_id": None,
        "flow_state": None,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _clear_flow_editor():
    for k in ["flow_id", "flow_name", "flow_description", "flow_input_desc", "flow_input_default",
              "flow_nodes", "flow_edges", "flow_edit_mode", "selected_node_id", "flow_state"]:
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
    st.session_state.selected_node_id = None
    st.session_state.flow_state = None
    st.session_state.flow_edit_mode = True
    # 同步 widget 值
    for meta_key in ["flow_id", "flow_name", "flow_description", "flow_input_desc", "flow_input_default"]:
        st.session_state[f"{meta_key}_widget"] = st.session_state.get(meta_key, "")


def _default_node_data():
    return {
        "agent_id": "",
        "role_description": "",
        "task_instruction": "",
        "tools": [],
        "kb_docs": [],
        "skills": [],
        "input_template": "{__input__}",
        "output_key": "output",
        "output_parser": "text",
    }


def _node_label(node: dict) -> str:
    role = node.get("role_description", "").strip()
    agent = node.get("agent_id", "").strip()
    return role or agent or node.get("id", "节点")


def _build_node_data(node: dict, step_results: dict = None) -> dict:
    """把内部节点配置转换为 StreamlitFlowNode 的 data 字段。"""
    nid = node["id"]
    label = _node_label(node)
    status_color = "#4C78A8"
    if step_results and nid in step_results:
        status_color = "#2E8B57" if step_results[nid].get("success") else "#E45756"

    return {
        "label": label,
        "agent_id": node.get("agent_id", ""),
        "role_description": node.get("role_description", ""),
        "task_instruction": node.get("task_instruction", ""),
        "tools": node.get("tools", []),
        "kb_docs": node.get("kb_docs", []),
        "skills": node.get("skills", []),
        "input_template": node.get("input_template", ""),
        "output_key": node.get("output_key", ""),
        "output_parser": node.get("output_parser", "text"),
        "status_color": status_color,
    }


def _node_style(node: dict, step_results: dict = None) -> dict:
    """节点样式。"""
    nid = node["id"]
    color = "#4C78A8"
    if step_results and nid in step_results:
        color = "#2E8B57" if step_results[nid].get("success") else "#E45756"
    return {
        "backgroundColor": color,
        "color": "#ffffff",
        "borderRadius": "8px",
        "padding": "10px",
        "width": f"{NODE_WIDTH}px",
        "minHeight": f"{NODE_HEIGHT}px",
        "fontSize": "12px",
        "border": "1px solid #2c3e50",
    }


def _nodes_edges_to_state(nodes: list, edges: list, step_results: dict = None) -> StreamlitFlowState:
    """把内部 nodes/edges 转换为 StreamlitFlowState。"""
    flow_nodes = []
    for i, node in enumerate(nodes):
        pos = node.get("position", {})
        x = pos.get("x", i * 280)
        y = pos.get("y", 100)
        flow_nodes.append(StreamlitFlowNode(
            id=node["id"],
            pos=(x, y),
            data=_build_node_data(node, step_results),
            node_type="default",
            source_position="bottom",
            target_position="top",
            style=_node_style(node, step_results),
            draggable=True,
            selectable=True,
            connectable=True,
            deletable=True,
        ))

    flow_edges = []
    for i, edge in enumerate(edges):
        src = edge.get("from") or edge.get("source")
        dst = edge.get("to") or edge.get("target")
        if src and dst:
            flow_edges.append(StreamlitFlowEdge(
                id=f"e-{src}-{dst}",
                source=src,
                target=dst,
                animated=True,
                marker_end={"type": "arrow", "width": 15, "height": 15},
                style={"strokeWidth": 2, "stroke": "#888"},
                deletable=True,
                focusable=True,
            ))

    return StreamlitFlowState(flow_nodes, flow_edges)


def _state_to_nodes_edges(state: StreamlitFlowState) -> tuple:
    """把 StreamlitFlowState 转换回内部 nodes/edges。"""
    nodes = []
    edges = []
    if state is None:
        return nodes, edges

    for fn in getattr(state, "nodes", []) or []:
        data = getattr(fn, "data", {}) or {}
        pos = getattr(fn, "position", {"x": 0, "y": 0})
        # streamlit-flow-component 会把 data['label'] 转成 data['content']
        role = data.get("role_description", "")
        if not role:
            role = data.get("label", "") or data.get("content", "")
        nodes.append({
            "id": getattr(fn, "id", ""),
            "agent_id": data.get("agent_id", ""),
            "role_description": role,
            "task_instruction": data.get("task_instruction", ""),
            "tools": data.get("tools", []),
            "kb_docs": data.get("kb_docs", []),
            "skills": data.get("skills", []),
            "input_template": data.get("input_template", "{__input__}"),
            "output_key": data.get("output_key", ""),
            "output_parser": data.get("output_parser", "text"),
            "position": {"x": pos.get("x", 0), "y": pos.get("y", 0)},
        })

    for fe in getattr(state, "edges", []) or []:
        edges.append({
            "from": getattr(fe, "source", ""),
            "to": getattr(fe, "target", ""),
        })

    return nodes, edges


def _sync_state_to_session(state: StreamlitFlowState):
    """把组件返回的 state 同步回 session_state 的 nodes/edges。"""
    nodes, edges = _state_to_nodes_edges(state)
    st.session_state.flow_nodes = nodes
    st.session_state.flow_edges = edges
    st.session_state.flow_state = state


def _get_flow_state(step_results: dict = None) -> StreamlitFlowState:
    """获取当前画布 state（优先用 session_state 中缓存的，避免重复构建）。"""
    if st.session_state.flow_state is None:
        st.session_state.flow_state = _nodes_edges_to_state(
            st.session_state.flow_nodes,
            st.session_state.flow_edges,
            step_results,
        )
    return st.session_state.flow_state


def _add_node(node_id: str, agent_id: str, x: float, y: float):
    """在画布上添加一个新节点。"""
    node = {
        "id": node_id,
        "agent_id": agent_id,
        "role_description": "新节点",
        "task_instruction": "请填写任务说明",
        "tools": [],
        "kb_docs": [],
        "skills": [],
        "input_template": "{__input__}",
        "output_key": "output",
        "output_parser": "text",
        "position": {"x": x, "y": y},
    }
    st.session_state.flow_nodes.append(node)
    st.session_state.flow_state = None


def _delete_node(nid: str):
    """删除节点及相关边。"""
    st.session_state.flow_nodes = [n for n in st.session_state.flow_nodes if n["id"] != nid]
    st.session_state.flow_edges = [
        e for e in st.session_state.flow_edges
        if e.get("from") != nid and e.get("to") != nid
    ]
    if st.session_state.selected_node_id == nid:
        st.session_state.selected_node_id = None
    st.session_state.flow_state = None


def _update_node_field(nid: str, field: str, value):
    """更新指定节点的某个字段。"""
    for node in st.session_state.flow_nodes:
        if node["id"] == nid:
            node[field] = value
            break
    st.session_state.flow_state = None


def _tab_editor():
    st.subheader("🎨 流程画布编辑器")

    if not HAS_FLOW_COMPONENT:
        st.error("请先安装 streamlit-flow-component: pip install streamlit-flow-component")
        return

    is_edit = st.session_state.get("flow_edit_mode", False)
    if is_edit:
        st.info(f"正在编辑流程：`{st.session_state.flow_id}`")

    root = _project_root()
    agents = agent_manager.list_agents(root)
    agent_options = [a["agent_id"] for a in agents]
    tools = list_tools_for_agent(root)
    tool_options = [t["name"] for t in tools]
    docs = kb_manager.list_documents(root)
    doc_options = [d["name"] for d in docs]
    skills = skill_manager.list_skills(root)
    skill_options = [s["name"] for s in skills]

    # 三列布局：左侧元信息 + 中间画布 + 右侧节点编辑器
    col_meta, col_canvas, col_editor = st.columns([1, 3, 1])

    with col_meta:
        # 为了避免 widget key 与 session_state 变量冲突，widget 使用 _widget 后缀
        for meta_key in ["flow_id", "flow_name", "flow_description", "flow_input_desc", "flow_input_default"]:
            widget_key = f"{meta_key}_widget"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.get(meta_key, "")

        with st.expander("流程元信息", expanded=True):
            flow_id = st.text_input(
                "流程 ID *",
                value=st.session_state.flow_id,
                key="flow_id_widget",
                help="唯一标识，将作为文件名",
                disabled=is_edit,
            )
            flow_name = st.text_input(
                "流程名称 *",
                value=st.session_state.flow_name,
                key="flow_name_widget",
            )
            flow_description = st.text_area(
                "流程描述",
                value=st.session_state.flow_description,
                key="flow_description_widget",
                height=80,
            )
            flow_input_desc = st.text_input(
                "初始输入说明",
                value=st.session_state.flow_input_desc,
                key="flow_input_desc_widget",
            )
            flow_input_default = st.text_input(
                "初始输入默认值",
                value=st.session_state.flow_input_default,
                key="flow_input_default_widget",
            )

        with st.expander("🤖 AI 一键规划", expanded=True):
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
                        res = plan_flow_from_natural_language(plan_requirement.strip(), root)
                    if res["success"]:
                        flow = res["flow"]
                        st.session_state.flow_id = flow.get("flow_id", "ai_flow")
                        st.session_state.flow_id_widget = st.session_state.flow_id
                        st.session_state.flow_name = flow.get("name", "AI 生成流程")
                        st.session_state.flow_name_widget = st.session_state.flow_name
                        st.session_state.flow_description = flow.get("description", "")
                        st.session_state.flow_description_widget = st.session_state.flow_description
                        inp = flow.get("input", {})
                        st.session_state.flow_input_desc = inp.get("description", "用户初始输入")
                        st.session_state.flow_input_desc_widget = st.session_state.flow_input_desc
                        st.session_state.flow_input_default = inp.get("default_value", "")
                        st.session_state.flow_input_default_widget = st.session_state.flow_input_default
                        st.session_state.flow_nodes = flow.get("nodes", [])
                        st.session_state.flow_edges = flow.get("edges", [])
                        st.session_state.selected_node_id = None
                        st.session_state.flow_state = None
                        st.session_state.flow_edit_mode = False
                        st.success("流程生成完成，请在画布上检查并保存。")
                        st.rerun()
                    else:
                        st.error(res.get("error", "生成失败"))
                        if res.get("raw"):
                            with st.expander("原始响应"):
                                st.text(res["raw"])

        # 添加节点
        with st.expander("➕ 添加节点"):
            new_node_id = st.text_input("节点 ID *", key="new_node_id", help="例如 advisor、executor")
            new_agent_id = st.selectbox("选择 Agent *", agent_options, key="new_agent_id") if agent_options else st.text_input("Agent ID *", key="new_agent_id_text")
            if st.button("添加节点", key="btn_add_node"):
                chosen_agent = new_agent_id if isinstance(new_agent_id, str) else st.session_state.get("new_agent_id_text", "")
                if not new_node_id or not chosen_agent:
                    st.error("节点 ID 和 Agent 不能为空。")
                elif new_node_id in {n["id"] for n in st.session_state.flow_nodes}:
                    st.error(f"节点 ID '{new_node_id}' 已存在。")
                else:
                    x = 100 + len(st.session_state.flow_nodes) * 50
                    y = 100 + len(st.session_state.flow_nodes) * 30
                    _add_node(new_node_id.strip(), chosen_agent.strip(), x, y)
                    st.success(f"节点 '{new_node_id}' 已添加")
                    st.rerun()

        # 保存/清空
        c_save, c_clear = st.columns(2)
        with c_save:
            if st.button("💾 保存流程", type="primary", key="btn_save_flow"):
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
            if st.button("🔄 清空", key="btn_clear_flow"):
                _clear_flow_editor()
                st.rerun()

    with col_canvas:
        st.markdown("#### 流程画布")
        st.caption("拖拽节点调整位置｜从节点底部/顶部拖拽连线｜点击节点在右侧面板编辑")

        flow_state = _get_flow_state()
        returned_state = streamlit_flow(
            "flow_canvas",
            state=flow_state,
            height=550,
            fit_view=True,
            show_controls=True,
            show_minimap=True,
            allow_new_edges=True,
            animate_new_edges=True,
            get_node_on_click=True,
            enable_node_menu=True,
            enable_pane_menu=True,
            pan_on_drag=True,
            allow_zoom=True,
        )

        # 同步画布返回的 state：无交互时返回同一对象；有交互时返回新 state
        if returned_state is not None and returned_state is not flow_state:
            st.session_state.flow_state = returned_state
            st.session_state.flow_nodes, st.session_state.flow_edges = _state_to_nodes_edges(returned_state)
            selected_id = getattr(returned_state, "selected_id", None)
            if selected_id and selected_id != st.session_state.selected_node_id:
                st.session_state.selected_node_id = selected_id
            st.rerun()

    with col_editor:
        st.markdown("#### 节点属性")
        selected_id = st.session_state.get("selected_node_id")
        selected_node = None
        if selected_id:
            for node in st.session_state.flow_nodes:
                if node["id"] == selected_id:
                    selected_node = node
                    break

        if selected_node is None:
            st.info("在画布上点击一个节点以编辑其属性。")
            return

        nid = selected_node["id"]
        with st.form(f"edit_node_{nid}"):
            st.text_input("节点 ID", value=nid, disabled=True)
            edit_agent = st.selectbox(
                "选择 Agent *",
                agent_options,
                index=agent_options.index(selected_node["agent_id"]) if selected_node["agent_id"] in agent_options else 0,
                key=f"edit_agent_{nid}",
            )
            edit_role = st.text_input("角色描述 *", value=selected_node.get("role_description", ""), key=f"edit_role_{nid}")
            edit_task = st.text_area("任务说明 *", value=selected_node.get("task_instruction", ""), height=80, key=f"edit_task_{nid}")
            edit_tools = st.multiselect("可调用工具", tool_options, default=selected_node.get("tools", []), key=f"edit_tools_{nid}")
            edit_docs = st.multiselect("引用知识库文档", doc_options, default=selected_node.get("kb_docs", []), key=f"edit_docs_{nid}")
            edit_skills = st.multiselect("引用 Skill", skill_options, default=selected_node.get("skills", []), key=f"edit_skills_{nid}")
            edit_template = st.text_area(
                "输入模板 *",
                value=selected_node.get("input_template", "{__input__}"),
                height=100,
                key=f"edit_template_{nid}",
            )
            edit_output_key = st.text_input("输出键名 *", value=selected_node.get("output_key", ""), key=f"edit_output_key_{nid}")

            submitted = st.form_submit_button("更新节点", type="primary")
            if submitted:
                if not edit_role.strip() or not edit_task.strip() or not edit_template.strip() or not edit_output_key.strip():
                    st.error("角色描述、任务说明、输入模板和输出键名不能为空。")
                else:
                    selected_node["agent_id"] = edit_agent
                    selected_node["role_description"] = edit_role.strip()
                    selected_node["task_instruction"] = edit_task.strip()
                    selected_node["tools"] = edit_tools
                    selected_node["kb_docs"] = edit_docs
                    selected_node["skills"] = edit_skills
                    selected_node["input_template"] = edit_template.strip()
                    selected_node["output_key"] = edit_output_key.strip()
                    st.session_state.flow_state = None
                    st.success(f"节点 '{nid}' 已更新")
                    st.rerun()

        if st.button("🗑️ 删除节点", key=f"btn_del_node_{nid}"):
            _delete_node(nid)
            st.success(f"节点 '{nid}' 已删除")
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

    if not HAS_FLOW_COMPONENT:
        st.error("请先安装 streamlit-flow-component: pip install streamlit-flow-component")
        return

    root = _project_root()
    flows = flow_manager.list_flows(root)
    flow_options = {f["name"]: f["flow_id"] for f in flows}

    selected = st.session_state.get("selected_run_flow", "")
    default_name = ""
    if selected and selected in flow_options.values():
        default_name = [k for k, v in flow_options.items() if v == selected][0]

    chosen_name = st.selectbox("选择流程", list(flow_options.keys()), index=list(flow_options.values()).index(selected) if default_name else 0)
    flow_id = flow_options[chosen_name]

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

        step_results = {step["node_id"]: step for step in result.get("steps", [])}

        st.markdown("---")
        st.subheader("🗺️ 执行流程可视化")
        run_state = _nodes_edges_to_state(cfg.get("nodes", []), cfg.get("edges", []), step_results)
        returned_run_state = streamlit_flow(
            "run_flow_canvas",
            state=run_state,
            height=500,
            fit_view=True,
            show_controls=True,
            show_minimap=True,
            allow_new_edges=False,
            animate_new_edges=True,
            get_node_on_click=True,
            enable_node_menu=False,
            enable_pane_menu=False,
            pan_on_drag=True,
            allow_zoom=True,
        )
        st.caption("🟩 绿色=成功  🟥 红色=失败  ⬜ 蓝色=未执行")

        if returned_run_state:
            clicked_id = getattr(returned_run_state, "selected_id", None)
            if clicked_id and clicked_id in step_results:
                st.session_state.run_selected_node = clicked_id

        selected_run_node = st.session_state.get("run_selected_node")
        if selected_run_node and selected_run_node in step_results:
            step = step_results[selected_run_node]
            with st.expander(f"节点 {selected_run_node} 执行详情", expanded=True):
                st.markdown(f"**Agent**: {step.get('agent_id', '')}")
                st.markdown(f"**输出键**: {step.get('output_key', '')}")
                st.markdown(f"**状态**: {'成功' if step.get('success') else '失败'}")
                if step.get("error"):
                    st.error(step["error"])
                if step.get("llm_response"):
                    with st.expander("Agent 响应"):
                        st.text(step["llm_response"])
                if step.get("tool_results"):
                    with st.expander("工具执行结果"):
                        st.json(step["tool_results"])
                if step.get("output"):
                    with st.expander("最终输出"):
                        st.text(step["output"])

        with st.expander("执行详情列表", expanded=True):
            for step in result.get("steps", []):
                if step.get("success"):
                    st.markdown(f"✅ **{step['node_id']}** ({step['agent_id']}) → `{step['output_key']}`")
                else:
                    st.markdown(f"❌ **{step['node_id']}** ({step['agent_id']})")
                    st.error(step.get("error", ""))

        if result.get("final_output"):
            st.markdown("---")
            st.subheader("🏁 最终输出")
            st.markdown(result["final_output"])


def page_agent_flow_platform():
    st.title("🎨 Agent 流程画布")
    st.markdown("画布式编排多智能体流程：拖拽节点、连接边、点击编辑、运行查看完整执行流。")

    _init_session()

    tabs = st.tabs(["🎨 流程编辑", "📋 流程列表", "▶️ 运行流程"])
    with tabs[0]:
        _tab_editor()
    with tabs[1]:
        _tab_list()
    with tabs[2]:
        _tab_run()
