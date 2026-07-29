"""智能体注册与管理前端页面。"""

import json
from pathlib import Path

import streamlit as st

from common.config_loader import get_project_root
from dev_platform import agent_manager


def _project_root() -> Path:
    return get_project_root()


def _init_session():
    """初始化页面状态。"""
    keys = {
        "agent_id": "",
        "agent_model": "gpt-4o-mini",
        "agent_model_type": "api",
        "agent_api_key": "",
        "agent_base_url": "https://api.zhizengzeng.com/v1",
        "agent_model_path": "",
        "agent_torch_dtype": "bfloat16",
        "agent_max_new_tokens": 2048,
        "agent_temperature": 0.2,
        "agent_top_p": 0.9,
        "agent_max_length": 8192,
        "agent_system_message": "",
        "agent_rag_enable": False,
        "agent_edit_mode": False,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_agent_into_form(agent_id: str):
    """将现有 Agent 配置加载到表单中。"""
    root = _project_root()
    cfg = agent_manager.load_agent_config(agent_id, root)
    ts = cfg.get("training_setting", {})
    st.session_state.agent_id = cfg.get("agent_id", agent_id)
    st.session_state.agent_model = cfg.get("model", "gpt-4o-mini")
    st.session_state.agent_model_type = ts.get("model_type", "api")
    st.session_state.agent_api_key = ts.get("api_key", "")
    st.session_state.agent_base_url = ts.get("base_url", "https://api.zhizengzeng.com/v1")
    st.session_state.agent_model_path = ts.get("model_path", "") or ""
    st.session_state.agent_torch_dtype = ts.get("torch_dtype", "bfloat16")
    st.session_state.agent_max_new_tokens = ts.get("max_new_tokens", 2048)
    st.session_state.agent_temperature = ts.get("temperature", 0.2)
    st.session_state.agent_top_p = ts.get("top_p", 0.9)
    st.session_state.agent_max_length = ts.get("max_length", 8192)
    st.session_state.agent_system_message = cfg.get("system_message", "")
    st.session_state.agent_rag_enable = cfg.get("rag", {}).get("enable", False)
    st.session_state.agent_edit_mode = True


def _clear_form():
    """清空表单到初始状态。"""
    for k in [
        "agent_id", "agent_model", "agent_model_type", "agent_api_key",
        "agent_base_url", "agent_model_path", "agent_torch_dtype",
        "agent_max_new_tokens", "agent_temperature", "agent_top_p",
        "agent_max_length", "agent_system_message", "agent_rag_enable",
        "agent_edit_mode",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    _init_session()


def _tab_register():
    """注册/编辑 Agent 表单。"""
    st.subheader("📝 注册 / 编辑 Agent")

    is_edit = st.session_state.get("agent_edit_mode", False)
    if is_edit:
        st.info(f"正在编辑 Agent：`{st.session_state.agent_id}`")

    with st.form("agent_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            agent_id = st.text_input(
                "Agent ID *",
                value=st.session_state.agent_id,
                key="agent_id",
                help="唯一标识，仅字母、数字、下划线，将作为配置文件名",
                disabled=is_edit,
            )
            model = st.text_input(
                "模型名称 *",
                value=st.session_state.agent_model,
                key="agent_model",
                help="例如 gpt-4o-mini、qwen2-7b 等",
            )
            model_type = st.selectbox(
                "模型类型 *",
                ["api", "local"],
                index=0 if st.session_state.agent_model_type == "api" else 1,
                key="agent_model_type",
                help="API 模式调用 OpenAI 兼容接口；local 模式加载本地模型",
            )
        with col2:
            api_key = st.text_input(
                "API Key",
                value=st.session_state.agent_api_key,
                key="agent_api_key",
                type="password",
                help="API 模式下必填",
            )
            base_url = st.text_input(
                "API 基础 URL / 网页 *",
                value=st.session_state.agent_base_url,
                key="agent_base_url",
                help="OpenAI 兼容接口地址，例如 https://api.zhizengzeng.com/v1",
            )
            model_path = st.text_input(
                "本地模型路径",
                value=st.session_state.agent_model_path,
                key="agent_model_path",
                help="本地模型模式下必填，例如 /path/to/model",
            )

        st.markdown("---")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            torch_dtype = st.selectbox(
                "torch dtype",
                ["bfloat16", "float16", "float32"],
                index=["bfloat16", "float16", "float32"].index(st.session_state.agent_torch_dtype),
                key="agent_torch_dtype",
            )
        with col4:
            max_new_tokens = st.number_input(
                "max_new_tokens",
                min_value=1,
                max_value=8192,
                value=int(st.session_state.agent_max_new_tokens),
                key="agent_max_new_tokens",
            )
        with col5:
            temperature = st.slider(
                "temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.agent_temperature),
                step=0.1,
                key="agent_temperature",
            )
        with col6:
            top_p = st.slider(
                "top_p",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.agent_top_p),
                step=0.1,
                key="agent_top_p",
            )

        col7, col8 = st.columns(2)
        with col7:
            max_length = st.number_input(
                "max_length",
                min_value=1,
                max_value=32768,
                value=int(st.session_state.agent_max_length),
                key="agent_max_length",
            )
        with col8:
            rag_enable = st.checkbox(
                "启用 RAG",
                value=st.session_state.agent_rag_enable,
                key="agent_rag_enable",
            )

        system_message = st.text_area(
            "System Message *",
            value=st.session_state.agent_system_message,
            key="agent_system_message",
            height=200,
            help="Agent 的系统提示词",
        )

        col_save, col_clear = st.columns([1, 1])
        with col_save:
            submitted = st.form_submit_button("💾 保存 Agent", type="primary")
        with col_clear:
            clear_clicked = st.form_submit_button("🔄 清空表单")

        if submitted:
            if not agent_id.strip() or not model.strip() or not system_message.strip():
                st.error("Agent ID、模型名称、System Message 不能为空。")
                return
            with st.spinner("正在保存..."):
                res = agent_manager.save_agent_config(
                    agent_id=agent_id.strip(),
                    model=model.strip(),
                    model_type=model_type,
                    api_key=api_key,
                    base_url=base_url.strip(),
                    model_path=model_path.strip(),
                    torch_dtype=torch_dtype,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_length=max_length,
                    system_message=system_message.strip(),
                    rag_enable=rag_enable,
                    project_root=_project_root(),
                )
            if res["success"]:
                st.success(res["message"])
                _clear_form()
                st.rerun()
            else:
                st.error(res["message"])

        if clear_clicked:
            _clear_form()
            st.rerun()


def _tab_manage():
    """管理已注册 Agent。"""
    st.subheader("📋 Agent 管理")

    agents = agent_manager.list_agents(_project_root())
    if not agents:
        st.info("暂无已注册 Agent。")
        return

    for info in agents:
        with st.expander(f"🤖 {info['agent_id']}  —  `{info['model']}` ({info['model_type']})"):
            st.markdown(f"**配置文件**: `{info['file']}`")
            st.markdown(f"**base_url**: `{info.get('base_url', '') or '默认'}`")
            st.markdown(f"**描述**: {info['description']}")

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("✏️ 编辑", key=f"edit_agent_{info['agent_id']}"):
                    _load_agent_into_form(info["agent_id"])
                    st.rerun()
            with c2:
                if st.button("🗑️ 删除", key=f"del_agent_{info['agent_id']}"):
                    res = agent_manager.delete_agent(info["agent_id"], _project_root())
                    if res["success"]:
                        st.success(res["message"])
                        st.rerun()
                    else:
                        st.error(res["message"])


def page_agent_platform():
    """Agent 注册与管理主页面。"""
    st.title("🤖 智能体注册平台")
    st.markdown("在此注册、编辑和管理 Gaia 平台的 LLM Agent。")

    _init_session()

    tabs = st.tabs(["📝 注册 / 编辑", "📋 Agent 管理"])
    with tabs[0]:
        _tab_register()
    with tabs[1]:
        _tab_manage()
