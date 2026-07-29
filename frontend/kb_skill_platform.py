"""知识库文档与 Skill 管理前端页面。"""

import json
from pathlib import Path

import streamlit as st

from common.config_loader import get_project_root
from dev_platform import kb_manager, skill_manager


def _project_root() -> Path:
    return get_project_root()


def _init_session():
    keys = {
        "kb_doc_name": "",
        "kb_doc_content": "",
        "skill_name": "",
        "skill_title": "",
        "skill_description": "",
        "skill_content": "",
        "skill_edit_mode": False,
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_skill_into_form(name: str):
    root = _project_root()
    data = skill_manager.load_skill(name, root)
    st.session_state.skill_name = data.get("name", name)
    st.session_state.skill_title = data.get("title", "")
    st.session_state.skill_description = data.get("description", "")
    st.session_state.skill_content = data.get("content", "")
    st.session_state.skill_edit_mode = True


def _clear_skill_form():
    for k in ["skill_name", "skill_title", "skill_description", "skill_content", "skill_edit_mode"]:
        if k in st.session_state:
            del st.session_state[k]
    _init_session()


def _tab_knowledge_base():
    st.subheader("📚 知识库文档")
    st.markdown("上传或粘贴文本，作为 Agent RAG 检索的素材。")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("上传 .txt 文档", type=["txt"], key="kb_uploader")
    with col2:
        doc_name = st.text_input(
            "文档名（不含扩展名）",
            value=st.session_state.kb_doc_name,
            key="kb_doc_name",
            help="仅支持字母、数字、中文、下划线和横线",
        )

    doc_content = st.text_area(
        "文档内容",
        value=st.session_state.kb_doc_content,
        key="kb_doc_content",
        height=250,
        placeholder="在此粘贴知识库文本内容，或先上传文件再修改...",
    )

    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="replace")
        st.session_state.kb_doc_content = content
        if not st.session_state.kb_doc_name:
            st.session_state.kb_doc_name = Path(uploaded.name).stem
        st.rerun()

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("💾 保存文档", type="primary", key="kb_save"):
            if not doc_name.strip():
                st.error("请填写文档名。")
                return
            if not doc_content.strip():
                st.error("文档内容不能为空。")
                return
            res = kb_manager.save_document(doc_name.strip(), doc_content.strip(), _project_root())
            if res["success"]:
                st.success(res["message"])
                st.session_state.kb_doc_name = ""
                st.session_state.kb_doc_content = ""
                st.rerun()
            else:
                st.error(res["message"])
    with c2:
        if st.button("🔄 清空", key="kb_clear"):
            st.session_state.kb_doc_name = ""
            st.session_state.kb_doc_content = ""
            st.rerun()

    st.divider()
    st.subheader("已上传文档")
    docs = kb_manager.list_documents(_project_root())
    if not docs:
        st.info("暂无知识库文档。")
        return

    for doc in docs:
        with st.expander(f"📄 {doc['name']}  ({doc['word_count']} 字)"):
            st.caption(f"文件：{doc['path']}")
            st.text(doc["summary"])
            c_view, c_del = st.columns([1, 1])
            with c_view:
                doc_path = _project_root() / doc["path"]
                st.download_button(
                    "⬇️ 下载",
                    data=doc_path.read_text(encoding="utf-8"),
                    file_name=doc["filename"],
                    key=f"kb_dl_{doc['name']}",
                )
            with c_del:
                if st.button("🗑️ 删除", key=f"kb_del_{doc['name']}"):
                    res = kb_manager.delete_document(doc["name"], _project_root())
                    if res["success"]:
                        st.success(res["message"])
                        st.rerun()
                    else:
                        st.error(res["message"])


def _tab_skills():
    st.subheader("🛠️ Skill 管理")
    st.markdown("编写可复用的提示片段或能力说明；启用 RAG 的 Agent 会自动检索相关 Skill。")

    is_edit = st.session_state.get("skill_edit_mode", False)
    if is_edit:
        st.info(f"正在编辑 Skill：`{st.session_state.skill_name}`")

    with st.form("skill_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            skill_name = st.text_input(
                "Skill 标识名 *",
                value=st.session_state.skill_name,
                key="skill_name",
                help="唯一标识，例如 shale_expert",
                disabled=is_edit,
            )
        with col2:
            skill_title = st.text_input(
                "显示标题 *",
                value=st.session_state.skill_title,
                key="skill_title",
                help="给人类看的标题",
            )

        skill_description = st.text_input(
            "简短描述",
            value=st.session_state.skill_description,
            key="skill_description",
            help="说明该 Skill 的用途",
        )
        skill_content = st.text_area(
            "Skill 内容 *",
            value=st.session_state.skill_content,
            key="skill_content",
            height=250,
            help="具体的提示词、能力说明或模板",
        )

        c_save, c_clear = st.columns([1, 1])
        with c_save:
            submitted = st.form_submit_button("💾 保存 Skill", type="primary")
        with c_clear:
            clear_clicked = st.form_submit_button("🔄 清空")

        if submitted:
            if not skill_name.strip() or not skill_title.strip() or not skill_content.strip():
                st.error("Skill 标识名、标题、内容不能为空。")
                return
            res = skill_manager.save_skill(
                name=skill_name.strip(),
                title=skill_title.strip(),
                description=skill_description.strip(),
                content=skill_content.strip(),
                project_root=_project_root(),
            )
            if res["success"]:
                st.success(res["message"])
                _clear_skill_form()
                st.rerun()
            else:
                st.error(res["message"])

        if clear_clicked:
            _clear_skill_form()
            st.rerun()

    st.divider()
    st.subheader("已创建 Skill")
    skills = skill_manager.list_skills(_project_root())
    if not skills:
        st.info("暂无 Skill。")
        return

    for sk in skills:
        with st.expander(f"🛠️ {sk['title']}  (`{sk['name']}`)"):
            st.caption(f"文件：{sk['path']}")
            st.markdown(f"**描述**：{sk['description'] or '无'}")
            c_edit, c_del = st.columns([1, 1])
            with c_edit:
                if st.button("✏️ 编辑", key=f"skill_edit_{sk['name']}"):
                    _load_skill_into_form(sk["name"])
                    st.rerun()
            with c_del:
                if st.button("🗑️ 删除", key=f"skill_del_{sk['name']}"):
                    res = skill_manager.delete_skill(sk["name"], _project_root())
                    if res["success"]:
                        st.success(res["message"])
                        st.rerun()
                    else:
                        st.error(res["message"])


def page_kb_skill_platform():
    st.title("📚 知识库 & Skill")
    st.markdown("管理知识库文档与可复用 Skill；启用 RAG 的 Agent 会自动引用相关内容。")

    _init_session()

    tabs = st.tabs(["📚 知识库文档", "🛠️ Skill 管理"])
    with tabs[0]:
        _tab_knowledge_base()
    with tabs[1]:
        _tab_skills()
