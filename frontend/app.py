"""Gaia Streamlit 前端：自然语言指令 + 一键运行 + MCP 工具知识图谱。"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

import streamlit as st

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "extensions"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from frontend.commands import parse_command
from frontend.knowledge_graph import (
    load_tool_info,
    render_group_legend,
    render_stage_legend,
)
from frontend.kg_plotly_view import build_plotly_graph
from frontend.kg_static_view import build_static_graph
from frontend.dev_platform import page_dev_platform


def run_subprocess_stream(cmd: list, cwd: Path):
    """运行子进程并实时返回输出行。"""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    for line in process.stdout:
        yield line
    process.wait()
    yield f"\n[退出码: {process.returncode}]\n"


def list_output_files(output_dir: Path, extensions=(".csv", ".json", ".txt")) -> list:
    """列出输出目录下指定扩展名的文件。"""
    files = []
    if output_dir.exists():
        for ext in extensions:
            files.extend(sorted(output_dir.rglob(f"*{ext}")))
    return files


def page_run():
    """运行控制台页面。"""
    st.title("🌍 Gaia 测井解释智能平台")
    st.markdown("基于大语言模型 + MCP 的测井解释重构项目。支持**流程法**与**MCP方法**两种运行方式。")

    col1, col2 = st.columns([1, 2])
    with col1:
        method = st.radio("选择运行模式", ["流程法", "MCP方法"], index=0)
        use_llm = st.checkbox(
            "🧠 使用 LLM 智能选工具（实验）",
            value=True,
            help="勾选后，MCP 方法会先让 LLM Agent 根据工具描述选择工具；失败时自动回退到关键词匹配。",
            key="use_llm_tool_selector"
        ) if method == "MCP方法" else False
        user_input = st.text_area(
            "自然语言指令",
            value="运行全流程" if method == "流程法" else "调用全部工具",
            height=100,
            placeholder="例如：运行全流程 / 只计算泥质含量和孔隙度 / 调用渗透率工具 / 计算GR均值"
        )
        parsed = parse_command(method, user_input, use_llm=use_llm, project_root=PROJECT_ROOT)
        st.info(f"**解析结果**：{parsed['action']}")
        if parsed.get("reasoning"):
            st.caption(f"💡 {parsed['reasoning']}")
        if parsed.get("llm_error"):
            st.warning(f"LLM 提示：{parsed['llm_error']}")

        run_clicked = st.button("🚀 一键运行", type="primary")

    with col2:
        log_area = st.empty()
        logs = []

        if run_clicked:
            with st.spinner("正在执行，请稍候..."):
                if parsed["mode"] == "flow":
                    cmd = [sys.executable, str(PROJECT_ROOT / "run_flow.py")]
                    if parsed.get("stop_at"):
                        cmd.extend(["--stop-at", parsed["stop_at"]])
                else:
                    # 生成临时 MCP 计划文件；优先使用 LLM 返回的 plan（可能含 overrides）
                    plan = parsed.get("plan") or [{"tool": t} for t in parsed["tools"]]
                    fd, plan_path = tempfile.mkstemp(suffix=".json", prefix="gaia_mcp_plan_")
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(plan, f, ensure_ascii=False, indent=2)
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "src" / "methods" / "mcp_orchestrator.py"),
                        "--plan", plan_path
                    ]

                for line in run_subprocess_stream(cmd, PROJECT_ROOT):
                    logs.append(line)
                    log_area.code("".join(logs[-200:]), language="log")

                # 清理临时计划文件
                if parsed["mode"] == "mcp" and os.path.exists(plan_path):
                    try:
                        os.remove(plan_path)
                    except Exception:
                        pass

            st.success("执行完成！")

    # 结果展示区
    st.divider()
    st.subheader("📁 输出成果")
    output_dir = PROJECT_ROOT / "data" / "processed_data"
    files = list_output_files(output_dir)
    if files:
        selected = st.selectbox("选择文件预览", [str(f.relative_to(PROJECT_ROOT)) for f in files])
        selected_path = PROJECT_ROOT / selected
        try:
            if selected.endswith(".csv"):
                import pandas as pd
                df = pd.read_csv(selected_path, low_memory=False)
                st.write(f"**{selected}** ({len(df)} 行)")
                st.dataframe(df.head(50), use_container_width=True)
            elif selected.endswith(".json"):
                with open(selected_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.json(data)
            else:
                with open(selected_path, "r", encoding="utf-8") as f:
                    st.text(f.read())
        except Exception as e:
            st.error(f"预览失败: {e}")
    else:
        st.info("暂无输出文件，请运行流程或工具。")


def page_knowledge_graph():
    """知识图谱页面：以 Plotly 交互图为主视图，支持点击节点查看源码。"""
    st.title("🕸️ MCP 工具知识图谱")
    st.markdown("**点击图谱中任意工具节点**，即可在下方查看该工具的完整源码；左侧列表也支持直接搜索与浏览。")
    st.markdown(f"{render_stage_legend()}  &nbsp;&nbsp;|&nbsp;&nbsp;  {render_group_legend()}", unsafe_allow_html=True)

    try:
        tool_info = load_tool_info(PROJECT_ROOT)
    except Exception as e:
        st.error(f"加载工具信息失败: {e}")
        return

    # 初始化选中的工具
    if "kg_selected_tool" not in st.session_state:
        st.session_state.kg_selected_tool = None

    # 顶部搜索栏
    search = st.text_input("🔍 搜索工具名称或描述", placeholder="例如：shale、孔隙度、permeability", value="")
    search_lower = search.strip().lower()

    filtered = {
        name: info for name, info in tool_info.items()
        if not search_lower
        or search_lower in name.lower()
        or search_lower in info["description"].lower()
        or search_lower in info["module"].lower()
        or search_lower in info["group"].lower()
        or search_lower in info["stage"].lower()
    }

    # 左侧边栏：工具列表与元信息
    with st.sidebar:
        st.subheader("📋 工具列表")
        if not filtered:
            st.warning("未找到匹配工具。")
        else:
            default_index = 0
            if st.session_state.kg_selected_tool in filtered:
                default_index = list(filtered.keys()).index(st.session_state.kg_selected_tool)

            options = [name for name, info in filtered.items()]
            selected_label = st.selectbox("选择工具", options, index=default_index)
            st.session_state.kg_selected_tool = selected_label

            info = filtered[selected_label]
            st.markdown("---")
            st.markdown(f"**名称**：`{info['name']}`")
            st.markdown(f"**阶段**：`{info['stage']}`")
            st.markdown(f"**功能组**：`{info['group']}`")
            st.markdown(f"**模块**：`{info['source_path']}`")
            st.markdown(f"**配置**：`{info['default_config']}`")
            st.markdown(f"**描述**：{info['description']}")

    # 主区域：Plotly 交互图谱
    highlighted = list(filtered.keys()) if filtered else None
    try:
        plotly_fig = build_plotly_graph(PROJECT_ROOT, highlight_tools=highlighted)
        selection = st.plotly_chart(
            plotly_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="kg_plotly",
            config={"displayModeBar": True},
        )
        if selection:
            sel_obj = getattr(selection, "selection", selection)
            points = sel_obj.get("points", []) if hasattr(sel_obj, "get") else getattr(sel_obj, "points", [])
            if points:
                point = points[0]
                customdata = point.get("customdata", [None]) if hasattr(point, "get") else getattr(point, "customdata", [None])
                clicked = customdata[0] if isinstance(customdata, (list, tuple)) else customdata
                if clicked and clicked != st.session_state.kg_selected_tool:
                    st.session_state.kg_selected_tool = clicked
                    st.rerun()
    except Exception as e:
        st.error(f"知识图谱生成失败: {e}")

    with st.expander("🖨️ 高清静态预览（导出/打印推荐）", expanded=False):
        st.caption("下方为 matplotlib 渲染的静态图，适合导出/打印。")
        try:
            fig = build_static_graph(PROJECT_ROOT, highlight_tools=highlighted)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"静态预览生成失败: {e}")

    # 图下方完整源码区
    selected_tool = st.session_state.kg_selected_tool
    if selected_tool and selected_tool in tool_info:
        st.divider()
        st.subheader(f"📄 {selected_tool} 源码")
        st.caption(f"文件：{tool_info[selected_tool]['source_path']}  |  模块：{tool_info[selected_tool]['module']}")
        st.code(tool_info[selected_tool]["source_code"], language="python")


def main():
    st.set_page_config(page_title="Gaia 测井解释平台", layout="wide")
    page = st.sidebar.radio("导航", ["运行控制台", "知识图谱", "二次开发平台"])
    if page == "运行控制台":
        page_run()
    elif page == "知识图谱":
        page_knowledge_graph()
    else:
        page_dev_platform()


if __name__ == "__main__":
    main()
