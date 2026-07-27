"""二次开发平台前端：支持用户自主新增工具、Agent 辅助开发、格式转换与注册表管理。"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

# 注意：app.py 已经把 project_root/src 和 project_root/extensions 加入 sys.path
from dev_platform import registry_manager, tool_generator, tool_validator, template
from common.config_loader import get_project_root, load_json, save_json


def _project_root() -> Path:
    return get_project_root()


def _set_state(key: str, value):
    """安全地设置 session_state，避免与已实例化的 widget key 冲突。

    Streamlit 不允许在 widget 实例化后修改其绑定的 session_state key。
    通过先删除 key 再赋值，并在随后调用 st.rerun()，可在下一次渲染时让 widget 读取新值。
    """
    if key in st.session_state:
        del st.session_state[key]
    st.session_state[key] = value


def _init_session():
    keys = {
        "dev_tool_name": "",
        "dev_description": "",
        "dev_requirement": "",
        "dev_python_code": "",
        "dev_config_json": "",
        "dev_module_name": "",
        "dev_function_name": "run",
        "convert_tool_name": "",
        "convert_description": "",
        "convert_source_code": "",
        "convert_python_code": "",
        "convert_config_json": "",
        "test_config_json": "",
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _module_file_for(module_name: str, project_root: Path) -> Path:
    """根据模块名确定应保存的 .py 文件路径。"""
    if module_name.startswith("extensions.tools."):
        module_name = module_name.split(".", 2)[2]
    elif module_name.startswith("extensions."):
        module_name = module_name.split(".", 1)[1]
    return project_root / "extensions" / "tools" / f"{module_name}.py"


def _config_file_for(tool_name: str, project_root: Path) -> Path:
    return project_root / "extensions" / "configs" / f"{tool_name}.json"


def _normalize_module_name(module_name: str) -> str:
    """用户工具模块统一以 extensions.tools. 开头。"""
    module_name = module_name.strip()
    if module_name.startswith("extensions.tools."):
        return module_name
    if module_name.startswith("extensions."):
        return f"extensions.tools.{module_name.split('.', 1)[1]}"
    return f"extensions.tools.{module_name}"


def _save_tool(tool_name: str, description: str, module_name: str,
               function_name: str, python_code: str, config_template: dict) -> dict:
    """保存工具源码、配置文件并注册到 MCP。"""
    root = _project_root()
    module_name = _normalize_module_name(module_name)
    module_file = _module_file_for(module_name, root)
    config_file = _config_file_for(tool_name, root)

    module_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    module_file.write_text(python_code, encoding="utf-8")
    config_file.write_text(json.dumps(config_template, indent=2, ensure_ascii=False), encoding="utf-8")

    default_config_rel = str(config_file.relative_to(root))
    return registry_manager.register_tool(
        name=tool_name,
        module=module_name,
        function=function_name,
        default_config=default_config_rel,
        description=description,
        source="user",
        project_root=root,
    )


def _tab_create_tool():
    st.subheader("🛠️ 新建工具")
    st.markdown("输入自然语言需求，Agent 会生成符合 Gaia 规范的工具代码；你也可以直接编辑后保存。")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("工具名称（英文小写+下划线）", key="dev_tool_name")
    with col2:
        st.text_input("工具描述", key="dev_description")

    st.text_area("需求描述", height=120, key="dev_requirement",
                 placeholder="例如：计算每口井的自然伽马均值，并输出到 CSV 文件。")

    c1, c2 = st.columns([1, 4])
    with c1:
        generate_clicked = st.button("🤖 Agent 生成代码", type="primary")
    with c2:
        if st.button("📝 使用空白模板"):
            name = st.session_state.dev_tool_name.strip() or "custom_tool"
            desc = st.session_state.dev_description.strip() or f"用户自定义工具: {name}"
            _set_state("dev_python_code", template.render_tool_code(name, desc))
            cfg = template.make_default_config_template(name, desc)
            _set_state("dev_config_json", json.dumps(cfg, indent=2, ensure_ascii=False))
            _set_state("dev_module_name", name)
            _set_state("dev_function_name", "run")
            st.rerun()

    if generate_clicked:
        if not st.session_state.dev_requirement.strip():
            st.warning("请先填写需求描述。")
        else:
            with st.spinner("Agent 正在生成工具代码，请稍候..."):
                result = tool_generator.generate_tool_from_requirement(
                    requirement=st.session_state.dev_requirement,
                    tool_name=st.session_state.dev_tool_name or None,
                    description=st.session_state.dev_description or None,
                    project_root=_project_root(),
                )
            if result["success"]:
                data = result["data"]
                _set_state("dev_python_code", data.get("python_code", ""))
                _set_state("dev_config_json", json.dumps(
                    data.get("config_template", {}), indent=2, ensure_ascii=False
                ))
                _set_state("dev_tool_name", data.get("tool_name", st.session_state.dev_tool_name))
                _set_state("dev_module_name", data.get("module_name", st.session_state.dev_tool_name))
                _set_state("dev_function_name", data.get("function_name", "run"))
                _set_state("dev_description", data.get("description", st.session_state.dev_description))
                st.success("代码生成完成，请在下方检查并编辑。")
                st.rerun()
            else:
                st.error(result["error"])
                if result["raw"]:
                    st.code(result["raw"], language="text")

    st.markdown("---")
    st.markdown("**生成/编辑区**")
    st.text_area("Python 源码", height=350, key="dev_python_code")
    st.text_area("配置文件模板 (JSON)", height=180, key="dev_config_json")

    add_pipe = st.checkbox("保存时追加到流程法（默认禁用）", key="dev_add_pipeline")

    col_save, col_validate = st.columns([1, 1])
    with col_validate:
        if st.button("🔍 仅校验"):
            code = st.session_state.dev_python_code
            if not code.strip():
                st.warning("源码为空。")
            else:
                res = tool_validator.validate_tool(code, st.session_state.dev_function_name)
                if res["valid"]:
                    st.success(res["message"])
                else:
                    st.error(res["message"])
                    for d in res.get("details", []):
                        if not d["valid"]:
                            st.write(f"- ❌ {d['message']}")

    with col_save:
        if st.button("💾 校验并保存", type="primary"):
            code = st.session_state.dev_python_code
            tool_name = st.session_state.dev_tool_name.strip()
            if not tool_name:
                st.error("请填写工具名称。")
                return
            val = tool_validator.validate_tool(code, st.session_state.dev_function_name)
            if not val["valid"]:
                st.error(val["message"])
                for d in val.get("details", []):
                    if not d["valid"]:
                        st.write(f"- ❌ {d['message']}")
                return
            try:
                cfg = json.loads(st.session_state.dev_config_json)
            except json.JSONDecodeError as e:
                st.error(f"配置文件 JSON 格式错误: {e}")
                return

            module_name = st.session_state.dev_module_name.strip() or tool_name
            function_name = st.session_state.dev_function_name.strip() or "run"
            with st.spinner("正在保存并注册工具..."):
                res = _save_tool(tool_name, st.session_state.dev_description.strip(),
                                 module_name, function_name, code, cfg)
            if res["success"]:
                st.success(res["message"])
                norm_module = _normalize_module_name(module_name)
                short_module = norm_module.rsplit(".", 1)[1]
                st.info(f"模块文件: extensions/tools/{short_module}.py")
                st.info(f"配置文件: extensions/configs/{tool_name}.json")
                if add_pipe:
                    pipe_res = registry_manager.add_pipeline_step(
                        step=tool_name,
                        tool=norm_module,
                        function=function_name,
                        config=str(_config_file_for(tool_name, _project_root()).relative_to(_project_root())),
                        enabled=False,
                        project_root=_project_root(),
                    )
                    st.success(pipe_res["message"])
            else:
                st.error(res["message"])


def _tab_convert_tool():
    st.subheader("🔄 格式转换")
    st.markdown("粘贴现有 Python 代码或上传 `.py` 文件，Agent 会将其转换为 Gaia 工具格式。")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("目标工具名称", key="convert_tool_name")
    with col2:
        st.text_input("目标工具描述", key="convert_description")

    uploaded = st.file_uploader("上传 Python 文件（可选）", type=["py"])
    if uploaded is not None:
        _set_state("convert_source_code", uploaded.read().decode("utf-8", errors="replace"))

    st.text_area("原始 Python 代码", height=250, key="convert_source_code",
                 placeholder="在此处粘贴原始函数或脚本...")

    if st.button("🤖 转换为 Gaia 工具"):
        name = st.session_state.convert_tool_name.strip()
        desc = st.session_state.convert_description.strip()
        code = st.session_state.convert_source_code.strip()
        if not name or not desc or not code:
            st.warning("请填写工具名称、描述和原始代码。")
        else:
            with st.spinner("Agent 正在转换代码..."):
                result = tool_generator.convert_code_to_tool(
                    source_code=code,
                    tool_name=name,
                    description=desc,
                    project_root=_project_root(),
                )
            if result["success"]:
                data = result["data"]
                _set_state("convert_python_code", data.get("python_code", ""))
                _set_state("convert_config_json", json.dumps(
                    data.get("config_template", {}), indent=2, ensure_ascii=False
                ))
                st.success("转换完成，请在下方检查并编辑。")
                st.rerun()
            else:
                st.error(result["error"])

    st.markdown("---")
    st.text_area("转换后的 Python 源码", height=350, key="convert_python_code")
    st.text_area("转换后的配置文件 (JSON)", height=180, key="convert_config_json")

    if st.button("💾 保存转换结果", type="primary"):
        name = st.session_state.convert_tool_name.strip()
        code = st.session_state.convert_python_code
        if not name:
            st.error("请填写工具名称。")
            return
        val = tool_validator.validate_tool(code, "run")
        if not val["valid"]:
            st.error(val["message"])
            return
        try:
            cfg = json.loads(st.session_state.convert_config_json)
        except json.JSONDecodeError as e:
            st.error(f"配置文件 JSON 格式错误: {e}")
            return
        with st.spinner("正在保存..."):
            res = _save_tool(name, st.session_state.convert_description.strip(),
                             name, "run", code, cfg)
        if res["success"]:
            st.success(res["message"])
        else:
            st.error(res["message"])


def _tab_tool_management():
    st.subheader("📋 工具管理")
    st.markdown("查看、测试、编辑或删除已注册工具。内置工具只读，用户工具可管理。")

    tools = registry_manager.list_tools(project_root=_project_root())
    if not tools:
        st.info("暂无已注册工具。")
        return

    for name, info in tools.items():
        source = info.get("source", "builtin")
        is_builtin = source == "builtin"
        color = "🔵" if is_builtin else "🟢"
        with st.expander(f"{color} {name}  —  {info.get('description', '')}  (`{source}`)"):
            st.markdown(f"**模块**: `{info.get('module')}`")
            st.markdown(f"**函数**: `{info.get('function')}`")
            st.markdown(f"**默认配置**: `{info.get('default_config')}`")

            # 源码查看
            module_name = info.get("module", "")
            if module_name.startswith("extensions."):
                module_path = _project_root() / "extensions" / "tools" / f"{module_name.split('.', 1)[1]}.py"
            else:
                module_path = _project_root() / "src" / "tools" / f"{module_name}.py"

            if module_path.exists():
                src_code = module_path.read_text(encoding="utf-8")
                st.code(src_code, language="python")
            else:
                st.warning(f"源码文件不存在: {module_path}")

            # 用户工具操作
            if not is_builtin:
                st.markdown("---")
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    if st.button(f"🗑️ 删除 {name}", key=f"del_{name}"):
                        res = registry_manager.unregister_tool(name, remove_files=True, project_root=_project_root())
                        if res["success"]:
                            st.success(res["message"])
                            st.rerun()
                        else:
                            st.error(res["message"])

                with c2:
                    test_key = f"test_cfg_{name}"
                    if test_key not in st.session_state:
                        cfg_path = _project_root() / info.get("default_config", "")
                        if cfg_path.exists():
                            st.session_state[test_key] = cfg_path.read_text(encoding="utf-8")
                        else:
                            st.session_state[test_key] = json.dumps({
                                "paths": {"output_directory": f"data/processed_data/{name}"},
                                "parameters": {}
                            }, indent=2, ensure_ascii=False)
                    if st.button(f"▶️ 测试运行 {name}", key=f"run_{name}"):
                        try:
                            cfg = json.loads(st.session_state[test_key])
                        except json.JSONDecodeError as e:
                            st.error(f"测试配置 JSON 错误: {e}")
                            cfg = None
                        if cfg is not None:
                            with st.spinner("正在子进程中安全运行..."):
                                run_res = tool_validator.sandbox_run(
                                    module_file=module_path,
                                    function_name=info.get("function", "run"),
                                    config=cfg,
                                    project_root=_project_root(),
                                    timeout=60,
                                )
                            if run_res["success"]:
                                st.success("测试运行成功")
                                st.json(run_res["result"])
                            else:
                                st.error(run_res["error"])

                with st.expander(f"编辑测试配置: {name}", expanded=False):
                    st.text_area("测试配置 JSON", key=test_key, height=200)


def _tab_registry_management():
    st.subheader("⚙️ 注册表管理")
    st.markdown("直接编辑 `ctl/mcp_config.json` 与 `ctl/pipeline_config.json`（请谨慎操作）。")

    root = _project_root()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MCP 注册表**")
        mcp_cfg = load_json("ctl/mcp_config.json", root)
        mcp_text = st.text_area("mcp_config.json", value=json.dumps(mcp_cfg, indent=2, ensure_ascii=False),
                                height=400, key="mcp_cfg_text")
    with col2:
        st.markdown("**流程法配置**")
        pipe_cfg = load_json("ctl/pipeline_config.json", root)
        pipe_text = st.text_area("pipeline_config.json", value=json.dumps(pipe_cfg, indent=2, ensure_ascii=False),
                                 height=400, key="pipe_cfg_text")

    if st.button("💾 保存注册表修改", type="primary"):
        try:
            mcp_new = json.loads(mcp_text)
            pipe_new = json.loads(pipe_text)
        except json.JSONDecodeError as e:
            st.error(f"JSON 格式错误: {e}")
            return
        save_json(mcp_new, "ctl/mcp_config.json", root)
        save_json(pipe_new, "ctl/pipeline_config.json", root)
        st.success("注册表已保存。")


def page_dev_platform():
    """二次开发平台主页面。"""
    st.title("🔧 二次开发平台")
    st.markdown("自主扩展工具包，或由智能体辅助生成/转换工具。")

    _init_session()

    tabs = st.tabs(["🛠️ 新建工具", "🔄 格式转换", "📋 工具管理", "⚙️ 注册表管理"])
    with tabs[0]:
        _tab_create_tool()
    with tabs[1]:
        _tab_convert_tool()
    with tabs[2]:
        _tab_tool_management()
    with tabs[3]:
        _tab_registry_management()
