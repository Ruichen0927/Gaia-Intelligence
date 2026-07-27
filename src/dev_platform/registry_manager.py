"""MCP / Pipeline 注册表管理：支持用户自主增删工具。"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import load_json, save_json, get_project_root


def _get_mcp_path(project_root: Path) -> Path:
    return project_root / "ctl" / "mcp_config.json"


def _get_pipeline_path(project_root: Path) -> Path:
    return project_root / "ctl" / "pipeline_config.json"


def load_mcp_config(project_root: Path = None) -> Dict[str, Any]:
    """加载 MCP 注册表。"""
    root = project_root or get_project_root()
    return load_json("ctl/mcp_config.json", root)


def save_mcp_config(config: Dict[str, Any], project_root: Path = None) -> Path:
    """保存 MCP 注册表。"""
    root = project_root or get_project_root()
    return Path(save_json(config, "ctl/mcp_config.json", root))


def load_pipeline_config(project_root: Path = None) -> List[Dict[str, Any]]:
    """加载流程法配置。"""
    root = project_root or get_project_root()
    return load_json("ctl/pipeline_config.json", root)


def save_pipeline_config(config: List[Dict[str, Any]], project_root: Path = None) -> Path:
    """保存流程法配置。"""
    root = project_root or get_project_root()
    return Path(save_json(config, "ctl/pipeline_config.json", root))


def list_tools(source: Optional[str] = None, project_root: Path = None) -> Dict[str, Dict[str, Any]]:
    """列出已注册工具。

    参数：
        source: 过滤来源，"builtin" / "user" / None 表示全部。
    """
    cfg = load_mcp_config(project_root)
    tools = cfg.get("tools", {})
    result = {}
    for name, info in tools.items():
        item = dict(info)
        item.setdefault("source", "builtin")  # 旧数据默认内置
        if source is None or item.get("source") == source:
            result[name] = item
    return result


def _validate_identifier(name: str) -> Optional[str]:
    """检查工具名/模块名是否为合法 Python 标识符。"""
    if not name:
        return "名称不能为空"
    if not name.replace("_", "").replace(".", "").isalnum():
        return "名称只能包含字母、数字、下划线和点号"
    if name[0].replace(".", "").isdigit():
        return "名称不能以数字开头"
    return None


def register_tool(
    name: str,
    module: str,
    function: str,
    default_config: str,
    description: str,
    source: str = "user",
    project_root: Path = None,
    add_to_pipeline: bool = False,
    pipeline_enabled: bool = False,
) -> Dict[str, Any]:
    """注册一个新工具到 MCP 注册表。

    参数：
        name: 工具唯一标识。
        module: Python 模块名；用户工具通常使用 "extensions.xxx_tool"。
        function: 工具入口函数名。
        default_config: 默认配置文件相对路径。
        description: 工具描述。
        source: "builtin" 或 "user"。
        add_to_pipeline: 是否同时追加到 pipeline_config.json。
        pipeline_enabled: 追加流程步骤时是否启用。
    """
    root = project_root or get_project_root()

    err = _validate_identifier(name)
    if err:
        return {"success": False, "message": err}

    cfg = load_mcp_config(root)
    tools = cfg.setdefault("tools", {})

    if name in tools:
        return {"success": False, "message": f"工具 '{name}' 已存在"}

    if module in [t.get("module") for t in tools.values()]:
        # 仅提醒，不强制阻止；不同工具可共享模块但函数不同
        pass

    tools[name] = {
        "module": module,
        "function": function,
        "default_config": default_config,
        "description": description,
        "source": source,
    }
    save_mcp_config(cfg, root)

    result = {
        "success": True,
        "message": f"工具 '{name}' 已注册到 ctl/mcp_config.json",
        "tool": tools[name],
    }

    if add_to_pipeline:
        step_result = add_pipeline_step(
            step=name,
            tool=module,
            function=function,
            config=default_config,
            enabled=pipeline_enabled,
            project_root=root,
        )
        result["pipeline"] = step_result

    return result


def unregister_tool(
    name: str,
    remove_files: bool = False,
    project_root: Path = None
) -> Dict[str, Any]:
    """注销一个工具。

    内置工具（source=builtin）不允许注销，避免误删核心能力。
    """
    root = project_root or get_project_root()
    cfg = load_mcp_config(root)
    tools = cfg.get("tools", {})

    if name not in tools:
        return {"success": False, "message": f"工具 '{name}' 不存在"}

    info = tools[name]
    source = info.get("source", "builtin")
    if source == "builtin":
        return {"success": False, "message": f"内置工具 '{name}' 不允许注销"}

    removed_files = []
    if remove_files:
        # 删除工具模块文件
        module = info.get("module", "")
        if module.startswith("extensions.tools."):
            module_name = module.split(".", 2)[2]
            tool_file = root / "extensions" / "tools" / f"{module_name}.py"
        elif module.startswith("extensions."):
            module_name = module.split(".", 1)[1]
            tool_file = root / "extensions" / "tools" / f"{module_name}.py"
        else:
            tool_file = None
        if tool_file and tool_file.exists():
            tool_file.unlink()
            removed_files.append(str(tool_file.relative_to(root)))
        # 删除默认配置文件
        cfg_rel = info.get("default_config", "")
        if cfg_rel:
            cfg_file = root / cfg_rel
            if cfg_file.exists():
                cfg_file.unlink()
                removed_files.append(str(cfg_file.relative_to(root)))

    del tools[name]
    save_mcp_config(cfg, root)

    # 同步从 pipeline 中移除
    pipeline_result = remove_pipeline_step(name, root)

    return {
        "success": True,
        "message": f"工具 '{name}' 已注销",
        "removed_files": removed_files,
        "pipeline": pipeline_result,
    }


def add_pipeline_step(
    step: str,
    tool: str,
    function: str,
    config: str,
    enabled: bool = False,
    project_root: Path = None,
) -> Dict[str, Any]:
    """向流程法配置追加一个步骤。"""
    root = project_root or get_project_root()
    pipeline = load_pipeline_config(root)

    if any(s.get("step") == step for s in pipeline):
        return {"success": False, "message": f"流程步骤 '{step}' 已存在"}

    pipeline.append({
        "step": step,
        "tool": tool,
        "function": function,
        "config": config,
        "enabled": enabled,
    })
    save_pipeline_config(pipeline, root)
    return {"success": True, "message": f"流程步骤 '{step}' 已追加（enabled={enabled}）"}


def remove_pipeline_step(step: str, project_root: Path = None) -> Dict[str, Any]:
    """从流程法配置中移除指定步骤。"""
    root = project_root or get_project_root()
    pipeline = load_pipeline_config(root)
    original_len = len(pipeline)
    pipeline = [s for s in pipeline if s.get("step") != step]
    if len(pipeline) == original_len:
        return {"success": False, "message": f"流程步骤 '{step}' 不存在"}
    save_pipeline_config(pipeline, root)
    return {"success": True, "message": f"流程步骤 '{step}' 已移除"}


def update_pipeline_enabled(step: str, enabled: bool, project_root: Path = None) -> Dict[str, Any]:
    """启用/禁用流程法中的某个步骤。"""
    root = project_root or get_project_root()
    pipeline = load_pipeline_config(root)
    for s in pipeline:
        if s.get("step") == step:
            s["enabled"] = enabled
            save_pipeline_config(pipeline, root)
            return {"success": True, "message": f"流程步骤 '{step}' enabled 已设置为 {enabled}"}
    return {"success": False, "message": f"流程步骤 '{step}' 不存在"}


def is_builtin(name: str, project_root: Path = None) -> bool:
    """判断工具是否为内置。"""
    tools = list_tools(project_root=project_root)
    if name not in tools:
        return False
    return tools[name].get("source", "builtin") == "builtin"
