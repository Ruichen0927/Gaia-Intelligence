"""工具代码与配置模板，用于二次开发平台兜底或快速初始化。"""

from string import Template


CUSTOM_TOOL_TEMPLATE = Template('''\
"""${description}"""

from pathlib import Path
from typing import Dict, Any


def ${function_name}(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """${description}

    Args:
        config: 工具配置字典，通常包含 paths 与 parameters。
        project_root: Gaia 项目根目录。

    Returns:
        dict: {"success": bool, "message": str, "output_files": list}
    """
    try:
        # TODO: 在此处实现工具核心逻辑
        # 示例：读取输出目录
        output_dir = Path(config["paths"]["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "success": True,
            "message": "工具执行成功",
            "output_files": []
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"工具执行失败: {e}",
            "output_files": []
        }
''')


def make_default_config_template(tool_name: str, description: str) -> dict:
    """生成用户工具的默认 JSON 配置模板。"""
    # 延迟导入，避免循环依赖
    from common.tool_registry import load_default_tool_paths
    return {
        "name": tool_name,
        "description": description,
        "paths": load_default_tool_paths(tool_name),
        "parameters": {}
    }


def render_tool_code(tool_name: str, description: str, function_name: str = "run") -> str:
    """渲染最简工具代码。"""
    return CUSTOM_TOOL_TEMPLATE.substitute(
        description=description or f"用户自定义工具: {tool_name}",
        function_name=function_name or "run"
    )
