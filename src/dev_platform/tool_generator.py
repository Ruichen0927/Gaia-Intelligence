"""智能体辅助工具开发：根据自然语言需求生成工具，或将现有代码转换为 Gaia 工具格式。"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root
from common.utils import extract_json_from_response


DEFAULT_AGENT_CONFIG = "ctl/agent_configs/tool_developer_agent_config.json"


def _load_agent(project_root: Path = None):
    # 延迟导入，避免在仅做模板/校验时触发 transformers 等重依赖
    from llm.caller import load_agent
    root = project_root or get_project_root()
    return load_agent(DEFAULT_AGENT_CONFIG, root)


def _load_example_tools(project_root: Path, max_chars: int = 3000) -> List[str]:
    """读取内置工具源码作为 few-shot 示例。"""
    examples = []
    candidates = [
        project_root / "src" / "tools" / "lithology_tool.py",
        project_root / "src" / "tools" / "data_tool.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            code = candidate.read_text(encoding="utf-8")
            if len(code) > max_chars:
                code = code[:max_chars] + "\n\n# ... (truncated for prompt length)\n"
            examples.append(code)
    return examples


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """从 LLM 返回文本中提取 JSON 对象。"""
    cleaned = extract_json_from_response(text)
    return json.loads(cleaned)


def _build_generation_prompt(
    requirement: str,
    tool_name: Optional[str],
    description: Optional[str],
    examples: List[str]
) -> str:
    example_text = "\n\n---\n\n".join(
        f"示例 {i+1}:\n{ex}" for i, ex in enumerate(examples)
    )
    name_hint = f"\n建议工具名：{tool_name}" if tool_name else ""
    desc_hint = f"\n建议描述：{description}" if description else ""

    return f"""请根据以下需求，为 Gaia 测井解释平台设计一个可插拔工具。

需求描述：
{requirement}{name_hint}{desc_hint}

参考示例（已有内置工具）：
{example_text}

请严格遵循以下规范：
1. 工具主函数签名必须是：def run(config: dict, project_root: Path) -> dict
2. config 字典中包含 paths 和 parameters，paths 中的路径已经是绝对路径，可直接使用。
3. 默认情况下，config["paths"] 至少会包含 input_file、input_directory 和 output_directory（见 ctl/default_tool_paths.json）。若工具需要读取文件，优先使用 config["paths"]["input_file"]；若需要输出文件，优先写入 config["paths"]["output_directory"] 并在代码中确保该目录存在（使用 Path.mkdir(parents=True, exist_ok=True)）。
4. 返回值必须是字典，至少包含 success(bool)、message(str)、output_files(list[str])。
5. 代码需要健壮，使用 try/except 捕获异常并返回 success=false。
6. 使用 pathlib.Path 和 project_root 处理路径。

请直接输出严格符合以下结构的 JSON，不要包含 markdown 代码块或任何解释文字：

{{
  "tool_name": "英文小写下划线分隔的工具唯一标识",
  "module_name": "与 tool_name 相同的 Python 模块名",
  "function_name": "run",
  "description": "工具简短描述",
  "python_code": "完整 Python 源码字符串",
  "config_template": {{
    "name": "工具名",
    "description": "描述",
    "paths": {{
      "input_file": "data/processed_data/data_tools/cleaned_well_logs.csv",
      "input_directory": "data/sample",
      "output_directory": "data/processed_data/<tool_name>"
    }},
    "parameters": {{}}
  }}
}}
"""


def _build_conversion_prompt(
    source_code: str,
    tool_name: str,
    description: str,
    examples: List[str]
) -> str:
    example_text = "\n\n---\n\n".join(
        f"示例 {i+1}:\n{ex}" for i, ex in enumerate(examples)
    )
    return f"""请将以下现有 Python 代码转换为符合 Gaia 测井解释平台规范的工具模块。

目标工具名：{tool_name}
目标描述：{description}

原始代码：
```python
{source_code}
```

参考示例（已有内置工具）：
{example_text}

转换要求：
1. 主函数签名改为：def run(config: dict, project_root: Path) -> dict
2. 原函数中的硬编码路径应改为从 config["paths"] 读取；config 中的路径已经是绝对路径。默认 config["paths"] 会包含 input_file、input_directory 和 output_directory，请尽量使用它们。
3. 输出前请确保 output_directory 存在（Path.mkdir(parents=True, exist_ok=True)）。
4. 原函数中的常数/阈值建议放到 config["parameters"] 中读取，并给出合理的默认值。
5. 返回值必须是字典，至少包含 success(bool)、message(str)、output_files(list[str])。
6. 添加 try/except，异常时返回 success=false 与错误信息。

请直接输出严格符合以下结构的 JSON，不要包含 markdown 代码块或任何解释文字：

{{
  "tool_name": "{tool_name}",
  "module_name": "{tool_name}",
  "function_name": "run",
  "description": "{description}",
  "python_code": "转换后的完整 Python 源码字符串",
  "config_template": {{
    "name": "{tool_name}",
    "description": "{description}",
    "paths": {{
      "input_file": "data/processed_data/data_tools/cleaned_well_logs.csv",
      "input_directory": "data/sample",
      "output_directory": "data/processed_data/{tool_name}"
    }},
    "parameters": {{}}
  }}
}}
"""


def generate_tool_from_requirement(
    requirement: str,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    project_root: Path = None,
) -> Dict[str, Any]:
    """根据自然语言需求生成工具代码与配置模板。

    返回：
        {"success": bool, "data": dict, "raw": str, "error": str|None}
    """
    root = project_root or get_project_root()
    try:
        agent = _load_agent(root)
        examples = _load_example_tools(root)
        prompt = _build_generation_prompt(requirement, tool_name, description, examples)
        raw = agent.generate_response(user_input=prompt)
        data = _extract_json_from_text(raw)
        return {"success": True, "data": data, "raw": raw, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "raw": "", "error": f"生成失败: {e}"}


def convert_code_to_tool(
    source_code: str,
    tool_name: str,
    description: str,
    project_root: Path = None,
) -> Dict[str, Any]:
    """将现有 Python 代码转换为 Gaia 工具格式。

    返回：
        {"success": bool, "data": dict, "raw": str, "error": str|None}
    """
    root = project_root or get_project_root()
    try:
        agent = _load_agent(root)
        examples = _load_example_tools(root)
        prompt = _build_conversion_prompt(source_code, tool_name, description, examples)
        raw = agent.generate_response(user_input=prompt)
        data = _extract_json_from_text(raw)
        return {"success": True, "data": data, "raw": raw, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "raw": "", "error": f"转换失败: {e}"}
