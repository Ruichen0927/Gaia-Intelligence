"""公共工具函数。"""
import json
import os
from pathlib import Path
from typing import Any, Dict


def extract_json_from_response(raw_response: Any) -> str:
    """从 LLM 返回文本中提取第一个 JSON 对象。"""
    if not isinstance(raw_response, str):
        return ""
    start_index = raw_response.find('{')
    end_index = raw_response.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return raw_response[start_index:end_index + 1]
    return raw_response


def get_project_root() -> Path:
    """获取 Gaia 项目根目录。

    优先读取环境变量 GAIA_PROJECT_ROOT；否则根据本文件位置推断
    （本文件位于 src/common/utils.py，向上两级即为项目根）。
    """
    env_root = os.environ.get("GAIA_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


def make_absolute(path: str, root: Path) -> str:
    """将相对路径转换为绝对路径；绝对路径保持不变。"""
    if not path:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _resolve_object(obj: Any, root: Path) -> Any:
    """递归解析对象中的相对路径。"""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, str):
                # 仅对已知路径类键或 paths 字典下的值做解析
                if (
                    k == "paths"
                    or k.endswith(("_path", "_file", "_directory", "_dir", "_config"))
                    or k in ("config", "data_directory", "output_directory",
                             "input_directory", "strategy_file", "agent_config",
                             "memory_folder", "baseline_config")
                ):
                    new_obj[k] = make_absolute(v, root)
                else:
                    new_obj[k] = v
            else:
                new_obj[k] = _resolve_object(v, root)
        return new_obj
    elif isinstance(obj, list):
        return [_resolve_object(item, root) for item in obj]
    return obj


def load_json(rel_path: str, root: Path = None) -> Dict[str, Any]:
    """加载相对项目根目录的 JSON 文件。"""
    root = root or get_project_root()
    file_path = root / rel_path
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tool_config(rel_path: str, root: Path = None) -> Dict[str, Any]:
    """加载工具配置文件，并解析其中的相对路径。"""
    cfg = load_json(rel_path, root)
    root = root or get_project_root()
    return _resolve_object(cfg, root)


def save_json(data: Any, rel_path: str, root: Path = None) -> str:
    """保存 JSON 到相对项目根目录的路径，返回保存后的绝对路径。"""
    root = root or get_project_root()
    file_path = root / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(file_path.resolve())


def ensure_dirs(*paths: str) -> None:
    """确保一个或多个目录存在。"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
