"""Agent 配置管理：列出、保存、删除前端注册的 Agent 配置文件。"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root, load_json, save_json


AGENT_CONFIG_DIR = "ctl/agent_configs"


def _get_agent_dir(project_root: Path) -> Path:
    return project_root / AGENT_CONFIG_DIR


def _validate_agent_id(agent_id: str) -> Optional[str]:
    """校验 Agent ID 是否合法（适合作为文件名）。"""
    if not agent_id:
        return "Agent ID 不能为空"
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", agent_id):
        return "Agent ID 只能以字母开头，包含字母、数字、下划线"
    return None


def list_agents(project_root: Path = None) -> List[Dict[str, Any]]:
    """列出所有已注册的 Agent 配置（不含敏感 api_key）。"""
    root = project_root or get_project_root()
    agent_dir = _get_agent_dir(root)
    agents = []
    if not agent_dir.exists():
        return agents

    for cfg_file in sorted(agent_dir.glob("*.json")):
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            info = {
                "agent_id": cfg.get("agent_id", cfg_file.stem),
                "model": cfg.get("model", ""),
                "model_type": cfg.get("training_setting", {}).get("model_type", "api"),
                "base_url": cfg.get("training_setting", {}).get("base_url", ""),
                "file": str(cfg_file.relative_to(root)),
                "description": cfg.get("system_message", "")[:80] + "...",
            }
            agents.append(info)
        except Exception:
            continue
    return agents


def resolve_agent_config_path(agent_id: str, project_root: Path = None) -> Path:
    """根据 agent_id 查找对应的配置文件路径。

    允许文件名与 agent_id 不一致（如 interpretation_advisor_agent_config.json
    中包含 agent_id InterpretationAdvisorAgent）。
    """
    root = project_root or get_project_root()
    agent_dir = _get_agent_dir(root)
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent 配置目录不存在: {agent_dir}")

    # 优先精确匹配 {agent_id}.json
    exact = agent_dir / f"{agent_id}.json"
    if exact.exists():
        return exact

    # 否则扫描目录，按配置文件内的 agent_id 匹配
    for cfg_file in agent_dir.glob("*.json"):
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            if cfg.get("agent_id") == agent_id:
                return cfg_file
        except Exception:
            continue

    raise FileNotFoundError(f"找不到 agent_id 为 '{agent_id}' 的 Agent 配置文件")


def load_agent_config(agent_id: str, project_root: Path = None) -> Dict[str, Any]:
    """加载指定 Agent 的完整配置。"""
    cfg_path = resolve_agent_config_path(agent_id, project_root)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def save_agent_config(
    agent_id: str,
    model: str,
    model_type: str,
    api_key: str,
    base_url: str,
    model_path: str,
    torch_dtype: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    system_message: str,
    rag_enable: bool = False,
    project_root: Path = None,
) -> Dict[str, Any]:
    """保存或更新 Agent 配置文件。

    参数：
        agent_id: Agent 唯一标识，会作为文件名。
        model: 模型名称，如 gpt-4o-mini。
        model_type: "api" 或 "local"。
        api_key: API 密钥（仅 api 模式有效）。
        base_url: API 基础 URL / 网页代理地址。
        model_path: 本地模型路径（仅 local 模式有效）。
        torch_dtype: 本地模型 torch dtype。
        max_new_tokens: 最大生成 token 数。
        temperature: 采样温度。
        top_p: nucleus sampling 参数。
        max_length: 最大上下文长度。
        system_message: 系统提示词。
        rag_enable: 是否启用 RAG。
        project_root: 项目根目录。

    返回：
        {"success": bool, "message": str, "file": str}
    """
    root = project_root or get_project_root()

    err = _validate_agent_id(agent_id)
    if err:
        return {"success": False, "message": err, "file": ""}

    if model_type not in ("api", "local"):
        return {"success": False, "message": "模型类型必须是 api 或 local", "file": ""}

    if model_type == "api" and not api_key.strip():
        return {"success": False, "message": "API 模式下 api_key 不能为空", "file": ""}

    if model_type == "local" and not model_path.strip():
        return {"success": False, "message": "本地模型模式下 model_path 不能为空", "file": ""}

    cfg = {
        "agent_id": agent_id,
        "model": model,
        "training_setting": {
            "model_type": model_type,
            "api_key": api_key,
            "base_url": base_url.strip(),
            "model_path": model_path.strip() if model_path else None,
            "torch_dtype": torch_dtype,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_length": int(max_length),
        },
        "rag": {
            "enable": bool(rag_enable)
        },
        "system_message": system_message,
    }

    cfg_path = _get_agent_dir(root) / f"{agent_id}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "message": f"Agent '{agent_id}' 已保存到 {cfg_path.relative_to(root)}",
        "file": str(cfg_path.relative_to(root)),
    }


def delete_agent(agent_id: str, project_root: Path = None) -> Dict[str, Any]:
    """删除指定 Agent 配置文件。"""
    root = project_root or get_project_root()
    cfg_path = _get_agent_dir(root) / f"{agent_id}.json"
    if not cfg_path.exists():
        return {"success": False, "message": f"Agent '{agent_id}' 不存在"}
    cfg_path.unlink()
    return {"success": True, "message": f"Agent '{agent_id}' 已删除"}


def agent_exists(agent_id: str, project_root: Path = None) -> bool:
    """判断 Agent 是否已存在。"""
    root = project_root or get_project_root()
    return (_get_agent_dir(root) / f"{agent_id}.json").exists()
