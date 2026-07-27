"""LLM Agent 调用封装，保持与原项目一致的调用方式。"""
from pathlib import Path
from typing import Union

from llm.agent import Agent


def load_agent(agent_config_path: Union[str, Path], project_root: Path = None) -> Agent:
    """加载 LLM Agent。

    参数：
        agent_config_path: Agent 配置文件路径（相对项目根目录或绝对路径）。
        project_root: Gaia 项目根目录；None 时自动推断。

    返回：
        初始化完成的 Agent 实例。
    """
    if project_root is None:
        from common.config_loader import get_project_root
        project_root = get_project_root()

    if not Path(agent_config_path).is_absolute():
        agent_config_path = (project_root / agent_config_path).resolve()

    memory_folder = (project_root / "intermediate_files" / "llms").resolve()
    memory_folder.mkdir(parents=True, exist_ok=True)

    return Agent(str(agent_config_path), memory_folder=str(memory_folder))
