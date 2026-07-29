"""轻量级 RAG 检索：从知识库文档和 Skill 中检索相关内容。"""

from pathlib import Path
from typing import Any, Dict

from common.config_loader import get_project_root


def retrieve_context(
    query: str,
    project_root: Path = None,
    top_k_docs: int = 2,
    top_k_skills: int = 2,
) -> str:
    """根据 query 从知识库和 Skill 中检索上下文文本。

    参数：
        query: 用户查询/问题。
        project_root: 项目根目录。
        top_k_docs: 最多返回几个文档片段。
        top_k_skills: 最多返回几个 Skill。

    返回：
        格式化的检索结果字符串；未找到时返回空字符串。
    """
    root = project_root or get_project_root()

    # 延迟导入，避免顶层循环依赖
    from dev_platform import kb_manager, skill_manager

    doc_results = kb_manager.search_documents(query, top_k=top_k_docs, project_root=root)
    skill_results = skill_manager.search_skills(query, top_k=top_k_skills, project_root=root)

    if not doc_results and not skill_results:
        return ""

    parts = []
    if doc_results:
        parts.append("【相关知识库片段】")
        for item in doc_results:
            parts.append(f"- 来自《{item['name']}》：{item['chunk']}")

    if skill_results:
        parts.append("【相关 Skill】")
        for skill in skill_results:
            title = skill.get("title") or skill.get("name", "未命名")
            desc = skill.get("description", "")
            content = skill.get("content", "")
            header = f"- {title}" + (f"（{desc}）" if desc else "")
            parts.append(header)
            if content:
                parts.append(f"  {content}")

    return "\n".join(parts)
