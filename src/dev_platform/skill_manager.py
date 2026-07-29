"""Skill 管理：用户可编写可复用的提示片段/能力说明。"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root


SKILL_DIR = "extensions/skills"


def _get_skill_dir(project_root: Path) -> Path:
    return project_root / SKILL_DIR


def _sanitize_name(name: str) -> str:
    """把任意字符串转为安全的 skill 名（不含扩展名）。"""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-\u4e00-\u9fff]", "", name)
    return name or "untitled_skill"


def _tokenize(text: str) -> List[str]:
    """把查询拆分为检索 token：中文按字拆分，英文/数字按非单词字符拆分。"""
    text = text.lower().strip()
    if not text:
        return []
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[a-z0-9_]+", text)
    seen = set()
    tokens = []
    for t in chars + words:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    return tokens


def list_skills(project_root: Path = None) -> List[Dict[str, Any]]:
    """列出所有 Skill（不含完整 content，避免加载过大）。"""
    root = project_root or get_project_root()
    skill_dir = _get_skill_dir(root)
    skills = []
    if not skill_dir.exists():
        return skills
    for skill_file in sorted(skill_dir.glob("*.json")):
        try:
            data = json.loads(skill_file.read_text(encoding="utf-8"))
            skills.append({
                "name": data.get("name", skill_file.stem),
                "title": data.get("title", skill_file.stem),
                "description": data.get("description", ""),
                "filename": skill_file.name,
                "path": str(skill_file.relative_to(root)),
            })
        except Exception:
            continue
    return skills


def load_skill(name: str, project_root: Path = None) -> Dict[str, Any]:
    """加载单个 Skill 完整内容。"""
    root = project_root or get_project_root()
    skill_path = _get_skill_dir(root) / f"{_sanitize_name(name)}.json"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill 不存在: {skill_path}")
    return json.loads(skill_path.read_text(encoding="utf-8"))


def load_all_skills(project_root: Path = None) -> List[Dict[str, Any]]:
    """加载全部 Skill 完整内容。"""
    root = project_root or get_project_root()
    skill_dir = _get_skill_dir(root)
    skills = []
    if not skill_dir.exists():
        return skills
    for skill_file in sorted(skill_dir.glob("*.json")):
        try:
            skills.append(json.loads(skill_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return skills


def save_skill(
    name: str,
    title: str,
    description: str,
    content: str,
    project_root: Path = None,
) -> Dict[str, Any]:
    """保存 Skill。"""
    root = project_root or get_project_root()
    skill_dir = _get_skill_dir(root)
    skill_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_name(name)
    if not safe_name:
        return {"success": False, "message": "Skill 名无效"}

    skill_path = skill_dir / f"{safe_name}.json"
    data = {
        "name": safe_name,
        "title": title.strip() or safe_name,
        "description": description.strip(),
        "content": content.strip(),
    }
    skill_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "success": True,
        "message": f"Skill '{safe_name}' 已保存到 {skill_path.relative_to(root)}",
        "name": safe_name,
        "path": str(skill_path.relative_to(root)),
    }


def delete_skill(name: str, project_root: Path = None) -> Dict[str, Any]:
    """删除 Skill。"""
    root = project_root or get_project_root()
    skill_path = _get_skill_dir(root) / f"{_sanitize_name(name)}.json"
    if not skill_path.exists():
        return {"success": False, "message": f"Skill '{name}' 不存在"}
    skill_path.unlink()
    return {"success": True, "message": f"Skill '{name}' 已删除"}


def format_skills_for_prompt(skills: List[Dict[str, Any]]) -> str:
    """将 Skill 列表格式化为 Agent 可读的文本。"""
    if not skills:
        return ""
    lines = ["【相关 Skill】"]
    for skill in skills:
        title = skill.get("title") or skill.get("name", "未命名")
        desc = skill.get("description", "")
        content = skill.get("content", "")
        lines.append(f"- {title}" + (f"（{desc}）" if desc else ""))
        if content:
            lines.append(f"  {content}")
    return "\n".join(lines)


def search_skills(
    query: str,
    top_k: int = 3,
    project_root: Path = None,
) -> List[Dict[str, Any]]:
    """基于关键词匹配检索相关 Skill。"""
    root = project_root or get_project_root()
    skills = load_all_skills(root)
    if not skills or not query:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidates = []
    for skill in skills:
        text = " ".join([
            skill.get("name", ""),
            skill.get("title", ""),
            skill.get("description", ""),
            skill.get("content", ""),
        ]).lower()
        score = sum(1 for t in query_tokens if t in text)
        if score > 0:
            candidates.append({"skill": skill, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return [c["skill"] for c in candidates[:top_k]]
