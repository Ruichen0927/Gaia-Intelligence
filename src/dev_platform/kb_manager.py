"""知识库文档管理：上传、列出、删除、关键词检索。"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.config_loader import get_project_root


KB_DIR = "extensions/knowledge_base"


def _get_kb_dir(project_root: Path) -> Path:
    return project_root / KB_DIR


def _sanitize_filename(name: str) -> str:
    """把任意字符串转为安全的文件名（不含扩展名）。"""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^\w\-\u4e00-\u9fff]", "", name)
    return name or "untitled"


def list_documents(project_root: Path = None) -> List[Dict[str, Any]]:
    """列出所有知识库文档。"""
    root = project_root or get_project_root()
    kb_dir = _get_kb_dir(root)
    docs = []
    if not kb_dir.exists():
        return docs
    for doc_file in sorted(kb_dir.glob("*.txt")):
        content = doc_file.read_text(encoding="utf-8")
        docs.append({
            "name": doc_file.stem,
            "filename": doc_file.name,
            "size": doc_file.stat().st_size,
            "word_count": len(content),
            "summary": content[:200].replace("\n", " ") + ("..." if len(content) > 200 else ""),
            "path": str(doc_file.relative_to(root)),
        })
    return docs


def save_document(name: str, content: str, project_root: Path = None) -> Dict[str, Any]:
    """保存知识库文档。"""
    root = project_root or get_project_root()
    kb_dir = _get_kb_dir(root)
    kb_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_filename(name)
    if not safe_name:
        return {"success": False, "message": "文档名无效"}

    doc_path = kb_dir / f"{safe_name}.txt"
    doc_path.write_text(content, encoding="utf-8")

    return {
        "success": True,
        "message": f"文档 '{safe_name}' 已保存到 {doc_path.relative_to(root)}",
        "name": safe_name,
        "path": str(doc_path.relative_to(root)),
    }


def delete_document(name: str, project_root: Path = None) -> Dict[str, Any]:
    """删除知识库文档。"""
    root = project_root or get_project_root()
    doc_path = _get_kb_dir(root) / f"{_sanitize_filename(name)}.txt"
    if not doc_path.exists():
        return {"success": False, "message": f"文档 '{name}' 不存在"}
    doc_path.unlink()
    return {"success": True, "message": f"文档 '{name}' 已删除"}


def _tokenize(text: str) -> List[str]:
    """把查询拆分为检索 token：中文按字拆分，英文/数字按空白与非单词字符拆分。"""
    text = text.lower().strip()
    if not text:
        return []
    # 中文单字 token
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    # 英文/数字/下划线 token
    words = re.findall(r"[a-z0-9_]+", text)
    # 去重并保持一定顺序
    seen = set()
    tokens = []
    for t in chars + words:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    return tokens


def _split_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """按滑动窗口把文本切分为片段。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def search_documents(
    query: str,
    top_k: int = 3,
    project_root: Path = None,
    chunk_size: int = 300,
) -> List[Dict[str, Any]]:
    """基于关键词匹配检索最相关的文档片段。

    返回：
        [{"name": ..., "filename": ..., "chunk": ..., "score": int}, ...]
    """
    root = project_root or get_project_root()
    docs = list_documents(root)
    if not docs or not query:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    candidates = []
    for doc in docs:
        doc_path = _get_kb_dir(root) / doc["filename"]
        text = doc_path.read_text(encoding="utf-8")
        chunks = _split_chunks(text, chunk_size)
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for t in query_tokens if t in chunk_lower)
            if score > 0:
                candidates.append({
                    "name": doc["name"],
                    "filename": doc["filename"],
                    "chunk": chunk,
                    "score": score,
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]
