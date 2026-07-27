"""MCP 服务端：使用 FastMCP 将所有工具暴露为 MCP Tool。"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp.server.fastmcp import FastMCP
from common.config_loader import get_project_root, load_json, load_tool_config
import importlib

mcp = FastMCP("gaia-welllog-mcp")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _make_tool_handler(tool_name: str, info: Dict[str, Any]):
    async def handler(config_path: str = None, overrides: dict = None) -> str:
        root = get_project_root()
        cfg_rel = config_path or info["default_config"]
        cfg = load_tool_config(cfg_rel, root)
        if overrides:
            _deep_merge(cfg, overrides)
        module_name = info['module'] if '.' in info['module'] else f"tools.{info['module']}"
        module = importlib.import_module(module_name)
        func = getattr(module, info['function'])
        result = await asyncio.to_thread(func, cfg, root)
        return json.dumps(result, ensure_ascii=False)

    handler.__name__ = tool_name
    handler.__doc__ = info.get("description", "")
    return handler


def register_tools() -> None:
    root = get_project_root()
    cfg = load_json("ctl/mcp_config.json", root)
    for name, info in cfg.get("tools", {}).items():
        fn = _make_tool_handler(name, info)
        mcp.add_tool(fn, name=name, description=info.get("description"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia MCP 服务端")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio", help="传输协议")
    args = parser.parse_args()

    register_tools()
    print(f"[MCP] Gaia MCP 服务已启动，transport={args.transport}", file=sys.stderr)
    mcp.run(transport=args.transport)
