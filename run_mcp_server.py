#!/usr/bin/env python3
"""便捷入口：启动 MCP 服务端。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "extensions"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from methods import mcp_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="启动 Gaia MCP 服务端")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")
    args = parser.parse_args()
    mcp_server.register_tools()
    print(f"[MCP] 启动 MCP 服务，transport={args.transport}", file=sys.stderr)
    mcp_server.mcp.run(transport=args.transport)
