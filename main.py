"""Gaia 命令行入口。"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "extensions"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="gaia", description="Gaia 测井解释重构项目")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # flow 子命令
    flow_parser = subparsers.add_parser("flow", help="流程法一键运行")
    flow_parser.add_argument("--config", default="ctl/pipeline_config.json", help="流程配置文件")
    flow_parser.add_argument("--stop-at", default=None, help="执行到指定步骤后停止")

    # mcp 子命令（本地调度）
    mcp_parser = subparsers.add_parser("mcp", help="MCP 本地调度器")
    mcp_parser.add_argument("--tool", help="调用的工具名")
    mcp_parser.add_argument("--config", help="工具配置文件相对路径")
    mcp_parser.add_argument("--overrides", default="{}", help="JSON 配置覆盖")
    mcp_parser.add_argument("--plan", help="MCP 计划文件相对路径")

    # mcp-server 子命令
    server_parser = subparsers.add_parser("mcp-server", help="启动 MCP 服务端")
    server_parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio")

    args = parser.parse_args()
    root = get_project_root()

    if args.command == "flow":
        from methods.flow_runner import run_pipeline
        result = run_pipeline(args.config, root, stop_at=args.stop_at)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "mcp":
        from methods.mcp_orchestrator import run_tool, run_plan, load_json
        if args.tool:
            overrides = json.loads(args.overrides)
            result = run_tool(args.tool, args.config, overrides, root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.plan:
            plan = load_json(args.plan, root)
            result = run_plan(plan, root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            mcp_parser.print_help()
    elif args.command == "mcp-server":
        from methods import mcp_server
        mcp_server.register_tools()
        print(f"[MCP] 启动 MCP 服务，transport={args.transport}")
        mcp_server.mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
