#!/usr/bin/env python3
"""便捷入口：运行流程法。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "extensions"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from methods.flow_runner import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="运行 Gaia 流程法")
    parser.add_argument("--config", default="ctl/pipeline_config.json", help="流程配置文件")
    parser.add_argument("--stop-at", default=None, help="执行到指定步骤后停止")
    args = parser.parse_args()
    run_pipeline(args.config, stop_at=args.stop_at)
