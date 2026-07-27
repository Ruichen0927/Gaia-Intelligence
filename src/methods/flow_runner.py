"""流程法运行器：顺序执行 pipeline_config.json 中定义的步骤。"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, Any

# 将 src/ 加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config_loader import get_project_root, load_json, load_tool_config


def run_tool_step(step_cfg: Dict[str, Any], project_root: Path):
    """根据步骤配置调用对应工具函数。"""
    tool_module_name = step_cfg['tool'] if '.' in step_cfg['tool'] else f"tools.{step_cfg['tool']}"
    function_name = step_cfg['function']
    config_rel_path = step_cfg['config']

    print(f"\n{'='*60}")
    print(f"[Flow] 执行步骤: {step_cfg['step']}")
    print(f"{'='*60}")

    module = importlib.import_module(tool_module_name)
    func = getattr(module, function_name)
    cfg = load_tool_config(config_rel_path, project_root)
    result = func(cfg, project_root)
    return result


def run_pipeline(pipeline_config_rel: str = "ctl/pipeline_config.json",
                 project_root: Path = None,
                 stop_at: str = None) -> Dict[str, Any]:
    """顺序执行流程配置中的所有启用步骤。

    参数：
        pipeline_config_rel: 流程配置文件相对项目根目录的路径。
        project_root: 项目根目录；None 时自动推断。
        stop_at: 若指定，执行到该步骤后停止（包含该步骤）。

    返回：
        包含 overall success 和各步骤结果的字典。
    """
    if project_root is None:
        project_root = get_project_root()

    pipeline = load_json(pipeline_config_rel, project_root)
    results = []
    overall_success = True

    for step in pipeline:
        if not step.get("enabled", True):
            print(f"\n[Flow] 跳过禁用步骤: {step['step']}")
            continue
        try:
            result = run_tool_step(step, project_root)
            results.append({"step": step["step"], "result": result})
            if not result.get("success", False):
                overall_success = False
                print(f"\n[Flow] 步骤 {step['step']} 返回失败，停止后续执行。")
                break
        except Exception as e:
            overall_success = False
            print(f"\n[Flow] 步骤 {step['step']} 执行失败: {e}")
            traceback.print_exc()
            results.append({"step": step["step"], "error": str(e)})
            break

        if stop_at and step["step"] == stop_at:
            print(f"\n[Flow] 已达到指定停止步骤: {stop_at}")
            break

    print(f"\n{'='*60}")
    print(f"[Flow] 流程执行结束. Overall success: {overall_success}")
    print(f"{'='*60}")
    return {"success": overall_success, "steps": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaia 流程法运行器")
    parser.add_argument("--config", default="ctl/pipeline_config.json", help="流程配置文件相对路径")
    parser.add_argument("--stop-at", default=None, help="执行到指定步骤后停止")
    args = parser.parse_args()
    run_pipeline(args.config, stop_at=args.stop_at)
