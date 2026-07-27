"""工具格式校验与沙箱测试：确保用户新增工具符合 Gaia 规范且可安全运行。"""

import ast
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict


def validate_syntax(code: str) -> Dict[str, Any]:
    """使用 ast.parse 检查代码语法。"""
    try:
        ast.parse(code)
        return {"valid": True, "message": "语法检查通过"}
    except SyntaxError as e:
        return {"valid": False, "message": f"语法错误 (行 {e.lineno}): {e.msg}"}
    except Exception as e:
        return {"valid": False, "message": f"语法检查异常: {e}"}


def validate_signature(code: str, function_name: str = "run") -> Dict[str, Any]:
    """通过 AST 检查目标函数签名是否符合 Gaia 规范。

    规范：
        def function_name(config: dict, project_root: Path) -> dict:
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "message": f"语法错误: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = node.args.args
            arg_names = [a.arg for a in args]
            if len(arg_names) < 2:
                return {
                    "valid": False,
                    "message": f"函数 '{function_name}' 参数不足，需要至少 (config, project_root)，当前: {arg_names}"
                }
            if arg_names[0] != "config" or arg_names[1] != "project_root":
                return {
                    "valid": False,
                    "message": (
                        f"函数 '{function_name}' 参数名不规范。"
                        f"应为 (config, project_root)，当前: {arg_names}"
                    )
                }
            return {"valid": True, "message": f"函数签名检查通过: {function_name}{tuple(arg_names)}"}

    return {"valid": False, "message": f"未找到函数 '{function_name}'"}


def validate_returns_dict(code: str, function_name: str = "run") -> Dict[str, Any]:
    """简单 AST 检查函数是否包含 return 语句。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "message": f"语法错误: {e}"}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
            return {
                "valid": has_return,
                "message": "函数包含 return 语句" if has_return else "函数缺少 return 语句"
            }
    return {"valid": False, "message": f"未找到函数 '{function_name}'"}


def sandbox_run(
    module_file: Path,
    function_name: str,
    config: Dict[str, Any],
    project_root: Path,
    timeout: int = 60
) -> Dict[str, Any]:
    """在子进程中安全地导入并运行用户工具。

    参数：
        module_file: 用户工具模块的绝对路径（位于 extensions/tools/ 或 src/tools/）。
        function_name: 要调用的函数名。
        config: 传给函数的配置字典。
        project_root: 项目根目录。
        timeout: 子进程超时秒数。

    返回：
        {"success": bool, "result": dict|None, "error": str|None}
    """
    module_file = Path(module_file).resolve()
    project_root = Path(project_root).resolve()

    # 判断模块属于内置还是扩展，决定 import 名称
    try:
        rel = module_file.relative_to(project_root)
    except ValueError:
        return {"success": False, "result": None, "error": "模块文件不在项目根目录下"}

    parts = rel.with_suffix("").parts
    if parts[0] == "extensions":
        import_name = ".".join(parts)
    elif parts[0] == "src" and parts[1] == "tools":
        import_name = ".".join(parts[1:])  # tools.xxx
    else:
        import_name = ".".join(parts)

    runner_script = f'''
import json
import sys
from pathlib import Path

project_root = Path({str(project_root)!r}).resolve()
# 加入项目根目录以支持 extensions.tools.xxx 形式的用户工具导入
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

config = json.loads({json.dumps(config, ensure_ascii=False)!r})

import importlib
mod = importlib.import_module({import_name!r})
func = getattr(mod, {function_name!r})
result = func(config, project_root)
print("__RESULT__" + json.dumps(result, ensure_ascii=False))
'''

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "result": None, "error": f"工具运行超时（>{timeout}s）"}
    except Exception as e:
        return {"success": False, "result": None, "error": f"启动子进程失败: {e}"}

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        return {"success": False, "result": None, "error": f"工具运行异常:\n{err}"}

    stdout = proc.stdout.strip()
    marker = "__RESULT__"
    idx = stdout.rfind(marker)
    if idx == -1:
        return {"success": False, "result": None, "error": f"未捕获到工具返回值，原始输出:\n{stdout}"}

    payload = stdout[idx + len(marker):]
    try:
        result = json.loads(payload)
        return {"success": True, "result": result, "error": None}
    except json.JSONDecodeError as e:
        return {"success": False, "result": None, "error": f"返回值不是合法 JSON: {e}\n原始输出:\n{payload}"}


def validate_tool(code: str, function_name: str = "run") -> Dict[str, Any]:
    """一站式校验：语法 + 签名 + return。"""
    checks = [
        validate_syntax(code),
        validate_signature(code, function_name),
        validate_returns_dict(code, function_name),
    ]
    failed = [c for c in checks if not c["valid"]]
    if failed:
        return {
            "valid": False,
            "message": "工具校验未通过",
            "details": failed
        }
    return {
        "valid": True,
        "message": "工具校验通过",
        "details": checks
    }
