"""统计分析工具包：统计摘要、相关性矩阵。"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common.config_loader import ensure_dirs


def summary_statistics(config: Dict, project_root: Path) -> Dict[str, Any]:
    """分井/分层统计摘要。"""
    print("\n===== 启动工具: stat_summary =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC"])
    group_by = params.get("group_by", "WELL_NAME")

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    if group_by not in df.columns:
        return {"success": False, "message": f"分组列 '{group_by}' 不存在", "output_files": []}

    result = {}
    for name, group in df.groupby(group_by):
        stats = {}
        for curve in curves:
            if curve not in group.columns:
                continue
            stats[curve] = {
                "count": int(group[curve].notna().sum()),
                "mean": round(group[curve].mean(), 4),
                "std": round(group[curve].std(), 4),
                "min": round(group[curve].min(), 4),
                "p10": round(group[curve].quantile(0.10), 4),
                "p50": round(group[curve].median(), 4),
                "p90": round(group[curve].quantile(0.90), 4),
                "max": round(group[curve].max(), 4),
            }
        result[str(name)] = stats

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  - 统计摘要已生成，共 {len(result)} 组。")
    return {"success": True, "message": f"统计摘要完成，共 {len(result)} 组", "output_files": [output_file]}


def correlation_matrix(config: Dict, project_root: Path) -> Dict[str, Any]:
    """曲线相关性矩阵（CSV + JSON + 热图 PNG）。"""
    print("\n===== 启动工具: stat_correlation =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_csv = paths["output_csv"]
    output_png = paths.get("output_png")
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC", "PE"])

    ensure_dirs(os.path.dirname(output_csv))

    df = pd.read_csv(input_file)
    available = [c for c in curves if c in df.columns]
    if len(available) < 2:
        return {"success": False, "message": "可用曲线不足 2 条", "output_files": []}

    corr = df[available].corr()
    corr.to_csv(output_csv, encoding='utf-8-sig')

    output_files = [output_csv]
    if output_png:
        ensure_dirs(os.path.dirname(output_png))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_files.append(output_png)

    print(f"  - 相关性矩阵已保存: {output_csv}")
    return {"success": True, "message": "相关性矩阵计算完成", "output_files": output_files}
