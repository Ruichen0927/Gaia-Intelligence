"""可视化工具包：交会图、直方图、Pickett 图。"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config_loader import ensure_dirs


def _save_figure(fig, output_path: str) -> str:
    ensure_dirs(os.path.dirname(output_path))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def crossplot(config: Dict, project_root: Path) -> Dict[str, Any]:
    """绘制两曲线交会图并计算相关系数。"""
    print("\n===== 启动工具: viz_crossplot =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    x_curve = params.get("x_curve", "GR")
    y_curve = params.get("y_curve", "DEN")
    color_curve = params.get("color_curve", None)
    alpha = params.get("alpha", 0.5)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    if x_curve not in df.columns or y_curve not in df.columns:
        return {"success": False, "message": f"列 {x_curve} 或 {y_curve} 不存在", "output_files": []}

    sub = df[[x_curve, y_curve]].dropna()
    corr = sub[x_curve].corr(sub[y_curve])

    fig, ax = plt.subplots(figsize=(6, 5))
    if color_curve and color_curve in df.columns:
        scatter = ax.scatter(sub[x_curve], sub[y_curve], c=df.loc[sub.index, color_curve],
                             cmap="viridis", alpha=alpha, s=10)
        plt.colorbar(scatter, ax=ax, label=color_curve)
    else:
        ax.scatter(sub[x_curve], sub[y_curve], alpha=alpha, s=10)
    ax.set_xlabel(x_curve)
    ax.set_ylabel(y_curve)
    ax.set_title(f"{x_curve} vs {y_curve} (r={corr:.3f})")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, output_file)

    info_path = output_file.replace(".png", "_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({"correlation": round(corr, 4), "points": len(sub)}, f, indent=2, ensure_ascii=False)

    print(f"  - 交会图已保存: {output_file}，相关系数: {corr:.3f}")
    return {"success": True, "message": f"交会图完成，r={corr:.3f}", "output_files": [output_file, info_path]}


def histogram(config: Dict, project_root: Path) -> Dict[str, Any]:
    """绘制指定曲线直方图。"""
    print("\n===== 启动工具: viz_histogram =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN"])
    bins = params.get("bins", 30)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, curve in zip(axes, curves):
        if curve in df.columns:
            ax.hist(df[curve].dropna(), bins=bins, edgecolor='k', alpha=0.7)
            ax.set_title(curve)
            ax.set_xlabel(curve)
            ax.set_ylabel("Frequency")
    _save_figure(fig, output_file)

    print(f"  - 直方图已保存: {output_file}")
    return {"success": True, "message": "直方图生成完成", "output_files": [output_file]}


def pickett_plot(config: Dict, project_root: Path) -> Dict[str, Any]:
    """绘制含水饱和度 Pickett 图（Rt vs Porosity 对数-对数）。"""
    print("\n===== 启动工具: viz_pickett_plot =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    rt_col = params.get("rt_col", "RT")
    por_col = params.get("por_col", "phie_final")
    a = params.get("a", 1.0)
    Rw = params.get("Rw", 0.1)
    m = params.get("m", 2.0)
    n = params.get("n", 2.0)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    if rt_col not in df.columns or por_col not in df.columns:
        return {"success": False, "message": f"列 {rt_col} 或 {por_col} 不存在", "output_files": []}

    sub = df[[rt_col, por_col]].dropna()
    sub = sub[(sub[rt_col] > 0) & (sub[por_col] > 0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sub[por_col], sub[rt_col], alpha=0.5, s=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(por_col)
    ax.set_ylabel(rt_col)
    ax.set_title("Pickett Plot")
    ax.grid(True, which="both", ls="--", alpha=0.3)

    # 绘制 20%, 50%, 80% Sw 等值线
    phi_range = np.logspace(np.log10(sub[por_col].min()), np.log10(sub[por_col].max()), 100)
    for Sw in [0.2, 0.5, 0.8]:
        Rt = a * Rw / (phi_range ** m) / (Sw ** n)
        ax.plot(phi_range, Rt, label=f"Sw={Sw:.0%}")
    ax.legend()

    _save_figure(fig, output_file)
    print(f"  - Pickett 图已保存: {output_file}")
    return {"success": True, "message": "Pickett 图生成完成", "output_files": [output_file]}
