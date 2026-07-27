"""特征工程工具包：派生曲线、归一化、深度窗统计特征。"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from common.config_loader import ensure_dirs


def compute_derived_curves(config: Dict, project_root: Path) -> Dict[str, Any]:
    """计算常用派生曲线：线性 VSH、密度孔隙度、ρma_app 等。"""
    print("\n===== 启动工具: feature_compute_derived_curves =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]

    gr_col = params.get("gr_col", "GR")
    den_col = params.get("den_col", "DEN")
    cnl_col = params.get("cnl_col", "CNL")
    vsh_col = params.get("vsh_col", "vsh_linear")
    phid_col = params.get("phid_col", "phid")
    rho_ma_app_col = params.get("rho_ma_app_col", "rho_ma_app")

    gr_min = params.get("gr_min", 30.0)
    gr_max = params.get("gr_max", 150.0)
    rho_ma = params.get("rho_ma", 2.65)
    rho_f = params.get("rho_f", 1.0)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    if gr_col in df.columns:
        igr = (df[gr_col] - gr_min) / (gr_max - gr_min)
        igr = igr.clip(0, 1)
        df[vsh_col] = igr
    if den_col in df.columns:
        df[phid_col] = (rho_ma - df[den_col]) / (rho_ma - rho_f)
        df[rho_ma_app_col] = (df[den_col] - df.get(phid_col, 0) * rho_f) / (1 - df.get(phid_col, 0) + 1e-9)
    if cnl_col in df.columns:
        df["phi_n"] = (df[cnl_col] / 100.0 - (-0.04)) / (1.0 - (-0.04))

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 派生曲线计算完成，新增列: {vsh_col}, {phid_col}, {rho_ma_app_col}")
    return {"success": True, "message": "派生曲线计算完成", "output_files": [output_file]}


def normalize_curves(config: Dict, project_root: Path) -> Dict[str, Any]:
    """对指定曲线进行 Min-Max 或 Z-Score 归一化。"""
    print("\n===== 启动工具: feature_normalize_curves =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC"])
    method = params.get("method", "minmax")
    suffix = params.get("suffix", "_norm")

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    for curve in curves:
        if curve not in df.columns:
            continue
        series = df[curve]
        if method == "minmax":
            min_v, max_v = series.min(), series.max()
            df[f"{curve}{suffix}"] = (series - min_v) / (max_v - min_v + 1e-9)
        elif method == "zscore":
            mean, std = series.mean(), series.std()
            df[f"{curve}{suffix}"] = (series - mean) / (std + 1e-9)

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - {method} 归一化完成。")
    return {"success": True, "message": f"归一化完成，方法: {method}", "output_files": [output_file]}


def window_features(config: Dict, project_root: Path) -> Dict[str, Any]:
    """按深度窗计算曲线统计特征。"""
    print("\n===== 启动工具: feature_window_features =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    depth_col = params.get("depth_col", "DEPTH")
    curves = params.get("curves", ["GR", "DEN"])
    window = params.get("window", 11)
    stats = params.get("stats", ["mean", "std", "min", "max"])

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file).sort_values(by=depth_col)
    for curve in curves:
        if curve not in df.columns:
            continue
        for stat in stats:
            col_name = f"{curve}_win{window}_{stat}"
            if stat == "mean":
                df[col_name] = df[curve].rolling(window=window, center=True, min_periods=1).mean()
            elif stat == "std":
                df[col_name] = df[curve].rolling(window=window, center=True, min_periods=1).std()
            elif stat == "min":
                df[col_name] = df[curve].rolling(window=window, center=True, min_periods=1).min()
            elif stat == "max":
                df[col_name] = df[curve].rolling(window=window, center=True, min_periods=1).max()

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 深度窗特征计算完成，窗长: {window}。")
    return {"success": True, "message": f"深度窗特征计算完成，窗长: {window}", "output_files": [output_file]}
