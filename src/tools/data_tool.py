"""数据处理工具包：加载、清洗、异常检测、缺失值填充、拆分、合并、质检。"""

import json
import glob
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common.config_loader import ensure_dirs


def _get_curve_names(params: Dict) -> Dict[str, str]:
    return params.get("curve_names", {
        "GR": "GR", "CGR": "CGR", "DEN": "DEN", "CNL": "CNL",
        "AC": "AC", "PE": "PE", "LAYER": "LAYER", "SH": "SH",
        "DEPTH": "DEPTH", "CAL": "CAL", "BIT": "BIT"
    })


def load_well_logs(config: Dict, project_root: Path) -> Dict[str, Any]:
    """批量读取测井文本文件，输出合并 CSV 与单井 CSV。"""
    print("\n===== 启动工具: data_load_well_logs =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_directory = paths["input_directory"]
    output_directory = paths["output_directory"]
    file_extension = params.get("file_extension", ".txt")
    output_filename = params.get("merged_filename", "merged_well_logs.csv")
    curve_names = _get_curve_names(params)
    core_curves = params.get("core_curves", ["GR", "DEN", "CNL", "AC", "LAYER"])
    core_curve_cols = [curve_names.get(c, c) for c in core_curves]

    ensure_dirs(output_directory)

    all_files = sorted(glob.glob(os.path.join(input_directory, f"*{file_extension}")))
    if not all_files:
        return {"success": False, "message": f"输入目录 '{input_directory}' 中未找到 *{file_extension} 文件。", "output_files": []}

    df_list = []
    exported = []
    for f in all_files:
        well_name = Path(f).stem
        try:
            df = pd.read_csv(f, header=0, skiprows=[1], sep=',')
        except Exception as e:
            return {"success": False, "message": f"读取文件 {f} 失败: {e}", "output_files": []}
        df["WELL_NAME"] = well_name
        df_list.append(df)

        single_path = os.path.join(output_directory, f"{well_name}.csv")
        df.to_csv(single_path, index=False, encoding='utf-8-sig')
        exported.append(single_path)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_path = os.path.join(output_directory, output_filename)
    merged_df.to_csv(merged_path, index=False, encoding='utf-8-sig')
    exported.insert(0, merged_path)

    print(f"  - 成功加载 {len(all_files)} 口井，合并后共 {len(merged_df)} 行。")
    return {"success": True, "message": f"加载 {len(all_files)} 口井数据", "output_files": exported}


def clean_curves(config: Dict, project_root: Path) -> Dict[str, Any]:
    """替换无效值、类型转换、按核心曲线 dropna。"""
    print("\n===== 启动工具: data_clean_curves =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curve_names = _get_curve_names(params)
    numeric_curves = params.get("numeric_curves", ["GR", "CGR", "DEN", "CNL", "AC", "PE", "CAL", "BIT"])
    invalid_values = params.get("invalid_values", [-99999.0, -9999.0, -999.25])
    core_curves = params.get("core_curves", ["GR", "DEN", "CNL", "AC", "LAYER"])
    core_curve_cols = [curve_names.get(c, c) for c in core_curves]

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    for col in df.columns:
        if col in [curve_names.get(c, c) for c in numeric_curves]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.replace(invalid_values, np.nan, inplace=True)
    before = len(df)
    df.dropna(subset=[c for c in core_curve_cols if c in df.columns], inplace=True)
    after = len(df)

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 清洗前 {before} 行，清洗后 {after} 行。")
    return {"success": True, "message": f"清洗完成，移除 {before - after} 行", "output_files": [output_file]}


def detect_outliers(config: Dict, project_root: Path) -> Dict[str, Any]:
    """使用 IQR 或 Z-Score 检测异常值并标记。"""
    print("\n===== 启动工具: data_detect_outliers =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC"])
    method = params.get("method", "iqr")
    z_threshold = params.get("z_threshold", 3.0)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    outlier_mask = pd.Series(False, index=df.index)
    for curve in curves:
        if curve not in df.columns:
            continue
        series = df[curve].dropna()
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[curve] < lower) | (df[curve] > upper)
        else:
            mean, std = series.mean(), series.std()
            mask = np.abs((df[curve] - mean) / (std + 1e-9)) > z_threshold
        outlier_mask = outlier_mask | mask
        df[f"{curve}_outlier"] = mask.astype(int)

    df["is_outlier"] = outlier_mask.astype(int)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 检测到 {outlier_mask.sum()} 个异常点。")
    return {"success": True, "message": f"异常检测完成，异常点 {outlier_mask.sum()} 个", "output_files": [output_file]}


def impute_missing(config: Dict, project_root: Path) -> Dict[str, Any]:
    """缺失值填充：插值、均值、中位数。"""
    print("\n===== 启动工具: data_impute_missing =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC"])
    method = params.get("method", "interpolation")

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    for curve in curves:
        if curve not in df.columns:
            continue
        if method == "interpolation":
            df[curve] = df[curve].interpolate(method='linear', limit_direction='both')
        elif method == "mean":
            df[curve] = df[curve].fillna(df[curve].mean())
        elif method == "median":
            df[curve] = df[curve].fillna(df[curve].median())
        elif method == "forward":
            df[curve] = df[curve].ffill().bfill()

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 使用 {method} 方法填充缺失值。")
    return {"success": True, "message": f"缺失值填充完成，方法: {method}", "output_files": [output_file]}


def split_by_formation(config: Dict, project_root: Path) -> Dict[str, Any]:
    """按 LAYER 列拆分成多个层段 CSV。"""
    print("\n===== 启动工具: data_split_by_formation =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_directory = paths["output_directory"]
    layer_col = params.get("layer_col", "LAYER")

    ensure_dirs(output_directory)

    df = pd.read_csv(input_file)
    if layer_col not in df.columns:
        return {"success": False, "message": f"列 '{layer_col}' 不存在", "output_files": []}

    exported = []
    for layer, group in df.groupby(layer_col):
        out_path = os.path.join(output_directory, f"layer_{layer}.csv")
        group.to_csv(out_path, index=False, encoding='utf-8-sig')
        exported.append(out_path)

    print(f"  - 拆分为 {len(exported)} 个层段文件。")
    return {"success": True, "message": f"按层拆分完成，共 {len(exported)} 层", "output_files": exported}


def merge_well_data(config: Dict, project_root: Path) -> Dict[str, Any]:
    """合并多井 interpreted CSV。"""
    print("\n===== 启动工具: data_merge_well_data =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_directory = paths["input_directory"]
    output_file = paths["output_file"]
    pattern = params.get("pattern", "*.csv")

    ensure_dirs(os.path.dirname(output_file))

    files = sorted(glob.glob(os.path.join(input_directory, pattern)))
    if not files:
        return {"success": False, "message": f"未找到 {pattern} 文件", "output_files": []}

    df_list = [pd.read_csv(f) for f in files]
    merged = pd.concat(df_list, ignore_index=True)
    merged.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 合并 {len(files)} 个文件，共 {len(merged)} 行。")
    return {"success": True, "message": f"合并 {len(files)} 个文件", "output_files": [output_file]}


def data_qc_report(config: Dict, project_root: Path) -> Dict[str, Any]:
    """生成数据质量报告。"""
    print("\n===== 启动工具: data_qc_report =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    curves = params.get("curves", ["GR", "DEN", "CNL", "AC", "PE"])

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    report = {"total_rows": len(df), "wells": sorted(df.get("WELL_NAME", pd.Series()).unique().tolist())}
    curve_stats = {}
    for curve in curves:
        if curve not in df.columns:
            continue
        series = df[curve]
        valid = series.notna()
        curve_stats[curve] = {
            "missing_count": int((~valid).sum()),
            "missing_rate": round((~valid).mean(), 4),
            "min": round(series.min(), 4) if valid.any() else None,
            "max": round(series.max(), 4) if valid.any() else None,
            "mean": round(series.mean(), 4) if valid.any() else None,
            "std": round(series.std(), 4) if valid.any() else None,
        }
    report["curve_stats"] = curve_stats

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  - 数据质量报告已生成: {output_file}")
    return {"success": True, "message": "数据质量报告生成完成", "output_files": [output_file]}
