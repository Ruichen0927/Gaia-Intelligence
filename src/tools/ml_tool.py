"""机器学习工具包：回归、分类、聚类、评估与模型持久化。"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common.config_loader import ensure_dirs


REGRESSORS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "svr": SVR,
}

CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def _load_xy(input_file: str, feature_columns: list, target_column: str, drop_na: bool = True):
    df = pd.read_csv(input_file)
    cols = [c for c in feature_columns if c in df.columns]
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在")
    if len(cols) == 0:
        raise ValueError("没有可用的特征列")
    sub = df[cols + [target_column]].copy()
    if drop_na:
        sub.dropna(inplace=True)
    X = sub[cols]
    y = sub[target_column]
    return X, y, cols


def train_regressor(config: Dict, project_root: Path) -> Dict[str, Any]:
    """训练回归模型。"""
    print("\n===== 启动工具: ml_train_regressor =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_model_file = paths["output_model_file"]
    output_metrics_file = paths.get("output_metrics_file")

    feature_columns = params.get("feature_columns", [])
    target_column = params.get("target_column", "phie_final")
    algorithm = params.get("algorithm", "random_forest")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    algorithm_params = params.get("algorithm_params", {})

    ensure_dirs(os.path.dirname(output_model_file))

    X, y, cols = _load_xy(input_file, feature_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    ModelClass = REGRESSORS.get(algorithm, RandomForestRegressor)
    model = ModelClass(**algorithm_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }

    with open(output_model_file, "wb") as f:
        pickle.dump({"model": model, "feature_columns": cols, "target_column": target_column}, f)

    output_files = [output_model_file]
    if output_metrics_file:
        ensure_dirs(os.path.dirname(output_metrics_file))
        with open(output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        output_files.append(output_metrics_file)

    print(f"  - 回归模型训练完成: {algorithm}，R²={metrics['r2']}")
    return {"success": True, "message": f"回归模型训练完成，R²={metrics['r2']}", "output_files": output_files}


def predict(config: Dict, project_root: Path) -> Dict[str, Any]:
    """使用训练好的回归模型预测。"""
    print("\n===== 启动工具: ml_predict =====")
    paths = config["paths"]

    input_file = paths["input_file"]
    model_file = paths["model_file"]
    output_file = paths["output_file"]

    ensure_dirs(os.path.dirname(output_file))

    with open(model_file, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_column = bundle.get("target_column", "prediction")

    df = pd.read_csv(input_file)
    missing = df[feature_columns].isna().any(axis=1)
    X = df.loc[~missing, feature_columns]
    pred = model.predict(X)
    df.loc[~missing, f"{target_column}_pred"] = pred

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 预测完成，输出: {output_file}")
    return {"success": True, "message": "回归预测完成", "output_files": [output_file]}


def evaluate_regressor(config: Dict, project_root: Path) -> Dict[str, Any]:
    """评估回归预测结果。"""
    print("\n===== 启动工具: ml_evaluate =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    actual_col = params.get("actual_col", "phie_final")
    pred_col = params.get("pred_col", "phie_final_pred")

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    sub = df[[actual_col, pred_col]].dropna()
    metrics = {
        "rmse": round(np.sqrt(mean_squared_error(sub[actual_col], sub[pred_col])), 4),
        "mae": round(mean_absolute_error(sub[actual_col], sub[pred_col]), 4),
        "r2": round(r2_score(sub[actual_col], sub[pred_col]), 4),
        "samples": len(sub),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  - 回归评估完成，R²={metrics['r2']}")
    return {"success": True, "message": f"回归评估完成，R²={metrics['r2']}", "output_files": [output_file]}


def train_classifier(config: Dict, project_root: Path) -> Dict[str, Any]:
    """训练分类模型。"""
    print("\n===== 启动工具: ml_train_classifier =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_model_file = paths["output_model_file"]
    output_metrics_file = paths.get("output_metrics_file")

    feature_columns = params.get("feature_columns", [])
    target_column = params.get("target_column", "rock_type")
    algorithm = params.get("algorithm", "random_forest")
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)
    algorithm_params = params.get("algorithm_params", {})

    ensure_dirs(os.path.dirname(output_model_file))

    X, y, cols = _load_xy(input_file, feature_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    ModelClass = CLASSIFIERS.get(algorithm, RandomForestClassifier)
    model = ModelClass(**algorithm_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open(output_model_file, "wb") as f:
        pickle.dump({"model": model, "feature_columns": cols, "target_column": target_column}, f)

    output_files = [output_model_file]
    if output_metrics_file:
        ensure_dirs(os.path.dirname(output_metrics_file))
        with open(output_metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        output_files.append(output_metrics_file)

    print(f"  - 分类模型训练完成: {algorithm}，acc={metrics['accuracy']}")
    return {"success": True, "message": f"分类模型训练完成，acc={metrics['accuracy']}", "output_files": output_files}


def predict_classifier(config: Dict, project_root: Path) -> Dict[str, Any]:
    """使用训练好的分类模型预测。"""
    print("\n===== 启动工具: ml_predict_classifier =====")
    paths = config["paths"]

    input_file = paths["input_file"]
    model_file = paths["model_file"]
    output_file = paths["output_file"]

    ensure_dirs(os.path.dirname(output_file))

    with open(model_file, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    target_column = bundle.get("target_column", "prediction")

    df = pd.read_csv(input_file)
    missing = df[feature_columns].isna().any(axis=1)
    X = df.loc[~missing, feature_columns]
    pred = model.predict(X)
    df.loc[~missing, f"{target_column}_pred"] = pred

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 分类预测完成，输出: {output_file}")
    return {"success": True, "message": "分类预测完成", "output_files": [output_file]}


def evaluate_classifier(config: Dict, project_root: Path) -> Dict[str, Any]:
    """评估分类预测结果。"""
    print("\n===== 启动工具: ml_evaluate_classifier =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]
    actual_col = params.get("actual_col", "rock_type")
    pred_col = params.get("pred_col", "rock_type_pred")

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    sub = df[[actual_col, pred_col]].dropna()
    y_true, y_pred = sub[actual_col], sub[pred_col]
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "samples": len(sub),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  - 分类评估完成，acc={metrics['accuracy']}")
    return {"success": True, "message": f"分类评估完成，acc={metrics['accuracy']}", "output_files": [output_file]}


def cluster_data(config: Dict, project_root: Path) -> Dict[str, Any]:
    """对曲线数据进行聚类。"""
    print("\n===== 启动工具: ml_cluster_data =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    input_file = paths["input_file"]
    output_file = paths["output_file"]

    feature_columns = params.get("feature_columns", [])
    algorithm = params.get("algorithm", "kmeans")
    n_clusters = params.get("n_clusters", 5)
    eps = params.get("eps", 0.5)
    min_samples = params.get("min_samples", 5)
    random_state = params.get("random_state", 42)

    ensure_dirs(os.path.dirname(output_file))

    df = pd.read_csv(input_file)
    cols = [c for c in feature_columns if c in df.columns]
    if len(cols) == 0:
        return {"success": False, "message": "没有可用的特征列", "output_files": []}

    sub = df[cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(sub)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    elif algorithm == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(X)
    else:
        return {"success": False, "message": f"不支持的聚类算法: {algorithm}", "output_files": []}

    df.loc[sub.index, "cluster_label"] = labels
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    summary = pd.Series(labels).value_counts().to_dict()
    print(f"  - 聚类完成，算法: {algorithm}，类别分布: {summary}")
    return {"success": True, "message": f"聚类完成，算法: {algorithm}", "output_files": [output_file]}


def save_load_model(config: Dict, project_root: Path) -> Dict[str, Any]:
    """模型持久化辅助工具：save / load。"""
    print("\n===== 启动工具: ml_save_load_model =====")
    paths = config["paths"]
    params = config.get("parameters", {})

    action = params.get("action", "save")
    model_file = paths["model_file"]

    if action == "save":
        input_file = paths.get("input_file")
        if not input_file:
            return {"success": False, "message": "保存模型时需要 input_file", "output_files": []}
        ensure_dirs(os.path.dirname(model_file))
        df = pd.read_csv(input_file)
        with open(model_file, "wb") as f:
            pickle.dump({"placeholder": True, "data_shape": df.shape}, f)
        return {"success": True, "message": "模型占位保存完成", "output_files": [model_file]}
    elif action == "load":
        with open(model_file, "rb") as f:
            bundle = pickle.load(f)
        return {"success": True, "message": "模型加载完成", "output_files": [model_file], "bundle": str(type(bundle))}
    else:
        return {"success": False, "message": f"不支持的动作: {action}", "output_files": []}
