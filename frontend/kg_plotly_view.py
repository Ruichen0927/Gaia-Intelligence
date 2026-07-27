"""知识图谱 Plotly 交互预览：可点击节点查看源码。

相比 matplotlib 静态图的优势：
- 矢量渲染，支持缩放/平移。
- 节点可点击，点击后返回工具 ID 给 Streamlit。
- 悬停显示工具描述与模块路径。
- 保留分组矩形背景、颜色、中文标签。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go


GROUP_ORDER = [
    "data", "feature", "viz", "stat", "ml",
    "shale", "porosity", "permeability", "saturation", "lithology", "final_report",
    "user", "other",
]

GROUP_COLORS = {
    "data": "#4C78A8",
    "feature": "#54A24B",
    "viz": "#Eeca3b",
    "stat": "#B279A2",
    "ml": "#E45756",
    "shale": "#F58518",
    "porosity": "#72B7B2",
    "permeability": "#79706E",
    "saturation": "#17BECF",
    "lithology": "#9D7660",
    "final_report": "#BCBD22",
    "user": "#FF6692",
    "other": "#999999",
}

GROUP_LABELS = {
    "data": "数据处理",
    "feature": "特征工程",
    "viz": "可视化",
    "stat": "统计分析",
    "ml": "机器学习",
    "shale": "泥质含量",
    "porosity": "孔隙度",
    "permeability": "渗透率",
    "saturation": "饱和度",
    "lithology": "岩性",
    "final_report": "最终报告",
    "user": "用户扩展",
    "other": "其他",
}


def _load_mcp_tools(project_root: Path) -> Dict[str, Dict]:
    cfg_path = project_root / "ctl" / "mcp_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("tools", {})


def _tool_group(tool_name: str, module: str = "") -> str:
    if module.startswith("extensions."):
        return "user"
    for prefix in GROUP_ORDER:
        if tool_name.startswith(prefix):
            return prefix
    return "other"


def _hex_to_rgba(hex_color: str, alpha: float = 0.10) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _darken(hex_color: str, factor: float = 0.75) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _grid_layout(nodes: List[str], x_step: float = 0.6, y_step: float = 0.45) -> Dict[str, Tuple[float, float]]:
    """将节点排成多行网格，返回相对坐标。"""
    n = len(nodes)
    cols = math.ceil(math.sqrt(n))
    pos = {}
    for idx, node in enumerate(nodes):
        row = idx // cols
        col = idx % cols
        x = (col - (cols - 1) / 2) * x_step
        y = ((math.ceil(n / cols) - 1) / 2 - row) * y_step
        pos[node] = (x, y)
    return pos


def _compute_layout(tools: Dict[str, Dict], tool_to_group: Dict[str, str]) -> Tuple[
    Dict[str, Tuple[float, float]],
    Dict[str, Tuple[float, float, float, float]],
    Dict[str, Tuple[float, float]],
]:
    """计算 Plotly 布局：组中心、组边界框、工具坐标。"""
    groups = sorted(set(tool_to_group.values()), key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)
    group_tools = {g: [t for t in tools if tool_to_group[t] == g] for g in groups}

    n_groups = len(groups)
    big_radius = 4.2
    group_centers = {}
    all_positions = {}

    for i, g in enumerate(groups):
        angle = 2 * math.pi * i / max(n_groups, 1) - math.pi / 2
        gx = big_radius * math.cos(angle)
        gy = big_radius * math.sin(angle)
        group_centers[g] = (gx, gy)
        local_pos = _grid_layout(group_tools[g])
        for tool, (lx, ly) in local_pos.items():
            all_positions[tool] = (gx + lx, gy + ly)

    # 计算分组边界框
    bounds = {}
    for g in groups:
        pts = [all_positions[t] for t in group_tools[g]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        padding = 0.45
        bounds[g] = (
            min(xs) - padding,
            min(ys) - padding,
            max(xs) + padding,
            max(ys) + padding,
        )

    return all_positions, bounds, group_centers


def _bezier_points(p0: Tuple[float, float], p1: Tuple[float, float], n: int = 30) -> Tuple[List[float], List[float]]:
    """计算二次贝塞尔曲线点列，用于绘制平滑边。"""
    x0, y0 = p0
    x1, y1 = p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    offset = 0.18 * length
    cx = mx - dy / length * offset
    cy = my + dx / length * offset

    xs, ys = [], []
    for i in range(n + 1):
        t = i / n
        x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
        y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
        xs.append(x)
        ys.append(y)
    return xs, ys


def build_plotly_graph(
    project_root: Path,
    highlight_tools: Optional[List[str]] = None,
    width: int = 1100,
    height: int = 900,
) -> go.Figure:
    """构建可点击的 Plotly 知识图谱。"""
    tools = _load_mcp_tools(project_root)
    tool_to_group = {name: _tool_group(name, info.get("module", "")) for name, info in tools.items()}
    positions, bounds, centers = _compute_layout(tools, tool_to_group)

    highlighted = set(highlight_tools) if highlight_tools else set(tools.keys())

    # 流程顺序边
    pipeline = json.load(open(project_root / "ctl" / "pipeline_config.json", encoding="utf-8"))
    order = [step["step"] for step in pipeline if step.get("enabled", True)]

    edges = []
    for i in range(len(order) - 1):
        src, dst = order[i], order[i + 1]
        if src in tools and dst in tools:
            edges.append((src, dst, "#888888", 2, "流程下一步"))

    # 同组 build/learn -> run/generate/calculation
    for tool in tools:
        if "build" in tool or "learn" in tool:
            g = tool_to_group[tool]
            for other in tools:
                if other != tool and tool_to_group[other] == g and ("run" in other or "calculation" in other or "generate" in other):
                    edges.append((tool, other, GROUP_COLORS.get(g, "#999999"), 2, "产出→消费"))

    # 跨组数据流
    for src, dst in [
        ("data_load_well_logs", "data_clean_curves"),
        ("data_clean_curves", "feature_compute_derived_curves"),
        ("feature_compute_derived_curves", "shale_run_interpretation"),
        ("feature_compute_derived_curves", "porosity_run_calculation"),
        ("feature_compute_derived_curves", "ml_train_regressor"),
        ("stat_summary", "viz_crossplot"),
    ]:
        if src in tools and dst in tools:
            edges.append((src, dst, "#A0A0A0", 1.5, "数据流"))

    fig = go.Figure()

    # 1. 分组背景框 + 标签
    annotations = []
    for g, (min_x, min_y, max_x, max_y) in bounds.items():
        color = GROUP_COLORS.get(g, "#999999")
        fig.add_shape(
            type="rect",
            x0=min_x, y0=min_y, x1=max_x, y1=max_y,
            fillcolor=_hex_to_rgba(color, 0.08),
            line=dict(color=color, width=1, dash="dash"),
            layer="below",
        )
        annotations.append(dict(
            x=(min_x + max_x) / 2,
            y=max_y + 0.15,
            text=GROUP_LABELS.get(g, g),
            showarrow=False,
            font=dict(size=12, color=color, family="Arial, sans-serif"),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        ))
    fig.update_layout(annotations=annotations)

    # 2. 边（每条边一个 trace，保证颜色与悬停正确）
    for src, dst, color, ew, title in edges:
        active = src in highlighted and dst in highlighted
        c = color if active else "#DDDDDD"
        w = ew if active else 0.8
        xs, ys = _bezier_points(positions[src], positions[dst])
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=c, width=w),
            hoverinfo="text",
            text=[title] * len(xs),
            showlegend=False,
            name=f"edge_{src}_{dst}",
            hoverlabel=dict(bgcolor=c),
        ))

    # 3. 工具节点
    node_x, node_y, node_colors, node_sizes, node_texts, node_labels, node_ids = [], [], [], [], [], [], []
    for tool in tools:
        g = tool_to_group[tool]
        color = GROUP_COLORS.get(g, "#999999")
        active = tool in highlighted
        x, y = positions[tool]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(color if active else "#DDDDDD")
        node_sizes.append(14 if active else 8)
        node_texts.append(f"{tool}<br>{tools[tool].get('description', '')}<br>模块: {tools[tool].get('module')}.py")
        node_labels.append(tool.replace("_", "<br>"))
        node_ids.append(tool)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(color=[_darken(c) for c in node_colors], width=1),
        ),
        text=node_labels,
        textposition="bottom center",
        textfont=dict(size=8, color="#333333", family="Arial, sans-serif"),
        hoverinfo="text",
        hovertext=node_texts,
        customdata=node_ids,
        showlegend=False,
        name="tools",
        selected=dict(marker=dict(color="#F7A7A6", size=18)),
    ))

    # 4. 布局配置
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-5.5, 5.5], fixedrange=False),
        yaxis=dict(visible=False, range=[-5.5, 5.5], fixedrange=False, scaleanchor="x", scaleratio=1),
        hovermode="closest",
        clickmode="event+select",
        dragmode="pan",
        title=dict(text="点击节点查看源码", font=dict(size=14, color="#666666")),
        uirevision="gaia_kg",
    )

    return fig
