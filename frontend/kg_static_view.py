"""知识图谱静态高清预览：使用 matplotlib + networkx 绘制真正的嵌套分组图。

用于：
- 验证层级/放射布局的嵌套效果。
- 在 streamlit-agraph 渲染受限时提供高质量兜底视图。
- 导出 PNG/SVG 静态图。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# 尝试设置支持中文的字体
for _chinese_font in ["WenQuanYi Micro Hei", "Noto Sans CJK JP", "SimHei", "Microsoft YaHei"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [_chinese_font]
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def _load_mcp_tools(project_root: Path) -> Dict[str, Dict]:
    cfg_path = project_root / "ctl" / "mcp_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("tools", {})


def _tool_group(tool_name: str, module: str = "") -> str:
    GROUP_ORDER = [
        "data", "feature", "viz", "stat", "ml",
        "shale", "porosity", "permeability", "saturation", "lithology", "final_report",
        "user", "other",
    ]
    if module.startswith("extensions."):
        return "user"
    for prefix in GROUP_ORDER:
        if tool_name.startswith(prefix):
            return prefix
    return "other"


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


def _compute_group_bounds(positions: Dict[str, Tuple[float, float]], padding: float = 0.12) -> Dict[str, Tuple[float, float, float, float]]:
    """根据组内工具坐标计算包围盒 (min_x, min_y, max_x, max_y)，并扩展 padding。"""
    bounds = {}
    for group, pts in positions.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w, h = max_x - min_x, max_y - min_y
        bounds[group] = (
            min_x - padding * w - 0.08,
            min_y - padding * h - 0.08,
            max_x + padding * w + 0.08,
            max_y + padding * h + 0.08,
        )
    return bounds


def _draw_curved_edge(ax, src_pos, dst_pos, color="#888888", lw=1.0, arrow=True):
    """绘制带小箭头的贝塞尔曲线。"""
    x1, y1 = src_pos
    x2, y2 = dst_pos
    # 控制点：中垂方向偏移，形成轻微曲线
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    offset = 0.15 * length
    cx = mx - dy / length * offset
    cy = my + dx / length * offset

    t = 0.55  # 箭头位置
    px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
    py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2

    # 用多段线近似二次贝塞尔
    ts = [i / 40 for i in range(41)]
    xs = [(1 - tt) ** 2 * x1 + 2 * (1 - tt) * tt * cx + tt ** 2 * x2 for tt in ts]
    ys = [(1 - tt) ** 2 * y1 + 2 * (1 - tt) * tt * cy + tt ** 2 * y2 for tt in ts]
    ax.plot(xs, ys, color=color, lw=lw, zorder=1)

    if arrow:
        # 在 px,py 处绘制小箭头
        dx_arr = x2 - px
        dy_arr = y2 - py
        arr_len = math.hypot(dx_arr, dy_arr) or 1.0
        head_len = 0.025
        angle = math.atan2(dy_arr, dx_arr)
        ax.plot(
            [px, px - head_len * math.cos(angle - 0.35), px - head_len * math.cos(angle + 0.35), px],
            [py, py - head_len * math.sin(angle - 0.35), py - head_len * math.sin(angle + 0.35), py],
            color=color, lw=lw, zorder=1,
        )


def build_static_graph(
    project_root: Path,
    highlight_tools: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 14),
    dpi: int = 120,
) -> plt.Figure:
    """构建并返回 matplotlib Figure 对象。"""
    tools = _load_mcp_tools(project_root)
    tool_to_group = {name: _tool_group(name, info.get("module", "")) for name, info in tools.items()}

    groups = sorted(set(tool_to_group.values()), key=lambda g: list(GROUP_COLORS).index(g) if g in GROUP_COLORS else 999)
    group_tools = {g: [t for t in tools if tool_to_group[t] == g] for g in groups}

    # 用 networkx 计算组内布局（小节点多则使用网格，避免圆形拥挤）
    all_positions: Dict[str, Tuple[float, float]] = {}

    def _grid_layout(nodes: List[str], x_step: float = 0.55, y_step: float = 0.45) -> Dict[str, Tuple[float, float]]:
        """将节点排成多行网格，返回相对坐标。"""
        n = len(nodes)
        cols = math.ceil(math.sqrt(n))
        pos = {}
        for idx, node in enumerate(nodes):
            row = idx // cols
            col = idx % cols
            # 居中
            x = (col - (cols - 1) / 2) * x_step
            y = ((math.ceil(n / cols) - 1) / 2 - row) * y_step
            pos[node] = (x, y)
        return pos

    # 将各组排列在一个大圆上
    n_groups = len(groups)
    big_radius = 4.0
    group_centers = {}
    for i, g in enumerate(groups):
        angle = 2 * math.pi * i / max(n_groups, 1) - math.pi / 2
        gx = big_radius * math.cos(angle)
        gy = big_radius * math.sin(angle)
        group_centers[g] = (gx, gy)
        local_pos = _grid_layout(group_tools[g])
        for tool, (lx, ly) in local_pos.items():
            all_positions[tool] = (gx + lx, gy + ly)

    # 创建主图
    G = nx.DiGraph()
    G.add_nodes_from(tools.keys())

    # 添加流程顺序边
    order = [step["step"] for step in json.load(open(project_root / "ctl" / "pipeline_config.json", encoding="utf-8")) if step.get("enabled", True)]
    for i in range(len(order) - 1):
        src, dst = order[i], order[i + 1]
        if src in tools and dst in tools:
            G.add_edge(src, dst, kind="flow")

    # 添加同组 build->run 边
    for tool in tools:
        if "build" in tool or "learn" in tool:
            g = tool_to_group[tool]
            for other in tools:
                if other != tool and tool_to_group[other] == g and ("run" in other or "calculation" in other or "generate" in other):
                    G.add_edge(tool, other, kind="internal")

    # 添加核心跨组数据流边
    for src, dst in [
        ("data_load_well_logs", "data_clean_curves"),
        ("data_clean_curves", "feature_compute_derived_curves"),
        ("feature_compute_derived_curves", "shale_run_interpretation"),
        ("feature_compute_derived_curves", "porosity_run_calculation"),
        ("feature_compute_derived_curves", "ml_train_regressor"),
        ("stat_summary", "viz_crossplot"),
    ]:
        if src in tools and dst in tools:
            G.add_edge(src, dst, kind="dataflow")

    highlighted = set(highlight_tools) if highlight_tools else set(tools.keys())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    # 绘制分组背景框
    bounds = _compute_group_bounds({g: [all_positions[t] for t in group_tools[g]] for g in groups})
    for g in groups:
        if g not in bounds:
            continue
        min_x, min_y, max_x, max_y = bounds[g]
        color = GROUP_COLORS.get(g, "#999999")
        box = FancyBboxPatch(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=_hex_to_rgba(color, 0.08),
            edgecolor=color,
            linewidth=1.5,
            linestyle="--",
            zorder=0,
        )
        ax.add_patch(box)
        # 组标签放在框上方
        ax.text(
            (min_x + max_x) / 2,
            max_y + 0.12,
            GROUP_LABELS.get(g, g),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.9),
            zorder=3,
        )

    # 绘制边
    for src, dst, data in G.edges(data=True):
        kind = data.get("kind", "flow")
        active = src in highlighted and dst in highlighted
        if kind == "flow":
            color = "#888888" if active else "#DDDDDD"
            lw = 1.2 if active else 0.6
        elif kind == "internal":
            g = tool_to_group[src]
            color = GROUP_COLORS.get(g, "#999999") if active else "#DDDDDD"
            lw = 1.0 if active else 0.5
        else:  # dataflow
            color = "#A0A0A0" if active else "#DDDDDD"
            lw = 0.9 if active else 0.5
        _draw_curved_edge(ax, all_positions[src], all_positions[dst], color=color, lw=lw, arrow=True)

    # 绘制节点
    for tool in tools:
        group = tool_to_group[tool]
        color = GROUP_COLORS.get(group, "#999999")
        is_active = tool in highlighted
        node_color = color if is_active else "#DDDDDD"
        border_color = _darken(color) if is_active else "#BBBBBB"
        x, y = all_positions[tool]
        ax.scatter(
            x, y,
            s=200 if is_active else 80,
            c=node_color,
            edgecolors=border_color,
            linewidths=1.5,
            zorder=2,
        )
        # 标签带白色背景，避免被边/框覆盖
        ax.text(
            x, y - 0.18,
            tool.replace("_", "\n"),
            ha="center",
            va="top",
            fontsize=7 if is_active else 5,
            color="#222222" if is_active else "#AAAAAA",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=3,
        )

    ax.autoscale_view(tight=True)
    fig.tight_layout(pad=0.5)
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> Tuple[float, float, float, float]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r / 255, g / 255, b / 255, alpha)


def _darken(hex_color: str, factor: float = 0.7) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"
