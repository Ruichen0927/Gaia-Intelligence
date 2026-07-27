"""知识图谱生成器：使用 streamlit-agraph 可视化 MCP 工具层级关联。

特性：
- 三层嵌套结构：工作阶段（Stage）→ 功能组（Group）→ 工具（Tool）。
- 支持三种布局：
  * 层级布局（默认）：从左到右展示 Stage → Group → Tool 的 workflow。
  * 放射簇布局：以中心为锚，功能组呈花瓣状分布，工具围绕组中心成团。
  * 力导向布局：保留物理仿真，便于自由探索。
- 包含边（细灰线、无箭头）表达嵌套；调用/数据流边（彩色曲线、小箭头）表达依赖。
- 点击工具节点可将工具 ID 返回给 Streamlit，用于展示源码。
- 搜索/筛选时只保留命中分支及其祖先/邻居。
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from streamlit_agraph import Node, Edge, Config
except ImportError:
    Node = Edge = Config = None


# ----------------------------- 分组与阶段定义 -----------------------------

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

# 阶段 = 工作流大阶段，每个组归属一个阶段
STAGE_OF_GROUP = {
    "data": "数据准备",
    "feature": "数据准备",
    "shale": "地质解释",
    "porosity": "地质解释",
    "permeability": "地质解释",
    "saturation": "地质解释",
    "lithology": "地质解释",
    "stat": "分析洞察",
    "ml": "分析洞察",
    "viz": "分析洞察",
    "final_report": "成果输出",
    "user": "用户扩展",
    "other": "其他",
}

STAGE_ORDER = ["数据准备", "地质解释", "分析洞察", "成果输出", "用户扩展", "其他"]

STAGE_COLORS = {
    "数据准备": "#2E5AAC",
    "地质解释": "#2E8B57",
    "分析洞察": "#8B4513",
    "成果输出": "#6B4C9A",
    "用户扩展": "#C41E3A",
    "其他": "#777777",
}


# ----------------------------- 工具加载 -----------------------------

def _load_mcp_tools(project_root: Path) -> Dict[str, Dict]:
    cfg_path = project_root / "ctl" / "mcp_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("tools", {})


def _load_pipeline_order(project_root: Path) -> List[str]:
    cfg_path = project_root / "ctl" / "pipeline_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        pipeline = json.load(f)
    return [step["step"] for step in pipeline if step.get("enabled", True)]


def _read_tool_sources(project_root: Path, tools: Dict[str, Dict]) -> Dict[str, Dict[str, str]]:
    sources = {}
    for tool_name, info in tools.items():
        module = info.get("module", "")
        if module.startswith("extensions.tools."):
            module_name = module.split(".", 2)[2]
            src_path = project_root / "extensions" / "tools" / f"{module_name}.py"
        elif module.startswith("extensions."):
            module_name = module.split(".", 1)[1]
            src_path = project_root / "extensions" / "tools" / f"{module_name}.py"
        else:
            src_path = project_root / "src" / "tools" / f"{module}.py"
        try:
            code = src_path.read_text(encoding="utf-8")
        except Exception as e:
            code = f"# 无法读取源码: {e}"
            src_path = Path(module)
        sources[tool_name] = {"path": str(src_path.relative_to(project_root)), "code": code}
    return sources


def _tool_group(tool_name: str, module: str = "") -> str:
    """根据工具名前缀或模块路径判断功能组。"""
    if module.startswith("extensions.") or module.startswith("extensions.tools."):
        return "user"
    for prefix in GROUP_ORDER:
        if tool_name.startswith(prefix):
            return prefix
    return "other"


def load_tool_info(project_root: Path) -> Dict[str, Dict]:
    """加载所有工具的元信息，返回 {tool_name: {...}}。"""
    tools = _load_mcp_tools(project_root)
    sources = _read_tool_sources(project_root, tools)
    info = {}
    for name, cfg in tools.items():
        group = _tool_group(name, cfg.get("module", ""))
        info[name] = {
            "name": name,
            "description": cfg.get("description", ""),
            "module": cfg.get("module", ""),
            "default_config": cfg.get("default_config", ""),
            "group": group,
            "stage": STAGE_OF_GROUP.get(group, "其他"),
            "source_path": sources[name]["path"],
            "source_code": sources[name]["code"],
        }
    return info


# ----------------------------- 辅助函数 -----------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """将 #RRGGBB 转换为 rgba(R,G,B,a)。"""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _darken(hex_color: str, factor: float = 0.75) -> str:
    """将 #RRGGBB 颜色按 factor 加深。"""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        r, g, b = [int(c * 2, 16) for c in hex_color]
    else:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _short_label(tool_name: str, max_len: int = 20) -> str:
    """生成紧凑标签。"""
    label = tool_name.replace("_", " ")
    if len(label) > max_len:
        label = label[: max_len - 1] + "…"
    return label


def _sunflower_positions(n: int, radius: float) -> List[Tuple[float, float]]:
    """黄金角螺旋，在圆内均匀分布 n 个点。"""
    positions = []
    golden_angle = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        r = radius * math.sqrt((i + 0.5) / max(n, 1))
        theta = i * golden_angle
        positions.append((r * math.cos(theta), r * math.sin(theta)))
    return positions


def is_group_node(node_id: str) -> bool:
    """判断节点 ID 是否为功能组容器节点。"""
    return isinstance(node_id, str) and node_id.startswith("__group__")


def is_stage_node(node_id: str) -> bool:
    """判断节点 ID 是否为阶段容器节点。"""
    return isinstance(node_id, str) and node_id.startswith("__stage__")


def is_tool_node(node_id: str) -> bool:
    """判断节点 ID 是否为真实工具节点。"""
    return not is_group_node(node_id) and not is_stage_node(node_id)


# ----------------------------- 放射布局计算 -----------------------------

def _compute_radial_layout(
    groups: List[str],
    group_tools: Dict[str, List[str]],
    width: float = 1200,
    height: float = 800,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """计算放射簇布局坐标。

    策略：
    - 画布中心为原点，各功能组均匀分布在大圆上。
    - 组内工具围绕组中心按 sunflower 分布。
    - 工具数多的组获得稍大的分布半径。
    """
    cx, cy = width / 2, height / 2
    n_groups = len(groups)
    big_radius = min(width, height) * 0.32

    group_radii = {g: 38 + 16 * math.sqrt(max(len(group_tools.get(g, [])), 1)) for g in groups}
    group_positions: Dict[str, Tuple[float, float]] = {}
    tool_positions: Dict[str, Tuple[float, float]] = {}

    for i, group in enumerate(groups):
        angle = 2 * math.pi * i / max(n_groups, 1) - math.pi / 2
        gx = cx + big_radius * math.cos(angle)
        gy = cy + big_radius * math.sin(angle)
        group_positions[group] = (gx, gy)

        tools = group_tools.get(group, [])
        if not tools:
            continue
        local_radius = group_radii[group] * 0.75
        positions = _sunflower_positions(len(tools), local_radius)
        # 将第一个节点（通常是入口）旋转到朝外方向，便于跨组边连接
        rotation = angle + math.pi / 2
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for x, y in positions]
        for name, (dx, dy) in zip(tools, rotated):
            tool_positions[name] = (gx + dx, gy + dy)

    return group_positions, tool_positions, group_radii


# ----------------------------- 边构建 -----------------------------

def _build_edges(
    tools: Dict[str, Dict],
    order: List[str],
    tool_to_group: Dict[str, str],
    highlighted: Set[str],
) -> List[Edge]:
    """构建包含边与调用/数据流边。"""
    edges = []
    edge_set: Set[Tuple[str, str]] = set()

    def _add_edge(
        src: str,
        dst: str,
        title: str,
        color_active: str,
        width: float = 1.0,
        arrows: Optional[Dict] = None,
        dashes: Optional[List[int]] = None,
    ):
        if (src, dst) in edge_set:
            return
        if src not in tools and not is_group_node(src) and not is_stage_node(src):
            return
        if dst not in tools and not is_group_node(dst) and not is_stage_node(dst):
            return
        edge_set.add((src, dst))
        active = src in highlighted and dst in highlighted
        kwargs = {}
        if arrows is not None:
            kwargs["arrows"] = arrows
        if dashes is not None:
            kwargs["dashes"] = dashes
        edges.append(Edge(
            source=src,
            target=dst,
            color=color_active if active else "#DDDDDD",
            width=width if active else 0.4,
            title=title,
            type="CURVE_SMOOTH",
            **kwargs,
        ))

    # 1) 包含边：Stage -> Group -> Tool
    group_to_stage = {g: STAGE_OF_GROUP.get(g, "其他") for g in set(tool_to_group.values())}
    for group, stage in group_to_stage.items():
        _add_edge(
            f"__stage__{stage}",
            f"__group__{group}",
            f"阶段：{stage}",
            "#CCCCCC",
            width=0.6,
            arrows={"to": {"enabled": False}},
        )
    for tool_name, group in tool_to_group.items():
        _add_edge(
            f"__group__{group}",
            tool_name,
            f"功能组：{GROUP_LABELS.get(group, group)}",
            GROUP_COLORS.get(group, "#999999"),
            width=0.7,
            arrows={"to": {"enabled": False}},
        )

    # 2) 流程顺序边（仅解释主流程）
    for i in range(len(order) - 1):
        src, dst = order[i], order[i + 1]
        if src in tools and dst in tools:
            _add_edge(src, dst, "流程下一步", "#888888", 1.0, arrows={"to": {"enabled": True, "scaleFactor": 0.4}})

    # 3) 同领域 build/learn -> run/generate/calculation 强关联
    for tool_name in tools:
        if "build" in tool_name or "learn" in tool_name:
            group = tool_to_group[tool_name]
            for other in tools:
                if other == tool_name or tool_to_group[other] != group:
                    continue
                if "run" in other or "calculation" in other or "generate" in other:
                    _add_edge(
                        tool_name,
                        other,
                        "产出→消费",
                        GROUP_COLORS.get(group, "#999999"),
                        1.0,
                        arrows={"to": {"enabled": True, "scaleFactor": 0.45}},
                    )

    # 4) 跨组数据流：只保留最核心依赖，避免“全连接”
    dataflow_edges = [
        ("data_load_well_logs", "data_clean_curves"),
        ("data_clean_curves", "feature_compute_derived_curves"),
        ("feature_compute_derived_curves", "shale_run_interpretation"),
        ("feature_compute_derived_curves", "porosity_run_calculation"),
        ("feature_compute_derived_curves", "ml_train_regressor"),
        ("stat_summary", "viz_crossplot"),
    ]
    for src, dst in dataflow_edges:
        if src in tools and dst in tools:
            _add_edge(
                src,
                dst,
                "数据流",
                "#A0A0A0",
                0.8,
                arrows={"to": {"enabled": True, "scaleFactor": 0.4}},
                dashes=[6, 6],
            )

    return edges


# ----------------------------- 节点构建 -----------------------------

def _build_nodes(
    tools: Dict[str, Dict],
    tool_to_group: Dict[str, str],
    highlighted: Set[str],
    layout_mode: str,
    radial_positions: Optional[Tuple[Dict, Dict, Dict]] = None,
) -> List[Node]:
    """构建 Stage / Group / Tool 三层节点。"""
    inactive_color = "#DDDDDD"
    inactive_font = "#BBBBBB"

    present_groups = sorted({tool_to_group[t] for t in tools},
                            key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)
    present_stages = sorted({STAGE_OF_GROUP.get(g, "其他") for g in present_groups},
                            key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 999)

    # 计算节点度数（仅针对工具节点）
    degree: Dict[str, int] = {name: 0 for name in tools}

    nodes: List[Node] = []

    # ---------- Stage 节点 ----------
    if layout_mode == "hierarchical":
        for stage in present_stages:
            stage_id = f"__stage__{stage}"
            color = STAGE_COLORS.get(stage, "#777777")
            nodes.append(Node(
                id=stage_id,
                label=stage,
                title=f"工作阶段：{stage}",
                color={
                    "background": color,
                    "border": color,
                    "highlight": {"background": color, "border": "#333333"},
                },
                shape="box",
                size=35,
                font={
                    "color": "#ffffff",
                    "size": 14,
                    "face": "Microsoft YaHei, sans-serif",
                    "bold": True,
                    "background": "rgba(0,0,0,0.25)",
                    "strokeWidth": 0,
                },
                margin=12,
                borderWidth=0,
                level=0,
                group=stage,
                shapeProperties={"borderRadius": 8},
            ))

    # ---------- Group 节点 ----------
    group_positions: Dict[str, Tuple[float, float]] = {}
    if layout_mode == "radial" and radial_positions is not None:
        group_positions, _, _ = radial_positions

    for group in present_groups:
        group_id = f"__group__{group}"
        color = GROUP_COLORS.get(group, "#999999")
        has_active_tool = any(t in highlighted for t in tools if tool_to_group[t] == group)
        hidden = False  # 隐藏逻辑在 build_knowledge_graph 中统一处理

        node_kwargs: Dict = {
            "title": f"功能组：{GROUP_LABELS.get(group, group)}",
            "color": {
                "background": color,
                "border": color,
                "highlight": {"background": color, "border": "#333333"},
            },
            "shape": "box",
            "size": 28,
            "font": {
                "color": "#ffffff",
                "size": 12,
                "face": "Microsoft YaHei, sans-serif",
                "bold": True,
                "background": "rgba(0,0,0,0.2)",
                "strokeWidth": 0,
            },
            "margin": 10,
            "borderWidth": 0,
            "group": group,
            "shapeProperties": {"borderRadius": 6},
        }
        if layout_mode == "hierarchical":
            node_kwargs["level"] = 1
        elif layout_mode == "radial":
            gx, gy = group_positions.get(group, (0, 0))
            node_kwargs["x"] = gx
            node_kwargs["y"] = gy

        nodes.append(Node(id=group_id, label=GROUP_LABELS.get(group, group), **node_kwargs))

    # ---------- Tool 节点 ----------
    _, tool_positions, _ = radial_positions if radial_positions else ({}, {}, {})

    for tool_name, info in tools.items():
        group = tool_to_group[tool_name]
        is_active = tool_name in highlighted
        color = GROUP_COLORS.get(group, "#999999") if is_active else inactive_color
        font_color = "#333333" if is_active else inactive_font

        node_kwargs = {
            "title": f"{info.get('description', tool_name)}\n模块: {info.get('module')}.py",
            "color": {
                "background": color,
                "border": _darken(color),
                "highlight": {"background": _darken(color), "border": "#333333"},
            } if is_active else inactive_color,
            "font": {
                "color": font_color,
                "size": 11,
                "face": "Microsoft YaHei, sans-serif",
                "background": "rgba(255,255,255,0.85)",
                "strokeWidth": 0,
            },
            "shape": "dot",
            "size": (14 + degree.get(tool_name, 0) * 1.5) if is_active else 8,
            "borderWidth": 1,
            "group": group,
        }
        if layout_mode == "hierarchical":
            node_kwargs["level"] = 2
        elif layout_mode == "radial":
            x, y = tool_positions.get(tool_name, (0, 0))
            node_kwargs["x"] = x
            node_kwargs["y"] = y

        nodes.append(Node(id=tool_name, label=_short_label(tool_name), **node_kwargs))

    return nodes


# ----------------------------- 核心 API -----------------------------

def build_knowledge_graph(
    project_root: Path,
    highlight_tools: Optional[List[str]] = None,
    hide_others: bool = True,
    layout_mode: str = "hierarchical",
) -> Tuple[List[Node], List[Edge], Config]:
    """构建 streamlit-agraph 所需的 nodes、edges 与 config。

    参数：
        layout_mode: "hierarchical"（层级树） / "radial"（放射簇） / "physics"（力导向）。
    """
    if Node is None or Edge is None or Config is None:
        raise ImportError("请安装 streamlit-agraph: pip install streamlit-agraph")

    tools = _load_mcp_tools(project_root)
    order = _load_pipeline_order(project_root)

    if layout_mode == "physics":
        # 物理布局只展示工具节点，保持简洁
        return _build_physics_graph(tools, order, highlight_tools, hide_others)

    tool_to_group = {name: _tool_group(name, info.get("module", "")) for name, info in tools.items()}

    # 高亮集合：默认全部高亮；搜索时仅高亮命中节点
    highlighted = set(highlight_tools) if highlight_tools else set(tools.keys())

    # 放射布局预计算坐标
    present_groups = sorted({tool_to_group[t] for t in tools},
                            key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 999)
    group_tools = {g: [t for t in tools if tool_to_group[t] == g] for g in present_groups}
    radial_positions = _compute_radial_layout(present_groups, group_tools) if layout_mode == "radial" else None

    # 先建边，用于计算度数（仅在需要时）
    edges = _build_edges(tools, order, tool_to_group, highlighted)

    # 计算工具度数
    degree = {name: 0 for name in tools}
    for e in edges:
        if not is_group_node(e.source) and not is_stage_node(e.source):
            degree[e.source] = degree.get(e.source, 0) + 1
        if not is_group_node(e.to) and not is_stage_node(e.to):
            degree[e.to] = degree.get(e.to, 0) + 1

    # 搜索/筛选：保留命中节点 + 直接邻居 + 所有保留工具节点的祖先容器
    if hide_others and highlight_tools:
        highlighted_set = set(highlight_tools)
        keep = set(highlighted_set)
        for e in edges:
            if e.source in highlighted_set or e.to in highlighted_set:
                keep.add(e.source)
                keep.add(e.to)
        # 为所有保留的工具节点补充 Group / Stage 容器
        for node_id in list(keep):
            if node_id in tool_to_group:
                group = tool_to_group[node_id]
                keep.add(f"__group__{group}")
                keep.add(f"__stage__{STAGE_OF_GROUP.get(group, '其他')}")
    else:
        keep = None

    # 构建节点
    nodes = _build_nodes(tools, tool_to_group, highlighted, layout_mode, radial_positions)

    # 应用隐藏
    if keep is not None:
        for node in nodes:
            if node.id not in keep:
                node.hidden = True
                node.font = {"color": "#EEEEEE", "size": 1}

    # 调整工具节点大小（基于度数）
    for node in nodes:
        if is_tool_node(node.id) and not getattr(node, "hidden", False):
            node.size = 14 + degree.get(node.id, 0) * 1.5

    # 构建 Config
    is_hierarchical = layout_mode == "hierarchical"
    is_radial = layout_mode == "radial"

    config_kwargs = {
        "width": "100%",
        "height": 800,
        "directed": True,
        "physics": False,
        "hierarchical": is_hierarchical,
        "nodeHighlightBehavior": True,
        "highlightColor": "#F7A7A6",
        "collapsible": False,
        "staticGraphWithDragAndDrop": is_radial,
        "maxVelocity": 50,
        "interaction": {
            "hover": True,
            "selectable": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.35},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.4}},
            "color": {"color": "#CCCCCC", "highlight": "#F7A7A6"},
        },
    }

    if is_hierarchical:
        config_kwargs.update({
            "direction": "LR",
            "levelSeparation": 260,
            "nodeSpacing": 150,
            "treeSpacing": 260,
            "sortMethod": "directed",
            "shakeTowards": "roots",
            "parentCentralization": True,
            "edgeMinimization": True,
            "blockShifting": True,
        })

    config = Config(**config_kwargs)
    return nodes, edges, config


def _build_physics_graph(
    tools: Dict[str, Dict],
    order: List[str],
    highlight_tools: Optional[List[str]],
    hide_others: bool,
) -> Tuple[List[Node], List[Edge], Config]:
    """力导向布局：仅展示工具节点与核心调用边。"""
    tool_to_group = {name: _tool_group(name, info.get("module", "")) for name, info in tools.items()}
    highlighted = set(highlight_tools) if highlight_tools else set(tools.keys())

    inactive_color = "#DDDDDD"
    inactive_font = "#BBBBBB"

    nodes = []
    for tool_name, info in tools.items():
        group = tool_to_group[tool_name]
        is_active = tool_name in highlighted
        color = GROUP_COLORS.get(group, "#999999") if is_active else inactive_color
        font_color = "#333333" if is_active else inactive_font
        hidden = hide_others and not is_active
        nodes.append(Node(
            id=tool_name,
            label=_short_label(tool_name),
            title=f"{info.get('description', tool_name)}\n模块: {info.get('module')}.py",
            color=color,
            font={"color": font_color, "size": 10, "face": "Microsoft YaHei, sans-serif"},
            shape="dot",
            size=18 if is_active else 10,
            hidden=hidden,
            group=group,
        ))

    edges = []
    edge_set = set()

    def _add(src, dst, title, color, width=1.0):
        if (src, dst) in edge_set or src not in tools or dst not in tools:
            return
        edge_set.add((src, dst))
        active = src in highlighted and dst in highlighted
        edges.append(Edge(
            source=src, target=dst,
            color=color if active else "#DDDDDD",
            width=width if active else 0.5,
            title=title, type="CURVE_SMOOTH",
        ))

    for i in range(len(order) - 1):
        src, dst = order[i], order[i + 1]
        _add(src, dst, "流程下一步", "#888888", 1.2)

    for tool_name in tools:
        if "build" in tool_name or "learn" in tool_name:
            group = tool_to_group[tool_name]
            for other in tools:
                if other != tool_name and tool_to_group[other] == group and (
                    "run" in other or "calculation" in other or "generate" in other
                ):
                    _add(tool_name, other, "产出→消费", GROUP_COLORS.get(group, "#999999"), 1.2)

    config = Config(
        width="100%",
        height=800,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        maxVelocity=50,
        interaction={"hover": True, "selectable": True, "dragNodes": True, "dragView": True, "zoomView": True},
        edges={"smooth": {"type": "continuous"}, "arrows": {"to": {"scaleFactor": 0.5}}},
    )
    return nodes, edges, config


# ----------------------------- 图例 -----------------------------

def render_group_legend() -> str:
    """返回分组颜色图例的 Markdown。"""
    items = []
    for group in GROUP_ORDER:
        if group not in GROUP_COLORS:
            continue
        color = GROUP_COLORS[group]
        label = GROUP_LABELS.get(group, group)
        items.append(f"<span style='color:{color}; font-size:18px;'>●</span> {label}")
    return " &nbsp;|&nbsp; ".join(items)


def render_stage_legend() -> str:
    """返回阶段颜色图例的 Markdown。"""
    items = []
    for stage in STAGE_ORDER:
        color = STAGE_COLORS.get(stage, "#777777")
        items.append(f"<span style='color:{color}; font-size:18px;'>■</span> {stage}")
    return " &nbsp;|&nbsp; ".join(items)
