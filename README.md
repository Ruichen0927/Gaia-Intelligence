# Gaia 测井解释重构项目

Gaia 是对 `welllog_agent` 测井解释流程的重构版本，目标是将原有脚本改造为**配置驱动、工具原子化、支持流程法与 MCP 方法两种运行方式**，并提供自然语言前端一键运行。

## 目录结构

```
Gaia/
├── ctl/                          # 全部 JSON 控制文件
│   ├── agent_configs/            # LLM Agent 配置文件
│   ├── tool_configs/             # 每个工具的输入/输出/参数配置
│   ├── pipeline_config.json      # 流程法编排
│   └── mcp_config.json           # MCP 工具注册表
├── src/
│   ├── llm/                      # LLM Agent 调用（与原项目一致）
│   ├── common/                   # 配置加载、路径解析、公共工具
│   ├── tools/                    # 各解释能力工具（shale/porosity/...）
│   ├── methods/                  # flow_runner / mcp_orchestrator / mcp_server
│   └── dev_platform/             # 二次开发平台后端（注册表、生成、校验）
├── extensions/                   # 用户扩展工具隔离目录
│   ├── tools/                    # 用户工具 Python 模块
│   └── configs/                  # 用户工具 JSON 配置
├── frontend/
│   ├── app.py                    # Streamlit 前端（运行控制台 + 知识图谱 + 二次开发）
│   ├── commands.py               # 自然语言指令解析
│   ├── knowledge_graph.py        # MCP 工具知识图谱与源码查看
│   └── dev_platform.py           # 二次开发平台前端
├── data/sample/                  # 测试用部分数据
├── main.py                       # CLI 入口
├── run_flow.py                   # 便捷运行流程法
└── run_mcp_server.py             # 便捷启动 MCP 服务
```

## 环境准备

本项目在 conda `Agent` 环境中运行。安装缺失依赖：

```bash
cd /home/ruichen/welllog_agent/Gaia
pip install -r requirements.txt
```

## 运行方式

### 1. 流程法（Pipeline / Flow）

一键执行完整流程：

```bash
python main.py flow
# 或
python run_flow.py
```

执行到指定阶段（例如只跑完孔隙度）：

```bash
python main.py flow --stop-at porosity_run_calculation
```

### 2. MCP 方法

调用单个工具：

```bash
python main.py mcp --tool shale_build_strategy
```

按 JSON 计划批量调用：

```bash
python main.py mcp --plan ctl/mcp_plan_example.json
```

启动 MCP 服务端（stdio 默认）：

```bash
python main.py mcp-server
# SSE 模式
python main.py mcp-server --transport sse
```

### 3. 前端界面

启动 Streamlit：

```bash
streamlit run frontend/app.py
```

在页面中：
- **运行控制台**：选择运行模式、输入自然语言指令（如“运行全流程”、“调用泥质含量工具”），点击**一键运行**。
- **知识图谱**：可视化展示 34 个 MCP 工具之间的层级与流程关联；**点击任意工具节点**可在左侧面板查看该工具的源码。

## 配置说明

- 所有 JSON 文件位于 `ctl/` 目录。
- `ctl/tool_configs/*.json` 控制每个工具的输入/输出路径和算法参数。
- 路径统一使用**相对 Gaia 项目根目录**的相对路径，由 `src/common/config_loader.py` 自动解析为绝对路径。
- 测试数据默认使用 `data/sample/` 中的 1 口井；生产环境可修改为 `../welllog_agent/welllog_agent/data/unit_consistent_dataset`。

## 测试数据

`data/sample/` 下已复制原项目 `unit_consistent_dataset` 中的 1 口井，用于快速联调。默认工具配置指向该目录，以减少 LLM 调用量。如需扩大测试，可复制更多井到 `data/sample/`。

## 4. 二次开发平台

进入 Streamlit 前端后，点击左侧导航 **二次开发平台**，可以：

1. **🛠️ 新建工具**：用自然语言描述需求，Agent 自动生成符合 Gaia 规范的工具代码与配置文件，经校验后保存到 `extensions/` 并自动注册到 MCP。
2. **🔄 格式转换**：粘贴或上传已有 Python 代码，Agent 会将其改写为 `run(config, project_root) -> dict` 的标准工具格式。
3. **📋 工具管理**：查看、测试、编辑、删除已注册工具。内置工具只读，用户工具可操作；测试运行在子进程沙箱中执行，确保安全。
4. **⚙️ 注册表管理**：可视化编辑 `ctl/mcp_config.json` 与 `ctl/pipeline_config.json`。

### 用户工具约定

- 用户工具统一存放在 `extensions/tools/`，配置存放在 `extensions/configs/`。
- 注册表中的 `module` 字段格式为 `extensions.tools.<模块名>`，与内置工具 `tools.<模块名>` 区分。
- 用户工具追加到流程法时默认 `enabled=false`，避免影响一键运行。

### CLI 验证

新增工具后，可用命令行直接验证：

```bash
python main.py mcp --tool <your_tool_name>
```

## 5. 更细粒化工具包、机器学习与层级知识图谱

系统已扩展 23 个细粒度工具，覆盖数据处理、特征工程、可视化、统计分析与基础机器学习：

| 功能组 | 工具示例 |
|---|---|
| 数据处理 | `data_load_well_logs`、`data_clean_curves`、`data_detect_outliers`、`data_impute_missing`、`data_qc_report` ... |
| 特征工程 | `feature_compute_derived_curves`、`feature_normalize_curves`、`feature_window_features` |
| 可视化 | `viz_crossplot`、`viz_histogram`、`viz_pickett_plot` |
| 统计分析 | `stat_summary`、`stat_correlation` |
| 机器学习 | `ml_train_regressor`、`ml_predict`、`ml_evaluate`、`ml_cluster_data`、`ml_train_classifier` ... |

### 使用示例

加载并清洗数据：

```bash
python main.py mcp --tool data_load_well_logs
python main.py mcp --tool data_clean_curves
```

训练回归模型预测孔隙度：

```bash
python main.py mcp --tool ml_train_regressor
python main.py mcp --tool ml_predict
python main.py mcp --tool ml_evaluate
```

绘制交会图：

```bash
python main.py mcp --tool viz_crossplot
```

### 知识图谱

进入 Streamlit 的 **知识图谱** 页面：

- **主视图**：基于 Plotly 渲染的交互图谱，每个功能组有圆角矩形背景框，工具节点可点击。
- **点击查看源码**：点击任意工具节点，页面自动选中该工具并在下方展示完整源码。
- **搜索筛选**：顶部搜索框实时过滤，未命中节点变灰/变小。
- **左侧边栏**：工具列表与元信息（阶段、功能组、模块路径、描述）。
- **高清静态预览**：基于 matplotlib 生成带圆角分组框的静态图，文字清晰、嵌套完整，适合导出/打印。
- 也可在左侧边栏直接选择工具查看源码。

## 注意事项

- 当前 LLM 调用方式与 `welllog_agent` 保持一致，均通过 `src/llm/agent.py` 的 `Agent` 类实现。
- 流程中包含大量 Agent 调用，测试时建议关闭 `use_few_shot` / `use_dynamic_few_shots` 并使用少量样本数据。
- API 密钥保存在 `ctl/agent_configs/*.json` 中，请勿泄露。
- 二次开发平台生成的代码会在子进程中测试运行，但保存前仍需人工检查，避免执行恶意代码。
