# AGENTS.md — 项目说明（供 AI 编码代理阅读）

> 本文面向对本项目一无所知的 AI 代理。所有信息均来自对仓库实际内容的核查，未做推测。

## 1. 项目概览

这是**天池「基于 Agentic AI 的卡车司机连续找货决策」大赛（复赛）**的参赛代码库。任务是实现一个**司机找货智能体（Agent）**：在事件驱动的仿真环境中，针对每位司机持续一个完整仿真周期，在每个决策点从 **接单（take_order）**、**休息（wait）**、**空驶（reposition）** 三类动作中择一执行，目标是最大化**月度净收益**（总收入 − 里程成本 − 偏好罚分），同时尽量满足司机的个性化偏好。

仓库包含两部分：

- **赛方提供的评测框架与文档**（`docs/`、`demo/server/`、`demo/simkit/`、`demo/calc_monthly_income.py`）：离线仿真编排、数据、规则、收益结算脚本。这部分**原则上不应修改**——正式评测使用赛方统一代码运行选手的 `agent/`。
- **选手实现的 Agent**（`demo/agent/`）：本仓库的主要开发对象，实现司机画像解析 + 偏好规则化 + 策略场决策引擎。

官方文档在 `docs/` 下（中文），按编号阅读：`01-赛题详情`、`02-数据说明`、`03-评测规则`、`04-提交方式`、`05-快速开始`、`更新日志`。修改行为前请先核对 `docs/更新日志.md`（最近一次 2026-06-08 发布了复赛新数据）。

## 2. 技术栈与依赖

- **语言**：纯 Python（开发环境为 Windows + Python 3.13）。
- **第三方依赖极少**：`demo/server/requirements.txt` 仅 `numpy>=1.26.0`、`requests>=2.32.0`；`demo/agent/requirements.txt` 明确写明**决策核心仅依赖 Python 标准库**。新增依赖前先确认必要——选手代码在赛方评测环境运行，依赖越少越安全。
- **LLM 调用**：通过 OpenAI 兼容的 Chat Completions 接口，由评测进程内的 `SimulationApiPort.model_chat_completion` 注入（server 侧 `ModelGatewayClient` 用 `requests` 透传）。模型配置在 `demo/server/config/config.json`（当前示例指向 DeepSeek；`config.example.json` 指向阿里云 DashScope/qwen）。密钥用环境变量 `DASHSCOPE_API_KEY` 或 `TIANCHI_MODEL_API_KEY` 注入（优先级高于配置文件）。
- **无构建系统**：没有 pyproject.toml / setup.py / package.json，通过 `sys.path` 拼接（见 `demo/server/main.py`、`demo/calc_monthly_income.py`）实现包间引用，包导入路径形如 `from simkit.ports import ...`、`from agent.xxx import ...`。
- **无单元测试框架、无 CI、无 lint 配置**。验证方式是跑仿真 + 结算脚本（见第 6 节）。

## 3. 目录结构与模块划分

```text
manbang-V7/
├─ docs/                        # 赛方文档（中文）：赛题、数据、评测、提交、快速开始
├─ demo/
│  ├─ agent/                    # ★ 选手代码：决策 Agent（主要开发对象）
│  │  ├─ model_decision_service.py   # 对外门面 ModelDecisionService（接口固定，server 按此类名加载）
│  │  ├─ loop.py                     # StrategyFieldEngine：策略场引擎主循环（~1600 行，核心）
│  │  ├─ virtual_manager.py          # Virtual Manager LLM：输出 virtual patch 维护策略场
│  │  ├─ virtual_registry.py         # 虚拟单注册表（rest / deadhead / cargo_modifier 三类）
│  │  ├─ plan_route.py / path_planner.py / cargo_graph.py   # 多跳路线搜索与货源图
│  │  ├─ driver_persona.py / llm_persona_extractor.py / _persona_adapter.py / preference_parser.py
│  │  │                                # 司机画像与偏好：自然语言 → 结构化规则（preference_parser ~4000 行）
│  │  ├─ long_memory.py / monthly_quota_plan.py / memory_tracker.py  # 记忆与月度配额规划
│  │  ├─ harness.py                  # 执行前复核 top route 第一跳（fails-open）
│  │  ├─ time_tools.py / model_io.py / log_color.py / _helpers.py    # 工具层
│  │  ├─ prompts/                    # LLM prompt 文本（txt）
│  │  └─ requirements.txt            # 仅注释：决策核心只用标准库
│  ├─ simkit/                   # 仿真域共享内核（赛方）：规则纯函数与状态
│  │  ├─ ports.py                    # 跨模块协议：AgentDecisionPort / SimulationApiPort（typing.Protocol）
│  │  ├─ simulation_actions.py       # 动作规则纯函数：haversine、耗时换算、query_cargo/take_order/wait/reposition
│  │  ├─ cargo_repository.py         # 货源内存索引（向量化 Haversine 取最近 K 条）
│  │  └─ driver_state_manager.py     # 司机状态管理
│  ├─ server/                   # 评测编排（赛方）
│  │  ├─ main.py                     # 仿真入口（CLI，无 HTTP 服务）
│  │  ├─ bench/                      # EvaluationRunner、SimulationOrchestrator（主循环）、
│  │  │                              #   EmbeddedDecisionEnvironment（SimulationApiPort 实现）、
│  │  │                              #   ModelGatewayClient、token 预算、单步超时、延迟记录
│  │  ├─ config/config.example.json  # 配置模板（config.json 为本地实际配置，勿入库）
│  │  └─ data/                       # 输入数据：cargo_dataset.jsonl（约 150 万条）、drivers.json
│  ├─ results/                  # 运行产物（actions_*.jsonl、run_summary_*.json、history/ 归档、logs/）
│  ├─ calc_monthly_income.py    # 收益结算与动作合法性校验脚本（本地自测用）
│  ├─ verify_persona.py         # 画像提取校验脚本（python demo/verify_persona.py drivers.json）
│  ├─ 思路.md                    # 选手的调优笔记（权重/rollout/记忆偏差等实验想法）
│  └─ README.md                  # demo 目录官方说明
```

**依赖方向（不可反向）**：`simkit` ← `agent` ← `server`。`agent` 只通过 `simkit.ports.SimulationApiPort` 与环境交互，不 import `server` 的任何模块。

**Agent 决策架构**（`loop.py` 模块 docstring 有完整说明）：LLM **不直接输出动作**——Virtual Manager LLM 维护一张「策略场」（虚拟单 + 货源奖惩 patch），`plan_route` 在策略场中做**确定性多跳最优搜索**，执行最优路线的第一跳；`harness` 在执行前复核（失败时 fails-open 不阻断）。每步流程：获取新货源 → 记忆/账本更新 → Virtual Manager 维护虚拟单 → plan_route 生成候选 → harness 复核 → 执行 + 落账。

**注意——未接线的遗留模块**：`agent/cargo_memory.py`、`agent/market_memory.py`、`agent/memory_tracker.py` 当前**没有任何模块 import 它们**（属于早期方案遗留；`思路.md` 中有基于 market_memory 的调优设想）。`agent/preference_llm.py` 引用的 `prompts/preference_llm_prompt.txt` 与 `harness.py` 引用的 `prompts/harness_review_prompt.txt` 在 `prompts/` 目录中**不存在**（相关代码均做了缺失容错）。改动前先确认模块是否真的在调用链上。

## 4. 运行与构建命令

没有构建步骤。标准流程（Windows / Git Bash 同样适用，注意 README 中的 `copy` 是 cmd 语法）：

```bash
# 1) 安装依赖
cd demo/server
pip install -r requirements.txt

# 2) 初始化配置（首次）
cp config/config.example.json config/config.json
#    填入 model_api_url / model_name / model_api_key，或设置环境变量 DASHSCOPE_API_KEY

# 3) 运行仿真（在 demo/server 下）
python main.py                          # 或 python main.py config/config.json
# 调试可用 CLI 覆盖参数，如：
# python main.py --simulation-days 3 --max-steps 200 --agent-dir ../agent

# 4) 收益结算与合法性校验（在 demo/ 下）
cd ..
python calc_monthly_income.py
```

`main.py` 关键 CLI 参数：`--agent-dir`（必须指向含 `model_decision_service.py` 的 agent 目录）、`--data-dir`、`--results-dir`、`--reposition-speed`、`--simulation-days`、`--simulation-max-steps`、`--max-steps`（调试）、`--model-api-url/--model-name/--model-timeout`。

产物（写入 `demo/results/`）：

- `actions_202603_<driver>_<timestamp>.jsonl`：逐步动作记录（结算校验的直接输入；文件名前缀固定为 `202603`，与仿真起点月份一致）
- `run_summary_202603.json`：本次仿真汇总（含 `simulation_duration_days`）
- `monthly_income_202603.json`：收益明细、偏好罚分逐条判定、token 统计、校验失败信息
- `history/<时间戳>/`：每次仿真开始前归档的旧结果

当前本地数据（`demo/server/data/`）：仅 **D001** 一位司机（`drivers.json` 中其偏好可见窗口覆盖 2026-03-01 至 2026-05-31）；货源约 **150 万条**，时间跨度 2026-03-01 ~ 2026-06-01。`config.json` 中 `simulation_duration_days=92`（约 3 个月，`time_tools.DURATION_DAYS = 92` 与之对应），`reposition_speed_km_per_hour=60`。

## 5. 关键规则与口径（改动代码时必须遵守）

完整规则见 `docs/02-数据说明.md` 与 `docs/03-评测规则.md`，核心口径：

- **时间**：仿真主时间单位为**分钟**（`simulation_progress_minutes`），起点 `2026-03-01 00:00:00`（`_SIMULATION_EPOCH`）。墙钟时间仅用于展示。
- **距离**：任意两点为 Haversine 大圆距离（km）；距离换算分钟时 `max(1, ceil(d / speed * 60))`（`distance_to_minutes`）。接单空驶零距离特例：≤1e-6 km 时耗时 0。
- **query_cargo**：`k` 默认 100，上限 600；浏览耗时 `scan_minutes = ceil(n / cargo_view_batch_size)`（batch 默认 10），计入当步耗时。Agent 侧每次 query 都会推进仿真时间，不要把查询当免费操作。
- **take_order 耗时** = 空驶到装货点 + 装货窗等待（早于窗开始则原地等；晚于窗结束接单失败）+ `cost_time_minutes`（含装卸）。完单时刻若超过仿真上界则无收益。
- **收益**：`gross_income`（按 `cargo_id` 回查原始价格，`price/100` 换算为元）− `distance_km * cost_per_km` − `preference_penalty`。
- **结算脚本**：`calc_monthly_income.py` 会逐步校验时间/位置/耗时/参数一致性，**校验失败按司机隔离**（单司机收益为 0，看结果 JSON 中 `validation_error`）。偏好规则以「文本偏好 + 固定规则映射」实现，偏好文本若变，赛方会同步改结算脚本——选手代码不要跟着写死规则。

## 6. 测试与验证策略

项目**没有自动化测试**。验证手段：

1. **小步快跑仿真**：`python main.py --simulation-days 1 --max-steps 100` 之类缩短周期跑通全流程，再逐步放大。日志在 `demo/results/logs/`（`server_runtime.log`、`simulation_orchestrator.log`）。
2. **结算校验**：`python calc_monthly_income.py` 后检查 `monthly_income_202603.json` 的 `validation_error`、`preference_check.rules`——这是动作合法性的权威判定。
3. **画像/偏好解析校验**：`python demo/verify_persona.py demo/server/data/drivers.json`（从仓库根目录运行），输出见 `demo/verify_persona_result.json`。
4. **对照实验**：`demo/思路.md` 记录了调权（income/efficiency 权重）、两步 rollout 等实验思路；`results/history/` 保留了历次仿真产物供对比。

## 7. 参赛硬约束（写决策代码时的红线）

- **禁止直读原始数据**：决策代码不得 `open` / 解析 `cargo_dataset.jsonl`、`drivers.json`（含写死路径、整表扫描缓存），一切信息经 `SimulationApiPort`（`query_cargo` / `get_driver_status` / `query_decision_history`）获取。
- **禁止硬编码偏好规则**：不得为特定 `driver_id` 写死 if/else 或固定时间窗；须基于运行时接口返回的 `preferences` 由策略/模型理解执行（复赛评测用未公开司机 D003/D004 等）。
- **接口契约固定**：`server` 通过 `ModelDecisionService(api).decide(driver_id) -> dict` 调用 agent，返回 `{"action": ..., "params": {...}, ...}`；`model_usage` 由服务端强制覆盖，选手侧上报无效。
- **资源上限**：复赛总仿真运行时长 ≤ 4 小时；每司机 token ≤ 500 万（`driver_max_total_tokens`，超预算会被终止）；单步决策墙钟超时 `decision_step_timeout_seconds=120`。
- **提交**：复赛 ZIP 包内为 `demo/` 根目录，**至少含 `demo/agent/`**，**不要**打包 `demo/results/` 与 `data/`；若改动 `simkit/` 等最小目录之外的内容，必须附说明文档（详见 `docs/04-提交方式.md`）。

## 8. 代码风格约定

- 中文注释与 docstring 为主；模块顶部用 docstring 说明职责（部分早期模块 docstring 为英文，新旧并存，跟随所在文件风格即可）。
- 类型标注风格：`from __future__ import annotations` + `dict[str, Any]` 等内置泛型；Python 3.10+ 语法（`X | None`）。
- 日志用标准 `logging`，logger 名按模块域命名（如 `llm_agent.loop`、`bench.evaluation_runner`、`agent.decision_service`）；agent 侧控制台彩色输出走 `log_color._tag()/_line_end()`。
- LLM 相关调用统一走 `model_io.call_content_with_retry`（重试 + 退避），失败策略多为 **fails-open**（不阻断决策，由确定性逻辑兜底）。
- 时间与日期运算集中在 `time_tools.py`，LLM 不做日期算术——新增需要时间计算的功能时沿用这一层。
- 改动最小化：`simkit/`、`server/` 属赛方代码，除非确有必要（且愿按提交规范附说明文档），不要修改；选手逻辑放在 `agent/` 内。

## 9. 安全注意事项

- **密钥**：真实 API key 不要写入仓库或提交包。`demo/server/config/config.json` 本地维护（当前文件含真实密钥，注意勿外泄）；优先用环境变量 `DASHSCOPE_API_KEY` / `TIANCHI_MODEL_API_KEY`。
- **运行产物**：`demo/results/` 是仿真输出，体积可能很大（`cargo_dataset.jsonl` 约 150 万行，数据文件不入提交包）；本仓库副本**没有 .gitignore 也不是 git 仓库**，如需入库请自行排除 `results/`、`data/`、`config.json`、`__pycache__/`。
- 仓库根目录的 `__pycache__/validate_personas.cpython-313.pyc` 是已删除脚本的残留编译产物，无对应源码。
