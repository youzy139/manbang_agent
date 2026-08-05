# 满帮复赛 · 卡车司机连续找货决策 Agent

天池「基于 Agentic AI 的卡车司机连续找货决策」大赛（复赛）参赛代码库。

任务：在事件驱动的仿真环境中，为每位司机持续决策一个完整仿真周期（92 天），每个决策点从
**接单（take_order）/ 休息（wait）/ 空驶（reposition）** 中择一执行，目标是最大化
**月度净收益**（总收入 − 里程成本 − 偏好罚分），同时尽量满足司机的个性化偏好。

## 决策架构

LLM **不直接输出动作**。架构分三层：

1. **Virtual Manager LLM**（`agent/virtual_manager.py`）：维护一张「策略场」——
   虚拟单（rest / deadhead / cargo_modifier，可用 combo 串联多步义务）+ 货源奖惩 patch。
   稀疏唤醒（日界 + 上/下午 + 场签名变化），省 token。
2. **plan_route**（`agent/plan_route.py` / `path_planner.py`）：在策略场中做**确定性多跳最优搜索**，
   执行最优路线的第一跳；`harness` 执行前复核（fails-open）。
3. **event_watcher**（`agent/event_watcher.py`）：**事件触发型偏好**（复赛黑盒数据新形态，
   偏好项带 `type:"事件触发型"` + 结构化 `trigger`）的确定性监听器——
   `on_date`（限期回访参考单装货地并停留）、`first_take_order_touch_city`（首单触城后注入禁令
   modifier）每步检测、幂等注入虚拟单，不依赖 LLM 自觉；未知事件类型交 Virtual Manager 按原文处理。

司机画像链路：自然语言偏好 → `driver_persona` / `llm_persona_extractor` LLM 提取画像 →
`_persona_adapter` / `preference_parser` 结构化规则 → 喂给 Virtual Manager 编译进策略场。

## 目录结构

```text
├─ docs/                    # 赛方文档（赛题/数据/评测规则/提交方式/快速开始/更新日志）
├─ demo/
│  ├─ agent/                # ★ 选手代码：决策 Agent（事件监听 / 策略场引擎 / 画像解析）
│  ├─ simkit/               # 仿真域共享内核（赛方）：规则纯函数与状态
│  ├─ server/               # 评测编排（赛方）：仿真入口 main.py、bench/、config/、data/
│  ├─ calc_monthly_income.py  # 收益结算与动作合法性校验（本地自测）
│  ├─ verify_persona.py     # 画像提取校验脚本
│  └─ 思路.md               # 调优笔记
└─ AGENTS.md                # 面向 AI 编码代理的详细项目说明（口径/红线/验证方法）
```

## 快速开始

```bash
# 1) 安装依赖（仅 numpy / requests；决策核心只用标准库）
cd demo/server
pip install -r requirements.txt

# 2) 配置模型（密钥优先用环境变量 DASHSCOPE_API_KEY / TIANCHI_MODEL_API_KEY）
cp config/config.example.json config/config.json   # 填入 model_api_url / model_name / key

# 3) 运行仿真（config.json 默认 92 天全量；调试可加 --simulation-days 1 --max-steps 100）
python main.py config/config.json

# 4) 收益结算与合法性校验
cd ..
python calc_monthly_income.py
```

产物在 `demo/results/`：逐步动作记录 `actions_*.jsonl`、汇总 `run_summary_*.json`、
收益明细与偏好罚分逐条判定 `monthly_income_*.json`。

## 本地验证结果（2026-08-05，真实 D001 数据 92 天全量）

- 净收益 **160,360 元**（总收入 193,139 − 里程成本 29,779 − 偏好罚分 3,000）
- 动作合法性校验全绿（`validation_error: null`）
- token 420 万 / 500 万预算；唯一失分点：4 月 >8h 长途接了 8 单（上限 5，罚 3,000）
- 事件触发型偏好：在仿黑盒格式的合成数据上验证两类 trigger 完整履约
  （回访装货地停留、触城禁令），本地官方数据无此偏好时 watcher 零开销静默

## 参赛硬约束（决策代码红线）

- 不得直读原始数据文件；一切信息经 `SimulationApiPort`（query_cargo / get_driver_status / query_decision_history）
- 不得为特定 driver_id 硬编码偏好规则
- 接口契约固定：`ModelDecisionService(api).decide(driver_id) -> dict`
- 资源上限：总仿真 ≤ 4 小时、每司机 token ≤ 500 万、单步决策 ≤ 120s

更详细的口径与开发约定见 [AGENTS.md](AGENTS.md) 与 [docs/](docs/index.md)。
