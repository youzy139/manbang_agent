"""评测入口：通过 CLI 指定 agent / 数据 / 结果目录，无需改 config.json。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SERVER_ROOT = Path(__file__).resolve().parent
_DEMO_ROOT = _SERVER_ROOT.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

from bench.evaluation_runner import EvaluationRunner
from bench.settings import _resolve_model_api_key, build_app_settings, load_settings


def _resolve_agent_dir(agent_dir: Path) -> Path:
    resolved = agent_dir.resolve()
    if resolved.name != "agent":
        raise ValueError(f"--agent-dir 必须指向 agent 目录，当前: {resolved}")
    if not (resolved / "model_decision_service.py").is_file():
        raise FileNotFoundError(f"缺少 agent 入口: {resolved / 'model_decision_service.py'}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="离线仿真评测（CLI 指定路径，无需改 config）")
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=_DEMO_ROOT / "agent",
        help="agent 目录；默认 demo/agent",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_SERVER_ROOT / "data",
        help="数据目录（cargo_dataset.jsonl + drivers.json）；默认 server/data",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEMO_ROOT / "results",
        help="结果输出目录；默认 demo/results",
    )
    parser.add_argument("--reposition-speed", type=float, default=None, help="空驶速度 km/h")
    parser.add_argument("--simulation-days", type=int, default=None, help="仿真天数")
    parser.add_argument("--simulation-max-steps", type=int, default=None, help="全局最大步数")
    parser.add_argument("--max-steps", type=int, default=None, help="调试：覆盖最大步数上限")
    parser.add_argument("--model-api-url", type=str, default=None, help="模型 API URL")
    parser.add_argument("--model-name", type=str, default=None, help="模型名称")
    parser.add_argument("--model-timeout", type=float, default=None, help="模型超时秒数")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="可选：仅用于读取模型/步数默认值（路径仍由 --agent-dir/--data-dir/--results-dir 指定）",
    )
    args = parser.parse_args()

    agent_dir = _resolve_agent_dir(args.agent_dir)
    data_dir = args.data_dir.resolve()
    results_dir = args.results_dir.resolve()
    cargo_dataset = data_dir / "cargo_dataset.jsonl"
    drivers_dataset = data_dir / "drivers.json"
    if not cargo_dataset.is_file():
        raise FileNotFoundError(f"缺少货源数据: {cargo_dataset}")
    if not drivers_dataset.is_file():
        raise FileNotFoundError(f"缺少司机数据: {drivers_dataset}")

    submission_demo_root = agent_dir.parent
    if str(submission_demo_root.resolve()) not in sys.path:
        sys.path.insert(0, str(submission_demo_root.resolve()))

    config_path = Path(args.config) if args.config else None
    defaults = load_settings(config_path)

    try:
        settings = build_app_settings(
            cargo_dataset_path=cargo_dataset,
            drivers_path=drivers_dataset,
            results_dir=results_dir,
            log_dir=results_dir / "logs",
            reposition_speed_km_per_hour=float(
                args.reposition_speed if args.reposition_speed is not None else defaults.reposition_speed_km_per_hour
            ),
            simulation_max_steps=int(
                args.simulation_max_steps if args.simulation_max_steps is not None else defaults.simulation_max_steps
            ),
            simulation_duration_days=int(
                args.simulation_days if args.simulation_days is not None else defaults.simulation_duration_days
            ),
            model_api_url=str(
                args.model_api_url if args.model_api_url is not None else defaults.model_api_url
            ).strip(),
            model_api_key=_resolve_model_api_key(defaults.model_api_key),
            model_name=str(args.model_name if args.model_name is not None else defaults.model_name).strip(),
            model_timeout_seconds=float(
                args.model_timeout if args.model_timeout is not None else defaults.model_timeout_seconds
            ),
            decision_step_timeout_seconds=defaults.decision_step_timeout_seconds,
        )
        runner = EvaluationRunner(
            settings=settings,
            max_steps=args.max_steps,
            decision_step_timeout_seconds=settings.decision_step_timeout_seconds,
        )
        runner.run()
        return 0
    except Exception:
        logging.exception("evaluation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
