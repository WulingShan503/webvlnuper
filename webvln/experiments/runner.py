"""第五章实验的驱动脚本。

一条命令跑完一组配置，输出可直接比对论文表格的 Markdown。

    python -m webvln.experiments.runner --data-dir Data --exp ablation

需要 torch 与 ``OPENAI_API_KEY``，由用户自行执行；本文件只负责编排，
各实验的差别全在筛选配置上（见 ``ablation.py``）。

四个子实验对应论文小节：

    baseline    5.2  关闭筛选
    ablation    5.3  k ∈ {3, 5, 8} 加基线
    two_stage   5.5  单阶段 vs 两阶段
    stage_one   5.4  仅规则过滤，看阶段一自身的 CR / RR
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from webvln.experiments.ablation import (
    TOPK_VALUES,
    ExperimentResult,
    baseline_config,
    compare_with_reference,
    format_table,
    rule_filter_only_config,
    screening_config_for_k,
    screening_summary_row,
    single_stage_config,
)
from webvln.experiments.reference import REFERENCE_ABLATION, REFERENCE_SCREENING


def build_experiment_configs(
    experiment: str, base: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """列出某组实验的 ``{名称: 筛选配置}``。

    名称与 ``reference.py`` 的键一致，这样 ``compare_with_reference``
    能直接对上论文数字。
    """
    if experiment == "baseline":
        return {"baseline": baseline_config(base)}
    if experiment == "ablation":
        configs = {"baseline": baseline_config(base)}
        for k in TOPK_VALUES:
            configs[f"llm_top{k}"] = screening_config_for_k(k, base)
        return configs
    if experiment == "two_stage":
        return {
            "single_stage": single_stage_config(5, base),
            "two_stage": screening_config_for_k(5, base),
        }
    if experiment == "stage_one":
        return {"rule_filter_only": rule_filter_only_config(base)}
    raise ValueError(f"未知实验：{experiment!r}")


def run_one(
    name: str,
    screening_config: Dict[str, Any],
    data_dir: str,
    setting: str = "seen",
    load_path: Optional[str] = None,
    n_iters: Optional[int] = None,
) -> ExperimentResult:
    """跑一个配置：装配环境、训练（或加载权重）、在 val 与 test 上评测。

    Args:
        name: 实验名。
        screening_config: 筛选配置。
        data_dir: 数据目录。
        setting: ``seen`` / ``unseen``。
        load_path: 已有权重。给出时跳过训练直接评测——
            Top-k 消融若共用同一份基线权重，只需评测四次而非训练四次。
        n_iters: 训练迭代数，默认 ``config.max_iters``。

    Returns:
        ExperimentResult。
    """
    # 延迟导入：torch 与 transformers 只在真正跑实验时才需要。
    from webvln.data import FeatureStore, NavigationGraph, WebVLNDataset
    from webvln.data.text import load_bert_tokenizer
    from webvln.models.config import WebVLNConfig
    from webvln.models.webvln_net import WebVLNNet
    from webvln.screening.config import build_screener
    from webvln.train.env import WebVLNEnv
    from webvln.train.trainer import Trainer

    config = WebVLNConfig()
    tokenizer = load_bert_tokenizer(config.bert_model_name)

    graph = NavigationGraph.from_dir(data_dir)
    features = FeatureStore.from_dir(data_dir, feature_size=config.feature_size)
    screener = build_screener(config=screening_config)

    splits = {
        split: WebVLNDataset.from_dir(
            data_dir, setting, split, tokenizer=tokenizer,
            batch_size=config.batch_size,
        )
        for split in ("train", "val", "test")
    }

    trainer = Trainer(WebVLNNet(config), WebVLNEnv(graph, features, screener), config)
    trainer.tokenizer = tokenizer

    if load_path:
        trainer.load(load_path)
    else:
        trainer.fit(
            splits["train"],
            {"val": splits["val"]},
            n_iters=n_iters,
            save_path=f"checkpoints/{name}.pt",
        )

    return ExperimentResult(
        name=name,
        val=trainer.validate(splits["val"]),
        test=trainer.validate(splits["test"]),
        screening=trainer.env.screening_stats(),
    )


def report(results: List[ExperimentResult], experiment: str) -> str:
    """把结果渲染成 Markdown：先是指标表，再是与论文的差异表。"""
    sections: List[str] = []

    sections.append("### 导航指标\n")
    sections.append(
        format_table(
            [r.as_row() for r in results], ["name", "val_SR", "test_SR", "SPL", "TL"]
        )
    )

    screening_rows = [
        screening_summary_row(r.name, r.screening) for r in results if r.screening
    ]
    if screening_rows:
        sections.append("\n### 筛选指标\n")
        sections.append(
            format_table(screening_rows, ["name", "avg_candidates", "CR", "RR"])
        )

    reference = (
        REFERENCE_SCREENING if experiment == "stage_one" else REFERENCE_ABLATION
    )
    diffs = compare_with_reference(results, reference=reference)
    if diffs:
        sections.append("\n### 与论文数字的差异\n")
        sections.append(
            format_table(
                diffs, ["name", "metric", "actual", "expected", "diff", "within_tolerance"]
            )
        )

    return "\n".join(sections)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WebvlnUper 第五章实验")
    parser.add_argument("--data-dir", required=True, help="WebVLN-v1 数据目录")
    parser.add_argument("--setting", default="seen", choices=["seen", "unseen"])
    parser.add_argument(
        "--exp",
        default="ablation",
        choices=["baseline", "ablation", "two_stage", "stage_one"],
        help="实验组，对应论文 5.2 / 5.3 / 5.5 / 5.4 节",
    )
    parser.add_argument(
        "--config", default="configs/screening.yaml", help="筛选配置基准"
    )
    parser.add_argument(
        "--load", default=None,
        help="已有权重路径。给出时跳过训练直接评测（消融可共用同一份权重）",
    )
    parser.add_argument("--iters", type=int, default=None, help="训练迭代数")
    parser.add_argument("--out", default=None, help="结果 JSON 的写出路径")
    args = parser.parse_args(argv)

    from webvln.screening.config import load_yaml

    base = load_yaml(args.config)
    configs = build_experiment_configs(args.exp, base)

    results = [
        run_one(
            name, cfg, args.data_dir, setting=args.setting,
            load_path=args.load, n_iters=args.iters,
        )
        for name, cfg in configs.items()
    ]

    print(report(results, args.exp))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(
                [
                    {"name": r.name, "val": r.val, "test": r.test,
                     "screening": r.screening}
                    for r in results
                ],
                fh,
                ensure_ascii=False,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
