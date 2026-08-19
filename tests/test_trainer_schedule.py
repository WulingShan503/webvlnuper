"""验证调度与模型选择的单元测试。

``Trainer`` 的前向需要 torch，但验证时机（论文 3.9 节：140,000 迭代后
每 1,000 步）与 Best Score = SR + WUPS0.9 的比较是纯 Python 逻辑，
这里用不构建损失函数的方式绕开 torch 依赖直接测。
"""

from webvln.models.config import WebVLNConfig
from webvln.train.trainer import Trainer


def trainer(**cfg_kwargs):
    """构造只用于调度逻辑的 Trainer。

    传入 ``loss_fn`` 占位对象，避免 ``__init__`` 去导入需要 torch 的
    ``WebVLNLoss``；model 与 env 在调度逻辑里用不到。
    """
    return Trainer(
        model=None,
        env=None,
        config=WebVLNConfig(**cfg_kwargs),
        loss_fn=object(),
    )


def test_no_validation_before_eval_start_iter():
    t = trainer()
    # 前 140,000 步不验证：早期模型的分数没有选择价值，
    # 而每次验证要跑完 1,262 条验证集。
    assert t.should_validate(1) is False
    assert t.should_validate(139_000) is False
    assert t.should_validate(140_000) is True


def test_validates_every_interval_after_start():
    t = trainer()
    assert t.should_validate(141_000) is True
    assert t.should_validate(141_500) is False
    assert t.should_validate(200_000) is True


def test_custom_schedule():
    t = trainer(eval_start_iter=100, eval_interval=50)
    assert t.should_validate(50) is False
    assert t.should_validate(100) is True
    assert t.should_validate(150) is True
    assert t.should_validate(175) is False


def test_update_best_uses_sr_plus_wups9():
    t = trainer()
    # 论文 3.9 节的准则。首次一定是最优。
    assert t.update_best({"SR": 38.35, "WUPS0.9": 24.43}) is True
    assert t.best_score == 38.35 + 24.43


def test_higher_sr_with_lower_wups_can_lose():
    t = trainer()
    t.update_best({"SR": 38.35, "WUPS0.9": 24.43})  # 62.78
    # SR 更高但 WUPS 掉得更多，合计更低，不应被选为最优——
    # 只看 SR 会选出「找对页面但答不对」的模型。
    assert t.update_best({"SR": 39.12, "WUPS0.9": 22.00}) is False
    assert t.update_best({"SR": 39.12, "WUPS0.9": 24.00}) is True


def test_equal_score_is_not_an_improvement():
    t = trainer()
    t.update_best({"SR": 38.0, "WUPS0.9": 24.0})
    assert t.update_best({"SR": 38.0, "WUPS0.9": 24.0}) is False


def test_missing_metrics_default_to_zero():
    t = trainer()
    assert t.update_best({}) is True
    assert t.best_score == 0.0
