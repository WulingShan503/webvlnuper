"""第三章模型配置的单元测试。

只覆盖 ``config.py`` —— 它不依赖 torch，可在本机运行。
``language.py`` / ``candidate_encoder.py`` 等需要 torch 的模块
待环境具备后再补测试（见 ROADMAP 阶段 3）。
"""

import pytest

from webvln.models.config import PAPER_CONFIG, WebVLNConfig


def test_default_matches_official_implementation():
    # 默认值取官方 r2r_src/，因为论文报告的 SR / SPL 是用那份代码跑出来的。
    cfg = WebVLNConfig()
    assert cfg.hidden_size == 768
    assert cfg.feature_size == 512  # env.py:79
    assert cfg.tokens_per_candidate == 3  # agent.py:_candidate_variable
    assert cfg.la_layers == 9 and cfg.vl_layers == 2  # vlnbert_init.py
    assert cfg.qa_layers == 2  # QADecoder(num_layers=2)
    assert cfg.learning_rate == 1e-5  # run/train.bash
    assert cfg.batch_size == 4
    assert cfg.max_iters == 200_000
    assert cfg.max_action_len == 10


def test_paper_config_records_thesis_values():
    # 论文 3.5 / 3.7 / 5.1 节的写法，供对照实验。
    assert PAPER_CONFIG.vl_layers == 4
    assert PAPER_CONFIG.qa_layers == 4
    assert PAPER_CONFIG.learning_rate == 1e-4
    assert PAPER_CONFIG.batch_size == 8
    assert PAPER_CONFIG.num_attention_heads == 8


def test_eval_schedule_follows_paper():
    # 论文 3.9 节：140,000 迭代后每 1,000 步验证。
    cfg = WebVLNConfig()
    assert cfg.eval_start_iter == 140_000
    assert cfg.eval_interval == 1_000


def test_attention_head_size_divides_evenly():
    cfg = WebVLNConfig(hidden_size=768, num_attention_heads=8)
    assert cfg.attention_head_size == 96


def test_rejects_indivisible_head_count():
    with pytest.raises(ValueError, match="必须能被"):
        WebVLNConfig(hidden_size=768, num_attention_heads=7)


def test_candidate_seq_len_matches_official_formula():
    # 官方：len(ob['candidate']) * 3 + 1，末位留给 [EOA]。
    cfg = WebVLNConfig()
    assert cfg.candidate_seq_len(0) == 1  # 仅 [EOA]
    assert cfg.candidate_seq_len(1) == 4
    assert cfg.candidate_seq_len(45) == 136  # 论文报告的平均候选数


def test_candidate_seq_len_respects_custom_token_count():
    cfg = WebVLNConfig(tokens_per_candidate=1)
    assert cfg.candidate_seq_len(45) == 46


def test_loss_weight_defaults_to_one():
    # 式 (3.8.1) L_total = L_nav + λ·L_ans
    assert WebVLNConfig().qa_loss_weight == 1.0


def test_as_dict_roundtrips():
    cfg = WebVLNConfig(vl_layers=4)
    assert WebVLNConfig(**cfg.as_dict()) == cfg
