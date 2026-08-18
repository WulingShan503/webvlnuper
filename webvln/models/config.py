"""WebVLN-Net 的结构与训练超参。

集中一处定义，避免论文数字与官方实现的差异散落在各模块里。
凡论文与官方 ``r2r_src/`` 不一致的项，默认值取**官方实现**
（论文报告的 SR / SPL 是用那份代码跑出来的），并在注释中注明论文写法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WebVLNConfig:
    """模型与训练配置。

    Attributes:
        hidden_size: 隐层维度。BERT-base 为 768（论文 3.2 节）。
        num_attention_heads: 注意力头数。
        intermediate_size: FFN 中间层维度，论文 3.5 节记为 3072。
        la_layers: 语言分支自注意力层数。官方 ``vlnbert_init.py`` 为 9。
        vl_layers: 跨模态层数。官方为 2；论文 3.5 节写 4。
        qa_layers: 回答头层数。官方 ``QADecoder(num_layers=2)``；论文写 4。
        feature_size: 单段候选特征维度。官方 ``env.py`` 中 ``feature_size=512``。
        tokens_per_candidate: 每个候选占用的 token 数。官方把文本 / 按钮图 /
            截图裁剪三段特征放成三个独立 token，而非论文 3.3 节的
            768+2048+2048=4864 维拼接向量。
        max_action_len: 单个 episode 最大步数。官方 ``--maxAction 10``。
        ignore_id: 交叉熵忽略的标签值，用于已结束的样本。
    """

    # --- 结构 ---
    hidden_size: int = 768
    num_attention_heads: int = 8
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.4
    attention_probs_dropout_prob: float = 0.4
    layer_norm_eps: float = 1e-12
    la_layers: int = 9
    vl_layers: int = 2
    qa_layers: int = 2

    # --- 候选特征 ---
    feature_size: int = 512
    tokens_per_candidate: int = 3
    feat_dropout: float = 0.4

    # --- 文本 ---
    vocab_size: int = 30522  # bert-base-uncased
    max_position_embeddings: int = 512
    max_instr_len: int = 80
    max_answer_len: int = 30
    bert_model_name: str = "bert-base-uncased"

    # --- 训练 ---
    # 官方 run/train.bash 用 1e-5；论文 3.9 / 5.1 节写 1e-4。
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    batch_size: int = 4  # 官方 4；论文写 8
    max_iters: int = 200_000
    # 论文 3.9 节：140,000 迭代后每 1,000 步验证一次
    eval_start_iter: int = 140_000
    eval_interval: int = 1_000
    max_action_len: int = 10

    # --- 损失 ---
    # 式 (3.8.1) L_total = L_nav + λ·L_ans
    qa_loss_weight: float = 1.0
    label_smoothing: float = 0.1
    ignore_id: int = -100

    # --- 特殊 token ---
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102

    device: str = "cuda"
    seed: int = 0
    #: 加载官方权重时的路径（如 Downloads/ckpt/best_val）
    load_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 "
                f"num_attention_heads ({self.num_attention_heads}) 整除"
            )

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def candidate_seq_len(self, n_candidates: int) -> int:
        """候选序列的 token 数。

        每个候选占 ``tokens_per_candidate`` 个 token，末尾额外留一位给 [EOA]。
        对应官方 ``_candidate_variable`` 中的 ``len(ob['candidate'])*3 + 1``。
        """
        return n_candidates * self.tokens_per_candidate + 1

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


#: 论文 3.5 / 3.7 节所述的结构（4 层跨模态、4 层回答头）。
#: 供 5.x 节对照实验使用——官方权重与之不兼容，需从头训练。
PAPER_CONFIG = WebVLNConfig(
    num_attention_heads=8,
    vl_layers=4,
    qa_layers=4,
    learning_rate=1e-4,
    batch_size=8,
)
