"""第三章：WebVLN-Net 基线模型复现。

模型基于 VLN-BERT (Hong et al., 2021)，含三个阶段：

    初始化 (initialisation)  BERT 编码问题 Q 与辅助描述 D，[CLS] 作为初始状态
    导航 (navigation)        状态 token 与候选 token 在跨模态 Transformer 中交互
    回答 (answering)         以最终状态 s_[EOA] 自回归解码答案

与论文描述的差异见各模块 docstring 中的说明：官方实现把候选的三段特征
拆成三个独立 token 而非拼接成 4864 维向量，动作 logits 直接取跨模态
注意力分数而非独立线性头。实现以官方为准，论文写法在注释中标注。
"""

from webvln.models.config import PAPER_CONFIG, WebVLNConfig

#: 需要 torch 的组件按需导入，避免仅读取配置时就拉起 torch。
#: 用法：``from webvln.models.webvln_net import WebVLNNet``
__all__ = ["WebVLNConfig", "PAPER_CONFIG"]
