"""3.3 节：候选元素的多模态特征。

论文 3.3 节的三段特征与维度：

    文本   BERT 编码元素文本 / alt          768
    按钮图 ResNet152 (ImageNet 预训练)      2048
    截图   元素区域在页面截图上的裁剪        2048
                                    拼接 = 4864  式 (3.3.1)

官方实现不做拼接：三段各占一个 token、每段统一投影到 512 维
（``env.py`` 中 ``feature_size=512``），预抽好后存成三张 pkl 表：

    text_feats.pkl              键为 clickable_id
    img_feats.pkl               键为图片文件名（候选 ``imgs[0]``）
    screenshot_crop_feats.pkl   键为 clickable_id

本模块提供两条路径：``FeatureStore`` 读取官方 pkl（复现基线用），
``ResNet152Extractor`` 从原始图片重抽（数据缺失或换特征做消融时用）。
两者都以 ``feature_size`` 为输出维度，下游 ``candidate_encoder`` 不必区分来源。
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: 官方三张特征表的文件名。
TEXT_FEATS_FILE = "text_feats.pkl"
IMG_FEATS_FILE = "img_feats.pkl"
SCREENSHOT_FEATS_FILE = "screenshot_crop_feats.pkl"

#: 官方统一的单段特征维度（``env.py:79``）。
DEFAULT_FEATURE_SIZE = 512

#: 论文式 (3.3.1) 的三段原始维度。
PAPER_TEXT_DIM = 768
PAPER_IMAGE_DIM = 2048
PAPER_CONCAT_DIM = PAPER_TEXT_DIM + PAPER_IMAGE_DIM * 2  # 4864


def load_pickle(path: str) -> Any:
    """读取 pkl 特征表。"""
    with open(path, "rb") as fh:
        return pickle.load(fh)


class FeatureStore:
    """三张官方特征表的统一查询入口。

    官方 ``make_candidate`` 里对缺失键的处理不一致：``text_features``
    用 ``in`` 判断后回退零向量，而 ``screenshot_features`` 直接下标索引——
    该键缺失就会 KeyError。这里统一为「缺失即零向量」并计数，
    使数据不全时训练仍能跑通，同时把缺失规模暴露出来而非静默吞掉。

    Attributes:
        feature_size: 单段特征维度。
        missing: 各表的缺失查询次数，供数据完整性检查。
    """

    def __init__(
        self,
        text_feats: Optional[Mapping[str, Any]] = None,
        img_feats: Optional[Mapping[str, Any]] = None,
        screenshot_feats: Optional[Mapping[str, Any]] = None,
        feature_size: int = DEFAULT_FEATURE_SIZE,
    ) -> None:
        self.text_feats = text_feats or {}
        self.img_feats = img_feats or {}
        self.screenshot_feats = screenshot_feats or {}
        self.feature_size = feature_size
        self.missing: Dict[str, int] = {"text": 0, "img": 0, "screenshot": 0}

    @classmethod
    def from_dir(
        cls, data_dir: str, feature_size: int = DEFAULT_FEATURE_SIZE
    ) -> "FeatureStore":
        """从数据目录加载三张表。

        缺失的表按空表处理：``--test_only`` 模式下官方就不加载 img_feats
        （``read_img_features`` 返回 None），此时全部按钮图特征为零向量。
        """
        return cls(
            text_feats=_load_if_exists(os.path.join(data_dir, TEXT_FEATS_FILE)),
            img_feats=_load_if_exists(os.path.join(data_dir, IMG_FEATS_FILE)),
            screenshot_feats=_load_if_exists(
                os.path.join(data_dir, SCREENSHOT_FEATS_FILE)
            ),
            feature_size=feature_size,
        )

    def zeros(self) -> List[float]:
        """缺失特征的占位向量。

        用零向量而非随机值：零在注意力里等价于「无信息」，
        随机噪声会让模型把缺失特征当成可区分的信号去拟合。
        """
        return [0.0] * self.feature_size

    def text(self, clickable_id: str) -> Any:
        """取候选文本特征。"""
        return self._get(self.text_feats, clickable_id, "text")

    def screenshot(self, clickable_id: str) -> Any:
        """取截图裁剪特征。"""
        return self._get(self.screenshot_feats, clickable_id, "screenshot")

    def image(self, imgs: Sequence[str]) -> Any:
        """取按钮图特征。

        官方只用 ``imgs[0]``：一个可点击元素可能含多张图（如商品图加图标），
        首张是主体图。无图元素返回零向量。
        """
        if not imgs:
            return self.zeros()
        return self._get(self.img_feats, imgs[0], "img")

    def candidate_features(
        self, clickable_id: str, imgs: Sequence[str] = ()
    ) -> List[Any]:
        """按官方顺序返回一个候选的三段特征。

        顺序为 [文本, 按钮图, 截图]，与 ``make_candidate`` 中
        ``[text_features, feature[idx], screenshot_features]`` 一致——
        ``candidate_encoder`` 依赖这个次序把三段放到对应的 token 位上。
        """
        return [
            self.text(clickable_id),
            self.image(imgs),
            self.screenshot(clickable_id),
        ]

    def coverage(self, clickable_ids: Sequence[str]) -> Dict[str, float]:
        """统计给定候选在三张表中的命中率。

        训练前跑一次即可发现「特征表与 map.json 版本不匹配」这类问题——
        否则模型会在大量零向量上训练，表现为 SR 停在随机水平却无报错。
        """
        n = len(clickable_ids) or 1
        return {
            "text": sum(1 for c in clickable_ids if c in self.text_feats) / n,
            "screenshot": sum(
                1 for c in clickable_ids if c in self.screenshot_feats
            ) / n,
        }

    def _get(self, table: Mapping[str, Any], key: str, kind: str) -> Any:
        if key in table:
            return table[key]
        self.missing[kind] += 1
        return self.zeros()


def _load_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    return load_pickle(path) or {}


class ResNet152Extractor:
    """ResNet152 图像特征抽取器（论文 3.3 节）。

    去掉最后的分类层，取全局平均池化后的 2048 维向量。
    官方给出的 pkl 已是 512 维，故默认再接一层线性投影对齐；
    ``project_to`` 设为 None 时保留原始 2048 维，供式 (3.3.1) 的
    拼接形式做对照实验。

    torch 与 torchvision 延迟导入：本机无 GPU 环境时其余数据代码仍可运行。
    """

    def __init__(
        self,
        project_to: Optional[int] = DEFAULT_FEATURE_SIZE,
        device: str = "cpu",
        weights: str = "IMAGENET1K_V1",
    ) -> None:
        self.project_to = project_to
        self.device = device
        self.weights = weights
        self._model = None
        self._transform = None
        self._projection = None

    def _build(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        backbone = models.resnet152(weights=self.weights)
        # 用 Identity 替换 fc 而非切 children()：后者会丢掉 flatten，
        # 输出仍带 1x1 空间维，拼接时维度对不上。
        backbone.fc = nn.Identity()
        backbone.eval().to(self.device)
        self._model = backbone

        # ImageNet 标准预处理。尺寸与归一化须与预训练一致，
        # 否则特征分布偏移，2048 维向量的语义不再可比。
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.project_to is not None:
            self._projection = nn.Linear(PAPER_IMAGE_DIM, self.project_to).to(
                self.device
            )
            self._projection.eval()

    def extract(self, images: Sequence[Any]) -> Any:
        """抽取一批 PIL 图像的特征。

        Args:
            images: PIL Image 列表。

        Returns:
            形状 [n, project_to or 2048] 的张量，已 detach。
        """
        import torch

        if self._model is None:
            self._build()

        batch = torch.stack([self._transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self._model(batch)
            if self._projection is not None:
                feats = self._projection(feats)
        return feats.cpu()

    def extract_dir(self, image_dir: str, exts: Sequence[str] = (".jpg", ".png")) -> Dict[str, Any]:
        """抽取整个目录的图片，返回 ``{文件名: 特征}``。

        键用文件名而非完整路径，与官方 ``img_feats.pkl`` 的键
        （候选记录里的 ``imgs`` 元素）保持一致。
        """
        from PIL import Image

        names = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith(tuple(exts))
        )
        out: Dict[str, Any] = {}
        for name in names:
            with Image.open(os.path.join(image_dir, name)) as img:
                feats = self.extract([img.convert("RGB")])
            out[name] = feats[0]
        return out
