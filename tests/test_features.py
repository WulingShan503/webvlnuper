"""特征表查询的单元测试。

只覆盖 ``FeatureStore``——``ResNet152Extractor`` 需要 torch 与预训练权重，
本机无法运行。重点验证官方 ``make_candidate`` 中不一致的缺失处理
（text 有回退、screenshot 无回退）在这里被统一为零向量并计数。
"""

import os
import pickle

from webvln.data.features import (
    DEFAULT_FEATURE_SIZE,
    IMG_FEATS_FILE,
    PAPER_CONCAT_DIM,
    SCREENSHOT_FEATS_FILE,
    TEXT_FEATS_FILE,
    FeatureStore,
)


def store(feature_size=4):
    return FeatureStore(
        text_feats={"c1": [1, 1, 1, 1]},
        img_feats={"img_0.jpg": [2, 2, 2, 2]},
        screenshot_feats={"c1": [3, 3, 3, 3]},
        feature_size=feature_size,
    )


def test_paper_concat_dim_matches_formula():
    # 式 (3.3.1)：768 + 2048 + 2048 = 4864
    assert PAPER_CONCAT_DIM == 4864
    assert DEFAULT_FEATURE_SIZE == 512  # 官方 env.py:79


def test_candidate_features_follow_official_order():
    fs = store()
    feats = fs.candidate_features("c1", ["img_0.jpg"])
    # 顺序须为 [文本, 按钮图, 截图]，candidate_encoder 依赖这个次序。
    assert feats == [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]


def test_missing_keys_fall_back_to_zeros_and_count():
    fs = store()
    assert fs.text("nope") == [0.0] * 4
    assert fs.screenshot("nope") == [0.0] * 4
    assert fs.image(["absent.jpg"]) == [0.0] * 4
    assert fs.missing == {"text": 1, "img": 1, "screenshot": 1}


def test_candidate_without_images_gets_zero_image_feature():
    fs = store()
    # 无图元素不算缺失：官方 getStates 里就直接填 np.zeros。
    assert fs.image([]) == [0.0] * 4
    assert fs.missing["img"] == 0


def test_image_uses_first_entry_only():
    fs = FeatureStore(img_feats={"a.jpg": [9], "b.jpg": [8]}, feature_size=1)
    assert fs.image(["a.jpg", "b.jpg"]) == [9]


def test_coverage_reports_hit_rate():
    fs = store()
    cov = fs.coverage(["c1", "c2", "c3", "c4"])
    assert cov["text"] == 0.25
    assert cov["screenshot"] == 0.25
    # 空输入不应除零。
    assert fs.coverage([])["text"] == 0.0


def test_from_dir_loads_three_tables(tmp_path):
    for name, table in (
        (TEXT_FEATS_FILE, {"c1": [1]}),
        (IMG_FEATS_FILE, {"i.jpg": [2]}),
        (SCREENSHOT_FEATS_FILE, {"c1": [3]}),
    ):
        with open(os.path.join(tmp_path, name), "wb") as fh:
            pickle.dump(table, fh)

    fs = FeatureStore.from_dir(str(tmp_path), feature_size=1)
    assert fs.candidate_features("c1", ["i.jpg"]) == [[1], [2], [3]]


def test_from_dir_tolerates_missing_tables(tmp_path):
    # --test_only 模式下官方不加载 img_feats，全部按钮图应为零向量。
    with open(os.path.join(tmp_path, TEXT_FEATS_FILE), "wb") as fh:
        pickle.dump({"c1": [7]}, fh)

    fs = FeatureStore.from_dir(str(tmp_path), feature_size=1)
    assert fs.text("c1") == [7]
    assert fs.image(["i.jpg"]) == [0.0]
    assert fs.screenshot("c1") == [0.0]
