"""数据集加载与批次取样的单元测试。

覆盖三处易错点：``_enc.json`` 优先于 ``{split}.json``、
批次末尾回绕取满、以及划分规模与论文 2.3 节（8,960 / 1,262 / 4,603）的核对。
"""

import json
import os

import pytest

from webvln.data.dataset import (
    SPLIT_SIZES,
    WebVLNDataset,
    load_split,
    prepare_episodes,
    save_encoded,
    split_path,
)
from webvln.data.episode import Episode
from tests.test_data_text import FakeTokenizer


def make_records(n, prefix="SAHB"):
    return [
        {
            "idx": i,
            "target": f"item {i}",
            "path": [f"{prefix}_0", f"{prefix}_{i + 1}"],
            "QA": [f"question about item {i}?", f"answer for item {i}"],
        }
        for i in range(n)
    ]


def write_split(tmp_path, split, records, setting="seen", encoded=False):
    d = tmp_path / setting
    d.mkdir(parents=True, exist_ok=True)
    name = f"{split}_enc.json" if encoded else f"{split}.json"
    (d / name).write_text(json.dumps(records), encoding="utf-8")


def episodes(n):
    return prepare_episodes(make_records(n), FakeTokenizer())


def test_split_path_layout():
    p = split_path("Data", "seen", "train", encoded=True)
    assert p == os.path.join("Data", "seen", "train_enc.json")


def test_load_split_prefers_encoded_file(tmp_path):
    write_split(tmp_path, "val", make_records(2), encoded=False)
    enc = [dict(r, text_enc=[101, 102], text="cached") for r in make_records(2)]
    write_split(tmp_path, "val", enc, encoded=True)

    records = load_split(str(tmp_path), "seen", "val")
    # _enc.json 是编码缓存，存在时应优先读取，避免重跑 WordPiece。
    assert records[0]["text"] == "cached"


def test_load_split_falls_back_to_raw(tmp_path):
    write_split(tmp_path, "test", make_records(3), encoded=False)
    assert len(load_split(str(tmp_path), "seen", "test")) == 3


def test_prepare_episodes_encodes_missing_fields():
    eps = episodes(2)
    assert eps[0].text.startswith("Target: item 0,")
    assert eps[0].text_enc[0] == 101
    assert eps[0].answer_enc[0] == 1  # [unused0]
    assert eps[0].answer_enc_w_eos[eps[0].answer_words - 1] == 2  # [unused1]


def test_prepare_episodes_keeps_existing_encoding():
    record = dict(make_records(1)[0], text="preset", text_enc=[101, 5, 102], text_words=3)
    eps = prepare_episodes([record])  # 无需 tokenizer
    assert eps[0].text == "preset"
    assert eps[0].text_enc == [101, 5, 102]


def test_prepare_episodes_requires_tokenizer_when_unencoded():
    with pytest.raises(ValueError):
        prepare_episodes(make_records(1))


def test_next_minibatch_wraps_to_full_batch():
    ds = WebVLNDataset(episodes(5), split="val", batch_size=2)
    assert len(ds.next_minibatch()) == 2
    assert len(ds.next_minibatch()) == 2
    # 只剩 1 条时须从头回绕补满：训练按迭代计数，每次迭代都要满批。
    last = ds.next_minibatch()
    assert len(last) == 2
    assert ds.ix == 1


def test_val_split_is_not_shuffled():
    eps = episodes(4)
    ds = WebVLNDataset(eps, split="val", batch_size=4)
    assert [e.idx for e in ds.episodes] == [0, 1, 2, 3]


def test_train_split_is_shuffled_deterministically():
    order_a = [e.idx for e in WebVLNDataset(episodes(20), split="train").episodes]
    order_b = [e.idx for e in WebVLNDataset(episodes(20), split="train").episodes]
    # 同种子必须同顺序，否则复现实验拿不到相同结果。
    assert order_a == order_b
    assert order_a != list(range(20))


def test_iter_batches_covers_each_episode_once():
    ds = WebVLNDataset(episodes(5), split="test", batch_size=2)
    seen = [e.idx for batch in ds.iter_batches() for e in batch]
    assert sorted(seen) == list(range(5))


def test_reset_epoch_rewinds_index():
    ds = WebVLNDataset(episodes(4), split="val", batch_size=2)
    ds.next_minibatch()
    ds.reset_epoch()
    assert ds.ix == 0
    assert [e.idx for e in ds.next_minibatch()] == [0, 1]


def test_tile_one_repeats_single_episode():
    ds = WebVLNDataset(episodes(3), split="val", batch_size=3)
    batch = ds.next_minibatch(tile_one=True)
    assert len({id(e) for e in batch}) == 1
    assert batch[0].idx == 0


def test_check_size_matches_thesis_numbers():
    assert SPLIT_SIZES == {"train": 8960, "val": 1262, "test": 4603}
    ds = WebVLNDataset(episodes(3), split="val")
    msg = ds.check_size()
    assert msg is not None and "1262" in msg
    with pytest.raises(ValueError):
        ds.check_size(strict=True)


def test_check_size_silent_when_split_unknown():
    ds = WebVLNDataset(episodes(2), split="debug")
    assert ds.check_size(strict=True) is None


def test_save_encoded_roundtrip(tmp_path):
    eps = episodes(2)
    path = save_encoded(eps, str(tmp_path), "seen", "train")
    assert os.path.exists(path)
    reloaded = prepare_episodes(json.load(open(path, encoding="utf-8")))
    assert [e.text_enc for e in reloaded] == [e.text_enc for e in eps]


def test_from_dir_builds_dataset(tmp_path):
    write_split(tmp_path, "train", make_records(6))
    ds = WebVLNDataset.from_dir(
        str(tmp_path), setting="seen", split="train",
        tokenizer=FakeTokenizer(), batch_size=3,
    )
    assert ds.size() == 6
    assert len(ds.next_minibatch()) == 3
    assert all(isinstance(e, Episode) for e in ds.episodes)
