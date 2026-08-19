"""Episode 数据结构的单元测试。

记录字段取自官方 ``prepare_dataset`` 的输出；``path`` 元素形如
``"<websiteID>_<页面序号>"``，官方从 ``path[0].split("_")[0]`` 取网站 ID。
"""

from webvln.data.episode import Episode, infer_website_id


def encoded_record():
    """已编码的记录（``{split}_enc.json`` 形态）。"""
    return {
        "idx": 17,
        "target": "Grey Striped Socks",
        "path": ["SAHB_0", "SAHB_12", "SAHB_45"],
        "Q": "how much do they cost?",
        "A": "they cost 12 dollars",
        "text": "Target: Grey Striped Socks, how much do they cost?",
        "text_enc": [101, 1, 2, 102, 0],
        "text_words": 4,
        "answer_enc": [1, 3, 4, 0],
        "answer_enc_w_eos": [3, 4, 2, 0],
        "answer_words": 3,
    }


def raw_record():
    """未编码的记录（原始 ``{split}.json`` 形态，问答在 QA 二元组里）。"""
    return {
        "idx": 3,
        "target": "Blue Cap",
        "path": ["ES_0", "ES_7"],
        "QA": ["is it in stock?", "yes it is available"],
    }


def test_from_dict_reads_encoded_fields():
    ep = Episode.from_dict(encoded_record())
    assert ep.idx == 17
    assert ep.question == "how much do they cost?"
    assert ep.answer == "they cost 12 dollars"
    assert ep.text_words == 4
    assert ep.answer_enc_w_eos == [3, 4, 2, 0]


def test_from_dict_unpacks_qa_tuple():
    ep = Episode.from_dict(raw_record())
    assert ep.question == "is it in stock?"
    assert ep.answer == "yes it is available"
    # 未编码时文本字段留空，由 prepare_episodes 补齐。
    assert ep.text_enc == []


def test_website_id_inferred_from_path_prefix():
    ep = Episode.from_dict(encoded_record())
    assert ep.website_id == "SAHB"
    assert infer_website_id(["ES_9"]) == "ES"
    assert infer_website_id([]) == ""


def test_start_and_target_url_ids():
    ep = Episode.from_dict(encoded_record())
    assert ep.start_url_id == "SAHB_0"
    assert ep.target_url_id == "SAHB_45"
    assert ep.path_length == 3


def test_teacher_url_id_returns_none_at_path_end():
    ep = Episode.from_dict(encoded_record())
    assert ep.teacher_url_id(0) == "SAHB_12"
    assert ep.teacher_url_id(1) == "SAHB_45"
    # 走到路径末端后教师动作是 [EOA]，不再指向任何候选。
    assert ep.teacher_url_id(2) is None
    assert ep.teacher_url_id(9) is None


def test_to_official_dict_roundtrip():
    record = encoded_record()
    out = Episode.from_dict(record).to_official_dict()
    # 字段名须与官方 prepare_dataset 输出一致，才能喂给未改动的 R2RBatch。
    for key in (
        "idx",
        "target",
        "path",
        "Q",
        "A",
        "text",
        "text_enc",
        "text_words",
        "answer_enc",
        "answer_words",
        "answer_enc_w_eos",
    ):
        assert out[key] == record[key]


def test_path_elements_coerced_to_str():
    ep = Episode(idx=1, path=[0, 12])
    assert ep.path == ["0", "12"]
    assert ep.website_id == "0"
