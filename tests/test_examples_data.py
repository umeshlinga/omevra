"""Tests that the bundled worked examples are internally true."""

import json
from pathlib import Path

from omevra.variant import parse_coding_hgvs, parse_protein_hgvs

DATA = Path(__file__).parent.parent / "omevra" / "data"


def _examples():
    with (DATA / "examples.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def test_examples_file_is_labelled_illustrative():
    payload = _examples()
    assert "Illustrative" in payload["label"]
    assert len(payload["examples"]) >= 4


def test_cftr_example_parses_and_is_in_frame():
    examples = {e["id"]: e for e in _examples()["examples"]}
    cftr = examples["cftr-phe508del"]
    coding = parse_coding_hgvs(cftr["coding_hgvs"])
    protein = parse_protein_hgvs(cftr["protein_hgvs"])
    assert coding.length_change == -3  # in-frame: multiple of three
    assert protein.position == 508
    assert cftr["disease"] == "Cystic fibrosis"


def test_brca1_example_is_a_frameshift_deletion():
    examples = {e["id"]: e for e in _examples()["examples"]}
    brca1 = examples["brca1-founder-del"]
    coding = parse_coding_hgvs(brca1["coding_hgvs"])
    assert coding.length_change == -2  # not a multiple of three
    assert coding.length_change % 3 != 0


def test_all_stated_notations_parse():
    for entry in _examples()["examples"]:
        if entry["coding_hgvs"]:
            parse_coding_hgvs(entry["coding_hgvs"])
        if entry["protein_hgvs"]:
            parse_protein_hgvs(entry["protein_hgvs"])


def test_demo_evidence_file_topics_have_items():
    with (DATA / "demo_evidence.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "Illustrative" in payload["label"] or "illustrative" in payload["label"]
    topics = payload["topics"]
    assert "cftr:p.phe508del" in topics
    for topic in topics.values():
        assert topic["items"], "each demo topic needs at least one item"
        for item in topic["items"]:
            assert item["citation"]
            assert item["statement"]
