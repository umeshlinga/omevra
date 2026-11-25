"""Tests for the prospective prediction ledger (provome.ledger).

The ledger's one promise: a prediction recorded today cannot be
quietly rewritten tomorrow.  These tests pin the chain mechanics —
append, linkage, verification — and, most importantly, that
tampering with a sealed entry is detected at that entry.
"""

import json

import pytest

from provome.ledger import (
    GENESIS_HASH,
    PredictionLedger,
    append_entry,
    entry_hash,
    verify_chain,
)

TS = "2026-11-25T12:00:00Z"
VERSIONS = {"clinvar": "2026-11", "model": "gbm-v2"}


def _seed(path, n=3):
    ledger = PredictionLedger(path)
    for i in range(n):
        ledger.append(
            f"clinvar:{1000 + i}",
            verdict="pathogenic",
            probability=0.9,
            timestamp=TS,
            data_versions=VERSIONS,
        )
    return ledger


def test_append_seals_and_chains_entries(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = _seed(path)
    entries = ledger.entries
    assert len(entries) == 3
    assert entries[0]["prev_hash"] == GENESIS_HASH
    for prev, cur in zip(entries, entries[1:]):
        assert cur["prev_hash"] == prev["hash"]
        assert cur["seq"] == prev["seq"] + 1
    for entry in entries:
        assert entry["hash"] == entry_hash(entry)
    ok, bad = ledger.verify()
    assert ok and bad is None


def test_ledger_persists_and_reopens(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _seed(path, n=2)
    reopened = PredictionLedger(path)
    assert len(reopened) == 2
    entry = reopened.append(
        "clinvar:2000", verdict="benign", timestamp=TS,
        data_versions=VERSIONS,
    )
    assert entry["seq"] == 2
    ok, bad = verify_chain(path)
    assert ok and bad is None


def test_append_entry_convenience_function(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = append_entry(
        path, "hbb:p.glu7val", probability=0.83, timestamp=TS,
        data_versions=VERSIONS,
    )
    assert entry["probability"] == 0.83
    assert entry["verdict"] is None


def test_prediction_must_say_something(tmp_path):
    ledger = PredictionLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError):
        ledger.append("clinvar:1", timestamp=TS)
    with pytest.raises(ValueError):
        ledger.append("clinvar:1", probability=1.5, timestamp=TS)
    with pytest.raises(ValueError):
        ledger.append("", verdict="pathogenic", timestamp=TS)


def test_tampering_with_a_sealed_entry_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _seed(path, n=3)
    lines = path.read_text().splitlines()
    victim = json.loads(lines[1])
    victim["probability"] = 0.1  # quietly rewrite history
    lines[1] = json.dumps(victim, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    ok, bad = verify_chain(path)
    assert not ok
    assert bad == 1  # detected at the tampered entry itself


def test_tampering_with_linkage_is_detected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _seed(path, n=3)
    lines = path.read_text().splitlines()
    victim = json.loads(lines[2])
    victim["prev_hash"] = GENESIS_HASH
    victim["hash"] = entry_hash(victim)  # reseal the lie
    lines[2] = json.dumps(victim, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    ok, bad = verify_chain(path)
    assert not ok
    assert bad == 2


def test_deleting_an_entry_breaks_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _seed(path, n=3)
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    ok, bad = verify_chain(path)
    assert not ok


def test_tampered_ledger_is_refused_on_open(tmp_path):
    path = tmp_path / "ledger.jsonl"
    _seed(path, n=2)
    lines = path.read_text().splitlines()
    victim = json.loads(lines[0])
    victim["verdict"] = "benign"
    lines[0] = json.dumps(victim, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        PredictionLedger(path)
