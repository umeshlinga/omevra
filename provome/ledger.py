"""The prospective prediction ledger: an append-only, hash-chained
record of what Provome predicted, written before the truth is known.

Why this module exists (CEO synthesis, 2026-11-25): Provome's moat
is a trust record, and a trust record is only worth anything if it
cannot be quietly rewritten after the fact.  Every scored variant
gets one ledger entry — its identity, the calibrated probability
or verdict Provome issued, the timestamp, and the data versions
the answer rested on — and each entry carries a SHA-256 hash
chained to the previous entry's hash.  Change any earlier entry
and every later hash breaks, so tampering is detectable by anyone
with the file and :func:`verify_chain`.  Predictions are therefore
*prospective*: the entry exists, timestamped and hash-locked,
before the variant's classification is settled by the field.

Design rules:

* **Append-only.**  There is no update or delete operation.  A
  correction is a new entry that supersedes an old one by
  reference, never a rewrite.
* **Hash chain.**  ``entry.hash = sha256(canonical(entry without
  hash) )`` where the canonical form is JSON with sorted keys, and
  the entry embeds ``prev_hash`` — the previous entry's hash, or
  the genesis constant for the first entry.
* **Honest contents.**  An entry must carry a calibrated
  probability (0..1) or a verdict string, or both; an entry with
  neither is refused, because a prediction that says nothing
  cannot later be scored.  Timestamps are caller-supplied ISO
  strings (the caller's clock is the caller's responsibility;
  the chain proves order and integrity, not wall-clock truth).
* **Storage.**  One JSON object per line (JSONL).  Deliberately
  boring: the file can be read, diffed, and verified with standard
  tools, and :func:`verify_chain` needs nothing but the file.

Standard library only.  Research software.  Not for clinical use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

__all__ = [
    "GENESIS_HASH",
    "canonical_entry_bytes",
    "entry_hash",
    "PredictionLedger",
    "append_entry",
    "verify_chain",
    "main",
]

#: The prev_hash of the first entry in any chain.
GENESIS_HASH = "0" * 64

_ENTRY_FIELDS = (
    "seq",
    "variant_id",
    "verdict",
    "probability",
    "timestamp",
    "data_versions",
    "prev_hash",
)


def canonical_entry_bytes(entry: dict) -> bytes:
    """The canonical byte form an entry's hash is computed over.

    Every field except ``hash`` itself, JSON with sorted keys and
    tight separators — the same bytes on every machine, so a hash
    computed anywhere verifies everywhere.
    """
    payload = {key: entry[key] for key in _ENTRY_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def entry_hash(entry: dict) -> str:
    """SHA-256 of an entry's canonical form (which embeds prev_hash)."""
    return hashlib.sha256(canonical_entry_bytes(entry)).hexdigest()


def _validate_prediction(
    verdict: Optional[str], probability: Optional[float]
) -> None:
    if verdict is None and probability is None:
        raise ValueError(
            "a ledger entry must carry a verdict or a calibrated "
            "probability (or both): a prediction that says nothing "
            "cannot be scored later"
        )
    if verdict is not None and not str(verdict).strip():
        raise ValueError("verdict must be a non-empty string")
    if probability is not None and not (0.0 <= probability <= 1.0):
        raise ValueError("probability must lie in [0, 1]")


class PredictionLedger:
    """An append-only prediction ledger backed by a JSONL file.

    Args:
        path: The ledger file.  Created on first append if absent;
            loaded and chain-checked on open if present — a ledger
            whose chain does not verify is refused, loudly, rather
            than extended.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._entries: list[dict] = []
        if self.path.exists():
            self._entries = _read_entries(self.path)
            ok, bad = verify_entries(self._entries)
            if not ok:
                raise ValueError(
                    f"ledger at {self.path} fails chain verification "
                    f"at entry {bad}; refusing to extend a tampered "
                    "or corrupt record"
                )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[dict, ...]:
        return tuple(self._entries)

    @property
    def head_hash(self) -> str:
        """The hash the next entry will chain to."""
        if not self._entries:
            return GENESIS_HASH
        return self._entries[-1]["hash"]

    def append(
        self,
        variant_id: str,
        *,
        verdict: Optional[str] = None,
        probability: Optional[float] = None,
        timestamp: str,
        data_versions: Optional[dict] = None,
    ) -> dict:
        """Append one prediction; returns the sealed entry.

        Args:
            variant_id: The variant's identity (a stable, citable
                identifier — an HGVS string, a ClinVar Variation
                ID, a VRS id; the ledger does not interpret it).
            verdict: The categorical prediction, if one was issued.
            probability: The calibrated probability, if one was
                issued.  At least one of verdict / probability is
                required.
            timestamp: ISO-8601 timestamp of the prediction,
                supplied by the caller.
            data_versions: The data versions the prediction rested
                on (for example ClinVar release, gnomAD version,
                model version), recorded verbatim.
        """
        if not variant_id or not str(variant_id).strip():
            raise ValueError("variant_id must be a non-empty string")
        _validate_prediction(verdict, probability)
        entry = {
            "seq": len(self._entries),
            "variant_id": str(variant_id),
            "verdict": verdict,
            "probability": probability,
            "timestamp": str(timestamp),
            "data_versions": dict(data_versions or {}),
            "prev_hash": self.head_hash,
        }
        entry["hash"] = entry_hash(entry)
        self._entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self) -> tuple[bool, Optional[int]]:
        """Verify this ledger's chain; see :func:`verify_entries`."""
        return verify_entries(self._entries)


def _read_entries(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def verify_entries(entries: Sequence[dict]) -> tuple[bool, Optional[int]]:
    """Check a chain of entries end to end.

    Returns ``(True, None)`` when every entry's sequence number,
    ``prev_hash`` linkage, and recomputed hash check out, and
    ``(False, seq)`` with the sequence number of the first entry
    that fails — the point from which the record can no longer be
    trusted.
    """
    prev_hash = GENESIS_HASH
    for expected_seq, entry in enumerate(entries):
        try:
            if entry["seq"] != expected_seq:
                return False, expected_seq
            if entry["prev_hash"] != prev_hash:
                return False, expected_seq
            if entry["hash"] != entry_hash(entry):
                return False, expected_seq
        except (KeyError, TypeError):
            return False, expected_seq
        prev_hash = entry["hash"]
    return True, None


def append_entry(
    path,
    variant_id: str,
    *,
    verdict: Optional[str] = None,
    probability: Optional[float] = None,
    timestamp: str,
    data_versions: Optional[dict] = None,
) -> dict:
    """Open the ledger at ``path``, append one entry, return it."""
    ledger = PredictionLedger(path)
    return ledger.append(
        variant_id,
        verdict=verdict,
        probability=probability,
        timestamp=timestamp,
        data_versions=data_versions,
    )


def verify_chain(path) -> tuple[bool, Optional[int]]:
    """Verify the ledger file at ``path`` without extending it."""
    return verify_entries(_read_entries(Path(path)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="provome.ledger",
        description="Append to or verify the prospective prediction ledger.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_append = sub.add_parser("append", help="append one prediction")
    p_append.add_argument("--file", required=True)
    p_append.add_argument("--variant", required=True)
    p_append.add_argument("--verdict")
    p_append.add_argument("--probability", type=float)
    p_append.add_argument("--timestamp", required=True)
    p_append.add_argument(
        "--data-versions",
        default="{}",
        help="JSON object of data versions, e.g. '{\"clinvar\": \"2026-11\"}'",
    )

    p_verify = sub.add_parser("verify", help="verify a ledger file's chain")
    p_verify.add_argument("--file", required=True)

    args = parser.parse_args(argv)
    if args.command == "append":
        entry = append_entry(
            args.file,
            args.variant,
            verdict=args.verdict,
            probability=args.probability,
            timestamp=args.timestamp,
            data_versions=json.loads(args.data_versions),
        )
        print(json.dumps(entry, sort_keys=True))
        return 0
    ok, bad = verify_chain(args.file)
    if ok:
        print(f"OK: chain verifies ({args.file})")
        return 0
    print(f"FAILED: chain broken at entry {bad} ({args.file})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
