"""Independent re-confirmation of literature citations (gate hardening).

Why this module exists: the live providers fetch literature from the
Europe PMC *search* endpoint and originally marked every returned
record VERIFIED on the strength of that one response.  That means a
single poisoned or fabricated search response — a compromised
mirror, a bad proxy, a poisoned index — could mint "verified"
evidence out of thin air: the red-team review (2026-11-25)
demonstrated fabricated PMIDs passing the gate in every live
provider, including a "PMID" that was not even numeric.

The rule now: a literature item counts as VERIFIED only when its
PMID has been **independently re-confirmed** against PubMed via
NCBI E-utilities — a different service, run by a different operator
(NCBI) than the search feed (Europe PMC, EMBL-EBI) — and the title
PubMed carries for that PMID plausibly matches the title the search
feed claimed.  One feed lying is no longer enough; two independent
services would have to lie consistently.

Status semantics follow :class:`provome.evidence.VerificationStatus`:

* ``VERIFIED`` — PubMed holds the PMID and the titles match.
* ``FAILED`` — the check was made and the source did not support
  the item: the PMID is not a numeric identifier, PubMed has no
  such record, or PubMed's title for it does not match.  Failed
  items are kept for audit and never count toward the gate.
* ``UNVERIFIED`` — the check could not be completed (the lookup
  itself failed, e.g. a network error).  Not counted either; the
  distinction matters for audit: unconfirmed is not disproven.

The lookup goes through the caller's injectable transport, like
every other fetch in this package, so tests and fixtures stay
deterministic and offline.  The cost is honest and stated: one
extra lookup per literature item (providers cap literature at two
items per topic, and cache per topic).

Standard library only.
"""

from __future__ import annotations

import html
import re
from typing import Callable, Optional, Tuple

from .evidence import VerificationStatus

__all__ = [
    "PUBMED_ESUMMARY_URL",
    "normalise_title",
    "titles_plausibly_match",
    "confirm_literature",
    "make_pubmed_confirmer",
]

#: NCBI E-utilities record lookup for PubMed — the independent
#: second source for literature re-confirmation.
PUBMED_ESUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=pubmed&retmode=json&id={pmid}"
)

_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalise_title(title: str) -> frozenset:
    """Reduce a paper title to a comparable token set.

    Europe PMC titles arrive with HTML entities (``&gt;``) and
    inline markup (``<i>``) that PubMed's record for the same paper
    does not carry; both sides are normalised the same way —
    unescape, strip tags, case-fold, tokenise — before comparison.
    """
    text = html.unescape(title or "")
    text = _TAG_RE.sub(" ", text)
    return frozenset(
        token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1
    )


def titles_plausibly_match(claimed: str, confirmed: str) -> bool:
    """Whether two titles plausibly name the same paper.

    Exact token-set equality always matches.  Otherwise the
    overlap coefficient — shared tokens over the smaller set —
    must reach 0.6 with at least two shared tokens, which absorbs
    punctuation, trailing subtitles, and minor rendering
    differences between the two services while rejecting a
    different paper riding on a real PMID.
    """
    a = normalise_title(claimed)
    b = normalise_title(confirmed)
    if not a or not b:
        return False
    if a == b:
        return True
    shared = len(a & b)
    return shared >= 2 and shared / min(len(a), len(b)) >= 0.6


def confirm_literature(
    transport: Callable,
    pmid: object,
    title: str,
) -> Tuple[VerificationStatus, str]:
    """Independently re-confirm one literature citation.

    Args:
        transport: The provider's injectable HTTP transport,
            called as ``transport(url, None)`` and expected to
            return parsed JSON.
        pmid: The PMID the search feed claimed for the record.
        title: The title the search feed claimed for the record.

    Returns:
        ``(status, note)`` — the verification status the item may
        carry, and an audit note explaining how it was reached.
    """
    pmid_text = str(pmid).strip()
    if not pmid_text.isdigit():
        return (
            VerificationStatus.FAILED,
            f"claimed PMID {pmid_text!r} is not a numeric PubMed "
            "identifier, so the citation cannot exist in PubMed; "
            "not counted as verified",
        )
    try:
        payload = transport(
            PUBMED_ESUMMARY_URL.format(pmid=pmid_text), None
        )
    except Exception:
        return (
            VerificationStatus.UNVERIFIED,
            f"independent re-confirmation of PMID {pmid_text} "
            "against PubMed could not be completed (lookup "
            "failed); item not counted as verified",
        )
    result = (payload or {}).get("result") or {}
    record = result.get(pmid_text)
    if not isinstance(record, dict) or "error" in record:
        return (
            VerificationStatus.FAILED,
            f"PMID {pmid_text} was not found in PubMed on "
            "independent re-confirmation; item not counted as "
            "verified",
        )
    confirmed_title = record.get("title") or ""
    if not titles_plausibly_match(title, confirmed_title):
        return (
            VerificationStatus.FAILED,
            f"PubMed's record for PMID {pmid_text} carries a "
            f"different title ({confirmed_title!r}) than the "
            "search feed claimed; item not counted as verified",
        )
    return (
        VerificationStatus.VERIFIED,
        f"PMID {pmid_text} independently re-confirmed against "
        "PubMed (NCBI E-utilities): the record exists and its "
        "title matches the search feed's",
    )


def make_pubmed_confirmer(
    transport: Callable,
) -> Callable[[str, str], VerificationStatus]:
    """Build a ``(pmid, title) -> VerificationStatus`` confirmer.

    This is the hook :func:`provome.rag.evidence_from_retrieval`
    accepts, so a locally indexed corpus can be held to the same
    independent re-confirmation standard as the live providers
    when a deployment wires a live transport into the bridge.
    """

    def _confirm(pmid: str, title: str) -> VerificationStatus:
        status, _note = confirm_literature(transport, pmid, title)
        return status

    return _confirm
