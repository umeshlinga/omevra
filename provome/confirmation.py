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

from .evidence import SupportDirection, VerificationStatus

__all__ = [
    "PUBMED_ESUMMARY_URL",
    "CLINVAR_ESUMMARY_URL",
    "normalise_title",
    "titles_plausibly_match",
    "record_is_retracted",
    "confirm_literature",
    "confirm_literature_detail",
    "make_pubmed_confirmer",
    "clinvar_variation_id",
    "variant_short_from_topic",
    "confirm_clinvar",
]

#: NCBI E-utilities record lookup for PubMed — the independent
#: second source for literature re-confirmation.
PUBMED_ESUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=pubmed&retmode=json&id={pmid}"
)

#: NCBI E-utilities record lookup for ClinVar — the served record a
#: ClinVar-shaped evidence item is re-confirmed against.
CLINVAR_ESUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=clinvar&retmode=json&id={variation_id}"
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


#: PubMed publication types that mark a record as retracted (the
#: article itself) or as a retraction notice.  Either way, the
#: record cannot support evidence.
_RETRACTED_PUBTYPES = frozenset(
    {"retracted publication", "retraction of publication"}
)


def record_is_retracted(record: dict) -> bool:
    """Whether a served PubMed record carries a retraction notice.

    Two shapes, both from the record itself: a ``pubtype`` of
    "Retracted Publication" (or "Retraction of Publication" for
    the notice), or a comments-corrections link whose ``reftype``
    is a retraction link (the retracted article's record points at
    its retraction notice).  This is an integrity flag on the
    record, never an accusation against any author: papers are
    retracted for honest error as well as for misconduct, and the
    check cannot tell the difference — it does not need to, because
    a retracted paper cannot support a claim either way.
    """
    pubtypes = {
        str(t).strip().lower() for t in record.get("pubtype") or []
    }
    if pubtypes & _RETRACTED_PUBTYPES:
        return True
    for link in record.get("commentscorrections") or []:
        if str(link.get("reftype", "")).strip().lower().startswith(
            "retraction"
        ):
            return True
    return False


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
    status, note, _retracted = confirm_literature_detail(
        transport, pmid, title
    )
    return status, note


def confirm_literature_detail(
    transport: Callable,
    pmid: object,
    title: str,
) -> Tuple[VerificationStatus, str, bool]:
    """Re-confirm one literature citation, with the retraction flag.

    Identical checks to :func:`confirm_literature`; the third
    return value says whether the served record carries a
    retraction notice, so callers that keep per-item audit notes
    (the verification agent) can attach the integrity warning to
    the item itself.  A retracted record returns ``FAILED``: the
    paper exists, but it has been withdrawn by its journal and
    cannot count as verified support.
    """
    pmid_text = str(pmid).strip()
    if not pmid_text.isdigit():
        return (
            VerificationStatus.FAILED,
            f"claimed PMID {pmid_text!r} is not a numeric PubMed "
            "identifier, so the citation cannot exist in PubMed; "
            "not counted as verified",
            False,
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
            False,
        )
    result = (payload or {}).get("result") or {}
    record = result.get(pmid_text)
    if not isinstance(record, dict) or "error" in record:
        return (
            VerificationStatus.FAILED,
            f"PMID {pmid_text} was not found in PubMed on "
            "independent re-confirmation; item not counted as "
            "verified",
            False,
        )
    if record_is_retracted(record):
        return (
            VerificationStatus.FAILED,
            f"integrity warning: PubMed's record for PMID "
            f"{pmid_text} carries a retraction notice, so the "
            "paper has been withdrawn and cannot count as "
            "verified support; this flag records the withdrawal, "
            "not any judgement about its cause",
            True,
        )
    confirmed_title = record.get("title") or ""
    if not titles_plausibly_match(title, confirmed_title):
        return (
            VerificationStatus.FAILED,
            f"PubMed's record for PMID {pmid_text} carries a "
            f"different title ({confirmed_title!r}) than the "
            "search feed claimed; item not counted as verified",
            False,
        )
    return (
        VerificationStatus.VERIFIED,
        f"PMID {pmid_text} independently re-confirmed against "
        "PubMed (NCBI E-utilities): the record exists and its "
        "title matches the search feed's",
        False,
    )


# ---------------------------------------------------------------------
# ClinVar re-confirmation (strengthening, 2026-11-25)
#
# The fraud battery's two misses shared one root: literature items
# are independently re-confirmed, but non-literature items were
# trusted at the provider's word.  A raw item could declare itself
# ClinVar, invent a Variation ID, stamp itself VERIFIED, and claim
# SUPPORTS for a record ClinVar actually classifies Benign — and
# nothing in the path ever asked ClinVar.  confirm_clinvar is the
# mirror of confirm_literature for the curated source: the item's
# claimed Variation ID is resolved against the served ClinVar
# record (NCBI E-utilities esummary, the same endpoint the live
# provider reads), the served record must be about the variant the
# item's topic names (identity match, using the live provider's own
# notation rules), and the item's direction is *derived from the
# served classification* — never taken from the item's own claim.
# ---------------------------------------------------------------------

_CLINVAR_ID_PATTERNS: tuple = (
    re.compile(r"\bClinVar\s+Variation\s+ID\s*:?\s*0*(\d{1,12})\b", re.IGNORECASE),
    re.compile(r"\bVCV0*(\d{1,12})\b", re.IGNORECASE),
    re.compile(r"clinvar/variation/0*(\d{1,12})\b", re.IGNORECASE),
)

_TOPIC_PROTEIN_RE = re.compile(r"^p\.([A-Za-z]{3}\d+\S*)$")


def clinvar_variation_id(*texts: str) -> Optional[str]:
    """The ClinVar Variation ID claimed in a citation or locator."""
    for text in texts:
        if not text:
            continue
        for pattern in _CLINVAR_ID_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
    return None


def variant_short_from_topic(topic_key: str) -> Optional[str]:
    """The protein-change short form a topic key names, if concrete.

    ``"cftr:p.phe508del"`` -> ``"Phe508del"``.  Returns ``None``
    when the topic does not identify a concrete protein change
    (a placeholder topic, a gene-only topic): without a claimed
    identity there is nothing to match a served record against,
    and the confirmer's honest answer is that it cannot run.
    """
    _gene, sep, hgvs = (topic_key or "").partition(":")
    if not sep:
        return None
    match = _TOPIC_PROTEIN_RE.match(hgvs.strip().lower())
    if not match:
        return None
    short = match.group(1)
    return short[:1].upper() + short[1:]


def confirm_clinvar(
    transport: Callable,
    variation_id: object,
    topic_key: str,
) -> Tuple[VerificationStatus, Optional[SupportDirection], str]:
    """Independently re-confirm one ClinVar-shaped evidence item.

    Args:
        transport: The injectable HTTP transport, called as
            ``transport(url, None)`` and expected to return
            parsed JSON (the ClinVar esummary payload).
        variation_id: The Variation ID the item claims.
        topic_key: The topic the item is evidence about; its
            protein change is the identity the served record
            must match.

    Returns:
        ``(status, direction, note)``.  ``direction`` is derived
        from the served record's germline classification via
        :func:`provome.live.clinvar_direction` whenever a record
        was actually read; it is ``None`` when no record could be
        read (transport failure), in which case the status is
        UNVERIFIED and the item does not count either way.

    Status semantics mirror :func:`confirm_literature`: VERIFIED
    only when the served record exists, is about this variant,
    and carries a germline classification; FAILED when the check
    was made and the record does not support the item (no such
    Variation ID, a different variant, no classification);
    UNVERIFIED when the check could not be completed.
    """
    from .live import acceptable_notations, clinvar_direction

    vid_text = str(variation_id).strip()
    if not vid_text.isdigit():
        return (
            VerificationStatus.FAILED,
            None,
            f"claimed ClinVar Variation ID {vid_text!r} is not a "
            "numeric identifier, so the record cannot exist in "
            "ClinVar; not counted as verified",
        )
    short = variant_short_from_topic(topic_key)
    if short is None:
        return (
            VerificationStatus.UNVERIFIED,
            None,
            f"topic {topic_key!r} does not name a concrete "
            "protein change, so the served ClinVar record cannot "
            "be identity-matched; item not counted as verified",
        )
    try:
        payload = transport(
            CLINVAR_ESUMMARY_URL.format(variation_id=vid_text), None
        )
    except Exception:
        return (
            VerificationStatus.UNVERIFIED,
            None,
            f"independent re-confirmation of ClinVar Variation "
            f"ID {vid_text} could not be completed (lookup "
            "failed); item not counted as verified",
        )
    result = (payload or {}).get("result") or {}
    record = result.get(vid_text)
    if not isinstance(record, dict) or "error" in record:
        return (
            VerificationStatus.FAILED,
            None,
            f"ClinVar Variation ID {vid_text} was not found in "
            "ClinVar on independent re-confirmation; item not "
            "counted as verified",
        )
    title = record.get("title") or ""
    protein_change = (record.get("protein_change") or "").lower()
    accepted = acceptable_notations(short)
    if not any(form in title.lower() for form in accepted) and (
        protein_change not in accepted
    ):
        return (
            VerificationStatus.FAILED,
            None,
            f"ClinVar's record for Variation ID {vid_text} "
            f"({title!r}) is not about the variant this item "
            f"claims ({short}); identity mismatch, item not "
            "counted as verified",
        )
    classification = record.get("germline_classification") or {}
    if isinstance(classification, dict):
        significance = classification.get("description", "")
    else:
        significance = str(classification)
    if not significance:
        return (
            VerificationStatus.FAILED,
            None,
            f"ClinVar's record for Variation ID {vid_text} "
            "carries no germline classification, so it cannot "
            "support or refute a classification claim; item not "
            "counted as verified",
        )
    direction = clinvar_direction(significance)
    return (
        VerificationStatus.VERIFIED,
        direction,
        f"ClinVar Variation ID {vid_text} independently "
        f"re-confirmed against ClinVar (NCBI E-utilities): the "
        f"record is about {short} and its served germline "
        f"classification is {significance!r}, from which the "
        f"item's direction ({direction.value}) is derived — the "
        "item's own claimed direction is not used",
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
