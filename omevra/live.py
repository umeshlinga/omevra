"""Live evidence provider: the Copilot's own browser (Step 3, Item 22).

The offline package answers from bundled illustrative data.  This
module is the production direction: a provider that fetches real
records from real sources at question time, checks them as it
fetches, and hands the Copilot evidence items whose citations carry
the source, the record identifier, and the fetch date.

Sources (all already registered in
:func:`omevra.evidence.default_source_registry`):

* **ClinVar** via NCBI E-utilities (esearch + esummary) — curated
  clinical significance for the variant.
* **gnomAD** via the Broad Institute GraphQL API — population allele
  frequency for the variant's GRCh38 coordinates (taken from the
  ClinVar record itself, never guessed).
* **Europe PMC** — peer-reviewed literature mentioning the gene and
  variant (registered under the ``pubmed-literature`` source).

Design rules, inherited from the hardening work:

* **Intent validation.**  A topic is recognised only when the
  question names a gene symbol AND a variant in HGVS-like notation
  (``p.`` or ``c.``).  Keyword stuffing, filler text with an alias
  buried in it, and vague questions produce no topic, which means
  the Copilot declines instead of pattern-matching its way to an
  answer (Step 1, Worker A residual finding).
* **Verification at fetch.**  An item is marked VERIFIED only when
  the record actually came back from the source in this fetch and
  the fields used were present.  A source that errors, times out, or
  returns nothing contributes no items — silence is never dressed
  up as evidence.
* **Fetch dates everywhere.**  Every citation records the date the
  record was fetched, so an answer can always be re-checked against
  the source as it stood.
* **No silent network in tests.**  The HTTP transport is injectable;
  the test-suite runs entirely on recorded fixtures.

Running this module directly performs a live run (network required):

    python -m omevra.live

Research software.  Not for clinical use.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional, Sequence

from .copilot import Copilot
from .evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection,
    VerificationStatus,
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GNOMAD_API = "https://gnomad.broadinstitute.org/api"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
USER_AGENT = "Provome/0.3 (research software; live evidence fetch)"

# A transport takes (url, optional POST body) and returns parsed JSON.
Transport = Callable[[str, Optional[dict]], dict]


def urllib_transport(url: str, body: Optional[dict] = None) -> dict:
    """Default transport: stdlib urllib, JSON in and out."""
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,11})\b")
_HGVS_P_RE = re.compile(
    r"\bp\.([A-Z][a-z]{2}\d+(?:[A-Z][a-z]{2}|del|dup|fs|Ter|\*|=))", re.IGNORECASE
)
_HGVS_C_RE = re.compile(r"\bc\.([0-9][0-9_+\->\*a-zA-Z]*)", re.IGNORECASE)


_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*",
}


def one_letter_notation(short: str) -> str:
    """``Phe508del`` -> ``F508del``; ClinVar indexes both spellings."""
    match = re.match(r"([A-Z][a-z]{2})(\d+)(.*)", short)
    if not match:
        return short
    ref, pos, rest = match.groups()
    alt = re.match(r"([A-Z][a-z]{2})(.*)", rest)
    if alt and alt.group(1) in _THREE_TO_ONE:
        rest = _THREE_TO_ONE[alt.group(1)] + alt.group(2)
    return _THREE_TO_ONE.get(ref, ref) + pos + rest


def acceptable_notations(short: str) -> set:
    """All spellings under which ClinVar may index this variant.

    Three-letter and one-letter forms, at the stated position and at
    position + 1: legacy protein numbering often excludes the
    initiator methionine, so the literature's HBB p.Glu6Val is
    ClinVar's p.Glu7Val.  Both spellings name the same residue
    change; accepting the +1 form is a numbering reconciliation,
    never a different variant (the amino acids must still match).
    """
    match = re.match(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|del|dup|fs|Ter|\*|=)(.*)", short)
    if not match:
        return {short.lower()}
    ref, pos, alt, rest = match.groups()
    forms = set()
    for delta in (0, 1):
        three = f"{ref}{int(pos) + delta}{alt}{rest}"
        forms.add(three.lower())
        forms.add(one_letter_notation(three).lower())
    return forms


@dataclass(frozen=True)
class VariantQuery:
    """A parsed, validated question target: one gene, one variant."""

    gene: str
    hgvs: str  # normalised, e.g. "p.Phe508del" or "c.68_69delAG"

    @property
    def topic_key(self) -> str:
        return f"{self.gene}:{self.hgvs}"

    @property
    def short_notation(self) -> str:
        """Protein change without the p. prefix, e.g. ``Phe508del``."""
        return self.hgvs[2:] if self.hgvs.startswith("p.") else self.hgvs


def parse_variant_query(question_text: str) -> Optional[VariantQuery]:
    """Extract a (gene, variant) target, or ``None`` if not a real target.

    Intent validation lives here: both a gene-like token and an
    HGVS-like variant must be present, and the gene token must not
    be part of the variant string itself.
    """
    if not isinstance(question_text, str):
        raise TypeError("question_text must be a string")
    if len(question_text.split()) < 3:
        return None
    protein = _HGVS_P_RE.search(question_text)
    coding = _HGVS_C_RE.search(question_text)
    hgvs: Optional[str] = None
    if protein:
        raw = protein.group(1)
        hgvs = "p." + raw[:1].upper() + raw[1:]
    elif coding:
        hgvs = "c." + coding.group(1)
    if hgvs is None:
        return None
    variant_span = (protein or coding).span()
    for match in _GENE_RE.finditer(question_text):
        if match.start() >= variant_span[0] and match.end() <= variant_span[1]:
            continue
        gene = match.group(1)
        if gene in {"DNA", "RNA", "USA", "FDA", "WHO"}:
            continue
        return VariantQuery(gene=gene, hgvs=hgvs)
    return None


@dataclass
class LiveEvidenceProvider:
    """Evidence provider that fetches from ClinVar, gnomAD, Europe PMC.

    Args:
        transport: Injectable HTTP transport (tests pass fixtures).
        today: Injectable date for citation fetch stamps.
        max_literature: Maximum literature items to return.
    """

    name = "live"
    transport: Transport = field(default=urllib_transport)
    today: Callable[[], date] = field(default=date.today)
    max_literature: int = 2
    _cache: dict = field(default_factory=dict, repr=False)

    # -- provider protocol -------------------------------------------------
    def topic_for_question(self, question_text: str) -> Optional[str]:
        query = parse_variant_query(question_text)
        return query.topic_key if query else None

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        if topic_key in self._cache:
            return self._cache[topic_key]
        gene, _, hgvs = topic_key.partition(":")
        query = VariantQuery(gene=gene, hgvs=hgvs)
        items: list[EvidenceItem] = []
        clinvar = self._clinvar_items(query)
        items.extend(clinvar)
        coords = self._clinvar_coords
        if coords:
            items.extend(self._gnomad_items(query, coords))
        items.extend(self._literature_items(query))
        result = tuple(items)
        self._cache[topic_key] = result
        return result

    # -- ClinVar ------------------------------------------------------------
    _clinvar_coords: Optional[tuple] = None

    def _clinvar_items(self, query: VariantQuery) -> list[EvidenceItem]:
        self._clinvar_coords = None
        stamp = self.today().isoformat()
        ids: list = []
        # ClinVar indexes variants under both three-letter and
        # one-letter protein spellings (Phe508del vs F508del); try
        # both before concluding the source holds nothing.
        for notation in {query.short_notation, one_letter_notation(query.short_notation)}:
            term = f"{query.gene}[gene] AND {notation}"
            search = self.transport(
                f"{EUTILS}/esearch.fcgi?db=clinvar&retmode=json&retmax=10"
                f"&term={urllib.request.quote(term)}",
                None,
            )
            ids = search.get("esearchresult", {}).get("idlist", [])
            if ids:
                break
        if not ids:
            return []
        summary = self.transport(
            f"{EUTILS}/esummary.fcgi?db=clinvar&retmode=json&id={','.join(ids)}",
            None,
        )
        result = summary.get("result", {})
        items: list[EvidenceItem] = []
        for uid in result.get("uids", [])[:10]:
            record = result.get(uid, {})
            title = record.get("title", "")
            # The search is loose by design; the record must actually
            # be about this variant, not merely a neighbour in the
            # same gene (Step 3 live-run finding: a Phe1450 deletion
            # matched a Phe508del query and was dropped here).
            protein_change = (record.get("protein_change") or "").lower()
            accepted = acceptable_notations(query.short_notation)
            title_l = title.lower()
            if not any(form in title_l for form in accepted) and protein_change not in accepted:
                continue
            # Haplotype / multi-variant records are not evidence about
            # this single variant, however they matched.
            if ";" in title or "," in protein_change:
                continue
            classification = record.get("germline_classification") or {}
            if isinstance(classification, dict):
                significance = classification.get("description", "")
                review = classification.get("review_status", "")
            else:
                significance = str(classification)
                review = ""
            if not significance or not title:
                continue
            for measure in record.get("variation_set", []):
                for loc in measure.get("variation_loc", []):
                    if loc.get("assembly_name") == "GRCh38":
                        self._clinvar_coords = (
                            loc.get("chr"),
                            loc.get("start"),
                            loc.get("ref"),
                            loc.get("alt"),
                        )
            items.append(
                EvidenceItem(
                    item_id=f"live-clinvar-{uid}",
                    topic_key=query.topic_key,
                    statement=(
                        f"ClinVar records {title} with germline "
                        f"classification: {significance}"
                        + (f" (review status: {review})" if review else "")
                        + "."
                    ),
                    source_id="clinvar",
                    citation=(
                        f"ClinVar Variation ID {uid} "
                        f"(fetched {stamp} via NCBI E-utilities)"
                    ),
                    kind=EvidenceKind.CURATED_CLASSIFICATION,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                    locator=f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/",
                )
            )
        return items

    # -- gnomAD ---------------------------------------------------------------
    def _gnomad_items(self, query: VariantQuery, coords: tuple) -> list[EvidenceItem]:
        chrom, pos, ref, alt = coords
        if not all([chrom, pos, ref, alt]):
            return []
        stamp = self.today().isoformat()
        variant_id = f"{chrom}-{pos}-{ref}-{alt}"
        gql = {
            "query": (
                "query($v: String!) { variant(variantId: $v, "
                "dataset: gnomad_r4) { variant_id genome { ac an af } "
                "exome { ac an af } } }"
            ),
            "variables": {"v": variant_id},
        }
        payload = self.transport(GNOMAD_API, gql)
        variant = (payload.get("data") or {}).get("variant")
        if not variant:
            return []
        freq = variant.get("genome") or variant.get("exome")
        if not freq or not freq.get("an"):
            return []
        af = freq["af"]
        return [
            EvidenceItem(
                item_id=f"live-gnomad-{variant_id}",
                topic_key=query.topic_key,
                statement=(
                    f"gnomAD v4 records variant {variant_id} at allele "
                    f"frequency {af:.6g} (AC {freq['ac']} / AN {freq['an']}), "
                    f"fetched live from the gnomAD API."
                ),
                source_id="gnomad",
                citation=f"gnomAD v4 variant {variant_id} (fetched {stamp})",
                kind=EvidenceKind.POPULATION_FREQUENCY,
                direction=SupportDirection.NEUTRAL,
                status=VerificationStatus.VERIFIED,
                locator=f"https://gnomad.broadinstitute.org/variant/{variant_id}",
            )
        ]

    # -- Europe PMC literature --------------------------------------------------
    def _literature_items(self, query: VariantQuery) -> list[EvidenceItem]:
        stamp = self.today().isoformat()
        q = f"{query.gene} AND {query.short_notation}"
        payload = self.transport(
            f"{EUROPEPMC}?query={urllib.request.quote(q)}"
            f"&format=json&pageSize={self.max_literature}&resultType=core",
            None,
        )
        results = payload.get("resultList", {}).get("result", [])
        items: list[EvidenceItem] = []
        for record in results:
            title = record.get("title", "").rstrip(".")
            pmid = record.get("pmid")
            year = record.get("pubYear", "")
            if not title or not pmid:
                continue
            items.append(
                EvidenceItem(
                    item_id=f"live-lit-{pmid}",
                    topic_key=query.topic_key,
                    statement=(
                        f"Peer-reviewed literature reports on {query.gene} "
                        f"{query.hgvs}: \"{title}\" ({year})."
                    ),
                    source_id="pubmed-literature",
                    citation=(
                        f"PMID {pmid} via Europe PMC (fetched {stamp})"
                    ),
                    kind=EvidenceKind.LITERATURE_REPORT,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                    locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
            )
        return items


def _demo() -> None:
    copilot = Copilot(provider=LiveEvidenceProvider())
    for question in (
        "What is known about CFTR p.Phe508del?",
        "Tell me about the weather in Indianapolis today please",
    ):
        answer = copilot.ask(question)
        print("=" * 72)
        print("Q:", question)
        print(answer.render())
        print()


if __name__ == "__main__":
    _demo()
