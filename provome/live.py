"""Live evidence provider: the Copilot's own browser (Step 3, Item 22).

The offline package answers from bundled illustrative data.  This
module is the production direction: a provider that fetches real
records from real sources at question time, checks them as it
fetches, and hands the Copilot evidence items whose citations carry
the source, the record identifier, and the fetch date.

Sources (all already registered in
:func:`provome.evidence.default_source_registry`):

* **ClinVar** via NCBI E-utilities (esearch + esummary) — curated
  clinical significance for the variant.
* **gnomAD** via the Broad Institute GraphQL API — population allele
  frequency for the variant's GRCh38 coordinates.  ClinVar's
  esummary locations carry chromosome and position but empty
  ref/alt alleles, so the alleles are resolved by cross-reference
  (Step 3b): the dbSNP rsID recorded in the matched ClinVar
  measure's ``variation_xrefs`` is looked up in the NCBI Variation
  API, the GRCh38 placement's SPDI alleles are read from the
  returned record, and — for indels — the chosen allele is
  left-normalised against GRCh38 reference sequence fetched from
  Ensembl REST, because gnomAD indexes variants in normalised
  (chrom, pos, ref, alt) form.  Every step works only from records
  actually returned; any failure along the chain contributes no
  gnomAD item rather than a guessed allele.
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

    python -m provome.live

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
VARIATION_API = "https://api.ncbi.nlm.nih.gov/variation/v0"
ENSEMBL_REGION = "https://rest.ensembl.org/sequence/region/human"
USER_AGENT = "Provome/0.3 (research software; live evidence fetch)"

# GRCh38 chromosome accessions -> gnomAD/Ensembl chromosome names.
GRCH38_ACCESSION_CHROM = {
    "NC_000001.11": "1", "NC_000002.12": "2", "NC_000003.12": "3",
    "NC_000004.12": "4", "NC_000005.10": "5", "NC_000006.12": "6",
    "NC_000007.14": "7", "NC_000008.11": "8", "NC_000009.12": "9",
    "NC_000010.11": "10", "NC_000011.10": "11", "NC_000012.12": "12",
    "NC_000013.11": "13", "NC_000014.9": "14", "NC_000015.10": "15",
    "NC_000016.10": "16", "NC_000017.11": "17", "NC_000018.10": "18",
    "NC_000019.10": "19", "NC_000020.11": "20", "NC_000021.9": "21",
    "NC_000022.11": "22", "NC_000023.11": "X", "NC_000024.10": "Y",
}

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


# -- Step 3b: dbSNP allele resolution --------------------------------------
#
# An SPDI tuple is (seq_id, position, deleted, inserted) with the
# 0-based interbase position SPDI uses, exactly as NCBI records it.

SpdiTuple = tuple


def parse_spdi(text: object) -> Optional[SpdiTuple]:
    """Parse an NCBI SPDI string on a chromosome accession.

    ``NC_000007.14:117559590:TCTT:T`` ->
    ``("NC_000007.14", 117559590, "TCTT", "T")``.  The deleted or
    inserted sequence may be empty (pure insertion / deletion);
    anything malformed, or not on an ``NC_`` accession, is None.
    """
    if not isinstance(text, str) or not text:
        return None
    parts = text.split(":", 3)
    if len(parts) != 4:
        return None
    seq_id, pos, deleted, inserted = parts
    if not seq_id.startswith("NC_"):
        return None
    try:
        position = int(pos)
    except ValueError:
        return None
    return (seq_id, position, deleted, inserted)


def dbsnp_rsid(measure: dict) -> Optional[int]:
    """The dbSNP rsID cross-referenced by a ClinVar measure, if any."""
    for xref in (measure or {}).get("variation_xrefs") or []:
        if xref.get("db_source") == "dbSNP":
            raw = str(xref.get("db_id", ""))
            if raw.isdigit():
                return int(raw)
    return None


def refsnp_grch38_alleles(payload: dict) -> list:
    """SPDI alleles of a refsnp record's GRCh38 chromosome placement.

    Reads ``primary_snapshot_data.placements_with_allele`` and keeps
    the placement on an ``NC_`` accession whose assembly traits name
    GRCh38 (preferring the primary top-level placement).  Returns
    [] when the record carries no such placement.
    """
    psd = (payload or {}).get("primary_snapshot_data") or {}
    chosen = None
    for placement in psd.get("placements_with_allele") or []:
        seq_id = placement.get("seq_id", "")
        if not seq_id.startswith("NC_"):
            continue
        annot = placement.get("placement_annot") or {}
        assemblies = [
            trait.get("assembly_name", "")
            for trait in annot.get("seq_id_traits_by_assembly", [])
        ]
        if not any(name.startswith("GRCh38") for name in assemblies):
            continue
        if placement.get("is_ptlp"):
            chosen = placement
            break
        if chosen is None:
            chosen = placement
    if chosen is None:
        return []
    alleles: list = []
    for entry in chosen.get("alleles", []):
        spdi = (entry.get("allele") or {}).get("spdi") or {}
        try:
            alleles.append(
                (
                    spdi["seq_id"],
                    int(spdi["position"]),
                    spdi.get("deleted_sequence", ""),
                    spdi.get("inserted_sequence", ""),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return alleles


def select_variant_allele(
    alleles: Sequence[SpdiTuple],
    clinvar_spdis: set,
    require_match: bool = False,
) -> Optional[SpdiTuple]:
    """Pick the one refsnp allele that is the ClinVar variant.

    The reference allele (deleted == inserted) is never a variant.
    A single remaining candidate is the variant — unless
    ``require_match`` is set (used for rsIDs found by a loose
    fallback search rather than by the ClinVar record itself).
    With several candidates, the variant is identified only when
    exactly one candidate's SPDI equals a canonical SPDI recorded
    on a matched ClinVar measure; anything else is ambiguous and
    yields None rather than a guess.
    """
    candidates: list = []
    for spdi in alleles:
        if spdi[2] == spdi[3]:
            continue
        if spdi not in candidates:
            candidates.append(spdi)
    if not candidates:
        return None
    if len(candidates) == 1 and not require_match:
        return candidates[0]
    matches = [c for c in candidates if c in clinvar_spdis]
    return matches[0] if len(matches) == 1 else None


def normalize_vcf(pos: int, ref: str, alt: str, base_at) -> Optional[tuple]:
    """Left-normalise a biallelic (pos, ref, alt) triple.

    ``pos`` is 1-based; ``base_at(p)`` returns the GRCh38 reference
    base at 1-based ``p`` or None outside the fetched window.  This
    is the standard parsimony + left-alignment gnomAD's index uses:
    trim shared suffixes (extending left when an allele empties),
    then shared prefixes (never emptying an allele), until stable.
    Returns None when the reference window runs out mid-shift.
    """
    for _ in range(10000):
        while ref and alt and ref[-1] == alt[-1]:
            ref, alt = ref[:-1], alt[:-1]
        if not ref or not alt:
            anchor = base_at(pos - 1)
            if anchor is None:
                return None
            pos -= 1
            ref, alt = anchor + ref, anchor + alt
            continue
        trimmed = False
        while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
            ref, alt = ref[1:], alt[1:]
            pos += 1
            trimmed = True
        if not trimmed and ref[-1] != alt[-1]:
            return (pos, ref, alt)
    return None


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
        # Per-source failure isolation (CEO fix, 2026-11-25): the
        # public APIs behind this provider flake under real network
        # conditions, and one source's transport failure must
        # contribute zero items from that source - never crash the
        # whole answer. This matches the Animal/Plant providers'
        # discipline ("any error contributes zero items").
        try:
            clinvar = self._clinvar_items(query)
        except Exception:
            clinvar = []
        items.extend(clinvar)
        coords = self._clinvar_coords
        if not (coords and coords[2] and coords[3]):
            # Real ClinVar locations carry no ref/alt alleles; try
            # the dbSNP cross-reference before giving up on gnomAD.
            coords = self._resolve_gnomad_coords(query) or coords
        if coords:
            try:
                items.extend(self._gnomad_items(query, coords))
            except Exception:
                pass
        try:
            items.extend(self._literature_items(query))
        except Exception:
            pass
        result = tuple(items)
        self._cache[topic_key] = result
        return result

    # -- ClinVar ------------------------------------------------------------
    _clinvar_coords: Optional[tuple] = None
    _clinvar_measures: Optional[list] = None

    def _clinvar_items(self, query: VariantQuery) -> list[EvidenceItem]:
        self._clinvar_coords = None
        self._clinvar_measures = []
        stamp = self.today().isoformat()
        ids: list = []
        # ClinVar indexes variants under both three-letter and
        # one-letter protein spellings (Phe508del vs F508del), and
        # the two searches return different record sets — the
        # one-letter search alone can miss the record carrying the
        # dbSNP cross-reference Step 3b needs.  Union both result
        # sets (three-letter first, deterministically) and let the
        # per-record filters below decide what is really about
        # this variant.
        seen: set = set()
        for notation in (
            query.short_notation,
            one_letter_notation(query.short_notation),
        ):
            term = f"{query.gene}[gene] AND {notation}"
            search = self.transport(
                f"{EUTILS}/esearch.fcgi?db=clinvar&retmode=json&retmax=10"
                f"&term={urllib.request.quote(term)}",
                None,
            )
            for uid in search.get("esearchresult", {}).get("idlist", []):
                if uid not in seen:
                    seen.add(uid)
                    ids.append(uid)
        if not ids:
            return []
        summary = self.transport(
            f"{EUTILS}/esummary.fcgi?db=clinvar&retmode=json&id={','.join(ids)}",
            None,
        )
        result = summary.get("result", {})
        items: list[EvidenceItem] = []
        for uid in result.get("uids", [])[:20]:
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
                self._clinvar_measures.append(measure)
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

    # -- Step 3b: allele resolution via dbSNP --------------------------------
    def _fetch_json(self, url: str) -> Optional[dict]:
        """GET JSON for the cross-reference calls, one retry.

        A failed cross-reference yields None (no coordinates, no
        gnomAD item) — never an exception escaping the provider and
        never a value dressed up as a record.
        """
        for _attempt in (1, 2):
            try:
                return self.transport(url, None)
            except Exception:
                continue
        return None

    def _resolve_gnomad_coords(self, query: VariantQuery) -> Optional[tuple]:
        """Resolve (chrom, pos, ref, alt) for gnomAD via dbSNP.

        Chain, every link from a returned record: dbSNP rsID from a
        matched ClinVar measure's xrefs -> NCBI Variation API
        refsnp record -> GRCh38 placement alleles -> the one allele
        that is this variant (see :func:`select_variant_allele`) ->
        VCF coordinates, left-normalised for indels.  If no matched
        measure carries an rsID, a position-constrained search of
        the snp database supplies candidates, but those are trusted
        only when an allele matches a ClinVar canonical SPDI
        exactly.  Any break in the chain returns None.
        """
        measures = self._clinvar_measures or []
        clinvar_spdis: set = set()
        rsids: list = []
        for measure in measures:
            spdi = parse_spdi(measure.get("canonical_spdi"))
            if spdi:
                clinvar_spdis.add(spdi)
            rsid = dbsnp_rsid(measure)
            if rsid is not None and rsid not in rsids:
                rsids.append(rsid)
        attempts = [(rsid, False) for rsid in rsids]
        if not attempts:
            attempts = [(rsid, True) for rsid in self._esearch_snp_rsids(query)]
        for rsid, require_match in attempts:
            payload = self._fetch_json(f"{VARIATION_API}/refsnp/{rsid}")
            if not payload:
                continue
            chosen = select_variant_allele(
                refsnp_grch38_alleles(payload),
                clinvar_spdis,
                require_match=require_match,
            )
            if chosen is None:
                continue
            chrom = GRCH38_ACCESSION_CHROM.get(chosen[0])
            if chrom is None:
                continue
            clinvar_chr = (self._clinvar_coords or (None,))[0]
            if clinvar_chr and str(clinvar_chr) != chrom:
                continue
            coords = self._spdi_to_gnomad_coords(chrom, chosen)
            if coords:
                return coords
        return None

    def _esearch_snp_rsids(self, query: VariantQuery) -> list:
        """Fallback rsID candidates from the NCBI snp database.

        Constrained to the gene and the ClinVar GRCh38 position.
        Loose by nature (a position can host several rsIDs), so
        callers must require an exact canonical-SPDI match before
        trusting any candidate.
        """
        coords = self._clinvar_coords
        if not coords or not coords[0] or not coords[1]:
            return []
        term = (
            f"{query.gene}[Gene Name] AND {coords[0]}[CHR] "
            f"AND {coords[1]}[CHRPOS]"
        )
        payload = self._fetch_json(
            f"{EUTILS}/esearch.fcgi?db=snp&retmode=json&retmax=10"
            f"&term={urllib.request.quote(term)}"
        )
        if not payload:
            return []
        out: list = []
        for raw in payload.get("esearchresult", {}).get("idlist", []):
            if str(raw).isdigit():
                out.append(int(raw))
        return out

    def _spdi_to_gnomad_coords(
        self, chrom: str, spdi: SpdiTuple
    ) -> Optional[tuple]:
        """Convert a genomic SPDI to gnomAD (chrom, pos, ref, alt).

        SNVs convert directly.  Indels are anchored and
        left-normalised against GRCh38 reference sequence fetched
        from Ensembl; the SPDI's deleted sequence must match that
        reference exactly, otherwise the allele is not trusted.
        """
        _, pos0, deleted, inserted = spdi
        if not deleted and not inserted:
            return None
        if not re.fullmatch(r"[ACGT]*", deleted):
            return None
        if not re.fullmatch(r"[ACGT]*", inserted):
            return None
        if len(deleted) > 5000:
            return None
        if len(deleted) == 1 and len(inserted) == 1:
            return (chrom, pos0 + 1, deleted, inserted)
        base_at = self._ensembl_base_at(chrom, pos0, len(deleted))
        if base_at is None:
            return None
        if deleted and inserted:
            pos, ref, alt = pos0 + 1, deleted, inserted
            if not self._reference_matches(base_at, pos, ref):
                return None
        elif deleted:
            anchor = base_at(pos0)
            if anchor is None:
                return None
            if not self._reference_matches(base_at, pos0 + 1, deleted):
                return None
            pos, ref, alt = pos0, anchor + deleted, anchor
        else:  # pure insertion: SPDI position is the interbase point
            anchor = base_at(pos0)
            if anchor is None:
                return None
            pos, ref, alt = pos0, anchor, anchor + inserted
        normalized = normalize_vcf(pos, ref, alt, base_at)
        if normalized is None:
            return None
        pos, ref, alt = normalized
        return (chrom, pos, ref, alt)

    @staticmethod
    def _reference_matches(base_at, start: int, sequence: str) -> bool:
        return all(
            base_at(start + offset) == base
            for offset, base in enumerate(sequence)
        )

    def _ensembl_base_at(self, chrom: str, pos0: int, deleted_len: int):
        """A ``base_at(1-based pos)`` lookup over one Ensembl fetch.

        Fetches the GRCh38 region spanning the SPDI with 60 bases
        of slack on each side (room for left-normalisation to walk
        through a repeat).  Returns None when the fetch fails or
        the returned sequence does not cover the region exactly.
        """
        start = max(1, pos0 + 1 - 60)
        end = pos0 + deleted_len + 60
        payload = self._fetch_json(
            f"{ENSEMBL_REGION}/{chrom}:{start}..{end}:1"
            f"?content-type=application/json"
        )
        if not payload:
            return None
        seq = payload.get("seq") or ""
        if len(seq) != end - start + 1:
            return None

        def base_at(pos1: int) -> Optional[str]:
            index = pos1 - start
            if 0 <= index < len(seq):
                return seq[index]
            return None

        return base_at

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
