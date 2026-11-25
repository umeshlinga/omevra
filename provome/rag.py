"""Literature-scale retrieval: the RAG core (LLM track, Stage A).

The Copilot's live provider (:mod:`provome.live`) fetches a handful
of records per question from named databases.  This module is the
other half of retrieval: an index over a *corpus* of documents —
papers, records, reports — that a question can be searched against,
so retrieval can grow from a few live lookups to a literature-scale
collection without changing how evidence enters the gate.

Honesty first, because this module is easy to oversell:

* **This is retrieval infrastructure, not a trained model.**  The
  bundled embedder is a lexical TF-IDF baseline (below).  Nothing in
  this file learns anything, and no coverage or accuracy figure is
  claimed for it.  A similarity score here means "these texts share
  weighted vocabulary", nothing more.
* **Retrieval is not verification, and neither is an identifier.**
  A document the index returns is a *candidate*.  It becomes evidence
  only through the bridge (:func:`evidence_from_retrieval`).  The
  bridge requires a well-formed PMID (a whole ``pmid:<digits>`` id or
  a PubMed / Europe PMC URL); documents without one are excluded.  A
  PMID-shaped string proves nothing by itself, so bridge items are
  UNVERIFIED unless a check has resolved the PMID at the source and
  matched the title.  The bridge accepts that check in two
  spellings: a :class:`PmidVerifier`
  (:class:`EuropePmcPmidVerifier` resolves against Europe PMC) or a
  confirmer callable — the production standard being the
  independent PubMed re-confirmation in
  :mod:`provome.confirmation`, the same check the live providers
  are held to.  The bridge also sets the item's source itself: a
  document cannot declare itself to be ClinVar.
* **The gate is unchanged.**  Bridge items are ordinary
  :class:`~provome.evidence.EvidenceItem` objects; the cite-or-decline
  gate in :mod:`provome.evidence` judges them exactly as it judges
  live-fetched items.

Design:

* :class:`Embedder` is a protocol with a single method,
  ``embed(text) -> tuple[float, ...]``.  Callers (the index, the
  bridge) depend only on that protocol, so a neural embedding model
  can be plugged in later — as another ``Embedder`` implementation —
  without changing any caller.  No external dependency is added for
  that future step today.
* :class:`LexicalEmbedder` is the standard-library baseline: TF-IDF
  over word tokens with cosine similarity, computed exactly and
  deterministically.  It works offline, needs no model download, and
  is the honest floor a future neural embedder must beat on a real
  benchmark before it replaces anything.
* :class:`RetrievalIndex` is in-memory.  Document vectors are cached
  and recomputed whenever the corpus changes (a lexical embedder's
  weights depend on the whole corpus, so an addition can change
  every vector).  A production deployment would persist vectors in
  an approximate-nearest-neighbour store; the interface here —
  ``add`` / ``search`` — is the part that must not change.

Running this module directly builds a tiny fixture index and runs
two searches against it (offline, no network):

    python -m provome.rag

Research software.  Not for clinical use.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Optional, Protocol, Sequence, runtime_checkable

from .evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection,
    VerificationStatus,
)

__all__ = [
    "Document",
    "Embedder",
    "LexicalEmbedder",
    "RetrievalIndex",
    "document_pmid",
    "evidence_from_retrieval",
    "PmidVerifier",
    "EuropePmcPmidVerifier",
    "LITERATURE_SOURCE_ID",
]

#: The only source the literature bridge emits.  A PMID identifies a
#: PubMed record, so whatever a document's ``source`` field says, a
#: PMID-backed item is primary literature.
LITERATURE_SOURCE_ID = "pubmed-literature"


@dataclass(frozen=True)
class Document:
    """One retrievable document in the index.

    Attributes:
        doc_id: Stable identifier for this document.  For literature,
            the convention is ``"pmid:<digits>"``; the bridge's
            verification rule looks for a PMID here or in ``url``.
        title: Document title.  Indexed together with ``text``.
        text: Body text the retrieval similarity is computed over.
        source: Id of the source this document came from, naming a
            source in a :class:`~provome.evidence.SourceRegistry`
            (for example ``"pubmed-literature"``).  The bridge copies
            it onto the evidence item unchanged, so the gate's
            independence accounting applies as usual.
        url: Canonical URL of the document at its source, if any.
        fetched_on: Date this copy was fetched from the source, so a
            citation can always be re-checked against the source as
            it stood.
    """

    doc_id: str
    title: str
    text: str
    source: str
    url: Optional[str] = None
    fetched_on: date = date(1970, 1, 1)

    def __post_init__(self) -> None:
        if not self.doc_id or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string")
        if not self.title or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if not self.text or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not self.source or not self.source.strip():
            raise ValueError("source must be a non-empty string")

    @property
    def indexed_text(self) -> str:
        """The text retrieval similarity is computed over."""
        return f"{self.title} {self.text}"


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedders the index can use.

    A single method: turn text into a fixed-length vector of floats.
    Any two texts embedded by the same embedder in the same state
    must produce vectors of the same length, comparable by cosine
    similarity.  This is the seam where a neural embedding model
    plugs in later: implement this protocol and neither
    :class:`RetrievalIndex` nor :func:`evidence_from_retrieval`
    changes at all.

    An embedder may optionally provide ``observe(text)``, called by
    the index once per added document, for corpus statistics (the
    lexical embedder's document frequencies).  Neural embedders
    normally need no such hook and simply do not define it.
    """

    def embed(self, text: str) -> tuple[float, ...]:
        """Return the embedding vector for ``text``."""
        ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens; punctuation is a separator."""
    return _TOKEN_RE.findall(text.lower())


class LexicalEmbedder:
    """TF-IDF embedder: the standard-library, offline baseline.

    Term weights are ``tf * idf`` with ``tf`` the term's share of the
    text and ``idf = log((1 + N) / (1 + df)) + 1`` over the observed
    corpus (``N`` documents, ``df`` documents containing the term).
    Vectors are L2-normalised and aligned to the observed vocabulary
    in sorted order, so embedding is fully deterministic.  Terms
    never observed in the corpus are ignored, exactly as a fitted
    vectoriser behaves.

    This embedder understands *vocabulary overlap only*.  It knows
    nothing about synonyms, negation, or meaning; "CFTR activator"
    and "CFTR inhibitor" look similar to it.  It exists so retrieval
    works today, offline, with zero dependencies — and as the floor
    a neural embedder must demonstrably beat before replacing it.
    """

    def __init__(self) -> None:
        self._df: Counter = Counter()
        self._doc_count = 0
        self._vocab: tuple[str, ...] = ()
        self._vocab_dirty = True

    # -- corpus statistics -------------------------------------------------
    def observe(self, text: str) -> None:
        """Record one corpus document's terms (called on index add)."""
        for term in set(_tokenise(text)):
            self._df[term] += 1
        self._doc_count += 1
        self._vocab_dirty = True

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Observed terms, sorted; the axes of every emitted vector."""
        if self._vocab_dirty:
            self._vocab = tuple(sorted(self._df))
            self._vocab_dirty = False
        return self._vocab

    # -- Embedder protocol ---------------------------------------------------
    def embed(self, text: str) -> tuple[float, ...]:
        vocab = self.vocabulary
        if not vocab:
            return ()
        counts = Counter(_tokenise(text))
        total = sum(counts.values())
        if total == 0:
            return tuple(0.0 for _ in vocab)
        n = self._doc_count
        vector: list[float] = []
        for term in vocab:
            count = counts.get(term, 0)
            if not count:
                vector.append(0.0)
                continue
            tf = count / total
            idf = math.log((1 + n) / (1 + self._df[term])) + 1.0 if n else 1.0
            vector.append(tf * idf)
        norm = math.sqrt(sum(w * w for w in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(w / norm for w in vector)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, clamped to [0, 1]; 0.0 for empty vectors."""
    if not a or not b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else 0.0
        y = b[i] if i < len(b) else 0.0
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return min(1.0, max(0.0, score))


class RetrievalIndex:
    """An in-memory index of documents, searched by similarity.

    Args:
        embedder: Any :class:`Embedder`.  Defaults to
            :class:`LexicalEmbedder`, the offline baseline.

    Search returns ``(document, score)`` pairs with scores in
    [0, 1], best first.  Ordering is deterministic: equal scores are
    broken by ``doc_id``, so the same corpus and query always return
    the same ranking.  Documents with no measurable similarity
    (score 0) are not retrieved at all — retrieval means overlap,
    and padding a result list with unrelated documents would dress
    absence up as a weak match.
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self._embedder: Embedder = (
            embedder if embedder is not None else LexicalEmbedder()
        )
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._vectors_dirty = True

    def __len__(self) -> int:
        return len(self._documents)

    def add(self, document: Document) -> None:
        """Add one document; raises ``ValueError`` on a duplicate id."""
        if document.doc_id in self._documents:
            raise ValueError(f"duplicate doc_id: {document.doc_id!r}")
        self._documents[document.doc_id] = document
        observe = getattr(self._embedder, "observe", None)
        if callable(observe):
            observe(document.indexed_text)
        # Corpus statistics may have changed every document's
        # vector; all cached vectors are stale until recomputed.
        self._vectors_dirty = True

    def add_many(self, documents: Iterable[Document]) -> None:
        """Add several documents, in order."""
        for document in documents:
            self.add(document)

    def _ensure_vectors(self) -> None:
        if not self._vectors_dirty:
            return
        self._vectors = {
            doc_id: self._embedder.embed(document.indexed_text)
            for doc_id, document in self._documents.items()
        }
        self._vectors_dirty = False

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """Return the ``k`` most similar documents, best first.

        Returns an empty list for an empty index or a query with no
        overlap with the corpus.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self._documents:
            return []
        self._ensure_vectors()
        query_vector = self._embedder.embed(query)
        scored: list[tuple[Document, float]] = []
        for doc_id, document in self._documents.items():
            score = _cosine(query_vector, self._vectors[doc_id])
            if score > 0.0:
                scored.append((document, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].doc_id))
        return scored[:k]


# Anchored: the whole doc_id must be a PMID, so "notpmid-1234-xyz"
# or "lab-pmid1234-copy" do not qualify.  PMIDs are 1-9 digits today;
# no leading zeros.
_PMID_IN_ID_RE = re.compile(r"^pmid[:_-]?([1-9]\d{0,8})$", re.IGNORECASE)
# Only canonical record URLs at PubMed or Europe PMC count; a
# "pmid=" parameter on an arbitrary host does not.
_PMID_IN_URL_RE = re.compile(
    r"^https?://(?:"
    r"pubmed\.ncbi\.nlm\.nih\.gov/([1-9]\d{0,8})"
    r"|(?:www\.)?europepmc\.org/(?:abstract|article)/MED/([1-9]\d{0,8})"
    r")/?(?:[?#].*)?$",
    re.IGNORECASE,
)


def document_pmid(document: Document) -> Optional[str]:
    """The PMID a document carries, or ``None``.

    Looks in ``doc_id`` first (``"pmid:12345678"``), then in ``url``
    (a PubMed URL or a ``pmid=`` parameter).  This is the bridge's
    verification rule, kept in one place so it can be audited: an
    identifier a reader can type into PubMed and land on this exact
    record, or nothing.
    """
    match = _PMID_IN_ID_RE.match(document.doc_id.strip())
    if match:
        return match.group(1)
    if document.url:
        match = _PMID_IN_URL_RE.match(document.url.strip())
        if match:
            return match.group(1) or match.group(2)
    return None


@runtime_checkable
class PmidVerifier(Protocol):
    """Checks a PMID against the record it claims to be.

    ``verify(pmid, title)`` returns:

    * ``True``  – the PMID resolves at the source and the record's
      title matches ``title``: the item may be VERIFIED;
    * ``False`` – the check ran and failed (no such record, or a
      different paper): the item is FAILED and can never be cited;
    * ``None``  – the check could not run (network error, rate
      limit): the item stays UNVERIFIED.
    """
    def verify(self, pmid: str, title: str) -> Optional[bool]:
        ...


def _title_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def title_similarity(a: str, b: str) -> float:
    """Jaccard overlap of title word tokens, in [0, 1]."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class EuropePmcPmidVerifier:
    """Resolve a PMID at Europe PMC and compare titles.

    Args:
        transport: ``transport(url, body) -> dict``; defaults to
            :func:`provome.live.urllib_transport`.  Tests inject one.
        min_title_similarity: Minimum title token overlap for the
            record to count as the same paper.
    """

    def __init__(
        self,
        transport: Optional[Callable[[str, Optional[dict]], dict]] = None,
        min_title_similarity: float = 0.6,
    ) -> None:
        if transport is None:
            from .live import urllib_transport  # lazy: avoids an import cycle
            transport = urllib_transport
        if not 0.0 < min_title_similarity <= 1.0:
            raise ValueError("min_title_similarity must be in (0, 1]")
        self.transport = transport
        self.min_title_similarity = min_title_similarity

    def verify(self, pmid: str, title: str) -> Optional[bool]:
        from urllib.parse import quote
        from .live import EUROPEPMC
        url = (
            f"{EUROPEPMC}?query={quote(f'EXT_ID:{pmid} AND SRC:MED')}"
            "&format=json&resultType=lite&pageSize=1"
        )
        try:
            payload = self.transport(url, None)
        except Exception:
            return None
        results = (payload or {}).get("resultList", {}).get("result", [])
        record = next((r for r in results if str(r.get("pmid")) == pmid), None)
        if record is None:
            return False
        return (
            title_similarity(record.get("title", ""), title)
            >= self.min_title_similarity
        )


def evidence_from_retrieval(
    index: RetrievalIndex,
    query: str,
    *,
    topic_key: str,
    k: int = 5,
    min_score: float = 0.0,
    verifier: Optional[PmidVerifier] = None,
    confirmer: Optional[Callable[[str, str], VerificationStatus]] = None,
) -> list[EvidenceItem]:
    """Turn top retrieved, PMID-carrying documents into evidence.

    The bridge rules, stated plainly (gate hardening, reconciled
    2026-11-25):

    1. Only documents carrying a well-formed PMID (see
       :func:`document_pmid`) become items; others are excluded,
       however well they scored.
    2. The item's source is always :data:`LITERATURE_SOURCE_ID`, set
       here.  A document whose ``source`` field names any other
       source is excluded: a PMID is a literature identifier, and a
       document cannot borrow another database's independence group
       by labelling itself.
    3. Status comes from an actual check, never from the
       identifier's shape and never from the retrieval score.  Two
       spellings of the check are accepted:

       * ``confirmer`` — a ``(pmid, title) -> VerificationStatus``
         callable, such as
         :func:`provome.confirmation.make_pubmed_confirmer`, which
         independently re-confirms the PMID against PubMed (NCBI,
         a different operator than the Europe PMC search feed) and
         requires the titles to match.  This is the production
         standard, the same one the live providers are held to.
         The confirmer's status is carried as returned and never
         silently upgraded.
       * ``verifier`` — a :class:`PmidVerifier` whose
         ``verify(pmid, title)`` returns ``True`` (VERIFIED),
         ``False`` (FAILED) or ``None`` (UNVERIFIED).

       With neither supplied, every item is UNVERIFIED: visible to
       a human, never counted by the gate.  An earlier bridge rule
       trusted corpus ingest and marked identifier-carrying
       documents VERIFIED on shape alone; the red-team review
       (2026-11-25) showed fabricated PMIDs passing under it, and
       the reconciliation of the two hardening efforts retired it
       in favour of this rule.

    Args:
        index: The index to search.
        query: The retrieval query.
        topic_key: Topic the items are evidence about.  Retrieval
            does not know the caller's topic, so the caller states
            it; items carry it unchanged and the gate only combines
            items that share it.
        k: Maximum documents to consider from the search.
        min_score: Drop retrieved documents scoring below this.
        verifier: Resolves each PMID at its source.  Optional;
            mutually exclusive with ``confirmer``.
        confirmer: Optional live re-confirmation callable; see
            rule 3 above.  Mutually exclusive with ``verifier``.

    Returns:
        :class:`EvidenceItem` objects, in retrieval order.
    """
    if not topic_key or not topic_key.strip():
        raise ValueError("topic_key must be a non-empty string")
    if verifier is not None and confirmer is not None:
        raise ValueError(
            "pass at most one of verifier and confirmer: both spell "
            "the same check, and accepting two verdicts would leave "
            "the item's status ambiguous"
        )
    items: list[EvidenceItem] = []
    for document, score in index.search(query, k=k):
        if score < min_score:
            continue
        pmid = document_pmid(document)
        if pmid is None:
            continue
        if document.source != LITERATURE_SOURCE_ID:
            continue
        if confirmer is not None:
            status = confirmer(pmid, document.title)
            check = (
                f"independently re-confirmed via the supplied "
                f"confirmer, which returned {status.value}"
            )
        elif verifier is not None:
            outcome = verifier.verify(pmid, document.title)
            if outcome is True:
                status = VerificationStatus.VERIFIED
                check = (
                    f"PMID {pmid} resolved at the source and the "
                    "title matched"
                )
            elif outcome is False:
                status = VerificationStatus.FAILED
                check = (
                    f"PMID {pmid} did not resolve to a record with "
                    "this title; the item can never be cited"
                )
            else:
                status = VerificationStatus.UNVERIFIED
                check = (
                    f"PMID {pmid} could not be resolved at the "
                    "source; unverified items do not count toward "
                    "the gate"
                )
        else:
            status = VerificationStatus.UNVERIFIED
            check = (
                f"PMID {pmid} has not been resolved at the source; "
                "unverified items do not count toward the gate"
            )
        items.append(
            EvidenceItem(
                item_id=f"rag-pmid-{pmid}",
                topic_key=topic_key,
                statement=(
                    f"Peer-reviewed literature reports: "
                    f"\"{document.title.rstrip('.')}\""
                ),
                source_id=LITERATURE_SOURCE_ID,
                citation=(
                    f"PMID {pmid} (indexed copy fetched "
                    f"{document.fetched_on.isoformat()})"
                ),
                kind=EvidenceKind.LITERATURE_REPORT,
                direction=SupportDirection.SUPPORTS,
                status=status,
                locator=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                notes=(
                    f"retrieved from the local index with similarity "
                    f"score {score:.3f}; {check}"
                ),
            )
        )
    return items


def _fixture_documents() -> list[Document]:
    """Small hand-written fixture corpus for the demo and tests.

    The PMIDs are fixture identifiers for offline demonstration of
    the mechanics; the documents are illustrative summaries, not
    fetched records.
    """
    stamp = date(2026, 11, 25)
    return [
        Document(
            doc_id="pmid:90000001",
            title="CFTR p.Phe508del impairs protein folding and trafficking",
            text=(
                "The Phe508del variant of CFTR misfolds in the "
                "endoplasmic reticulum and is degraded before reaching "
                "the cell surface. Corrector compounds partially rescue "
                "folding and trafficking of the mutant channel."
            ),
            source="pubmed-literature",
            url="https://pubmed.ncbi.nlm.nih.gov/90000001/",
            fetched_on=stamp,
        ),
        Document(
            doc_id="pmid:90000002",
            title="Ivacaftor potentiates CFTR channel gating",
            text=(
                "Ivacaftor is a potentiator that increases the open "
                "probability of CFTR channels at the cell surface, "
                "including gating variants of the cystic fibrosis "
                "transmembrane conductance regulator."
            ),
            source="pubmed-literature",
            url="https://pubmed.ncbi.nlm.nih.gov/90000002/",
            fetched_on=stamp,
        ),
        Document(
            doc_id="pmid:90000003",
            title="Venetoclax inhibits BCL2 in chronic lymphocytic leukemia",
            text=(
                "Venetoclax is a selective BCL2 inhibitor that restores "
                "apoptosis in malignant B cells by displacing pro-apoptotic "
                "proteins from the BCL2 binding groove."
            ),
            source="pubmed-literature",
            url="https://pubmed.ncbi.nlm.nih.gov/90000003/",
            fetched_on=stamp,
        ),
        Document(
            doc_id="pmid:90000004",
            title="The HBB Glu6Val substitution causes sickle cell disease",
            text=(
                "A single amino acid substitution in beta globin "
                "produces haemoglobin S, which polymerises when "
                "deoxygenated and deforms red blood cells into the "
                "sickle shape that drives vaso-occlusion."
            ),
            source="pubmed-literature",
            url="https://pubmed.ncbi.nlm.nih.gov/90000004/",
            fetched_on=stamp,
        ),
        Document(
            doc_id="pmid:90000005",
            title="Haemoglobin polymerisation in sickle cell anaemia",
            text=(
                "Deoxygenated haemoglobin S forms rigid polymers "
                "inside red blood cells. Polymerisation kinetics, "
                "haemoglobin concentration, and fetal haemoglobin "
                "levels determine sickling severity."
            ),
            source="pubmed-literature",
            url="https://pubmed.ncbi.nlm.nih.gov/90000005/",
            fetched_on=stamp,
        ),
        Document(
            doc_id="lab-notebook-2026-014",
            title="Lab notebook: CFTR folding assay observations",
            text=(
                "Internal notebook entry on CFTR Phe508del folding "
                "assay conditions and corrector response. Unpublished "
                "working notes with no external identifier."
            ),
            source="pubmed-literature",
            url=None,
            fetched_on=stamp,
        ),
    ]


def _demo() -> None:
    print("Provome retrieval demo (fixture corpus, offline)")
    print("=" * 72)
    index = RetrievalIndex()
    index.add_many(_fixture_documents())
    print(f"Indexed {len(index)} fixture documents.")
    print("Note: PMIDs below are fixture identifiers; the point is the")
    print("mechanics — ranking, and the bridge's identifier rule.")
    for query in (
        "CFTR Phe508del folding corrector trafficking",
        "venetoclax BCL2 apoptosis inhibitor",
    ):
        print("-" * 72)
        print("Q:", query)
        for document, score in index.search(query, k=3):
            print(f"  {score:.3f}  [{document.doc_id}] {document.title}")
    print("-" * 72)
    items = evidence_from_retrieval(
        index,
        "CFTR Phe508del folding corrector trafficking",
        topic_key="CFTR:p.Phe508del",
        k=5,
    )
    print(
        f"Bridge: {len(items)} evidence item(s) for CFTR:p.Phe508del, "
        "all UNVERIFIED because no PMID verifier was supplied (the "
        "fixture PMIDs are not real records). The notebook entry, "
        "though retrieved, carries no PMID and is excluded:"
    )
    for item in items:
        print(f"  - [{item.status.value}] {item.citation_label()}")
    print()


if __name__ == "__main__":
    _demo()
