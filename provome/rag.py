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
* **Retrieval is not verification.**  A document the index returns
  is a *candidate*.  It becomes evidence only through the bridge
  (:func:`evidence_from_retrieval`), and the bridge marks a document
  verified only when it carries a real, checkable identifier — a
  PMID in its ``doc_id`` or its URL.  A retrieved document with no
  such identifier is excluded from the bridge entirely: it may be
  shown to a human, but it never becomes a gate-countable item.
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
]


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


_PMID_IN_ID_RE = re.compile(r"pmid[:_-]?(\d{4,10})", re.IGNORECASE)
_PMID_IN_URL_RE = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|pmid=)(\d{4,10})", re.IGNORECASE
)


def document_pmid(document: Document) -> Optional[str]:
    """The PMID a document carries, or ``None``.

    Looks in ``doc_id`` first (``"pmid:12345678"``), then in ``url``
    (a PubMed URL or a ``pmid=`` parameter).  This is the bridge's
    verification rule, kept in one place so it can be audited: an
    identifier a reader can type into PubMed and land on this exact
    record, or nothing.
    """
    match = _PMID_IN_ID_RE.search(document.doc_id)
    if match:
        return match.group(1)
    if document.url:
        match = _PMID_IN_URL_RE.search(document.url)
        if match:
            return match.group(1)
    return None


def evidence_from_retrieval(
    index: RetrievalIndex,
    query: str,
    *,
    topic_key: str,
    k: int = 5,
    min_score: float = 0.0,
    confirmer: Optional[Callable[[str, str], VerificationStatus]] = None,
) -> list[EvidenceItem]:
    """Turn top retrieved, identifier-carrying documents into evidence.

    The bridge rule, stated plainly: **a retrieved document becomes a
    VERIFIED evidence item only when it carries a real identifier —
    a PMID in its ``doc_id`` or its URL** (see :func:`document_pmid`).
    Retrieval similarity is a ranking signal, never a verification
    signal: a high score says the text looks relevant, it does not
    say the record exists.  Documents without a PMID are excluded
    here, however well they scored; a human can still read them via
    :meth:`RetrievalIndex.search`, but they never reach the gate
    through this bridge.

    Honest limit of the default rule (gate hardening audit,
    2026-11-25): an identifier's *shape* is not proof the record
    exists — a fabricated corpus document carrying ``pmid:99999999``
    satisfies the default rule, because the bridge's trust boundary
    is corpus ingest: the corpus is supposed to be built from real
    fetches, stamped with ``fetched_on``.  A deployment that cannot
    vouch for its corpus ingest should pass ``confirmer`` — a
    ``(pmid, title) -> VerificationStatus`` callable such as
    :func:`provome.confirmation.make_pubmed_confirmer` — and each
    document is then held to the same independent PubMed
    re-confirmation standard as the live providers' literature.
    The confirmer's status replaces VERIFIED; its items are carried
    with that status and never silently upgraded.

    Args:
        index: The index to search.
        query: The retrieval query.
        topic_key: Topic the items are evidence about.  Retrieval
            does not know the caller's topic, so the caller states
            it; items carry it unchanged and the gate only combines
            items that share it.
        k: Maximum documents to consider from the search.
        min_score: Drop retrieved documents scoring below this.
        confirmer: Optional live re-confirmation callable; see
            above.  ``None`` (the default) applies the ingest-trust
            bridge rule exactly as before.

    Returns:
        :class:`EvidenceItem` objects, in retrieval order.
    """
    if not topic_key or not topic_key.strip():
        raise ValueError("topic_key must be a non-empty string")
    items: list[EvidenceItem] = []
    for document, score in index.search(query, k=k):
        if score < min_score:
            continue
        pmid = document_pmid(document)
        if pmid is None:
            continue
        if confirmer is not None:
            status = confirmer(pmid, document.title)
            notes = (
                f"retrieved from the local index with similarity "
                f"score {score:.3f}; independently re-confirmed "
                f"via the supplied confirmer, which returned "
                f"{status.value}"
            )
        else:
            status = VerificationStatus.VERIFIED
            notes = (
                f"retrieved from the local index with similarity "
                f"score {score:.3f}; marked verified under the "
                f"bridge rule because the document carries PMID "
                f"{pmid} — the identifier, not the score, is the "
                f"verification"
            )
        items.append(
            EvidenceItem(
                item_id=f"rag-pmid-{pmid}",
                topic_key=topic_key,
                statement=(
                    f"Peer-reviewed literature reports: "
                    f"\"{document.title.rstrip('.')}\""
                ),
                source_id=document.source,
                citation=(
                    f"PMID {pmid} (indexed copy fetched "
                    f"{document.fetched_on.isoformat()})"
                ),
                kind=EvidenceKind.LITERATURE_REPORT,
                direction=SupportDirection.SUPPORTS,
                status=status,
                locator=(
                    document.url
                    or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ),
                notes=notes,
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
        f"Bridge: {len(items)} verified evidence item(s) for "
        "CFTR:p.Phe508del (the notebook entry, though retrieved, "
        "carries no PMID and is excluded):"
    )
    for item in items:
        print(f"  - {item.citation_label()}")
    print()


if __name__ == "__main__":
    _demo()
