"""Plant division, v1 (Step 9, second half): species-aware evidence.

This is the first provider for the Plant division.  It answers one
kind of question — "what evidence exists for GENE / TRAIT in
SPECIES?" — by fetching live records from Ensembl Plants (gene
records for plant species, via the Ensembl REST API) and Europe PMC
literature restricted to the species, and passing them through the
same cite-or-decline gate as every other Provome answer.  The gate
is not lowered for plants: at least three verified items from at
least two independent source groups, or an honest decline.

Blunt honesty, stated here because it shapes every result: plant
evidence is sparse and trait-focused.  Ensembl Plants holds gene
records for the supported species, but curated phenotype/trait
associations are effectively absent (the phenotype endpoint
returned no records for the best-studied plant gene in our live
checks), gene symbols are inconsistent across species (famous
genes such as rice SD1 are not findable by symbol at all), and the
agricultural literature is a fraction of the human literature.
Most plant questions will decline, and a decline is the correct
output — this provider never pads a thin record up to the gate.
Coverage is NOT claimed for this division: it is in build and
unbenchmarked, and every answer says so.

Intent validation follows the Step 3 rule: a topic is recognised
only when the question names BOTH a supported species (common or
scientific name) AND a gene symbol or a trait phrase.  A gene with
no species produces no topic here, so plant evidence can never be
silently gathered for the wrong organism.

Run the live demo:

    python -m provome.plant

Research software.  Agricultural research use only.  Not agronomic
or commercial growing advice.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

from .copilot import Copilot, CopilotQuestion
from .domains import Division
from .evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection,
    VerificationStatus,
)

ENSEMBL = "https://rest.ensembl.org"
EUROPE_PMC = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?format=json&pageSize=3&query="
)
USER_AGENT = "Provome/0.8 (research software; plant evidence fetch)"

BOUNDARY = (
    "Agricultural research use only: this is not agronomic or "
    "commercial growing advice. Coverage is NOT claimed — the "
    "Plant division is in build and unbenchmarked."
)


@dataclass(frozen=True)
class SpeciesInfo:
    """One supported species: topic token, Ensembl slug, names."""

    key: str  # canonical token used in topic keys, e.g. "rice"
    ensembl: str  # Ensembl REST species slug (verified live)
    scientific: str
    aliases: tuple  # common/scientific names recognised in questions


# Ensembl slugs verified against the Ensembl Plants species list
# (info/species?division=EnsemblPlants) on 2026-11-25.  Tomato's
# production name really is the long GCA slug below.
SPECIES: tuple = (
    SpeciesInfo(
        "rice", "oryza_sativa", "Oryza sativa",
        ("rice", "oryza sativa"),
    ),
    SpeciesInfo(
        "maize", "zea_mays", "Zea mays",
        ("maize", "corn", "zea mays"),
    ),
    SpeciesInfo(
        "wheat", "triticum_aestivum", "Triticum aestivum",
        ("wheat", "triticum aestivum", "bread wheat"),
    ),
    SpeciesInfo(
        "soybean", "glycine_max", "Glycine max",
        ("soybean", "soybeans", "soya", "glycine max"),
    ),
    SpeciesInfo(
        "arabidopsis", "arabidopsis_thaliana", "Arabidopsis thaliana",
        ("arabidopsis", "arabidopsis thaliana", "thale cress"),
    ),
    SpeciesInfo(
        "tomato", "solanum_lycopersicum_gca000188115v5cm",
        "Solanum lycopersicum",
        ("tomato", "tomatoes", "solanum lycopersicum"),
    ),
    SpeciesInfo(
        "potato", "solanum_tuberosum", "Solanum tuberosum",
        ("potato", "potatoes", "solanum tuberosum"),
    ),
)
_BY_KEY = {info.key: info for info in SPECIES}

_GENE_TOKEN = re.compile(r"\b([A-Z][A-Z0-9]{1,11})\b")
_NOT_GENES = {"DNA", "RNA", "USA", "FDA", "WHO", "II", "III", "IV", "PCR", "NGS"}

_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "known", "about", "in",
    "of", "for", "and", "or", "to", "does", "do", "can", "you",
    "tell", "me", "evidence", "exists", "there", "with", "this",
    "that", "gene", "genes", "variant", "variants", "cause",
    "causes", "caused", "please", "how", "why", "when", "which",
    "plant", "plants", "crop", "crops",
}


def _find_species(question_text: str) -> Optional[SpeciesInfo]:
    """Return the species a question names, longest alias wins."""
    lowered = question_text.lower()
    best: Optional[SpeciesInfo] = None
    best_len = -1
    for info in SPECIES:
        for alias in info.aliases:
            pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
            if re.search(pattern, lowered) and len(alias) > best_len:
                best = info
                best_len = len(alias)
    return best


def _find_condition(question_text: str, species: SpeciesInfo) -> Optional[str]:
    """Extract a trait phrase when no gene symbol is present.

    Deliberately simple and transparent: drop the species alias,
    question words, and stopwords; what remains, if it is one to
    four words, is the trait phrase.  Anything messier produces
    no topic, and the Copilot declines rather than guessing.
    """
    lowered = question_text.lower()
    for alias in sorted(species.aliases, key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        lowered = re.sub(pattern, " ", lowered)
    words = [
        w for w in re.findall(r"[a-z][a-z\-]*", lowered)
        if w not in _STOPWORDS
    ]
    if 1 <= len(words) <= 4:
        return " ".join(words)
    return None


@dataclass(frozen=True)
class PlantQuery:
    """A parsed, validated plant question target."""

    species: SpeciesInfo
    subject: str  # gene symbol (uppercase) or trait phrase
    is_gene: bool

    @property
    def topic_key(self) -> str:
        subject = self.subject if self.is_gene else self.subject.replace(" ", "-")
        return f"plant:{self.species.key}:{subject}"


def parse_plant_query(question_text: str) -> Optional[PlantQuery]:
    """Extract a (species, gene-or-trait) target, or ``None``.

    Requires BOTH a recognised species and a gene symbol or a
    trait phrase.  Either one alone produces no topic.
    """
    if not isinstance(question_text, str):
        raise TypeError("question_text must be a string")
    species = _find_species(question_text)
    if species is None:
        return None
    for match in _GENE_TOKEN.finditer(question_text):
        token = match.group(1)
        if token not in _NOT_GENES:
            return PlantQuery(species=species, subject=token, is_gene=True)
    condition = _find_condition(question_text, species)
    if condition:
        return PlantQuery(species=species, subject=condition, is_gene=False)
    return None


def _default_transport(url: str, body: Optional[dict] = None) -> dict:
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class PlantEvidenceProvider:
    """Live Ensembl Plants + literature provider for plant questions."""

    name = "plant-live"

    # This provider's evidence is gathered from species-specific
    # records (Ensembl Plants species genomes + species-restricted
    # literature), so it is scoped to the Plant division. The
    # division remains unbenchmarked; the boundary text says so.
    scoped_divisions = frozenset({Division.PLANT})

    def __init__(
        self,
        transport: Callable = _default_transport,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._raw_transport = transport
        self.today = today
        self._cache: dict = {}

    def transport(self, url: str, body: Optional[dict] = None):
        """One retry on transient failure (public APIs throttle
        bursts); a second failure propagates to the caller's own
        error handling, which contributes no items."""
        try:
            return self._raw_transport(url, body)
        except Exception:
            return self._raw_transport(url, body)

    # -- provider protocol ------------------------------------------------
    def topic_for_question(self, question_text: str) -> Optional[str]:
        query = parse_plant_query(question_text)
        return query.topic_key if query else None

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        if topic_key in self._cache:
            return self._cache[topic_key]
        parts = topic_key.split(":", 2)
        if len(parts) != 3 or parts[0] != "plant" or parts[1] not in _BY_KEY:
            self._cache[topic_key] = ()
            return ()
        species = _BY_KEY[parts[1]]
        subject = parts[2]
        is_gene = subject == subject.upper() and "-" not in subject
        query = PlantQuery(
            species=species,
            subject=subject if is_gene else subject.replace("-", " "),
            is_gene=is_gene,
        )
        items: list[EvidenceItem] = []
        if query.is_gene:
            items.extend(self._ensembl_items(query, topic_key))
        items.extend(self._literature_items(query, topic_key))
        result = tuple(items)
        self._cache[topic_key] = result
        return result

    # -- Ensembl Plants -----------------------------------------------------
    def _ensembl_items(
        self, query: PlantQuery, topic_key: str
    ) -> list[EvidenceItem]:
        stamp = self.today().isoformat()
        slug = query.species.ensembl
        gene = query.subject
        try:
            record = self.transport(
                f"{ENSEMBL}/lookup/symbol/{slug}/{gene}"
                "?content-type=application/json",
                None,
            )
        except Exception:
            return []
        if not isinstance(record, dict):
            return []
        gene_id = record.get("id")
        if not gene_id:
            return []
        items: list[EvidenceItem] = []
        description = (record.get("description") or "").strip()
        statement = (
            f"Ensembl Plants records the {gene} gene in "
            f"{query.species.scientific} ({record.get('display_name', gene)}"
            f", {record.get('biotype', 'gene')}, assembly "
            f"{record.get('assembly_name', 'n/a')}, Ensembl gene {gene_id})"
            + (f": {description}" if description else "")
            + "."
        )
        items.append(
            EvidenceItem(
                item_id=f"plant-ensembl-gene-{gene_id}",
                topic_key=topic_key,
                statement=statement,
                source_id="ensembl-plants",
                citation=(
                    f"Ensembl Plants gene {gene_id} in {slug} "
                    f"(fetched {stamp} via the Ensembl REST API)"
                ),
                kind=EvidenceKind.OTHER,
                direction=SupportDirection.SUPPORTS,
                status=VerificationStatus.VERIFIED,
                locator=f"https://plants.ensembl.org/id/{gene_id}",
            )
        )
        items.extend(self._phenotype_items(query, topic_key, gene_id))
        return items

    def _phenotype_items(
        self, query: PlantQuery, topic_key: str, gene_id: str
    ) -> list[EvidenceItem]:
        """Trait associations where Ensembl has them.  In practice
        this endpoint returns no records for plant genes (verified
        live for Arabidopsis FLC); the code path exists so any
        record that IS returned is used, and absence contributes
        zero items rather than a padded claim."""
        stamp = self.today().isoformat()
        slug = query.species.ensembl
        try:
            payload = self.transport(
                f"{ENSEMBL}/phenotype/gene/{slug}/{gene_id}"
                "?content-type=application/json",
                None,
            )
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        items: list[EvidenceItem] = []
        seen: set = set()
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            phenotype = entry.get("phenotype")
            if isinstance(phenotype, dict):
                description = phenotype.get("description", "")
            else:
                description = str(phenotype or "")
            description = description.strip()
            if not description or description in seen:
                continue
            seen.add(description)
            source_name = ""
            source = entry.get("source")
            if isinstance(source, dict):
                source_name = source.get("name", "")
            items.append(
                EvidenceItem(
                    item_id=(
                        f"plant-ensembl-pheno-{gene_id}-"
                        f"{len(items)}"
                    ),
                    topic_key=topic_key,
                    statement=(
                        f"Ensembl Plants records a trait association "
                        f"for {query.subject} in "
                        f"{query.species.scientific}: \"{description}\""
                        + (f" (source: {source_name})" if source_name else "")
                        + "."
                    ),
                    source_id="ensembl-plants",
                    citation=(
                        f"Ensembl Plants phenotype associations for "
                        f"gene {gene_id} in {slug} (fetched {stamp} "
                        f"via the Ensembl REST API)"
                    ),
                    kind=EvidenceKind.CLINICAL_OBSERVATION,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                    locator=f"https://plants.ensembl.org/id/{gene_id}",
                )
            )
            if len(items) >= 2:
                break
        return items

    # -- literature -----------------------------------------------------------
    def _literature_items(
        self, query: PlantQuery, topic_key: str
    ) -> list[EvidenceItem]:
        stamp = self.today().isoformat()
        subject = query.subject
        search_query = (
            f'"{subject}" AND ("{query.species.scientific}" OR '
            f'"{query.species.key}")'
        )
        try:
            payload = self.transport(
                EUROPE_PMC + urllib.parse.quote(search_query), None
            )
        except Exception:
            return []
        items: list[EvidenceItem] = []
        for record in payload.get("resultList", {}).get("result", []):
            pmid = record.get("pmid")
            title = record.get("title")
            if not pmid or not title:
                continue
            items.append(
                EvidenceItem(
                    item_id=f"plant-lit-{pmid}",
                    topic_key=topic_key,
                    statement=(
                        f"Peer-reviewed literature on {subject} in "
                        f"{query.species.scientific}: \"{title}\" "
                        f"({record.get('pubYear', 'n.d.')})."
                    ),
                    source_id="pubmed-literature",
                    citation=(
                        f"PMID {pmid} via Europe PMC (fetched {stamp})"
                    ),
                    kind=EvidenceKind.LITERATURE_REPORT,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                )
            )
            if len(items) >= 2:
                break
        return items


class PlantCopilot:
    """The Plant division surface: gated answers + printed boundary."""

    def __init__(self, provider: Optional[PlantEvidenceProvider] = None) -> None:
        self.provider = provider if provider is not None else PlantEvidenceProvider()
        self.copilot = Copilot(provider=self.provider)

    def ask(self, question: str) -> str:
        parsed = parse_plant_query(question)
        if parsed is not None:
            # Carry the species so the answer is labelled with the
            # Plant division, not the Human default.
            answer = self.copilot.ask(
                CopilotQuestion(text=question, species=parsed.species.scientific)
            )
        else:
            answer = self.copilot.ask(question)
        return answer.render() + "\n\nBoundary: " + BOUNDARY


def _demo() -> None:
    plant = PlantCopilot()
    for question in (
        "What is known about FLC in Arabidopsis?",
        "What is known about SD1 in rice?",
        "What is known about FLC?",
    ):
        print("=" * 72)
        print("Q:", question)
        print(plant.ask(question))
        print()


if __name__ == "__main__":
    _demo()
