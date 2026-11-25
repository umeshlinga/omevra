"""Drug Discovery division, v1 (Step 5): target evidence.

Computational only, and honest about it.  This module answers one
kind of question — "what drug-discovery evidence exists for
targeting GENE?" — by fetching live records from ChEMBL (targets,
mechanisms, approved and clinical-stage compounds) and Europe PMC
literature, and passing them through the same cite-or-decline gate
as every other Provome answer.

What an answer here is: target evidence and candidate
prioritisation — which targets have real chemistry and clinical
history behind them, ranked by development phase and evidence
depth.  What it is never: a drug recommendation, a claim that any
compound treats any patient, or a substitute for synthesis, lab
and animal testing, human trials, and regulatory approval through
licensed partners.  Those boundaries are printed on every answer
this module produces.

Run the live demo:

    python -m provome.drug

Research software.  Not for clinical use.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional, Sequence

from .confirmation import confirm_literature
from .copilot import Copilot
from .evidence import (
    EvidenceItem,
    EvidenceKind,
    SupportDirection,
    VerificationStatus,
)

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
EUROPE_PMC = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    "?format=json&pageSize=3&query="
)
USER_AGENT = "Provome/0.6 (research software; drug target evidence fetch)"

_INTENT_WORDS = (
    "drug", "target", "druggable", "inhibitor", "compound",
    "candidate", "repurpos", "therapy", "therapeutic", "treat",
    "medicine", "pharma", "small molecule", "antibody",
)
_GENE_TOKEN = re.compile(r"\b([A-Z][A-Z0-9]{1,11})\b")
_NOT_GENES = {"DNA", "RNA", "USA", "FDA", "WHO", "II", "III", "IV", "PCR", "NGS"}

BOUNDARY = (
    "Computational evidence only: this is target evidence and "
    "candidate prioritisation, not a drug recommendation. A "
    "patient-ready drug additionally needs synthesis, lab and "
    "animal testing, human trials, and regulatory approval "
    "through licensed partners."
)


def _default_transport(url: str, body: Optional[dict] = None) -> dict:
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_drug_query(question_text: str) -> Optional[str]:
    """Return the gene symbol a drug-discovery question targets.

    Requires BOTH a drug-intent word and a gene-like token; a bare
    gene mention is a variant question, not a drug question.
    """
    if not isinstance(question_text, str):
        raise TypeError("question_text must be a string")
    lowered = question_text.lower()
    if not any(word in lowered for word in _INTENT_WORDS):
        return None
    for match in _GENE_TOKEN.finditer(question_text):
        token = match.group(1)
        if token not in _NOT_GENES:
            return token
    return None


@dataclass
class _Target:
    gene: str
    chembl_id: str
    pref_name: str
    target_type: str


class DrugEvidenceProvider:
    """Live ChEMBL + literature provider for target questions."""

    name = "drug-live"

    def __init__(
        self,
        transport: Callable = _default_transport,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._raw_transport = transport
        self.today = today
        self._cache: dict = {}

    def transport(self, url: str, body: Optional[dict] = None) -> dict:
        """One retry on transient failure (EBI services throttle
        bursts; a single retry recovered every flake seen in Step 5
        live testing).  A second failure propagates to the caller's
        own error handling, which contributes no items."""
        try:
            return self._raw_transport(url, body)
        except Exception:
            return self._raw_transport(url, body)

    # -- provider protocol ------------------------------------------------
    def topic_for_question(self, question_text: str) -> Optional[str]:
        gene = parse_drug_query(question_text)
        return f"drug:{gene}" if gene else None

    def items_for_topic(self, topic_key: str) -> Sequence[EvidenceItem]:
        if topic_key in self._cache:
            return self._cache[topic_key]
        gene = topic_key.split(":", 1)[1]
        items: list[EvidenceItem] = []
        target = self._find_target(gene)
        if target is not None:
            items.extend(self._chembl_items(target))
        items.extend(self._literature_items(gene, topic_key))
        result = tuple(items)
        self._cache[topic_key] = result
        return result

    # -- ChEMBL -------------------------------------------------------------
    def _uniprot_accession(self, gene: str) -> Optional[str]:
        """Resolve a human gene symbol to its UniProt accession.

        ChEMBL's API leaves component gene synonyms empty, so the
        honest join key between a gene and a ChEMBL target is the
        UniProt accession of the encoded protein.
        """
        url = (
            "https://rest.uniprot.org/uniprotkb/search?format=json&size=1"
            f"&fields=accession&query=gene_exact:{urllib.parse.quote(gene)}"
            "%20AND%20organism_id:9606%20AND%20reviewed:true"
        )
        try:
            payload = self.transport(url, None)
        except Exception:
            return None
        results = payload.get("results", [])
        return results[0].get("primaryAccession") if results else None

    def _find_target(self, gene: str) -> Optional[_Target]:
        accession = self._uniprot_accession(gene)
        if not accession:
            return None
        try:
            payload = self.transport(
                f"{CHEMBL}/target/search.json?q={urllib.parse.quote(gene)}&limit=10",
                None,
            )
        except Exception:
            return None
        for record in payload.get("targets", []):
            if record.get("organism") != "Homo sapiens":
                continue
            chembl_id = record.get("target_chembl_id", "")
            try:
                full = self.transport(f"{CHEMBL}/target/{chembl_id}.json", None)
            except Exception:
                continue
            for component in full.get("target_components", []):
                if component.get("accession") == accession:
                    return _Target(
                        gene=gene,
                        chembl_id=chembl_id,
                        pref_name=record.get("pref_name", gene),
                        target_type=record.get("target_type", ""),
                    )
        return None

    def _chembl_items(self, target: _Target) -> list[EvidenceItem]:
        stamp = self.today().isoformat()
        items: list[EvidenceItem] = []
        try:
            activities = self.transport(
                f"{CHEMBL}/activity.json?target_chembl_id={target.chembl_id}&limit=1",
                None,
            )
            total = activities.get("page_meta", {}).get("total_count", 0)
        except Exception:
            total = 0
        items.append(
            EvidenceItem(
                item_id=f"drug-chembl-target-{target.chembl_id}",
                topic_key=f"drug:{target.gene}",
                statement=(
                    f"ChEMBL records {target.gene} as a human drug target "
                    f"({target.pref_name}, {target.target_type}, "
                    f"{target.chembl_id}) with {total} reported bioactivity "
                    f"measurements — real chemistry exists against this target."
                ),
                source_id="chembl",
                citation=(
                    f"ChEMBL target {target.chembl_id} (fetched {stamp} "
                    f"via the ChEMBL API)"
                ),
                kind=EvidenceKind.EXPERIMENTAL_MODEL,
                direction=SupportDirection.SUPPORTS,
                status=VerificationStatus.VERIFIED,
            )
        )
        try:
            mechanisms = self.transport(
                f"{CHEMBL}/mechanism.json?target_chembl_id={target.chembl_id}&limit=40",
                None,
            ).get("mechanisms", [])
        except Exception:
            mechanisms = []
        approved: list[str] = []
        clinical = 0
        seen: set = set()
        for mech in mechanisms:
            mol_id = mech.get("molecule_chembl_id")
            if not mol_id or mol_id in seen:
                continue
            seen.add(mol_id)
            try:
                molecule = self.transport(f"{CHEMBL}/molecule/{mol_id}.json", None)
            except Exception:
                continue
            phase = molecule.get("max_phase")
            name = molecule.get("pref_name") or mol_id
            action = mech.get("mechanism_of_action") or mech.get("action_type") or ""
            if phase is not None and float(phase) >= 4:
                approved.append(f"{name} ({action})" if action else name)
            elif phase is not None and float(phase) >= 2:
                clinical += 1
        if approved:
            items.append(
                EvidenceItem(
                    item_id=f"drug-chembl-approved-{target.chembl_id}",
                    topic_key=f"drug:{target.gene}",
                    statement=(
                        f"Approved medicines already act on {target.gene}: "
                        + "; ".join(approved[:4])
                        + ". The target is clinically validated by "
                        "regulator-approved drugs."
                    ),
                    source_id="chembl",
                    citation=(
                        f"ChEMBL mechanisms for {target.chembl_id} "
                        f"(fetched {stamp} via the ChEMBL API)"
                    ),
                    kind=EvidenceKind.CLINICAL_OBSERVATION,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                )
            )
        if clinical:
            items.append(
                EvidenceItem(
                    item_id=f"drug-chembl-clinical-{target.chembl_id}",
                    topic_key=f"drug:{target.gene}",
                    statement=(
                        f"A further {clinical} compound(s) acting on "
                        f"{target.gene} reached phase 2-3 clinical "
                        f"development in ChEMBL's records — an active "
                        f"candidate pipeline exists beyond the approved set."
                    ),
                    source_id="chembl",
                    citation=(
                        f"ChEMBL mechanisms for {target.chembl_id} "
                        f"(fetched {stamp} via the ChEMBL API)"
                    ),
                    kind=EvidenceKind.CLINICAL_OBSERVATION,
                    direction=SupportDirection.SUPPORTS,
                    status=VerificationStatus.VERIFIED,
                )
            )
        return items

    # -- literature -----------------------------------------------------------
    def _literature_items(self, gene: str, topic_key: str) -> list[EvidenceItem]:
        stamp = self.today().isoformat()
        query = (
            f'TITLE:"{gene}" AND (inhibitor OR modulator OR antagonist '
            f"OR agonist OR drug OR target OR potentiator OR corrector)"
        )
        try:
            payload = self.transport(
                EUROPE_PMC + urllib.parse.quote(query), None
            )
        except Exception:
            return []
        items: list[EvidenceItem] = []
        for record in payload.get("resultList", {}).get("result", []):
            pmid = record.get("pmid")
            title = record.get("title")
            if not pmid or not title:
                continue
            # Gate hardening (2026-11-25): independent PubMed
            # re-confirmation before a paper counts as VERIFIED;
            # see provome.confirmation.
            status, note = confirm_literature(self.transport, pmid, title)
            items.append(
                EvidenceItem(
                    item_id=f"drug-lit-{pmid}",
                    topic_key=topic_key,
                    statement=(
                        f"Peer-reviewed literature on drugging {gene}: "
                        f"\"{title}\" ({record.get('pubYear', 'n.d.')})."
                    ),
                    source_id="pubmed-literature",
                    citation=(
                        f"PMID {pmid} via Europe PMC (fetched {stamp})"
                    ),
                    kind=EvidenceKind.LITERATURE_REPORT,
                    direction=SupportDirection.SUPPORTS,
                    status=status,
                    notes=note,
                )
            )
            if len(items) >= 2:
                break
        return items


class DrugCopilot:
    """The Drug Discovery surface: gated answers + printed boundary."""

    def __init__(self, provider: Optional[DrugEvidenceProvider] = None) -> None:
        self.provider = provider if provider is not None else DrugEvidenceProvider()
        self.copilot = Copilot(provider=self.provider)

    def ask(self, question: str) -> str:
        answer = self.copilot.ask(question)
        return answer.render() + "\n\nBoundary: " + BOUNDARY


def _demo() -> None:
    drug = DrugCopilot()
    for question in (
        "What drug discovery evidence exists for targeting CFTR?",
        "Is BCL2 a druggable target with approved medicines?",
        "What drug discovery evidence exists for targeting ZZZQX9?",
    ):
        print("=" * 72)
        print("Q:", question)
        print(drug.ask(question))
        print()


if __name__ == "__main__":
    _demo()
