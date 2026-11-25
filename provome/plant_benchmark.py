"""Plant division benchmark, v1: does the Plant Copilot behave?

This is the Plant counterpart of :mod:`provome.animal_benchmark`.
That harness measures the Animal surface on fixed probes; this
one measures the whole Plant surface — question in, verdict out —
on fixed probes whose ground truth was recorded from real sources
on 2026-11-25 (Ensembl Plants, Europe PMC; the source URL is kept
on every probe).

Each probe is a question with a known correct verdict:

* ``answer`` probes name a gene/species pair whose association is
  real and whose recorded live payloads genuinely pass the gate —
  at least ``MIN_VERIFIED_ITEMS`` verified items from at least
  ``MIN_INDEPENDENT_SOURCES`` independent groups.  In v1 that
  means an Ensembl Plants gene record plus two PMID-bearing
  literature items: Ensembl Plants serves no trait associations
  for any probed gene (see the fixture notes), so the gene record
  is the only Ensembl item there is.  The Copilot must answer.
* ``decline`` probes must decline, for one of the recorded
  reasons: a fabricated gene symbol, a human gene asked in a
  plant, a real gene in the wrong (or an unsupported) species,
  no species named at all, a trait topic (literature-only in v1,
  a single source group), or — the uncomfortable ones — real,
  famous genes the v1 sources cannot reach: rice SD1 and maize
  TB1 have no symbol record in Ensembl Plants at all, and
  Arabidopsis FRI's recorded literature window carries a single
  PMID-bearing paper, leaving the reachable evidence below the
  gate.  A benchmark that only contains probes the system passes
  would be rigged; the misses are in here on purpose.

Fixture mode (the default, and what CI scores) replays payloads
recorded verbatim from the live APIs on 2026-11-25 through the
real provider and the real gate — deterministic, no network.
Live mode (``--live``) runs the same probes against the live
APIs; its score can move with API state and is not the scored
benchmark.

What a perfect score does NOT establish: coverage.  This is a
fixed, builder-chosen probe set — a smoke benchmark of verdict
behaviour, not external validation.  The Plant division stays
"in build and unbenchmarked" in every public wording until
external validation exists and the Founder/CEO make the call.

The probe set and fixtures are frozen at ``plant-benchmark-v1``;
changing any probe, payload, or expected verdict is a new
version, never an edit.

Run it:

    python -m provome.plant_benchmark          # fixture mode
    python -m provome.plant_benchmark --live   # live mode

Research software.  Agricultural research use only.
"""

from __future__ import annotations

import copy
import json
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from .evidence import default_source_registry, evaluate_gate
from .plant import PlantCopilot, PlantEvidenceProvider

__all__ = [
    "BENCHMARK_VERSION",
    "PlantBenchmarkReport",
    "PlantProbe",
    "ProbeResult",
    "fixture_transport",
    "probes",
    "run_benchmark",
]

BENCHMARK_VERSION = "plant-benchmark-v1"
FIXTURE_DATE = date(2026, 11, 25)


@dataclass(frozen=True)
class PlantProbe:
    """One fixed Plant test: question in, expected verdict out.

    ``ground_truth`` states what is true in the world and why the
    expected verdict follows; ``source_url`` is where that truth
    was verified live on 2026-11-25.
    """

    name: str
    category: str  # "answer" | "decline"
    question: str
    expected_answer: bool
    ground_truth: str
    source_url: str


def probes() -> tuple:
    """The fixed probe set for plant-benchmark-v1. No network."""
    return (
        # --- must answer: real association, gate genuinely passes --
        PlantProbe(
            "arabidopsis-flc-flowering", "answer",
            "What is known about FLC in Arabidopsis?",
            True,
            "FLC (FLOWERING LOCUS C) is the central MADS-box "
            "floral repressor of the Arabidopsis vernalization "
            "pathway; Ensembl Plants records the gene in "
            "Arabidopsis thaliana (AT5G10140, TAIR10) and the "
            "recorded literature window carries three "
            "PMID-bearing papers on it.",
            "https://plants.ensembl.org/id/AT5G10140",
        ),
        PlantProbe(
            "arabidopsis-ttg1-trichome", "answer",
            "What is known about TTG1 in Arabidopsis?",
            True,
            "TTG1 (TRANSPARENT TESTA GLABRA 1) is the WD40 "
            "repeat protein of the MBW complex regulating "
            "trichome development and anthocyanin production "
            "in Arabidopsis; Ensembl Plants records the gene "
            "(AT5G24520) and three PMID-bearing papers were "
            "recorded for the provider's query.",
            "https://plants.ensembl.org/id/AT5G24520",
        ),
        PlantProbe(
            "arabidopsis-ft-florigen", "answer",
            "What is known about FT in Arabidopsis?",
            True,
            "FT (FLOWERING LOCUS T) encodes florigen, the "
            "mobile flowering signal in Arabidopsis; Ensembl "
            "Plants records the gene (AT1G65480) and two "
            "PMID-bearing papers were recorded — the gate "
            "passes at exactly its minimum (3 items, 2 groups).",
            "https://plants.ensembl.org/id/AT1G65480",
        ),
        PlantProbe(
            "arabidopsis-cop1-light", "answer",
            "What is known about COP1 in Arabidopsis?",
            True,
            "COP1 (CONSTITUTIVE PHOTOMORPHOGENIC 1) is the E3 "
            "ubiquitin ligase at the centre of Arabidopsis "
            "light signalling; Ensembl Plants records the gene "
            "(AT2G32950) and two PMID-bearing papers were "
            "recorded — again exactly at the gate minimum.",
            "https://plants.ensembl.org/id/AT2G32950",
        ),
        PlantProbe(
            "rice-xa21-blight-resistance", "answer",
            "What is known about XA21 in rice?",
            True,
            "XA21 is the rice leucine-rich repeat receptor "
            "kinase conferring broad resistance to bacterial "
            "blight (Xanthomonas oryzae pv. oryzae); Ensembl "
            "Plants records it in Oryza sativa (Os11g0569733, "
            "IRGSP-1.0, display name XA21) and two "
            "PMID-bearing papers were recorded.",
            "https://plants.ensembl.org/id/Os11g0569733",
        ),
        PlantProbe(
            "tomato-sp-self-pruning", "answer",
            "What is known about SP in tomato?",
            True,
            "SP (SELF-PRUNING) is the TFL1-family gene that "
            "controls determinate growth habit in tomato — a "
            "foundational breeding trait; Ensembl Plants "
            "records it in Solanum lycopersicum "
            "(gene-Solyc06g074350.3) and two PMID-bearing "
            "papers were recorded. (The recorded literature "
            "is full-text matched and off-topic — the "
            "benchmark scores verdict behaviour, not "
            "relevance; see the worker report.)",
            "https://plants.ensembl.org/id/gene-Solyc06g074350.3",
        ),
        PlantProbe(
            "soybean-e1-maturity", "answer",
            "What is known about E1 in soybean?",
            True,
            "E1 is the major soybean maturity locus, a "
            "B3-type transcription factor repressing "
            "flowering; Ensembl Plants records it in Glycine "
            "max (GLYMA_06G207800) and three PMID-bearing "
            "papers were recorded.",
            "https://plants.ensembl.org/id/GLYMA_06G207800",
        ),
        PlantProbe(
            "soybean-dt1-determinacy", "answer",
            "What is known about DT1 in soybean?",
            True,
            "DT1 (GmTFL1b) controls stem determinacy in "
            "soybean; Ensembl Plants records the gene in "
            "Glycine max (GLYMA_19G194300, display name Dt1) "
            "and three PMID-bearing papers were recorded.",
            "https://plants.ensembl.org/id/GLYMA_19G194300",
        ),
        # --- must decline -------------------------------------------
        PlantProbe(
            "rice-sd1-no-symbol-record", "decline",
            "What is known about SD1 in rice?",
            False,
            "SD1 (semi-dwarf1, the GA20ox2 gene) is the Green "
            "Revolution rice gene — real and famous — but "
            "Ensembl Plants' oryza_sativa symbol index has no "
            "SD1 record (live lookup returned HTTP 400 on "
            "2026-11-25; the gene sits under a locus ID, not "
            "the symbol). The only reachable evidence is "
            "literature: 2 items, 1 source group, below the "
            "gate. A documented v1 coverage gap, kept in the "
            "set so the benchmark cannot hide it.",
            "https://rest.ensembl.org/lookup/symbol/oryza_sativa/SD1",
        ),
        PlantProbe(
            "maize-tb1-no-symbol-record", "decline",
            "What is known about TB1 in maize?",
            False,
            "TB1 (teosinte branched1) is the classic maize "
            "domestication gene — real and famous — but "
            "Ensembl Plants' zea_mays symbol lookup returned "
            "HTTP 400 for both 'TB1' and 'tb1' on 2026-11-25, "
            "so the gene record is unreachable by symbol and "
            "the reachable evidence is literature only "
            "(1 group). A second documented v1 coverage gap: "
            "two of the seven supported species' most famous "
            "genes cannot be reached by symbol at all.",
            "https://rest.ensembl.org/lookup/symbol/zea_mays/TB1",
        ),
        PlantProbe(
            "arabidopsis-fri-below-gate", "decline",
            "What is known about FRI in Arabidopsis?",
            False,
            "FRI (FRIGIDA) is a real Arabidopsis flowering "
            "gene and Ensembl Plants DOES record it "
            "(AT4G00650) — but the recorded Europe PMC window "
            "for the provider's query holds only one "
            "PMID-bearing record (a preprint and a PMC record "
            "without a PMID precede it), so the reachable "
            "evidence is the gene record plus one paper: "
            "2 items, below the gate's 3. The system must "
            "decline a real gene rather than pad.",
            "https://plants.ensembl.org/id/AT4G00650",
        ),
        PlantProbe(
            "rice-brca1-human-gene", "decline",
            "What is known about BRCA1 in rice?",
            False,
            "BRCA1 is a human DNA-repair gene; Ensembl "
            "Plants' oryza_sativa symbol index has no BRCA1 "
            "record (live lookup HTTP 400, 2026-11-25). "
            "Full-text literature matches leak in, but at one "
            "source group the gate must decline.",
            "https://rest.ensembl.org/lookup/symbol/oryza_sativa/BRCA1",
        ),
        PlantProbe(
            "rice-tp53-human-gene", "decline",
            "What is known about TP53 in rice?",
            False,
            "TP53 is the mammalian tumour-suppressor gene; "
            "plants have no TP53 and Ensembl Plants' "
            "oryza_sativa symbol index returned HTTP 400 on "
            "2026-11-25. Literature full-text matches alone "
            "(one group) must not pass the gate.",
            "https://rest.ensembl.org/lookup/symbol/oryza_sativa/TP53",
        ),
        PlantProbe(
            "rice-flc-wrong-species", "decline",
            "What is known about FLC in rice?",
            False,
            "FLC is a Brassicaceae floral repressor; rice has "
            "no FLC and Ensembl Plants' oryza_sativa symbol "
            "index returned HTTP 400 on 2026-11-25. "
            "Comparative literature that mentions both leaks "
            "in at one source group, so the gate must "
            "decline — the Arabidopsis record (the answer "
            "probe above) must never be borrowed across "
            "species.",
            "https://rest.ensembl.org/lookup/symbol/oryza_sativa/FLC",
        ),
        PlantProbe(
            "wheat-zzzqx9-fabricated", "decline",
            "What is known about ZZZQX9 in wheat?",
            False,
            "ZZZQX9 is a fabricated symbol: no Ensembl "
            "Plants record (HTTP 400) and zero Europe PMC "
            "results, both live-verified 2026-11-25. Any "
            "answer here would be invented.",
            "https://rest.ensembl.org/lookup/symbol/triticum_aestivum/ZZZQX9",
        ),
        PlantProbe(
            "rice-fakgene1-fabricated", "decline",
            "What is known about FAKGENE1 in rice?",
            False,
            "FAKGENE1 is a fabricated symbol: no Ensembl "
            "Plants record (HTTP 400) and zero Europe PMC "
            "results, both live-verified 2026-11-25.",
            "https://rest.ensembl.org/lookup/symbol/oryza_sativa/FAKGENE1",
        ),
        PlantProbe(
            "tomato-notreal7-fabricated", "decline",
            "What is known about NOTREAL7 in tomato?",
            False,
            "NOTREAL7 is a fabricated symbol: no Ensembl "
            "Plants record (HTTP 400) and zero Europe PMC "
            "results, both live-verified 2026-11-25.",
            "https://rest.ensembl.org/lookup/symbol/solanum_lycopersicum_gca000188115v5cm/NOTREAL7",
        ),
        PlantProbe(
            "wheat-drought-tolerance-trait", "decline",
            "What is known about drought tolerance in wheat?",
            False,
            "Drought tolerance in wheat is real and heavily "
            "studied (e.g. PMID 43050865, recorded for this "
            "query), but v1 trait topics gather literature "
            "only — the Ensembl Plants path is gene-only — "
            "so the evidence is a single source group and "
            "the gate declines by design until a second "
            "trait source exists. A visible, deliberate gap.",
            "https://pubmed.ncbi.nlm.nih.gov/43050865/",
        ),
        PlantProbe(
            "rice-salt-tolerance-trait", "decline",
            "What is known about salt tolerance in rice?",
            False,
            "Salt tolerance in rice is real and heavily "
            "studied (e.g. PMID 42721655, recorded for this "
            "query), but as a trait topic it is "
            "literature-only in v1 and must decline; gene "
            "questions (the XA21 probe above) are the "
            "answerable route.",
            "https://pubmed.ncbi.nlm.nih.gov/42721655/",
        ),
        PlantProbe(
            "no-species-flc", "decline",
            "What is known about FLC?",
            False,
            "No species is named, so no plant topic may be "
            "recognised (the Step 3 intent rule): plant "
            "evidence must never be gathered for an unstated "
            "organism, however real the gene is.",
            "https://plants.ensembl.org/id/AT5G10140",
        ),
        PlantProbe(
            "barley-flc-unsupported-species", "decline",
            "What is known about FLC in barley?",
            False,
            "Barley (Hordeum vulgare) is not one of the "
            "seven species the Plant provider supports, so "
            "no topic may be recognised and no evidence may "
            "be gathered — the Arabidopsis FLC record must "
            "not be silently gathered for an unsupported "
            "organism.",
            "https://plants.ensembl.org/id/AT5G10140",
        ),
    )


# ---------------------------------------------------------------------
# Recorded fixtures (live payloads, recorded 2026-11-25)
#
# "lookup":    "<ensembl slug>/<GENE>" -> gene record, or null where
#              the live symbol lookup returned HTTP 400 (no record).
#              Records are projected to the five fields the
#              provider consumes (id / display_name / biotype /
#              assembly_name / description), values verbatim.
# "phenotype": "<ensembl gene id>" -> association list as returned,
#              or null where the live call returned HTTP 400.
#              Every resolved gene returned an EMPTY list (HTTP
#              200) or an HTTP 400 server error (tomato SP and the
#              two soybean genes): Ensembl Plants served no trait
#              association for any probed gene. Recorded as-is.
# "literature": "<subject>|<species key>" -> Europe PMC result
#              list (pmid / title / pubYear per record, values
#              verbatim) for the provider's exact query. The
#              species key is part of the fixture key because the
#              provider's query is species-restricted and the same
#              subject (FLC) is probed in two species.
# ---------------------------------------------------------------------

_FIXTURE_JSON = r"""{"lookup":{"arabidopsis_thaliana/FLC":{"id":"AT5G10140","display_name":"FLC","biotype":"protein_coding","assembly_name":"TAIR10","description":"K-box region and MADS-box transcription factor family protein [Source:NCBI gene (formerly Entrezgene);Acc:830878]"},"arabidopsis_thaliana/FRI":{"id":"AT4G00650","display_name":"FRI","biotype":"protein_coding","assembly_name":"TAIR10","description":"FRIGIDA-like protein [Source:NCBI gene (formerly Entrezgene);Acc:828044]"},"arabidopsis_thaliana/TTG1":{"id":"AT5G24520","display_name":"TTG1","biotype":"protein_coding","assembly_name":"TAIR10","description":"Transducin/WD40 repeat-like superfamily protein [Source:NCBI gene (formerly Entrezgene);Acc:832523]"},"arabidopsis_thaliana/FT":{"id":"AT1G65480","display_name":"FT","biotype":"protein_coding","assembly_name":"TAIR10","description":"PEBP (phosphatidylethanolamine-binding protein) family protein [Source:NCBI gene (formerly Entrezgene);Acc:842859]"},"arabidopsis_thaliana/COP1":{"id":"AT2G32950","display_name":"COP1","biotype":"protein_coding","assembly_name":"TAIR10","description":"Transducin/WD40 repeat-like superfamily protein [Source:NCBI gene (formerly Entrezgene);Acc:817857]"},"solanum_lycopersicum_gca000188115v5cm/SP":{"id":"gene-Solyc06g074350.3","display_name":"SP","biotype":"protein_coding","assembly_name":"SL4.0","description":"protein SELF-PRUNING [Source:NCBI gene (formerly Entrezgene);Acc:544038]"},"oryza_sativa/XA21":{"id":"Os11g0569733","display_name":"XA21","biotype":"protein_coding","assembly_name":"IRGSP-1.0"},"glycine_max/E1":{"id":"GLYMA_06G207800","display_name":"E1","biotype":"protein_coding","assembly_name":"Glycine_max_v2.1","description":"transcription factor E1 [Source:NCBI gene (formerly Entrezgene);Acc:102666452]"},"glycine_max/DT1":{"id":"GLYMA_19G194300","display_name":"Dt1","biotype":"protein_coding","assembly_name":"Glycine_max_v2.1"},"zea_mays/TB1":null,"oryza_sativa/SD1":null,"oryza_sativa/BRCA1":null,"oryza_sativa/TP53":null,"oryza_sativa/FLC":null,"triticum_aestivum/ZZZQX9":null,"oryza_sativa/FAKGENE1":null,"solanum_lycopersicum_gca000188115v5cm/NOTREAL7":null},"phenotype":{"AT5G10140":[],"AT4G00650":[],"AT5G24520":[],"AT1G65480":[],"AT2G32950":[],"gene-Solyc06g074350.3":null,"Os11g0569733":[],"GLYMA_06G207800":null,"GLYMA_19G194300":null},"literature":{"FLC|arabidopsis":[{"pmid":"42663084","title":"Local Trade-Offs Shape Flower Size Evolution Across Arabidopsis thaliana Distribution.","pubYear":"2026"},{"pmid":"42250340","title":"The anion channel SLAH3 regulates flowering time in Arabidopsis thaliana.","pubYear":"2026"},{"pmid":"42010214","title":"Formation of Adaptive Defense to Low Temperatures in Arabidopsis thaliana.","pubYear":"2026"}],"FRI|arabidopsis":[{"pmid":null,"title":"Disruption of lysine biosynthesis modulates DNA methylation and developmental programs in Arabidopsis thaliana","pubYear":"2026"},{"pmid":null,"title":"Temperature-mediated plasticity of floral and fitness traits in Arabidopsis thaliana","pubYear":"2026"},{"pmid":"42198889","title":"Differentiation for drought strategy but conserved plasticity to heat and drought in locally adapted populations of Arabidopsis thaliana.","pubYear":"2026"}],"TTG1|arabidopsis":[{"pmid":"42231887","title":"Antagonistic regulation of anthocyanin biosynthesis by HY5 and BPC1 in &lt;i&gt;Arabidopsis thaliana&lt;/i&gt;.","pubYear":"2026"},{"pmid":"40992548","title":"Overexpression of OsTTG1 influences seed size and silique length in Arabidopsis thaliana.","pubYear":"2025"},{"pmid":"41649624","title":"Combined R2R3-MYB transcription factor mutants reveal the regulatory structure of the Arabidopsis thaliana flavonoid biosynthesis pathway.","pubYear":"2026"}],"FT|arabidopsis":[{"pmid":"42047565","title":"NIMA-related kinases redundantly regulate vegetative and reproductive development in Arabidopsis thaliana.","pubYear":"2026"},{"pmid":null,"title":"Temperature-mediated plasticity of floral and fitness traits in Arabidopsis thaliana","pubYear":"2026"},{"pmid":"42250340","title":"The anion channel SLAH3 regulates flowering time in Arabidopsis thaliana.","pubYear":"2026"}],"COP1|arabidopsis":[{"pmid":null,"title":"Light-dependent activation of  <i>TCP4</i>  by the GATA25 transcription factor promotes petiole and hypocotyl elongation in  <i>Arabidopsis thaliana</i>","pubYear":"2026"},{"pmid":"42551010","title":"KIX8 coordinates leaf development and light-induced stress responses in Medicago truncatula and Arabidopsis thaliana.","pubYear":"2026"},{"pmid":"42892657","title":"Light modulates organ-specific gibberellin-abscisic acid crosstalk during drought response in Arabidopsis thaliana.","pubYear":"2026"}],"SP|tomato":[{"pmid":"42790656","title":"Endophytic Myco- Bio- Control Agents (BCA) against phytopathogens of Solanum lycopersicum L.","pubYear":"2026"},{"pmid":"43028960","title":"Silicon Enhances Water Use Efficiency by Improving Thermal Responses in &lt;i&gt;Solanum lycopersicum&lt;/i&gt; L.","pubYear":"2026"},{"pmid":null,"title":"Biostimulant Effects of Plant-Growth-Promoting Bacteria (PGPB) Consortia in Zea mays, Brassica oleracea, and Solanum lycopersicum","pubYear":"2026"}],"XA21|rice":[{"pmid":"42728868","title":"History of near-isogenic lines carrying genes for resistance to bacterial blight and blast in rice (Oryza sativa L.).","pubYear":"2026"},{"pmid":"42941748","title":"From Gene-Based Breeding to Haplotype-Based Breeding for Rice (&lt;i&gt;Oryza sativa&lt;/i&gt;) Improvement in the Genomics Era.","pubYear":"2026"},{"pmid":null,"title":"Marker-assisted backcross breeding for bacterial blight, blast and tolerance to low soil phosphorus in rice (Oryza sativa L.)","pubYear":"2025"}],"E1|soybean":[{"pmid":"42799890","title":"Genome-wide association study of diverse soybean [Glycine max (L.) Merrill] accessions for agronomic and seed composition traits in Rwanda.","pubYear":"2026"},{"pmid":"42444738","title":"Identification of a locus associated with chlorosis and antioxidant capacity using RNA-seq and BSA-seq in soybean [&lt;i&gt;Glycine max&lt;/i&gt; (L.) Merr].","pubYear":"2025"},{"pmid":"42286450","title":"Quantitative trait loci analysis identifying novel genomic regions for off-target dicamba tolerance in soybean [Glycine max (L.) Merr.].","pubYear":"2026"}],"DT1|soybean":[{"pmid":"42799890","title":"Genome-wide association study of diverse soybean [Glycine max (L.) Merrill] accessions for agronomic and seed composition traits in Rwanda.","pubYear":"2026"},{"pmid":"42663742","title":"Characterization of disomic alien addition lines from an intersubgeneric cross between Glycine max and G. tomentella and phenotypic variation within those lines and their progenies.","pubYear":"2026"},{"pmid":"41560908","title":"Integrating linkage mapping and GWAS reveals novel genetic architecture of seed weight in soybean (&lt;i&gt;Glycine max&lt;/i&gt; L.).","pubYear":"2025"}],"TB1|maize":[{"pmid":"42278377","title":"Application of NGS Technology, Association Mapping, and Physical Mapping Technologies to Identify Candidate Genes Associated with Maize (&lt;i&gt;Zea mays&lt;/i&gt; L.) Hybrid Yield.","pubYear":"2026"},{"pmid":"41043001","title":"Vegetative to generative1 (Vgt1) is an enhancer affecting flowering time and jasmonate signaling in maize by promoting the expression of Zea mays Related to APETALA 2.7.","pubYear":"2025"},{"pmid":"37508564","title":"The Genetic Structures and Molecular Mechanisms Underlying Ear Traits in Maize (<i>Zea mays</i> L.).","pubYear":"2023"}],"SD1|rice":[{"pmid":"42941748","title":"From Gene-Based Breeding to Haplotype-Based Breeding for Rice (&lt;i&gt;Oryza sativa&lt;/i&gt;) Improvement in the Genomics Era.","pubYear":"2026"},{"pmid":null,"title":"Integrative molecular mechanisms of lodging resistance in rice (Oryza sativa L.) culms: from DELLA\u2013OsGSK2\u2013BZR1 signaling to cell wall engineering and QTL-based breeding","pubYear":null},{"pmid":"41527021","title":"OsDof1 enhances rice (Oryza sativa L.) lodging resistance through suppression of tryptophan-dependent auxin biosynthesis.","pubYear":"2026"}],"BRCA1|rice":[{"pmid":null,"title":"Regulatory actions of rare earth elements (La and Gd) on the cell cycle of root tips in rice seedlings (Oryza sativa L.)","pubYear":"2022"},{"pmid":"35917980","title":"Regulatory actions of rare earth elements (La and Gd) on the cell cycle of root tips in rice seedlings (Oryza sativa L.).","pubYear":"2022"},{"pmid":"41693401","title":"Hidden players in plant response to sulfur deficit and beyond: insights into the function of the proteins Sulfur Deficiency Induced and Response to Low Sulfur.","pubYear":"2026"}],"TP53|rice":[{"pmid":null,"title":"Regulatory actions of rare earth elements (La and Gd) on the cell cycle of root tips in rice seedlings (Oryza sativa L.)","pubYear":"2022"},{"pmid":"35917980","title":"Regulatory actions of rare earth elements (La and Gd) on the cell cycle of root tips in rice seedlings (Oryza sativa L.).","pubYear":"2022"},{"pmid":"41440174","title":"Maize miRNAs Might Regulate Human Genes Involved in Prostate Cancer: An In Silico Approach.","pubYear":"2025"}],"FLC|rice":[{"pmid":"38413866","title":"Alternative polyadenylation profiles of susceptible and resistant rice (Oryza sativa L.) in response to bacterial leaf blight using RNA-seq.","pubYear":"2024"},{"pmid":"42511797","title":"Epigenetic Control of Cold Stress Tolerance in Plants: Emerging Mechanisms and Applications for Crop Improvement.","pubYear":"2026"},{"pmid":"42246288","title":"Mineral nutrients as regulators of plant flowering time: A molecular perspective.","pubYear":"2026"}],"ZZZQX9|wheat":[],"FAKGENE1|rice":[],"NOTREAL7|tomato":[],"drought tolerance|wheat":[{"pmid":"43050865","title":"UAV phenomics and GWAS uncover genetic architecture of early drought tolerance in wheat (Triticum aestivum L.).","pubYear":"2026"},{"pmid":"42896143","title":"Transcription factor TabZIP12 orchestrates the ABA signaling pathway to regulate drought tolerance in wheat (Triticum aestivum L.).","pubYear":"2026"},{"pmid":"42970105","title":"Integrated GWAS and RNA-Seq analyses identify candidate genes for drought tolerance in spring wheat (Triticum aestivum L.) from the cold and arid region of China.","pubYear":"2026"}],"salt tolerance|rice":[{"pmid":"42721655","title":"Cloning and functional characterization of OsERFSR1, a negative regulator of salt stress tolerance in rice (Oryza sativa L.).","pubYear":"2026"},{"pmid":"42243414","title":"Sundarbans-derived halotolerant Bacillus inaquosorum SWA3TU mitigates arsenic and lead stress in Oryza sativa L.","pubYear":"2026"},{"pmid":"41684070","title":"Ubiquitin-like SUMO protease expansion in rice (Oryza sativa L.).","pubYear":"2026"}]}}"""

_FIXTURES = json.loads(_FIXTURE_JSON)

# PubMed titles for the independent re-confirmation lookup the
# provider now performs (gate hardening, 2026-11-25): the recorded
# Europe PMC records above are real papers, and the honest PubMed
# in this fixture world resolves exactly those PMIDs to the same
# recorded titles.  A PMID the recorded literature never served
# does not resolve.
_PUBMED_TITLES = {
    record["pmid"]: record["title"]
    for records in _FIXTURES["literature"].values()
    for record in records
    if record.get("pmid")
}


def fixture_transport(url: str, body: Optional[dict] = None) -> dict:
    """Replay the recorded live payloads. No network, deterministic.

    Mirrors the live APIs' behaviour, including their failures: a
    recorded HTTP 400 raises, exactly as the real transport's
    ``urlopen`` raises on an HTTP error, and the provider's own
    error handling turns that into zero items.
    """
    if "/lookup/symbol/" in url:
        key = url.split("/lookup/symbol/", 1)[1].split("?", 1)[0]
        record = _FIXTURES["lookup"][key]
        if record is None:
            raise RuntimeError(f"recorded HTTP 400 for lookup {key}")
        return copy.deepcopy(record)
    if "/phenotype/gene/" in url:
        gene_id = (
            url.split("/phenotype/gene/", 1)[1].split("/", 1)[1]
            .split("?", 1)[0]
        )
        payload = _FIXTURES["phenotype"][gene_id]
        if payload is None:
            raise RuntimeError(f"recorded HTTP 400 for phenotype {gene_id}")
        return copy.deepcopy(payload)
    if "esummary" in url and "db=pubmed" in url:
        pmid = url.rsplit("id=", 1)[1]
        title = _PUBMED_TITLES.get(pmid)
        if title is None:
            return {"result": {"uids": []}}
        return {"result": {"uids": [pmid], pmid: {"title": title}}}
    if "europepmc" in url:
        parts = urllib.parse.unquote(url).split('"')
        subject, species_key = parts[1], parts[5]
        results = _FIXTURES["literature"][f"{subject}|{species_key}"]
        return {"resultList": {"result": copy.deepcopy(results)}}
    raise AssertionError(url)


def _fixture_factory() -> PlantEvidenceProvider:
    return PlantEvidenceProvider(
        transport=fixture_transport, today=lambda: FIXTURE_DATE
    )


def _live_factory() -> PlantEvidenceProvider:
    return PlantEvidenceProvider()


@dataclass(frozen=True)
class ProbeResult:
    """The measured outcome for one probe."""

    name: str
    category: str
    expected_answer: bool
    answered: bool
    correct: bool
    verified_items: int
    source_groups: int
    gate_passed: Optional[bool]  # None when no topic was recognised
    detail: str


@dataclass(frozen=True)
class PlantBenchmarkReport:
    """The benchmark outcome, stated exactly as measured."""

    version: str
    mode: str  # "fixture" | "live"
    total: int
    correct: int
    failures: tuple  # probe names whose verdict was wrong
    results: tuple  # ProbeResult per probe, in probe order

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_benchmark(
    mode: str = "fixture",
    provider_factory: Optional[Callable[[], PlantEvidenceProvider]] = None,
) -> PlantBenchmarkReport:
    """Run every probe through the real PlantCopilot and score it.

    Scoring is against the recorded ground truth, never against
    the system's own say-so — and it is audited: for each probe
    the gate is re-evaluated independently on the provider's own
    items, and an answer whose gate arithmetic does not genuinely
    pass is scored wrong even if the surface claimed ANSWER.
    A provider rigged to answer everything therefore fails every
    decline probe it flips.
    """
    if mode not in ("fixture", "live"):
        raise ValueError(f"unknown benchmark mode: {mode!r}")
    if provider_factory is None:
        provider_factory = _fixture_factory if mode == "fixture" else _live_factory
    registry = default_source_registry()
    results = []
    failures = []
    correct = 0
    for probe in probes():
        provider = provider_factory()
        copilot = PlantCopilot(provider=provider)
        text = copilot.ask(probe.question)
        answered = text.startswith("[ANSWER]")
        verified = 0
        groups = 0
        gate_passed: Optional[bool] = None
        topic = provider.topic_for_question(probe.question)
        if topic is not None:
            decision = evaluate_gate(
                provider.items_for_topic(topic), topic, registry
            )
            verified = decision.verified_count
            groups = decision.independent_source_count
            gate_passed = decision.passed
        detail = ""
        ok = answered == probe.expected_answer
        if answered and gate_passed is not True:
            ok = False
            detail = (
                "answered but the gate arithmetic does not pass "
                "on the provider's own items"
            )
        if ok:
            correct += 1
        else:
            failures.append(probe.name)
        results.append(
            ProbeResult(
                name=probe.name,
                category=probe.category,
                expected_answer=probe.expected_answer,
                answered=answered,
                correct=ok,
                verified_items=verified,
                source_groups=groups,
                gate_passed=gate_passed,
                detail=detail,
            )
        )
    return PlantBenchmarkReport(
        version=BENCHMARK_VERSION,
        mode=mode,
        total=len(results),
        correct=correct,
        failures=tuple(failures),
        results=tuple(results),
    )


_HONESTY_NOTE = (
    "What this establishes: on a fixed set of 21 probes whose\n"
    "ground truth and payloads were recorded from live sources\n"
    "(Ensembl Plants, Europe PMC) on 2026-11-25, the Plant\n"
    "Copilot's verdicts match the recorded truth, with the gate\n"
    "arithmetic independently re-checked per probe.\n"
    "What it does NOT establish: coverage of the Plant division.\n"
    "The probes were chosen by the builders; fixture payloads are\n"
    "a one-day snapshot; a perfect score here is a smoke\n"
    "benchmark, not external validation. The division remains\n"
    "'in build and unbenchmarked' in public wording until\n"
    "external validation exists and the Founder/CEO decide\n"
    "otherwise."
)


def _demo() -> None:
    mode = "live" if "--live" in sys.argv else "fixture"
    report = run_benchmark(mode=mode)
    print(f"Provome Plant division benchmark ({report.version}, {mode} mode)")
    print("=" * 72)
    for r in report.results:
        verdict = "ANSWER " if r.answered else "DECLINE"
        expected = "ANSWER " if r.expected_answer else "DECLINE"
        mark = "ok  " if r.correct else "FAIL"
        gate = (
            f"{r.verified_items} verified, {r.source_groups} group(s)"
            if r.gate_passed is not None
            else "no topic recognised"
        )
        print(f"  [{mark}] {r.name:36s} {verdict} (expected {expected}) {gate}")
        if r.detail:
            print(f"         {r.detail}")
    print("-" * 72)
    print(f"  probes:   {report.total}")
    print(f"  correct:  {report.correct}")
    print(f"  accuracy: {report.accuracy:.0%}")
    print(f"  failures: {list(report.failures)}")
    print(f"  RESULT:   {'PASS' if report.passed else 'FAIL'}")
    print()
    print(_HONESTY_NOTE)
    if mode == "live":
        print(
            "Live mode note: payloads came from the live APIs just\n"
            "now; scores can move with API state and new literature.\n"
            "The fixture run is the scored benchmark."
        )


if __name__ == "__main__":
    _demo()
