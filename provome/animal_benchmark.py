"""Animal division benchmark, v1: does the Animal Copilot behave?

This is the Animal counterpart of :mod:`provome.benchmark`.  That
harness measures the cite-or-decline gate on synthetic items; this
one measures the whole Animal surface — question in, verdict out —
on fixed probes whose ground truth was recorded from real sources
on 2026-11-25 (OMIA, Ensembl, Europe PMC; the source URL is kept
on every probe).

Each probe is a question with a known correct verdict:

* ``answer`` probes name a gene/species pair whose association is
  real (OMIA-curated in most cases) and whose recorded live
  payloads genuinely pass the gate — at least
  ``MIN_VERIFIED_ITEMS`` verified items from at least
  ``MIN_INDEPENDENT_SOURCES`` independent groups.  The Copilot
  must answer.
* ``decline`` probes must decline, for one of the recorded
  reasons: a fabricated gene symbol, a real gene in a species
  that does not have it, no species named at all, a condition
  topic (literature-only in v1, a single source group), or —
  the uncomfortable one — a real OMIA association the v1 sources
  cannot reach (feline PKD1: Ensembl's cat symbol index has no
  PKD1 record, so the reachable evidence is below the gate).
  A benchmark that only contains probes the system passes would
  be rigged; the misses are in here on purpose.

Fixture mode (the default, and what CI scores) replays payloads
recorded verbatim from the live APIs on 2026-11-25 through the
real provider and the real gate — deterministic, no network.
Live mode (``--live``) runs the same probes against the live
APIs; its score can move with API state and is not the scored
benchmark.

What a perfect score does NOT establish: coverage.  This is a
fixed, builder-chosen probe set — a smoke benchmark of verdict
behaviour, not external validation.  The Animal division stays
"in build and unbenchmarked" in every public wording until
external validation exists and the Founder/CEO make the call.

The probe set and fixtures are frozen at ``animal-benchmark-v1``;
changing any probe, payload, or expected verdict is a new
version, never an edit.

Run it:

    python -m provome.animal_benchmark          # fixture mode
    python -m provome.animal_benchmark --live   # live mode

Research software.  Veterinary research use only.
"""

from __future__ import annotations

import copy
import json
import sys
import urllib.parse
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from .animal import AnimalCopilot, AnimalEvidenceProvider
from .evidence import default_source_registry, evaluate_gate

__all__ = [
    "BENCHMARK_VERSION",
    "AnimalBenchmarkReport",
    "AnimalProbe",
    "ProbeResult",
    "fixture_transport",
    "probes",
    "run_benchmark",
]

BENCHMARK_VERSION = "animal-benchmark-v1"
FIXTURE_DATE = date(2026, 11, 25)


@dataclass(frozen=True)
class AnimalProbe:
    """One fixed Animal test: question in, expected verdict out.

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
    """The fixed probe set for animal-benchmark-v1. No network."""
    return (
        # --- must answer: real association, gate genuinely passes --
        AnimalProbe(
            "dog-abcb1-mdr1", "answer",
            "What is known about ABCB1 drug sensitivity in dogs?",
            True,
            "ABCB1 (MDR1) drug sensitivity in dogs is among the "
            "best-documented pharmacogenetic traits in veterinary "
            "medicine; Ensembl records the gene in Canis lupus "
            "familiaris (ENSCAFG00845007972) and current canine "
            "literature reports ABCB1 variants (PMID 42436707).",
            "https://www.ensembl.org/id/ENSCAFG00845007972",
        ),
        AnimalProbe(
            "horse-prkdc-scid", "answer",
            "What is known about PRKDC severe combined "
            "immunodeficiency in horses?",
            True,
            "OMIA 000220-9796: severe combined immunodeficiency "
            "disease, autosomal, PRKDC-related, in the horse; "
            "Ensembl records PRKDC in Equus caballus and carries "
            "the OMIA phenotype association for the gene.",
            "https://omia.org/OMIA000220/9796/",
        ),
        AnimalProbe(
            "cattle-itgb2-blad", "answer",
            "What is known about ITGB2 leukocyte adhesion "
            "deficiency in cattle?",
            True,
            "OMIA 000595-9913: leukocyte adhesion deficiency, "
            "type I (BLAD), ITGB2-related, in cattle; Ensembl "
            "records the gene and the OMIA association.",
            "https://omia.org/OMIA000595/9913/",
        ),
        AnimalProbe(
            "pig-ryr1-malignant-hyperthermia", "answer",
            "What is known about RYR1 malignant hyperthermia "
            "in pigs?",
            True,
            "OMIA 000621-9823: malignant hyperthermia (porcine "
            "stress syndrome), RYR1-related, in the pig; Ensembl "
            "records the gene and the OMIA association, and the "
            "porcine literature is on-topic (PMID 40035276).",
            "https://omia.org/OMIA000621/9823/",
        ),
        AnimalProbe(
            "sheep-prnp-scrapie", "answer",
            "What is known about PRNP scrapie susceptibility "
            "in sheep?",
            True,
            "OMIA 000944-9940: spongiform encephalopathy "
            "(scrapie) susceptibility/resistance, PRNP-related, "
            "in sheep; Ensembl records the gene and the OMIA "
            "association.",
            "https://omia.org/OMIA000944/9940/",
        ),
        AnimalProbe(
            "dog-sod1-degenerative-myelopathy", "answer",
            "What is known about SOD1 degenerative myelopathy "
            "in dogs?",
            True,
            "OMIA 000263-9615: degenerative myelopathy in the "
            "dog, SOD1-associated; Ensembl records the gene and "
            "current literature reports the canine SOD1:c.118G>A "
            "variant (PMID 42746632).",
            "https://omia.org/OMIA000263/9615/",
        ),
        AnimalProbe(
            "cat-mybpc3-hcm", "answer",
            "What is known about MYBPC3 hypertrophic "
            "cardiomyopathy in cats?",
            True,
            "OMIA 002951-9685: hypertrophic cardiomyopathy, "
            "MYBPC3-related, in the cat; Ensembl records the "
            "gene and two OMIA associations for it.",
            "https://omia.org/OMIA002951/9685/",
        ),
        AnimalProbe(
            "horse-scn4a-hypp", "answer",
            "What is known about SCN4A hyperkalemic periodic "
            "paralysis in horses?",
            True,
            "OMIA 000785-9796: hyperkalemic periodic paralysis "
            "(HYPP), SCN4A-related, in the horse; Ensembl "
            "records the gene and the OMIA association.",
            "https://omia.org/OMIA000785/9796/",
        ),
        AnimalProbe(
            "dog-rpe65-retinal", "answer",
            "What is known about RPE65 retinal degeneration "
            "in dogs?",
            True,
            "RPE65-associated inherited retinal degeneration in "
            "dogs is a long-standing canine model (gene-therapy "
            "literature, PMID 40461693); Ensembl records the "
            "gene in Canis lupus familiaris (ENSCAFG00845013646).",
            "https://www.ensembl.org/id/ENSCAFG00845013646",
        ),
        AnimalProbe(
            "horse-plod1-fragile-foal", "answer",
            "What is known about PLOD1 fragile foal syndrome "
            "in horses?",
            True,
            "OMIA 001982-9796: PLOD1-related Ehlers-Danlos "
            "(fragile foal syndrome) in the horse; Ensembl "
            "records the gene and the OMIA association, and "
            "the recent literature is on-topic (PMID 42951219).",
            "https://omia.org/OMIA001982/9796/",
        ),
        AnimalProbe(
            "chicken-pmel-dominant-white", "answer",
            "What is known about PMEL dominant white feather "
            "colour in chickens?",
            True,
            "OMIA 000373-9031: feather colour, dominant white, "
            "PMEL-related, in the chicken; Ensembl records the "
            "gene and the OMIA association.",
            "https://omia.org/OMIA000373/9031/",
        ),
        # --- must decline -------------------------------------------
        AnimalProbe(
            "cat-pkd1-below-gate", "decline",
            "What is known about PKD1 polycystic kidney "
            "disease in cats?",
            False,
            "The association is REAL (OMIA 000807-9685: "
            "polycystic kidney disease in the cat, PKD1-related) "
            "but Ensembl's felis_catus symbol index has no PKD1 "
            "record (live lookup returned HTTP 400 on "
            "2026-11-25), so the only reachable evidence is "
            "literature: 2 items, 1 source group, below the "
            "gate. A documented v1 coverage gap, kept in the "
            "set so the benchmark cannot hide it.",
            "https://omia.org/OMIA000807/9685/",
        ),
        AnimalProbe(
            "chicken-sry-wrong-species", "decline",
            "What is known about SRY in chickens?",
            False,
            "SRY is the mammalian Y-chromosome sex-determining "
            "gene; chickens have no SRY (sex is Z-dosage "
            "determined) and Ensembl's gallus_gallus symbol "
            "index has no SRY record (live lookup HTTP 400, "
            "2026-11-25). Full-text literature matches leak "
            "in, but at one source group the gate must decline.",
            "https://rest.ensembl.org/lookup/symbol/gallus_gallus/SRY",
        ),
        AnimalProbe(
            "dog-arhgap11b-human-specific", "decline",
            "What is known about ARHGAP11B in dogs?",
            False,
            "ARHGAP11B is a human-specific gene from a recent "
            "segmental duplication; there is no canine record "
            "in Ensembl (live lookup HTTP 400, 2026-11-25). "
            "Literature full-text matches alone (one group) "
            "must not pass the gate.",
            "https://rest.ensembl.org/lookup/symbol/canis_lupus_familiaris/ARHGAP11B",
        ),
        AnimalProbe(
            "cattle-zzzqx9-fabricated", "decline",
            "What is known about ZZZQX9 in cattle?",
            False,
            "ZZZQX9 is a fabricated symbol: no Ensembl record "
            "(HTTP 400) and zero Europe PMC results, both "
            "live-verified 2026-11-25. Any answer here would "
            "be invented.",
            "https://rest.ensembl.org/lookup/symbol/bos_taurus/ZZZQX9",
        ),
        AnimalProbe(
            "dog-fakgene1-fabricated", "decline",
            "What is known about FAKGENE1 in dogs?",
            False,
            "FAKGENE1 is a fabricated symbol: no Ensembl "
            "record (HTTP 400) and zero Europe PMC results, "
            "both live-verified 2026-11-25.",
            "https://rest.ensembl.org/lookup/symbol/canis_lupus_familiaris/FAKGENE1",
        ),
        AnimalProbe(
            "horse-notreal7-fabricated", "decline",
            "What is known about NOTREAL7 in horses?",
            False,
            "NOTREAL7 is a fabricated symbol: no Ensembl "
            "record (HTTP 400) and zero Europe PMC results, "
            "both live-verified 2026-11-25.",
            "https://rest.ensembl.org/lookup/symbol/equus_caballus/NOTREAL7",
        ),
        AnimalProbe(
            "dog-hip-dysplasia-condition", "decline",
            "What is known about hip dysplasia in dogs?",
            False,
            "Hip dysplasia in the dog is real (OMIA "
            "000473-9615), but v1 condition topics gather "
            "literature only — a single source group — so the "
            "gate declines by design until a second condition "
            "source exists.",
            "https://omia.org/OMIA000473/9615/",
        ),
        AnimalProbe(
            "sheep-scrapie-condition", "decline",
            "What is known about scrapie in sheep?",
            False,
            "Scrapie in sheep is a real prion disease (OMIA "
            "000944-9940 covers the spongiform encephalopathy "
            "entry), but as a condition topic it is "
            "literature-only in v1 and must decline; the gene "
            "form (the PRNP probe above) is the answerable "
            "route.",
            "https://omia.org/OMIA000944/9940/",
        ),
        AnimalProbe(
            "no-species-abcb1", "decline",
            "What is known about ABCB1?",
            False,
            "No species is named, so no animal topic may be "
            "recognised (the Step 3 intent rule): animal "
            "evidence must never be gathered for an unstated "
            "organism, however real the gene is.",
            "https://www.ensembl.org/id/ENSCAFG00845007972",
        ),
        AnimalProbe(
            "no-species-pkd1", "decline",
            "What is known about PKD1?",
            False,
            "No species is named, so no animal topic may be "
            "recognised. The cat association (OMIA 000807-9685) "
            "must not be assumed from a bare gene symbol.",
            "https://omia.org/OMIA000807/9685/",
        ),
    )


# ---------------------------------------------------------------------
# Recorded fixtures (live payloads, recorded 2026-11-25)
#
# "lookup":    "<ensembl slug>/<GENE>" -> gene record, or null where
#              the live symbol lookup returned HTTP 400 (no record).
# "phenotype": "<ensembl gene id>" -> association list as returned,
#              or null where the live call returned HTTP 400.
# "literature": "<subject>" -> Europe PMC result list (pmid / title /
#              pubYear per record) for the provider's exact query.
# ---------------------------------------------------------------------

_FIXTURE_JSON = r"""{"lookup":{"canis_lupus_familiaris/ABCB1":{"id":"ENSCAFG00845007972","display_name":"ABCB1","biotype":"protein_coding","assembly_name":"ROS_Cfam_1.0","description":"ATP binding cassette subfamily B member 1 [Source:NCBI gene (formerly Entrezgene);Acc:403879]"},"felis_catus/PKD1":null,"equus_caballus/PRKDC":{"id":"ENSECAG00000020168","display_name":"PRKDC","biotype":"protein_coding","assembly_name":"EquCab3.0","description":"protein kinase, DNA-activated, catalytic subunit [Source:VGNC Symbol;Acc:VGNC:21853]"},"bos_taurus/ITGB2":{"id":"ENSBTAG00000017060","display_name":"ITGB2","biotype":"protein_coding","assembly_name":"ARS-UCD2.0","description":"integrin subunit beta 2 [Source:VGNC Symbol;Acc:VGNC:30327]"},"sus_scrofa/RYR1":{"id":"ENSSSCG00000058790","display_name":"RYR1","biotype":"protein_coding","assembly_name":"Sscrofa11.1","description":"ryanodine receptor 1 [Source:VGNC Symbol;Acc:VGNC:111766]"},"ovis_aries/PRNP":{"id":"ENSOARG00020078629","display_name":"PRNP","biotype":"protein_coding","assembly_name":"ARS-UI_Ramb_v3.0","description":"prion protein (Kanno blood group) [Source:NCBI gene (formerly Entrezgene);Acc:493887]"},"canis_lupus_familiaris/SOD1":{"id":"ENSCAFG00845022725","display_name":"SOD1","biotype":"protein_coding","assembly_name":"ROS_Cfam_1.0","description":"superoxide dismutase 1 [Source:NCBI gene (formerly Entrezgene);Acc:403559]"},"felis_catus/MYBPC3":{"id":"ENSFCTG00005018865","display_name":"MYBPC3","biotype":"protein_coding","assembly_name":"F.catus_Fca126_mat1.0","description":"myosin binding protein C3 [Source:HGNC Symbol;Acc:HGNC:7551]"},"equus_caballus/SCN4A":{"id":"ENSECAG00000005968","display_name":"SCN4A","biotype":"protein_coding","assembly_name":"EquCab3.0","description":"sodium voltage-gated channel alpha subunit 4 [Source:VGNC Symbol;Acc:VGNC:22738]"},"canis_lupus_familiaris/RPE65":{"id":"ENSCAFG00845013646","display_name":"RPE65","biotype":"protein_coding","assembly_name":"ROS_Cfam_1.0","description":"retinoid isomerohydrolase RPE65 [Source:NCBI gene (formerly Entrezgene);Acc:403803]"},"equus_caballus/PLOD1":{"id":"ENSECAG00000022842","display_name":"PLOD1","biotype":"protein_coding","assembly_name":"EquCab3.0","description":"procollagen-lysine,2-oxoglutarate 5-dioxygenase 1 [Source:VGNC Symbol;Acc:VGNC:21585]"},"gallus_gallus/PMEL":{"id":"ENSGALG00010024745","display_name":"PMEL","biotype":"protein_coding","assembly_name":"bGalGal1.mat.broiler.GRCg7b","description":"premelanosome protein [Source:NCBI gene (formerly Entrezgene);Acc:396007]"},"gallus_gallus/SRY":null,"canis_lupus_familiaris/ARHGAP11B":null,"bos_taurus/ZZZQX9":null,"canis_lupus_familiaris/FAKGENE1":null,"equus_caballus/NOTREAL7":null},"phenotype":{"ENSCAFG00845007972":null,"ENSECAG00000020168":[{"source":"OMIA","location":"9:36236425-36438594","description":"Severe combined immunodeficiency disease, autosomal, PRKDC-related","attributes":{"external_id":"OMIA000220"},"Gene":"ENSECAG00000020168"}],"ENSBTAG00000017060":[{"description":"Resistance/susceptibility to Mannheimia haemolytica leukotoxin","Gene":"ENSBTAG00000017060","attributes":{"external_id":"OMIA002073"},"location":"1:144759562-144788699","source":"OMIA"},{"description":"Leukocyte adhesion deficiency, type I","Gene":"ENSBTAG00000017060","attributes":{"external_id":"OMIA000595"},"location":"1:144759562-144788699","source":"OMIA"}],"ENSSSCG00000058790":[{"description":"Malignant hyperthermia","Gene":"ENSSSCG00000058790","attributes":{"external_id":"OMIA000621"},"source":"OMIA","location":"6:47343768-47458458"}],"ENSOARG00020078629":[{"Gene":"ENSOARG00020078629","attributes":{"external_id":"OMIA000944"},"description":"Spongiform encephalopathy, susceptibility/resistance to","location":"13:46638185-46660977","source":"OMIA"}],"ENSCAFG00845022725":null,"ENSFCTG00005018865":[{"attributes":{"external_id":"OMIA002951"},"Gene":"ENSFCTG00005018865","description":"Cardiomyopathy, hypertrophic, MYBPC3-related, autosomal dominant","location":"D1:99239685-99281527","source":"OMIA"},{"source":"OMIA","location":"D1:99239685-99281527","description":"Cardiomyopathy, hypertrophic, MYBPC3-related, autosomal recessive","attributes":{"external_id":"OMIA002952"},"Gene":"ENSFCTG00005018865"}],"ENSECAG00000005968":[{"source":"OMIA","Gene":"ENSECAG00000005968","location":"11:15432921-15478423","description":"Hyperkalemic Periodic Paralysis, HYPP","attributes":{"external_id":"OMIA000785"}}],"ENSCAFG00845013646":null,"ENSECAG00000022842":[{"location":"2:39917275-39955111","Gene":"ENSECAG00000022842","source":"OMIA","description":"kyphoscoliotic Ehlers-Danlos syndrome kEDS, PLOD1-related","attributes":{"external_id":"OMIA001982"}}],"ENSGALG00010024745":[{"description":"Feather colour, dominant white","Gene":"ENSGALG00010024745","attributes":{"external_id":"OMIA000373"},"location":"34:1339431-1345613","source":"OMIA"}]},"literature":{"ABCB1":[{"pmid":"42770779","title":"ABCB1 Frameshift Deletion in Skye Terriers With Potential Pharmacogenetic Relevance.","pubYear":"2026"},{"pmid":"42436707","title":"Association Between the nt230(del4) Mutation and c.-6-180T&gt;G Polymorphism in the Canine ABCB1 Gene.","pubYear":"2026"},{"pmid":"42038050","title":"Cloning, structural and functional characterization of the ABCB1 transporter of the Eurasian bullfinch (<i>Pyrrhula pyrrhula</i>).","pubYear":"2026"}],"PKD1":[{"pmid":"41669239","title":"Case Report: Budd-Chiari-like syndrome in a cat with polycystic kidney and liver disease.","pubYear":"2025"},{"pmid":"42657244","title":"An integrin beta-1-anchored bicaudal C1-polycystin-1 module essential for tubular morphogenesis in polycystic kidney disease.","pubYear":"2026"},{"pmid":"42515596","title":"Evolutionary Analysis Reveals a Single Amino Acid in the AAV Entry Receptor (AAVR) of Cats That Disrupts Binding of a Major Phylogenetic Group of AAVs.","pubYear":"2026"}],"PRKDC":[{"pmid":"41765359","title":"DNA-PKcs promotes therapy resistance and metastatic recurrence in neuroblastoma.","pubYear":"2026"},{"pmid":"42010471","title":"Genomic diversity and structure in arabian horses revealed by whole-genome sequencing: establishment of an allele frequency database of common genetic variation.","pubYear":"2026"},{"pmid":"42602493","title":"LMO2 regulates STAT3 localization to the nucleus in breast cancer cells.","pubYear":"2026"}],"ITGB2":[{"pmid":"42939423","title":"Whole-Genome Resequencing Reveals Genomic Signatures Underlying High-Altitude Adaptation in Ethiopian Indigenous Cattle.","pubYear":"2026"},{"pmid":"40379730","title":"Genome wide locus-specific ancestry analysis revealed adaptive admixtures in crossbred cattle of India.","pubYear":"2025"},{"pmid":"42613629","title":"Transcriptomic analysis of naturally occurring bovine respiratory disease in Australian feedlot cattle identifies bovine viral diarrhoea virus-induced immunosuppression and Mollicutes as the dominant drivers of lung inflammation.","pubYear":"2026"}],"RYR1":[{"pmid":"40035276","title":"The Incidence of Volatile Anesthesia Porcine Stress Syndrome in Pigs (Sus scrofa domesticus) Gives Implications for Physiology during Anesthesia.","pubYear":"2025"},{"pmid":"33532660","title":"Wild pigs (<i>Sus scrofa</i>) population as reservoirs for deleterious mutations in the RYR1 gene associated with Porcine Stress Syndrome.","pubYear":"2021"},{"pmid":"42155544","title":"Very fast chilling regulates Ca&lt;sup&gt;2+&lt;/sup&gt; release by maintaining high ryanodine receptor 1 expression to promote meat tenderization at early postmortem.","pubYear":"2026"}],"PRNP":[{"pmid":"42676933","title":"From markers to mechanisms: a comprehensive review of major genes and quantitative trait loci shaping the modern sheep (<i>Ovis aries</i>).","pubYear":"2026"},{"pmid":"42840686","title":"Whole-genome resequencing of the Kazakh coarse-wool fat-tailed sheep identifies functional variants in disease-associated genes.","pubYear":"2026"},{"pmid":"42974130","title":"First report of polymorphisms and genetic characteristics of the <i>prion-like</i> protein gene (<i>PRND</i>) in bats.","pubYear":"2026"}],"SOD1":[{"pmid":"35549755","title":"A Mass Spectrometric Approach to the Proteomic Profiling of the <i>Canis lupus familiaris</i> Acquired Enamel Pellicle on Hydroxyapatite Discs.","pubYear":"2022"},{"pmid":"42746632","title":"SOD1:c.118G&gt;A mutation in the Cimarrón Uruguayo dog: prevalence, drift dynamics, and structural effects.","pubYear":"2026"},{"pmid":"42083737","title":"Genotype and Allele Frequencies of Canine Degenerative Myelopathy-Associated SOD1 Gene Variant in Irish Wolfhound Dogs.","pubYear":"2026"}],"MYBPC3":[{"pmid":null,"title":"The Maine Coon Cat Harboring the  <i>MYBPC3</i>  -A31P Mutation: A Genotype-Stratified Phenotypic Characterization of Hypertrophic Cardiomyopathy","pubYear":"2026"},{"pmid":"42501791","title":"Development and validation of PCR-RFLP assays for detection of MYBPC3 variants associated with hypertrophic cardiomyopathy in Maine Coon and Ragdoll cats.","pubYear":"2026"},{"pmid":"41946259","title":"Generation of two induced pluripotent stem cell lines from hypertrophic cardiomyopathy patients carrying MYBPC3 mutations.","pubYear":"2026"}],"SCN4A":[{"pmid":"42672330","title":"Correction: Evolution, hibernation, and inactivation of voltage-gated Na channels.","pubYear":"2026"},{"pmid":"42328694","title":"Integrative Advances in Equine Genomics From Reference Assemblies to Evolutionary History and Key Traits.","pubYear":"2026"},{"pmid":"41610137","title":"Aberrant skeletal muscle morphogenesis and myofiber differentiation characterize equine myotonic dystrophy.","pubYear":"2026"}],"RPE65":[{"pmid":"42959692","title":"Against premature progress: what is genuine progress in ophthalmology.","pubYear":"2026"},{"pmid":"40461693","title":"Gene therapy advances using canine and feline animal models of inherited retinal degeneration.","pubYear":"2025"},{"pmid":"42346603","title":"Precision Medicine in Inherited Retinal Disease: Advances, Challenges, and Future Directions.","pubYear":"2026"}],"PLOD1":[{"pmid":"42951219","title":"A subclinical biomechanical phenotype detected by AI-assisted video gait analysis in horses heterozygous for the Fragile Foal Syndrome Allele in PLOD1.","pubYear":"2026"},{"pmid":"41497457","title":"Surveillance of a <i>PLOD</i> gene variant linked to fragile foal syndrome in Silesian horses in Poland: implications for genetic monitoring and breeding strategies.","pubYear":"2025"},{"pmid":"41724255","title":"Fragile foal syndrome: manifestations, heterozygous advantage and the future of breeding policies.","pubYear":"2026"}],"PMEL":[{"pmid":"41265139","title":"PMEL governs autosomal dominant inheritance of white-tail independent of yellow body plumage in chickens (Gallus gallus domesticus).","pubYear":"2026"},{"pmid":"42374170","title":"Genomic and transcriptomic analysis suggests potential regulatory relationships among MC1R, TUBB3, and PMEL associated with black-flecked plumage in chickens.","pubYear":"2026"},{"pmid":"42341979","title":"Cross-cohort transcriptomic meta-analysis identifies conserved melanogenesis signatures and developmental background in black-boned chicken breast muscle.","pubYear":"2026"}],"SRY":[{"pmid":"27856505","title":"Transcriptomic changes throughout post-hatch development in Gallus gallus pituitary.","pubYear":"2017"},{"pmid":"29614966","title":"Molecular signatures of epithelial oviduct cells of a laying hen (Gallus gallus domesticus) and quail (Coturnix japonica).","pubYear":"2018"},{"pmid":"30241500","title":"Transcriptional profiling of liver during the critical embryo-to-hatchling transition period in the chicken (Gallus gallus).","pubYear":"2018"}],"ARHGAP11B":[{"pmid":"42113807","title":"Comparative genomics of human brain and immune gene preservation across species.","pubYear":"2026"},{"pmid":"34215180","title":"Modelling segmental duplications in the human genome.","pubYear":"2021"},{"pmid":"42736356","title":"Transcriptome of fetal cortex of tree shrew underlying the emergence of outer subventricular zone.","pubYear":"2026"}],"ZZZQX9":[],"FAKGENE1":[],"NOTREAL7":[],"hip dysplasia":[{"pmid":"33726677","title":"Chromosome-length genome assembly and structural variations of the primal Basenji dog (Canis lupus familiaris) genome.","pubYear":"2021"},{"pmid":"41277709","title":"Femoral neck anteversion among five canid taxa.","pubYear":"2026"},{"pmid":"32236524","title":"Canfam_GSD: De novo chromosome-length genome assembly of the German Shepherd Dog (Canis lupus familiaris) using a combination of long reads, optical mapping, and Hi-C.","pubYear":"2020"}],"scrapie":[{"pmid":"42676933","title":"From markers to mechanisms: a comprehensive review of major genes and quantitative trait loci shaping the modern sheep (<i>Ovis aries</i>).","pubYear":"2026"},{"pmid":"42840686","title":"Whole-genome resequencing of the Kazakh coarse-wool fat-tailed sheep identifies functional variants in disease-associated genes.","pubYear":"2026"},{"pmid":"30261281","title":"In silico investigation of the prion protein glycosylation profiles in relation to scrapie disease resistance in domestic sheep (Ovis aries).","pubYear":"2018"}]}}"""

_FIXTURES = json.loads(_FIXTURE_JSON)


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
    if "europepmc" in url:
        subject = urllib.parse.unquote(url).split('"')[1]
        results = _FIXTURES["literature"][subject]
        return {"resultList": {"result": copy.deepcopy(results)}}
    raise AssertionError(url)


def _fixture_factory() -> AnimalEvidenceProvider:
    return AnimalEvidenceProvider(
        transport=fixture_transport, today=lambda: FIXTURE_DATE
    )


def _live_factory() -> AnimalEvidenceProvider:
    return AnimalEvidenceProvider()


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
class AnimalBenchmarkReport:
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
    provider_factory: Optional[Callable[[], AnimalEvidenceProvider]] = None,
) -> AnimalBenchmarkReport:
    """Run every probe through the real AnimalCopilot and score it.

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
        copilot = AnimalCopilot(provider=provider)
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
    return AnimalBenchmarkReport(
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
    "(OMIA, Ensembl, Europe PMC) on 2026-11-25, the Animal\n"
    "Copilot's verdicts match the recorded truth, with the gate\n"
    "arithmetic independently re-checked per probe.\n"
    "What it does NOT establish: coverage of the Animal division.\n"
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
    print(f"Provome Animal division benchmark ({report.version}, {mode} mode)")
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
