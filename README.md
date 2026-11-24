# Omevra

An evidence engine for biology, and the **Omevra Copilot**: one copilot
that answers questions about variants, genes, and diseases only when
the evidence behind an answer is strong enough to cite - and declines,
with reasons, when it is not.

Research software. Not for clinical use.

## Three divisions, one engine

Omevra is not human-only and not cancer-only. The same engine serves
three divisions:

- **Human** - all human disease: rare disease, cancer, cardiac,
  neurological, metabolic, immunological, infectious disease host
  genetics, and more.
- **Animal** - veterinary and livestock: companion animals, cattle,
  poultry, pigs, horses, sheep and goats, and aquaculture.
- **Plant** - crops and agriculture: staple crops such as rice, maize,
  and wheat, model plants, plant pathology, and breeding.

One engine, honestly shared: molecular consequence prediction,
population frequency mathematics, calibration metrics, phenotype
semantic similarity, and the cite-or-decline evidence gate are
species-agnostic, and the division registry in `omevra/domains.py`
says exactly that, in data. It also says what does not transfer:
ACMG/AMP classification is human-clinical specific and is never
applied to animal or plant questions.

Honest status: **Human is the live, benchmarked division today.**
Animal and Plant are the expansion divisions, built on the same
engine. This package does not claim benchmarked coverage for them.

## The cite-or-decline principle

Every answer the Copilot gives is built from evidence items. An
answer is issued only when at least **3 verified evidence items from
at least 2 independent sources** support it. Otherwise the output is
a structured decline naming the exact shortfall - too few items, too
few independent sources, no recognised topic - instead of a vaguer,
weaker answer. Claims are constructed from evidence items and carry
their citations by construction; a claim without a citation cannot
be built.

## Architecture

```
omevra/
    copilot.py      The Omevra Copilot: question in, gated answer or
                    structured decline out. Pluggable evidence
                    providers; optional LLM rephrasing hook is an
                    interface only and off by default.
    evidence.py     EvidenceItem model, source registry with
                    independence groups, verification status,
                    deduplication, and the gate itself.
    domains.py      Human / Animal / Plant division registry:
                    scope, problem areas, example species, and which
                    engines apply to which division.
    variant.py      VCF record parsing, HGVS c. and p. parsing and
                    validation, chromosome and allele models.
    consequence.py  The full standard codon table, translation, and
                    consequence calling (synonymous, missense,
                    nonsense, frameshift, start-loss, stop-loss,
                    in-frame indels, splice-region) from real
                    sequence.
    acmg.py         ACMG/AMP classification two ways: the Richards
                    et al. 2015 combining rules and the Tavtigian
                    et al. 2018 Bayesian points system.
    population.py   Maximum credible allele frequency (Whiffin et
                    al. 2017) and BA1 / BS1 / PM2-style evaluation.
    calibration.py  AUROC (rank method), Brier score, log loss, ECE
                    and full calibration tables, from scratch.
    phenotype.py    Ontology information content, most informative
                    common ancestor, Resnik similarity, and
                    best-match average over term sets.
    data/           Bundled illustrative worked examples and the
                    offline demo evidence set.
tests/              A pytest suite covering every module with
                    biologically true assertions.
```

The core package uses the **Python standard library only**
(Python 3.10 or later). There are no runtime dependencies to install,
pin, or audit.

## Quick start

Run the Copilot demo (offline, no network, bundled example data):

```bash
python -m omevra.copilot
```

It prints one ANSWER case (CFTR p.Phe508del) and one DECLINE case
(BRCA1 c.68_69delAG, where the bundled demo evidence is deliberately
below the gate), and shows division context being carried through.

Use it as a library:

```python
from omevra import Copilot, CopilotQuestion

copilot = Copilot()
answer = copilot.ask("What is CFTR p.Phe508del and does it cause cystic fibrosis?")
print(answer.verdict)        # Verdict.ANSWER
for claim in answer.claims:
    print(claim.statement, claim.citations)

# Division context (defaults to Human):
from omevra.domains import Division
q = CopilotQuestion(text="...", division=Division.PLANT)
q2 = CopilotQuestion(text="...", species="Bos taurus")  # Animal, by species
```

Or install the console entry point:

```bash
pip install .
omevra-copilot
```

## Running the tests

```bash
python -m venv .venv
.venv/bin/pip install pytest
.venv/bin/python -m pytest
```

The suite covers, among other things: codon translation of known
sequences; Tavtigian point totals for known criterion combinations;
Whiffin formula arithmetic; AUROC of 1.0 on perfectly separated data
and a hand-computed ECE example; and the Copilot answering on demo
evidence, declining below the gate, and carrying division context.

## Benchmark numbers

The only platform numbers stated for Omevra are for the variant
evidence agent benchmark: **AUROC 0.9705** and **ECE 0.0248**. No
other metrics are claimed for this package, and the demo data in
`omevra/data/` is illustrative example data, not benchmark data.

## Honest limits

- **Research software, not for clinical use.** Nothing in this
  package is a clinical classification, a diagnosis, or medical
  advice, for humans, animals, or plants.
- **The demo provider is illustrative.** `OfflineDemoProvider` reads
  a small bundled example set so the Copilot can be run and tested
  offline. It is not live database access, and this package ships no
  live access to ClinVar, gnomAD, OMIM, or any other database; the
  source registry describes provenance, it does not fetch.
- **ACMG scoring is not ACMG judgement.** `omevra/acmg.py` scores
  criteria a curator supplies, under both published frameworks. It
  does not decide whether a criterion applies to a real variant, and
  real classifications require current ClinGen specifications and
  expert review.
- **Consequence calling is single-transcript and CDS-only.** It does
  not model splicing outcomes, nonsense mediated decay, or
  alternative transcripts.
- **The bundled phenotype ontology is an example subset** with
  illustrative annotation counts, labelled as such in the code. It
  is not the Human Phenotype Ontology.
- **The population module implements the Whiffin framework
  transparently**, with every input explicit. Production frequency
  filtering should cross-check the published allele frequency
  application and current gene specific guidance.
- **Animal and Plant divisions are expansion work.** The
  species-agnostic engines apply today; division specific ontologies,
  references, and classification frameworks are future work, and are
  not claimed here as done.

## References

- Richards S, et al. Standards and guidelines for the interpretation
  of sequence variants (ACMG/AMP). Genetics in Medicine 2015;
  17:405-424.
- Tavtigian SV, et al. Modeling the ACMG/AMP variant classification
  guidelines as a Bayesian classification framework. Genetics in
  Medicine 2018; 20:1054-1060.
- Whiffin N, et al. Using high-resolution variant frequencies to
  empower clinical genome interpretation. Genetics in Medicine 2018;
  20:171-180.
- den Dunnen JT, et al. HGVS recommendations for the description of
  sequence variants: 2016 update. Human Mutation 2016; 37:564-569.
- Danecek P, et al. The variant call format and VCFtools.
  Bioinformatics 2011; 27:2156-2158.
- Resnik P. Using information content to evaluate semantic
  similarity in a taxonomy. IJCAI 1995.
- Kohler S, et al. The Human Phenotype Ontology in 2021. Nucleic
  Acids Research 2021; 49:D1207-D1217.
