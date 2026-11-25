# Changelog

All notable changes to Provome are recorded here. Versions follow
semantic versioning.

## [0.17.0] - 2026-11-25

Three-doors packaging v1 (roadmap Item 31 in installable
form). `provome/doors.py`: thin wiring only - a test greps
the module to prove no gate logic lives there. Three console
scripts, `provome-human`, `provome-animal`, `provome-plant`,
each wiring only its own division's copilot, boundary, and
three benchmark-verified examples; every output framed as
its door, and cross-division questions decline through the
door layer in both directions (suite-tested, clean-install
verified). Honest limit, stated in the release: the doors
ship in one wheel by design (one engine); the per-door
extras are name markers today, not separate distributions,
and packaging a door is not a coverage claim - Animal and
Plant boundaries still say coverage is NOT claimed.
Suite 361.

## [0.16.0] - 2026-11-25

Plant benchmark v1 + two provider fixes it forced.
`provome/plant_benchmark.py`: 21 frozen probes (8 ANSWER,
13 DECLINE), same audited harness as Animal; fixture score
21/21 - a smoke benchmark, NOT external validation; coverage
wording unchanged. The set keeps its gaps visible on purpose:
rice SD1 and maize TB1 (real genes the Ensembl Plants symbol
index cannot resolve), Arabidopsis FRI (record exists,
literature window too thin), trait topics (decline by
design). FIXES in `provome/plant.py`: (1) the gene-symbol
pattern was all-uppercase-only, making real symbols like
Rht-B1, GmFT2a and tb1 unaskable - widened with a gene-shape
predicate, regression-tested both ways (live check: Rht-B1
now parses and reaches the gate, declining honestly on the
index limit); (2) the phenotype parser read the same nested
payload shape that broke Animal - corrected defensively to
the real top-level shape (Ensembl Plants serves no
associations today, so the path is unproven live for Plant
and contributes zero items either way). Every Plant answer
still sits exactly at the gate minimum - zero margin is the
division's known weakness, recorded, not hidden. Suite 339.

## [0.15.0] - 2026-11-25

Animal benchmark v1 + a production bug it caught and we
fixed the same day. `provome/animal_benchmark.py`: 21 frozen
probes with recorded ground truth (11 ANSWER cases from real
OMIA associations, 10 DECLINE traps including fabricated
genes, wrong-species genes, and one real association v1
honestly cannot reach); the harness re-audits the gate
arithmetic per probe, and a rigged always-answer provider
fails it. Fixture score 21/21 - a smoke benchmark, NOT
external validation; coverage wording unchanged. THE BUG:
Ensembl phenotype items never fired in production - the
parser read a nested payload shape that existed only in an
invented test fixture, while the real API serves the
description at the entry's top level with a plain-string
source. Every real OMIA phenotype association was silently
dropped. Fixed in `provome/animal.py`, the fixture corrected
to the recorded real shape, a verbatim-payload regression
test added, and the fix live-verified (equine PRKDC now
returns its SCID phenotype association). Suite 328.

## [0.14.0] - 2026-11-25

Plant division v1 + the flagship's gnomAD fix (Step 3b).
`provome/plant.py`: Plant provider (Ensembl Plants + species-
restricted literature) mirroring Animal, division-scoped to
Plant, agricultural RUO boundary; live-verified ANSWER on
Arabidopsis FLC and an honest DECLINE on rice SD1 (no symbol
record exists - the gate does not pad). Step 3b: the live Human
provider now resolves ref/alt alleles through dbSNP (ClinVar
xref -> NCBI refsnp GRCh38 SPDI -> left-normalisation against
the reference), so the gnomAD item fires generally - live-
verified CFTR p.Phe508del (AF 0.00788) and HBB p.Glu6Val
(AF 0.01272); ambiguous multi-allelic cases still contribute
nothing rather than guess. Also fixed: a nondeterministic
ClinVar notation search, and per-source failure isolation in
the live provider (one flaky API costs only its own items,
never the whole answer - regression-tested). Plant and Animal
remain in build and unbenchmarked; coverage is not claimed.
Suite 319. (Released as 0.14.0: the 0.9.0-0.13.0 numbers were
taken by the parallel voice-session release line.)

## [0.13.0] - 2026-11-25

Gate benchmark, internal v1 (priority item #1 under the
Founder's one-by-one order): `provome/benchmark.py` measures the
cite-or-decline gate's behaviour on 10 fixed probes - must-answer,
must-decline (too few verified, one source group, unverified
only, wrong topic, refuting only, empty), and must-disclose
(passes but surfaces verified refutation instead of hiding it).
Result at ship: 10/10 correct, zero disclosure failures. Fixture
probes test gate behaviour, not biomedical facts; the public
benchmark replays this harness on public data after Quality
Division and Founder sign-off. Suite 295.

## [0.12.1] - 2026-11-25

Fix: the Quality Division honesty test constructed WorkTask
without its required fields; corrected. No behaviour change.

## [0.12.0] - 2026-11-25

Quality Division built in code (Founder's direction: quality
covers the ENTIRE company, every department). `provome/quality.py`
registers 2,000 independent checkers across all 18 departments -
orchestra, research division, the three doors, drug discovery,
bridge, and all 11 disease corps - each with a distinct
(department, method, beat) assignment; every department is
covered by all 12 checking methods, so any important claim can be
verified by two different methods (no-identical-checks law). One
Chief Quality Copilot, a senior QA lead per department, and the
stop-the-line veto structure. A checker with no live dispatched
check passes nothing. Suite 293.

## [0.11.0] - 2026-11-25

Diabetes stands as its own corps at 20,000 researchers
(Founder's order). It began as focus areas inside the Metabolic
corps; it now has its own registry, focuses (type 1, type 2,
gestational, insulin resistance, complications, prevention,
paediatric, monitoring technology) and 200 senior leads.
Disease corps total: 70,000 across 11 corps. Suite 289.

## [0.10.0] - 2026-11-25

Disease corps (Founder's direction): `provome/diseases.py` gives
every major disease area its own corps of 5,000 researchers with
distinct (focus, function, beat) assignments. First wave deployed
and verified: 10 corps, 50,000 researchers - Human: oncology,
cardiovascular, infectious diseases (COVID-19 a named focus),
neurological, rare diseases, metabolic; Animal: infectious,
genetic; Plant: pathology, crop resistance. Cancer/oncology/
tumour form ONE Oncology corps (same field; the company's
no-duplicate law). Corps stand up per disease through the stage
gates via the generic builder. Suite 289.

## [0.9.0] - 2026-11-25

Research Division deployed (Founder's direction): `provome/research.py`
registers 5,000 scientific researcher units in the orchestra -
Human 2,000, Animal 1,000, Plant 1,000, Drug Discovery 600,
cross-division Bridge 400 - each with a unique (specialty, beat)
assignment across 50 specialties, one senior research lead per
hundred. Units are honest by construction: asked to run without a
live dispatched assignment, a unit reports exactly that and
produces no claims. Registration is the organisation standing up,
not a claim of completed research. Suite 286.

## [0.8.0] - 2026-11-25

LLM track Stage A + Animal division v1 + a division-scoping fix.
`provome/rag.py`: retrieval index with a pluggable embedder
protocol (stdlib TF-IDF baseline built in; neural embedders plug
in without caller changes) and an evidence bridge whose rule is
that similarity ranks but never verifies - only documents with a
real PMID become verified evidence. `provome/animal.py`: Animal
division provider (Ensembl REST + species-restricted Europe PMC
literature) under the unchanged cite-or-decline gate, veterinary
RUO boundary on every output; live-verified on canine ABCB1
(ANSWER, 3 items / 2 sources). Division scoping fix (CEO):
providers now declare `scoped_divisions`; the Copilot declines
out-of-scope divisions per provider instead of either leaking
human evidence into Animal answers or blocking genuinely
animal-scoped evidence. Animal remains in build and
unbenchmarked - coverage is not claimed. Suite 281.

## [0.7.0] - 2026-11-25

Step 6: freemium tiers as real product structure. Built by two
specialist workers, CEO-verified. `provome/tiers.py`: Free $0 /
Researcher $79 / Lab $499 entitlements (daily live questions,
batch size, audit-trail access, seats, priority support) with
denials that name the limit and the unlocking tier.
`provome/usage.py`: per-account daily metering with JSON
persistence and honest quota decisions (limit reached -> message
names the next-day reset; nothing silently dropped). Checkout
and billing remain Step 8 and are not claimed. 17 new tests;
suite 260.

## [0.6.1] - 2026-11-25

Copilot coverage fact sheet updated for Step 5: Drug Discovery
target evidence is live (computational only); Animal and Plant
remain honestly not covered.

## [0.6.0] - 2026-11-25

Step 5: the Drug Discovery division, v1 (`provome/drug.py`).
Target evidence under the same cite-or-decline gate: live ChEMBL
targets/mechanisms (joined to genes via reviewed UniProt
accessions), approved and phase 2-3 compounds, and title-matched
Europe PMC literature. Every answer prints the computational-only
boundary. Verified live: CFTR (ivacaftor, lumacaftor, tezacaftor)
and BCL2 (venetoclax) ANSWER; unknown gene DECLINEs. ChEMBL added
to the source registry. 4 new tests; suite 243.

## [0.5.0] - 2026-11-25

The rename release: Omevra becomes **Provome** everywhere —
package (`provome`), repository, documentation, and website —
executing the Founder's 2026-11-25 decision (name-risk finding:
Omivera biotech + OMEVRA-DM pharma brand; see the roadmap's
NAME_OPTIONS_v1.md). All 239 tests pass under the new name.

### Added
- The Provome website (this repo's index.html): how it works, a
  real live-run proof section, honestly-labelled divisions, the
  freemium pricing shape, Research Use Only footer.

## [0.4.1] - 2026-11-25

Full audit of Steps 1-4 (three senior audit workers + CEO review).
Fixes only; no new features.

- `__init__.__version__` had remained "0.2.0" through three
  releases; synced to the real version, with a new
  version-consistency test so the drift fails loudly next time.
- README "Honest limits" still claimed the package ships no live
  database access - stale since v0.3.0. Rewritten to describe both
  providers accurately; architecture list now includes live.py and
  converse.py; worker-scale note corrected (10,000-worker stress
  test, not 1,000).
- Guide SIMPLE depth now prints the key source it refers to.

## [0.4.0] - 2026-11-25

Step 4: the human Copilot (Items 21 + 23). `provome.converse.Guide`
wraps the gated Copilot in a conversational layer at three depths
(simple / standard / expert). It adds no claims: every factual
sentence comes from a finished gated answer; declines stay declines
and are explained with the exact missing piece and the question
shape that unblocks them. Tool questions ("What is Provome?", "How
do you work?", pricing, coverage, sources) are answered from a
static product fact sheet with honest coverage limits. Every guided
answer carries run transparency (provider, gate arithmetic,
confidence-is-a-heuristic note) and limits. 7 new tests; suite 237.

## [0.3.0] - 2026-11-25

Step 3: the live brain. The Copilot gains its own browser
(`provome.live.LiveEvidenceProvider`, Item 22): at question time it
fetches real records from ClinVar (NCBI E-utilities), gnomAD (Broad
GraphQL, when the ClinVar record supplies usable coordinates), and
Europe PMC literature, and answers only from what actually came
back — every citation carries its record id and fetch date.

### Added
- Intent validation: a topic is recognised only when the question
  names a gene AND an HGVS-like variant; everything else declines
  (closes the Step 1 Worker A residual finding for the live path).
- ClinVar record verification: loose search hits are filtered to
  the exact variant (title / protein-change match), multi-variant
  haplotype records are rejected, and legacy protein numbering is
  reconciled (literature HBB p.Glu6Val = ClinVar p.Glu7Val), with
  both three-letter and one-letter spellings searched.
- Live demos verified end-to-end: CFTR p.Phe508del ANSWER from the
  real ClinVar record (Pathogenic, Variation ID 4072070) plus live
  literature; HBB p.Glu6Val ANSWER (Variation ID 15333); weather
  question DECLINE.
- 6 fixture-based tests (no network in CI). Suite: 230 passed.

### Known limitation
- ClinVar's summary feed does not expose ref/alt alleles, so the
  gnomAD item fires only when coordinates are otherwise available;
  an allele-resolution source (dbSNP cross-reference) is Step 3b.

## [0.2.1] - 2026-11-25

Step 1 hardening round. Three senior test workers ran adversarial,
fuzz, and stress suites against v0.2.0 (61 gate cases, 18,706 fuzz
calls, 17 orchestra stress/scale checks incl. a 10,000-worker
registration test); the CEO reviewed every finding, fixed what was
real, and re-ran all suites.

### Fixed
- Gate integrity: unregistered (fabricated) sources no longer count
  toward the cite-or-decline gate; only registry sources can vouch.
- Cross-division leakage: human clinical evidence is never issued
  as an answer in the Animal or Plant divisions; the Copilot
  declines there until division-scoped evidence is benchmarked.
- Evidence counting: one citation restated in other words is one
  evidence line (citation-identity deduplication), and verified
  refuting evidence that matches or outweighs supporting evidence
  now blocks an answer instead of only warning.
- Topic matching uses word boundaries (an alias inside a larger
  token no longer matches).
- Validation: Claim rejects empty citation strings, EvidenceItem
  rejects whitespace ids, Copilot.ask raises a clean TypeError on
  non-string input.
- Wrong-type arguments across variant/consequence/population/
  calibration now raise the documented ValueError/VariantParseError
  instead of leaking AttributeError/TypeError (12 fuzz patterns);
  calibration_table caps n_bins at 10,000 (resource guard).

### Added
- tests/test_hardening_step1.py: 12 regression tests. Suite is now
  224 tests. Worker suites after fixes: gate 58/61 (remaining 3 are
  the documented demo-provider intent limitation, not carried into
  the live provider), fuzz 0 failures, orchestra stress 17/17.

### Known, scheduled
- A permanently-failing worker is excluded per run but re-assigned
  on the next run (no cross-run failure history). Correctness is
  unaffected; registry deprioritisation is scheduled in Item 9b.
semantic versioning.

## [0.2.0] - 2026-11-24

### Added
- One Copilot, an organisation inside: senior role registry
  (provome/roles.py) where every role is senior by construction, from
  Chief Copilot to division leads and senior specialists.
- Team registry (provome/teams.py): 14 teams including Cancer, Tumor
  Biology, Rare Disease, DNA and Variant, Animal Health, Livestock,
  Plant Crops and Plant Pathology, each with a senior lead and
  declared problem areas, plus a router with a named fallback team
  per division.
- Orchestration engine (provome/orchestra.py): the Chief Copilot
  routes each question to a team, assigns primary and cross-check
  workers, requires agreement from at least 2 independent workers
  per key claim, escalates disagreements to the team lead with a
  tiebreak worker, and excludes failed or erroring workers with the
  reason recorded.
- Six-stage QC pipeline: evidence gate, citation check, calculation
  check, consistency check, cross-worker agreement, senior review.
  The final answer is produced only when all six pass; otherwise the
  Copilot declines naming every failed stage.
- Full audit trail for every assignment, result, disagreement,
  escalation, exclusion and sign-off.
- Test suite grows to 212 tests, including a 1,000-worker scale
  test and edge cases (empty registry, worker exceptions) that must
  decline cleanly rather than crash.

## [0.1.0] - 2026-11-24

### Added
- Provome Copilot core with the cite-or-decline evidence gate: an
  answer requires at least 3 verified evidence items from at least 2
  independent source groups, otherwise the Copilot declines with
  named reasons.
- Evidence engine: EvidenceItem model, source registry with
  independence groups, fingerprint deduplication, gate evaluation.
- Three divisions on one engine (provome/domains.py): Human (live,
  benchmarked), Animal and Plant (expansion). ACMG marked
  human-clinical only and refused for Animal and Plant questions.
- ACMG/AMP classification: Richards 2015 combining rules and the
  Tavtigian Bayesian points system, with framework disagreements
  surfaced instead of hidden.
- Population genetics: Whiffin maximum credible allele frequency,
  BA1/BS1/PM2-style threshold evaluation.
- Variant and consequence science: VCF and HGVS c./p. parsing, full
  standard codon table, molecular consequence calling.
- Calibration science from scratch: AUROC, Brier score, log loss,
  ECE/MCE and calibration tables.
- Phenotype semantic similarity: information content, MICA, Resnik
  similarity on a labelled illustrative ontology subset.
- Test suite: 167 tests covering every module.
- CI: GitHub Actions runs the suite and the Copilot demo smoke test
  on Python 3.10, 3.11 and 3.12 for every push and pull request.
