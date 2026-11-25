# Changelog

All notable changes to Omevra are recorded here. Versions follow
semantic versioning.

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
  (omevra/roles.py) where every role is senior by construction, from
  Chief Copilot to division leads and senior specialists.
- Team registry (omevra/teams.py): 14 teams including Cancer, Tumor
  Biology, Rare Disease, DNA and Variant, Animal Health, Livestock,
  Plant Crops and Plant Pathology, each with a senior lead and
  declared problem areas, plus a router with a named fallback team
  per division.
- Orchestration engine (omevra/orchestra.py): the Chief Copilot
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
- Omevra Copilot core with the cite-or-decline evidence gate: an
  answer requires at least 3 verified evidence items from at least 2
  independent source groups, otherwise the Copilot declines with
  named reasons.
- Evidence engine: EvidenceItem model, source registry with
  independence groups, fingerprint deduplication, gate evaluation.
- Three divisions on one engine (omevra/domains.py): Human (live,
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
