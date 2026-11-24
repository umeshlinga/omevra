# Changelog

All notable changes to Omevra are recorded here. Versions follow
semantic versioning.

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
