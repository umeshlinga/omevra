# CLAUDE.md — Provome: standing instructions for coding agents

This file is read by Claude Code and by any other coding agent working
in this repository. It states the company's red lines. They are not
style preferences. A change that crosses one of them does not ship,
however green the build looks.

Provome's law is **cite, or decline**: an answer backed by verified
evidence, with a checkable source for every claim, or an honest
decline that names what is missing. See `MISSION.md`.

## Red lines

1. **The evidence gate is never bypassed.**
   Every answer, claim, batch result, watch-mode alert and API/MCP
   response passes through `evaluate_gate` in `provome/evidence.py`.
   There must be no code path where model-written or agent-written
   text reaches a user without going through the gate. If such a path
   exists, that is a bug, not a feature. Agents propose; the gate
   judges; the Copilot speaks; the human decides.

2. **Nothing is labelled VERIFIED without real resolution against
   its source.**
   A status is earned by checking, never by a provider's stamp, a
   well-formed identifier, or ingest-time trust.
   - Literature: a PMID is verified only after
     `provome/confirmation.py::confirm_literature` re-confirms it
     against PubMed (record exists, title matches). A fabricated
     PMID, a title mismatch, or an unrecoverable identifier must
     never survive as VERIFIED.
   - Known gap, stated plainly: non-literature evidence (for example
     a ClinVar item) has no equivalent independent confirmer today.
     Items from registered non-literature sources are currently
     trusted at the provider's word at the item/gate layer; the live
     provider mitigates this by deriving direction from the served
     classification. Do not describe this gap as closed. It is
     pinned by KNOWN GAP tests in `tests/test_fraud_battery.py`, and
     closing it means building a confirmer that mirrors
     `confirm_literature`, not relabelling the trust.
   - FAILED verification is sticky: a failed item is never
     resurrected by a later verified-looking copy of the same record.
   - Duplicates are merged on canonical IDs before evidence is
     counted, in either direction.

3. **No stub or placeholder unit may be presented as a working
   feature.**
   If a unit does no live work, it does not ship as a feature, is not
   counted in any headcount, registry, README, website or pitch, and
   is not described in the present tense as if it worked. Planned
   work lives in a roadmap document, labelled as a plan. (The
   registry/org-chart modules flagged in the evaluation report are
   known stubs, are not part of the answer path, and their removal
   or relabelling is a Founder decision, not an agent decision.)

4. **Tests and the evaluation / gold fixtures are never edited to
   make a failing build pass.**
   If a test fails, the code is wrong until proven otherwise. Never
   weaken an assertion, delete a case, re-label an expected outcome,
   or edit a fixture/gold set so a failure disappears. The protected
   files are listed in `CODEOWNERS` and enforced by the pre-commit
   guard (below); changing them requires the Founder's explicit
   override and a written reason in the commit message.

5. **Every claim in code or docs must be measurable or labelled a
   plan.**
   Numbers come from a run, a test, or a cited source, with the
   dataset and mode stated. Do not copy a metric from one component
   to another: AUROC 0.9705 is the separate GBM model's score in
   `vus-evidence-agent`; it is not, and must never be presented as,
   the Copilot's score. Precision is always reported with coverage
   beside it. Vendor/self-reported figures are labelled as such.

## Working rules

- Core package is standard-library only (`pyproject.toml`). Do not
  add a runtime dependency without the Founder's approval.
- Pin exact model IDs in configuration, never "latest". Re-run the
  evaluation set before switching any pinned model.
- Keep agent roles few, evaluated, and read-only by default: each
  role has its own tests, hard step/cost budgets, and a named human
  who reads its failures. Fetching from a structured source
  (ClinVar, gnomAD) is ordinary code, not an agent.
- Call the fraud work "integrity checks", never "fraud detection":
  we flag and down-weight; we do not accuse. Publish blind spots
  alongside checks.
- Log decisions: one paragraph per architectural choice, a short
  write-up for every wrong answer (what failed, what test now
  catches it).
- Small, single-idea changes. Tests first where the behaviour is
  new. Run the full suite before committing:
  `PYTHONPATH=. python -m pytest -q` (436 tests at the time of
  writing).

## Pre-commit guard

`scripts/precommit_guard.py` refuses commits that modify protected
files — `tests/test_fraud_battery.py`, the gate module
`provome/evidence.py` (and its confirmation layer
`provome/confirmation.py`), or evaluation fixture files — unless
the Founder sets the explicit override:

```sh
PROVOME_FOUNDER_OVERRIDE=1 git commit ...
```

Install it as the repository hook (per clone, not tracked by git):

```sh
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
# or run the guard directly against the staged changes:
python3 scripts/precommit_guard.py
```

An override is an auditable Founder act: state the reason in the
commit message. An agent must never set the override on its own
initiative.
