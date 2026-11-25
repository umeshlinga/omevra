#!/usr/bin/env python3
"""Pre-commit guard for Provome's protected files.

Refuses a commit that modifies any protected file unless the Founder
has set the explicit override environment variable:

    PROVOME_FOUNDER_OVERRIDE=1 git commit ...

Protected means (see CLAUDE.md, red line 4, and CODEOWNERS):

  * the fraud battery            tests/test_fraud_battery.py
  * the gate module              provome/evidence.py
  * the gate confirmation layer  provome/confirmation.py
  * evaluation / gold fixtures   provome/data/*.json,
                                 provome/benchmark.py,
                                 provome/animal_benchmark.py,
                                 provome/plant_benchmark.py,
                                 tests/test_benchmark.py,
                                 tests/test_animal_benchmark.py,
                                 tests/test_plant_benchmark.py,
                                 multi-agent-evidence/assets/eval_set.csv

Usage:
    python3 scripts/precommit_guard.py [--staged | FILE ...]

With no arguments (or --staged) the guard inspects the staged changes
(`git diff --cached`). With explicit FILE arguments it checks those
paths instead, which is how the guard is demonstrated and tested
without making a commit.

Exit codes: 0 = allowed, 1 = blocked, 2 = could not determine the
staged file list (fail closed when running as a hook on staged data).
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys

OVERRIDE_ENV = "PROVOME_FOUNDER_OVERRIDE"

PROTECTED_PATTERNS: tuple[str, ...] = (
    # Fraud battery
    "tests/test_fraud_battery.py",
    # Gate module and its confirmation layer
    "provome/evidence.py",
    "provome/confirmation.py",
    # Evaluation / gold fixtures and benchmark definitions
    "provome/data/*.json",
    "provome/benchmark.py",
    "provome/animal_benchmark.py",
    "provome/plant_benchmark.py",
    "tests/test_benchmark.py",
    "tests/test_animal_benchmark.py",
    "tests/test_plant_benchmark.py",
    "multi-agent-evidence/assets/eval_set.csv",
)


def is_protected(path: str) -> bool:
    """Return True if *path* (repo-relative, forward slashes) is protected."""
    normalized = path.strip().lstrip("./")
    return any(
        fnmatch.fnmatchcase(normalized, pattern)
        for pattern in PROTECTED_PATTERNS
    )


def staged_files() -> list[str]:
    """Return the repo-relative paths staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            "precommit_guard: could not read staged changes "
            f"(git exited {result.returncode}); failing closed.",
            file=sys.stderr,
        )
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise SystemExit(2)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--staged"]
    files = args if args else staged_files()
    blocked = [f for f in files if is_protected(f)]

    if not blocked:
        return 0

    if os.environ.get(OVERRIDE_ENV) == "1":
        print(
            "precommit_guard: PROVOME_FOUNDER_OVERRIDE=1 is set; "
            "allowing protected files (Founder override must be "
            "explained in the commit message):",
            file=sys.stderr,
        )
        for path in blocked:
            print(f"  - {path}", file=sys.stderr)
        return 0

    print(
        "precommit_guard: BLOCKED — this commit modifies protected "
        "Provome files:",
        file=sys.stderr,
    )
    for path in blocked:
        print(f"  - {path}", file=sys.stderr)
    print(
        "\nThe gate, the fraud battery, and the evaluation/gold "
        "fixtures are never edited to make a failing build pass "
        "(CLAUDE.md, red line 4).\nIf you are the Founder and this "
        "change is deliberate, re-run with:\n"
        f"    {OVERRIDE_ENV}=1 git commit ...\n"
        "and state the reason in the commit message.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
