"""Version strings must agree everywhere (audit finding, 2026-11-25).

__init__ carried 0.2.0 through three releases while pyproject,
CHANGELOG, and the git tags moved on.  This test makes that class
of drift fail loudly instead of shipping quietly.
"""

import re
from pathlib import Path

import omevra

ROOT = Path(__file__).resolve().parent.parent


def test_package_version_matches_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject version not found"
    assert omevra.__version__ == match.group(1)


def test_changelog_top_entry_matches():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    match = re.search(r"^## \[([^\]]+)\]", changelog, re.MULTILINE)
    assert match, "changelog entry not found"
    assert match.group(1) == omevra.__version__
