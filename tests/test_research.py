"""Tests for the Research Division registry (5,000 researchers)."""

from provome.orchestra import WorkContext, WorkTask
from provome.research import (
    DIVISION_SIZES,
    RESEARCH_DIVISION_SIZE,
    build_research_division,
    specialties_for,
)


def test_full_division_registers():
    registry = build_research_division()
    assert len(registry) == RESEARCH_DIVISION_SIZE == 5000
    for division, size in DIVISION_SIZES.items():
        assert len(registry.workers_for(f"research:{division}")) == size


def test_assignments_are_unique():
    registry = build_research_division()
    assignments = {
        (w.division, w.specialty, w.beat) for w in registry.all()
    }
    assert len(assignments) == 5000


def test_senior_leads_present_per_division():
    registry = build_research_division()
    for division, size in DIVISION_SIZES.items():
        leads = [
            w
            for w in registry.workers_for(f"research:{division}")
            if w.role_id == "senior-research-lead"
        ]
        assert len(leads) == size // 100


def test_every_specialty_staffed():
    registry = build_research_division()
    for division in DIVISION_SIZES:
        for specialty in specialties_for(division):
            assert registry.workers_for(f"specialty:{specialty}")


def test_unit_is_honest_without_live_assignment():
    registry = build_research_division()
    worker = registry.get("research-human-0001")
    result = worker.run(
        WorkTask(task_id="t1", kind="evidence", topic_key=None,
                 team_id="research-human"),
        WorkContext(question=None, topic_key=None, items=(),
                    source_registry=None, team=None),
    )
    assert result.ok is False
    assert result.claims == ()
    assert "standing research unit" in result.notes
