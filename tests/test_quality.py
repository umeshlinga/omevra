"""Tests for the Quality Division registry."""

from provome.orchestra import WorkContext, WorkTask
from provome.quality import (
    DEPARTMENTS,
    METHODS,
    QUALITY_DIVISION_SIZE,
    build_quality_division,
)


def test_quality_division_totals_and_coverage():
    registry = build_quality_division()
    assert len(registry) == QUALITY_DIVISION_SIZE
    covered = {w.department for w in registry.all()}
    assert covered == set(DEPARTMENTS)


def test_assignments_unique_and_methods_differ():
    registry = build_quality_division()
    assignments = {
        (w.department, w.method, w.beat) for w in registry.all()
    }
    assert len(assignments) == QUALITY_DIVISION_SIZE
    # Item 36: every department is checked by every distinct method,
    # so any claim can be verified by two different methods.
    for dept in DEPARTMENTS:
        methods = {
            w.method for w in registry.all() if w.department == dept
        }
        assert methods == set(METHODS)


def test_leadership_structure():
    registry = build_quality_division()
    chiefs = [
        w for w in registry.all()
        if w.role_id == "chief-quality-copilot"
    ]
    assert len(chiefs) == 1
    leads = [
        w for w in registry.all() if w.role_id == "senior-qa-lead"
    ]
    assert len(leads) == len(DEPARTMENTS) - 1  # chief leads dept 1


def test_checker_without_assignment_passes_nothing():
    registry = build_quality_division()
    worker = registry.all()[0]
    result = worker.run(
        WorkTask(task_id="t1", kind="check"),
        WorkContext(
            question=None, topic_key="t", items=(),
            source_registry=None, team=None,
        ),
    )
    assert result.ok is False
    assert result.claims == ()
