"""Tests for the gate benchmark harness."""

from provome.benchmark import probes, run_benchmark


def test_benchmark_all_probes_correct():
    report = run_benchmark()
    assert report.passed
    assert report.accuracy == 1.0
    assert report.failures == ()
    assert report.disclosure_failures == ()


def test_probe_set_covers_all_categories():
    categories = {p.category for p in probes()}
    assert categories == {"answer", "decline", "disclose"}
    expected = {p.expected_pass for p in probes()}
    assert expected == {True, False}
