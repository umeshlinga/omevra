"""Tests for disease corps (small corps for speed; the full
5,000-per-corps deployment is verified by the module demo)."""

from provome.diseases import FIRST_WAVE, build_disease_corps


def test_corps_assignments_unique_and_led():
    corps = build_disease_corps("oncology", "human", size=200)
    assert len(corps) == 200
    assignments = {
        (w.specialty, w.beat) for w in corps.registry.all()
    }
    assert len(assignments) == 200
    leads = [
        w for w in corps.registry.all()
        if w.role_id == "senior-research-lead"
    ]
    assert len(leads) == 2


def test_first_wave_manifest_arithmetic():
    from provome.diseases import CORPS_SIZE, CORPS_SIZES

    assert len(FIRST_WAVE) == 11
    total = sum(
        CORPS_SIZES.get(name, CORPS_SIZE) for name, _ in FIRST_WAVE
    )
    assert total == 70000
    assert CORPS_SIZES["diabetes"] == 20000
    divisions = {d for _, d in FIRST_WAVE}
    assert divisions == {"human", "animal", "plant"}


def test_covid_is_a_focus_not_a_duplicate_corps():
    corps = build_disease_corps("infectious-diseases", "human", size=160)
    focuses = {
        w.specialty.split("/")[0] for w in corps.registry.all()
    }
    assert "covid-19" in focuses
