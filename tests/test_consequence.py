"""Tests for provome.consequence on a small hand-built CDS.

CDS used throughout (15 bases, protein M A Y G then stop):
    ATG GCT TAT GGT TAA
positions: codon1 1-3, codon2 4-6, codon3 7-9, codon4 10-12,
codon5 (stop) 13-15.
"""

import pytest

from provome.consequence import (
    CODON_TABLE,
    ConsequenceTerm,
    apply_coding_variant,
    call_consequence,
    find_orfs,
    is_synonymous_codon_change,
    translate,
    translate_codon,
)
from provome.variant import VariantParseError, parse_coding_hgvs

CDS = "ATGGCTTATGGTTAA"


def test_codon_table_is_complete_and_correct():
    assert len(CODON_TABLE) == 64
    assert translate_codon("ATG") == "M"   # methionine, start
    assert translate_codon("TTT") == "F"   # phenylalanine
    assert translate_codon("TGG") == "W"   # tryptophan, single codon
    assert translate_codon("TAA") == "*"   # stop
    assert translate_codon("TAG") == "*"
    assert translate_codon("TGA") == "*"
    assert translate_codon("GCT") == "A"   # alanine
    # Leucine has six codons in the standard code.
    leucine = [c for c, aa in CODON_TABLE.items() if aa == "L"]
    assert len(leucine) == 6


def test_translate_known_sequence():
    assert translate(CDS, to_stop=True) == "MAYG"
    # TTT = Phe, TGG = Trp, TAA = stop.
    assert translate("TTTTGGTAA", to_stop=True) == "FW"


def test_translate_rejects_partial_codon():
    with pytest.raises(VariantParseError):
        translate("ATGGC")


def test_translate_codon_rejects_ambiguous():
    with pytest.raises(VariantParseError):
        translate_codon("ATN")


def test_synonymous_codon_change():
    assert is_synonymous_codon_change("GCT", "GCC")  # both alanine
    assert not is_synonymous_codon_change("GCT", "GTT")  # Ala vs Val
    assert not is_synonymous_codon_change("GCT", "GCT")


def test_synonymous_variant_call():
    result = call_consequence(CDS, "c.6T>C")  # GCT -> GCC, both Ala
    assert result.term is ConsequenceTerm.SYNONYMOUS
    assert result.alt_protein == result.ref_protein


def test_missense_variant_call():
    result = call_consequence(CDS, "c.5C>T")  # GCT -> GTT, Ala -> Val
    assert result.term is ConsequenceTerm.MISSENSE
    assert result.first_affected_residue == 2
    assert result.ref_residue == "A"
    assert result.alt_residue == "V"


def test_nonsense_variant_call():
    result = call_consequence(CDS, "c.9T>A")  # TAT -> TAA, Tyr -> stop
    assert result.term is ConsequenceTerm.STOP_GAINED
    assert result.alt_protein == "MA"


def test_start_loss_call():
    result = call_consequence(CDS, "c.1A>G")  # ATG -> GTG
    assert result.term is ConsequenceTerm.START_LOST
    assert result.alt_protein.startswith("V")


def test_stop_loss_call():
    result = call_consequence(CDS, "c.13T>C")  # TAA -> CAA (Gln)
    assert result.term is ConsequenceTerm.STOP_LOST


def test_frameshift_deletion_call():
    result = call_consequence(CDS, "c.5del")
    assert result.term is ConsequenceTerm.FRAMESHIFT


def test_inframe_deletion_call():
    result = call_consequence(CDS, "c.7_9delTAT")
    assert result.term is ConsequenceTerm.INFRAME_DELETION
    assert result.alt_protein == "MAG"
    assert result.protein_length_change == -1


def test_inframe_insertion_call():
    result = call_consequence(CDS, "c.6_7insGGG")  # adds a Gly codon
    assert result.term is ConsequenceTerm.INFRAME_INSERTION
    assert result.alt_protein == "MAGYG"


def test_splice_region_for_intronic():
    result = call_consequence(CDS, "c.9+2T>C")
    assert result.term is ConsequenceTerm.SPLICE_REGION
    assert result.alt_protein is None


def test_reference_mismatch_raises():
    with pytest.raises(VariantParseError):
        call_consequence(CDS, "c.5A>T")  # position 5 is C, not A


def test_apply_variant_directly():
    variant = parse_coding_hgvs("c.7_9delTAT")
    edited = apply_coding_variant(CDS, variant)
    assert edited == "ATGGCT" + "GGT" + "TAA"


def test_find_orfs():
    orfs = find_orfs("CCATGGCTTAAATGGGG")
    assert orfs
    start, end, protein = orfs[0]
    assert protein.startswith("M")


def test_out_of_range_variant_raises():
    with pytest.raises(VariantParseError):
        call_consequence(CDS, "c.99A>T")
