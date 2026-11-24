"""Tests for omevra.variant: VCF, HGVS c. and p. parsing."""

import pytest

from omevra.variant import (
    CodingVariantKind,
    ProteinVariantKind,
    VariantParseError,
    complement,
    normalise_chromosome,
    parse_coding_hgvs,
    parse_protein_hgvs,
    parse_vcf_line,
    reverse_complement,
    validate_dna_sequence,
)


def test_validate_dna_uppercases():
    assert validate_dna_sequence("acgt") == "ACGT"


def test_validate_dna_rejects_bad_bases():
    with pytest.raises(VariantParseError):
        validate_dna_sequence("ACGX")


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"
    assert complement("ATGC") == "TACG"


def test_normalise_chromosome():
    assert normalise_chromosome("chr17") == "17"
    assert normalise_chromosome("CHR7") == "7"
    assert normalise_chromosome("chrM") == "MT"
    assert normalise_chromosome("X") == "X"
    with pytest.raises(VariantParseError):
        normalise_chromosome("chrUn")


def test_parse_vcf_snv():
    record = parse_vcf_line("17\t43071077\t.\tC\tT\t.\t.\t.")
    assert record.chromosome == "17"
    assert record.position == 43071077
    assert record.variant_class() == "snv"
    assert record.end == 43071077


def test_parse_vcf_anchored_deletion():
    record = parse_vcf_line("7\t117559590\t.\tATCT\tA")
    assert record.variant_class() == "deletion"
    assert record.is_anchored_indel()
    assert record.deleted_sequence() == "TCT"


def test_parse_vcf_anchored_insertion():
    record = parse_vcf_line("7\t117559590\t.\tA\tATCT")
    assert record.variant_class() == "insertion"
    assert record.is_anchored_indel()
    assert record.inserted_sequence() == "TCT"


def test_parse_vcf_rejects_multiallelic_and_headers():
    with pytest.raises(VariantParseError):
        parse_vcf_line("1\t100\t.\tA\tC,G")
    with pytest.raises(VariantParseError):
        parse_vcf_line("#CHROM\tPOS\tID\tREF\tALT")


def test_vcf_ref_alt_must_differ():
    with pytest.raises(VariantParseError):
        parse_vcf_line("1\t100\t.\tA\tA")


def test_coding_substitution():
    variant = parse_coding_hgvs("c.76A>C")
    assert variant.kind is CodingVariantKind.SUBSTITUTION
    assert variant.start == 76 and variant.end == 76
    assert variant.ref == "A" and variant.alt == "C"
    assert variant.length_change == 0


def test_coding_cftr_deletion():
    # CFTR p.Phe508del: a three-base in-frame deletion.
    variant = parse_coding_hgvs("c.1521_1523delCTT")
    assert variant.kind is CodingVariantKind.DELETION
    assert (variant.start, variant.end) == (1521, 1523)
    assert variant.length_change == -3


def test_coding_brca1_founder_deletion():
    # BRCA1 c.68_69delAG (185delAG): a two-base frameshifting deletion.
    variant = parse_coding_hgvs("c.68_69delAG")
    assert variant.kind is CodingVariantKind.DELETION
    assert variant.length_change == -2


def test_coding_insertion_requires_flanking_positions():
    variant = parse_coding_hgvs("c.231_232insG")
    assert variant.kind is CodingVariantKind.INSERTION
    assert variant.length_change == 1
    with pytest.raises(VariantParseError):
        parse_coding_hgvs("c.231_235insG")


def test_coding_duplication_and_delins():
    dup = parse_coding_hgvs("c.546dupT")
    assert dup.kind is CodingVariantKind.DUPLICATION
    assert dup.length_change == 1
    delins = parse_coding_hgvs("c.112_114delinsTG")
    assert delins.kind is CodingVariantKind.DELINS
    assert delins.length_change == -1


def test_coding_intronic_offset():
    variant = parse_coding_hgvs("c.76+2T>C")
    assert variant.is_intronic
    assert variant.offset == 2
    downstream = parse_coding_hgvs("c.76-1G>A")
    assert downstream.offset == -1


def test_coding_deletion_length_mismatch_is_caught():
    with pytest.raises(VariantParseError):
        parse_coding_hgvs("c.100_102delAC")


def test_coding_garbage_is_rejected():
    with pytest.raises(VariantParseError):
        parse_coding_hgvs("c.hello")
    with pytest.raises(VariantParseError):
        parse_coding_hgvs("")


def test_protein_missense():
    variant = parse_protein_hgvs("p.Gly12Val")
    assert variant.kind is ProteinVariantKind.MISSENSE
    assert variant.position == 12
    assert variant.ref_one_letter() == "G"
    assert variant.alt_one_letter() == "V"


def test_protein_cftr_deletion():
    variant = parse_protein_hgvs("p.Phe508del")
    assert variant.kind is ProteinVariantKind.DELETION
    assert variant.position == 508
    assert variant.ref_one_letter() == "F"


def test_protein_nonsense_and_synonymous():
    nonsense = parse_protein_hgvs("p.Tyr123Ter")
    assert nonsense.kind is ProteinVariantKind.NONSENSE
    assert nonsense.alt_one_letter() == "*"
    syn = parse_protein_hgvs("p.Gly12=")
    assert syn.kind is ProteinVariantKind.SYNONYMOUS


def test_protein_frameshift_with_new_stop():
    variant = parse_protein_hgvs("p.Arg97ProfsTer23")
    assert variant.kind is ProteinVariantKind.FRAMESHIFT
    assert variant.new_stop_distance == 23


def test_protein_start_loss():
    variant = parse_protein_hgvs("p.Met1Val")
    assert variant.kind is ProteinVariantKind.START_LOSS


def test_protein_range_deletion():
    variant = parse_protein_hgvs("p.Lys23_Arg24del")
    assert variant.kind is ProteinVariantKind.DELETION
    assert variant.position == 23 and variant.end_position == 24


def test_protein_garbage_is_rejected():
    with pytest.raises(VariantParseError):
        parse_protein_hgvs("p.Xyz123Abc")
    with pytest.raises(VariantParseError):
        parse_protein_hgvs("p.Met1?")
