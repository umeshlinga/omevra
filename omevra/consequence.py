"""Molecular consequence calling from real sequence.

This module answers a concrete question: given a coding sequence
(CDS) and a coding DNA variant, what happens to the protein?

It contains the full standard genetic code (all 64 codons, NCBI
translation table 1), a translator that refuses partial or invalid
input instead of guessing, an editor that applies a parsed
:class:`~omevra.variant.CodingVariant` to the CDS after checking the
stated reference bases against the actual sequence, and a consequence
caller that compares the reference and edited proteins residue by
residue.

Consequence terms follow the Sequence Ontology names in common use:
``synonymous_variant``, ``missense_variant``, ``stop_gained``,
``frameshift_variant``, ``start_lost``, ``stop_lost``,
``inframe_deletion``, ``inframe_insertion``, ``inframe_indel``, and
``splice_region_variant``.

Honest scope: this is single-transcript, CDS-only consequence calling.
It does not model splicing outcomes, nonsense mediated decay, or
alternative transcripts.  Intronic variants are reported as
splice-region candidates based on their offset, not edited into the
sequence.  Those limits are stated, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from .variant import (
    CodingVariant,
    CodingVariantKind,
    VariantParseError,
    parse_coding_hgvs,
    validate_dna_sequence,
)

__all__ = [
    "CODON_TABLE",
    "STOP_CODONS",
    "START_CODON",
    "translate",
    "translate_codon",
    "find_orfs",
    "ConsequenceTerm",
    "ConsequenceResult",
    "apply_coding_variant",
    "call_consequence",
    "is_synonymous_codon_change",
]


# The standard genetic code, NCBI translation table 1.  Codons are
# listed in the canonical TCAG ordering used by the NCBI table so the
# table can be audited against the published source at a glance.
_BASE_ORDER = "TCAG"
_AMINO_ACIDS_IN_ORDER = (
    "FFLLSSSSYY**CC*W"  # T..
    "LLLLPPPPHHQQRRRR"  # C..
    "IIIMTTTTNNKKSSRR"  # A..
    "VVVVAAAADDEEGGGG"  # G..
)


def _build_codon_table() -> dict[str, str]:
    table: dict[str, str] = {}
    index = 0
    for first in _BASE_ORDER:
        for second in _BASE_ORDER:
            for third in _BASE_ORDER:
                table[first + second + third] = _AMINO_ACIDS_IN_ORDER[index]
                index += 1
    return table


#: Mapping of all 64 codons to one-letter amino acids; ``*`` is stop.
CODON_TABLE: dict[str, str] = _build_codon_table()

#: The three stop codons of the standard code.
STOP_CODONS = frozenset(codon for codon, aa in CODON_TABLE.items() if aa == "*")

#: The standard initiation codon.
START_CODON = "ATG"


def translate_codon(codon: str) -> str:
    """Translate one codon to a one-letter amino acid (``*`` = stop).

    Raises:
        VariantParseError: if the input is not exactly one valid
            codon.  Ambiguous bases are rejected: a codon containing
        ``N`` has no single correct translation.
    """
    if not isinstance(codon, str) or len(codon) != 3:
        raise VariantParseError(f"a codon is exactly 3 bases, got {codon!r}")
    normalised = validate_dna_sequence(codon, field_name="codon")
    return CODON_TABLE[normalised]


def translate(
    sequence: str,
    *,
    to_stop: bool = False,
    require_start: bool = False,
) -> str:
    """Translate a DNA coding sequence in frame from position 1.

    Args:
        sequence: DNA bases; length need not be a multiple of three,
            but a trailing partial codon raises rather than being
            silently dropped, unless ``to_stop`` ended translation
            before it.
        to_stop: Stop at the first stop codon (excluded from the
            result).  When False, stop codons appear as ``*``.
        require_start: Raise unless the first codon is ATG.

    Returns:
        The protein sequence in one-letter codes, without a terminal
        stop unless one occurs internally and ``to_stop`` is False.
    """
    seq = validate_dna_sequence(sequence, field_name="sequence")
    if require_start and not seq.startswith(START_CODON):
        raise VariantParseError("sequence does not start with ATG")
    protein: list[str] = []
    for index in range(0, len(seq) - 2, 3):
        amino = CODON_TABLE[seq[index:index + 3]]
        if amino == "*" and to_stop:
            return "".join(protein)
        protein.append(amino)
    remainder = len(seq) % 3
    if remainder and not (to_stop and protein and protein[-1] == "*"):
        # A trailing partial codon in a claimed CDS is a data error.
        # It is tolerated only when translation already stopped.
        raise VariantParseError(
            f"sequence length {len(seq)} is not a multiple of 3; "
            f"trailing partial codon of {remainder} base(s)"
        )
    return "".join(protein)


def find_orfs(sequence: str, *, min_protein_length: int = 1) -> list[tuple[int, int, str]]:
    """Find open reading frames in all three forward frames.

    An ORF here is an ATG followed by in-frame codons ending at a
    stop codon.  Returns a list of ``(start_index_0based, end_index,
    protein)`` tuples where ``end_index`` is the index just past the
    stop codon.  This is a teaching-grade scanner, not a gene finder:
    it does not score Kozak context or splice structure.
    """
    seq = validate_dna_sequence(sequence, allow_n=False, field_name="sequence")
    results: list[tuple[int, int, str]] = []
    for frame in range(3):
        index = frame
        while index + 3 <= len(seq):
            if seq[index:index + 3] == START_CODON:
                protein: list[str] = []
                cursor = index
                terminated = False
                while cursor + 3 <= len(seq):
                    amino = CODON_TABLE[seq[cursor:cursor + 3]]
                    if amino == "*":
                        terminated = True
                        cursor += 3
                        break
                    protein.append(amino)
                    cursor += 3
                if terminated and len(protein) >= min_protein_length:
                    results.append((index, cursor, "".join(protein)))
                    index = cursor
                    continue
            index += 3
    results.sort()
    return results


def is_synonymous_codon_change(ref_codon: str, alt_codon: str) -> bool:
    """True when two different codons encode the same amino acid."""
    if ref_codon.upper() == alt_codon.upper():
        return False
    return translate_codon(ref_codon) == translate_codon(alt_codon)


class ConsequenceTerm(str, Enum):
    """Sequence Ontology style consequence terms produced here."""

    SYNONYMOUS = "synonymous_variant"
    MISSENSE = "missense_variant"
    STOP_GAINED = "stop_gained"
    STOP_LOST = "stop_lost"
    START_LOST = "start_lost"
    FRAMESHIFT = "frameshift_variant"
    INFRAME_DELETION = "inframe_deletion"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_INDEL = "inframe_indel"
    SPLICE_REGION = "splice_region_variant"
    PROTEIN_ALTERING = "protein_altering_variant"
    NO_EFFECT = "no_sequence_effect"


@dataclass(frozen=True)
class ConsequenceResult:
    """Result of consequence calling for one variant on one CDS.

    Attributes:
        term: The primary (most severe) consequence term.
        all_terms: Every term that applies, most severe first.
        ref_protein: Reference protein (no terminal stop).
        alt_protein: Edited protein (no terminal stop), or ``None``
            when the variant cannot be edited into the CDS, for
            example an intronic variant.
        first_affected_residue: 1-based index of the first residue
            that differs, or ``None`` when the proteins are identical.
        ref_residue: Reference residue at that position, if any.
        alt_residue: Alternate residue at that position, if any.
        protein_length_change: ``len(alt) - len(ref)`` when both
            proteins are known, else ``None``.
        explanation: Plain language account of how the call was made.
        warnings: Caveats, for example a reference mismatch.
    """

    term: ConsequenceTerm
    all_terms: tuple[ConsequenceTerm, ...]
    ref_protein: str
    alt_protein: Optional[str] = None
    first_affected_residue: Optional[int] = None
    ref_residue: Optional[str] = None
    alt_residue: Optional[str] = None
    protein_length_change: Optional[int] = None
    explanation: str = ""
    warnings: tuple[str, ...] = ()


def _check_reference(
    cds: str,
    variant: CodingVariant,
    warnings: list[str],
) -> None:
    """Check any stated reference bases against the CDS.

    A mismatch means either the sequence or the notation is wrong.
    That is fatal for substitutions (the whole call depends on the
    base) but only a warning for deletions, where some notations
    state the deleted sequence redundantly.
    """
    if variant.ref is None:
        return
    if variant.kind is CodingVariantKind.SUBSTITUTION:
        actual = cds[variant.start - 1:variant.start]
        if actual != variant.ref:
            raise VariantParseError(
                f"reference mismatch at c.{variant.start}: notation "
                f"says {variant.ref}, sequence has {actual}"
            )
        return
    if variant.kind in (CodingVariantKind.DELETION, CodingVariantKind.DUPLICATION):
        actual = cds[variant.start - 1:variant.end]
        if actual != variant.ref:
            warnings.append(
                f"stated reference {variant.ref} does not match the "
                f"sequence segment {actual} at c.{variant.start}_"
                f"{variant.end}; the sequence was trusted"
            )


def apply_coding_variant(cds: str, variant: CodingVariant) -> str:
    """Apply a coding variant to a CDS and return the edited sequence.

    The edit follows HGVS semantics:

    * substitution replaces the single base at ``start``;
    * deletion removes ``start..end``;
    * insertion inserts ``alt`` between ``start`` and ``end``, which
      must be the two flanking positions;
    * duplication copies ``start..end`` and inserts the copy directly
      after ``end``;
    * delins removes ``start..end`` and inserts ``alt`` in its place;
    * inversion reverses and complements ``start..end``.

    Raises:
        VariantParseError: on out-of-range positions, a substitution
            reference mismatch, or an intronic variant (which has no
            CDS edit).
    """
    seq = validate_dna_sequence(cds, field_name="cds")
    if variant.is_intronic:
        raise VariantParseError(
            "intronic variants have no coding sequence edit"
        )
    if variant.end > len(seq):
        raise VariantParseError(
            f"variant end c.{variant.end} exceeds CDS length {len(seq)}"
        )
    _check_reference(seq, variant, warnings=[])
    start0 = variant.start - 1
    end0 = variant.end  # slice end (exclusive) for start..end inclusive

    if variant.kind is CodingVariantKind.SUBSTITUTION:
        if variant.alt is None:
            raise VariantParseError("substitution needs an alternate base")
        return seq[:start0] + variant.alt + seq[start0 + 1:]
    if variant.kind is CodingVariantKind.DELETION:
        return seq[:start0] + seq[end0:]
    if variant.kind is CodingVariantKind.INSERTION:
        if variant.alt is None:
            raise VariantParseError("insertion needs inserted bases")
        # HGVS insertion positions are the two flanking bases: the
        # inserted sequence goes between them, that is, after the
        # first flanking position (variant.start).
        return seq[:variant.start] + variant.alt + seq[variant.start:]
    if variant.kind is CodingVariantKind.DUPLICATION:
        segment = seq[start0:end0]
        return seq[:end0] + segment + seq[end0:]
    if variant.kind is CodingVariantKind.DELINS:
        if variant.alt is None:
            raise VariantParseError("delins needs replacement bases")
        return seq[:start0] + variant.alt + seq[end0:]
    if variant.kind is CodingVariantKind.INVERSION:
        from .variant import reverse_complement

        segment = seq[start0:end0]
        return seq[:start0] + reverse_complement(segment) + seq[end0:]
    raise VariantParseError(f"unsupported variant kind: {variant.kind}")  # pragma: no cover


_SEVERITY_ORDER = (
    ConsequenceTerm.FRAMESHIFT,
    ConsequenceTerm.STOP_GAINED,
    ConsequenceTerm.START_LOST,
    ConsequenceTerm.STOP_LOST,
    ConsequenceTerm.MISSENSE,
    ConsequenceTerm.INFRAME_DELETION,
    ConsequenceTerm.INFRAME_INSERTION,
    ConsequenceTerm.INFRAME_INDEL,
    ConsequenceTerm.PROTEIN_ALTERING,
    ConsequenceTerm.SPLICE_REGION,
    ConsequenceTerm.SYNONYMOUS,
    ConsequenceTerm.NO_EFFECT,
)


def _first_difference(
    ref_protein: str,
    alt_protein: str,
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Locate the first residue at which two proteins differ."""
    limit = min(len(ref_protein), len(alt_protein))
    for index in range(limit):
        if ref_protein[index] != alt_protein[index]:
            return index + 1, ref_protein[index], alt_protein[index]
    if len(ref_protein) != len(alt_protein):
        index = limit
        ref_res = ref_protein[index] if index < len(ref_protein) else None
        alt_res = alt_protein[index] if index < len(alt_protein) else None
        return index + 1, ref_res, alt_res
    return None, None, None


def call_consequence(
    cds: str,
    variant: CodingVariant | str,
    *,
    splice_boundary_distance: int = 8,
) -> ConsequenceResult:
    """Call the molecular consequence of a variant on a CDS.

    Args:
        cds: The coding sequence, starting at the ATG (c.1 is the
            first base) and normally ending with a stop codon.
        variant: A parsed :class:`CodingVariant` or an HGVS ``c.``
            string.
        splice_boundary_distance: Not used to edit sequence; intronic
            variants within this many bases of the exon boundary are
            the canonical splice region, and all intronic variants
            parsed here are reported as splice-region candidates with
            their offset in the explanation.

    Returns:
        A :class:`ConsequenceResult`.  The primary term is the most
        severe applicable term, using the severity order in
        ``_SEVERITY_ORDER`` (frameshift and stop-gained ahead of
        missense, and so on).
    """
    seq = validate_dna_sequence(cds, field_name="cds")
    if isinstance(variant, str):
        parsed = parse_coding_hgvs(variant)
    else:
        parsed = variant
    ref_protein_full = translate(seq, to_stop=False)
    ref_protein = ref_protein_full.rstrip("*")
    warnings: list[str] = []
    if "*" in ref_protein:
        warnings.append(
            "the reference CDS contains an internal stop codon; "
            "consequence calls after that point are unreliable"
        )
    if not seq.startswith(START_CODON):
        warnings.append("the reference CDS does not start with ATG")

    if parsed.is_intronic:
        return ConsequenceResult(
            term=ConsequenceTerm.SPLICE_REGION,
            all_terms=(ConsequenceTerm.SPLICE_REGION,),
            ref_protein=ref_protein,
            alt_protein=None,
            explanation=(
                f"intronic variant at offset {parsed.offset:+d} from "
                f"c.{parsed.start}; reported as a splice-region "
                "candidate. Splicing outcomes are not modelled by "
                "this package."
            ),
            warnings=tuple(warnings),
        )

    _check_reference(seq, parsed, warnings)
    alt_seq = apply_coding_variant(seq, parsed)
    # The edited sequence may not be a multiple of three (frameshift).
    # Translate codon by codon and stop at the first stop, tolerating
    # a trailing partial codon because a frameshift creates one.
    alt_protein = _translate_lenient(alt_seq)
    first, ref_res, alt_res = _first_difference(ref_protein, alt_protein)
    length_change = len(alt_protein) - len(ref_protein)

    terms: list[ConsequenceTerm] = []
    net = parsed.length_change
    frameshift = net is not None and net % 3 != 0 and parsed.kind in (
        CodingVariantKind.DELETION,
        CodingVariantKind.INSERTION,
        CodingVariantKind.DUPLICATION,
        CodingVariantKind.DELINS,
    )
    if alt_protein == ref_protein:
        terms.append(ConsequenceTerm.SYNONYMOUS)
        explanation = (
            "the edited protein is identical to the reference protein"
        )
    elif frameshift:
        terms.append(ConsequenceTerm.FRAMESHIFT)
        explanation = (
            f"net length change of {net} base(s) is not a multiple "
            "of three, so the reading frame shifts at the edit site"
        )
    else:
        if first == 1 and ref_protein[:1] == "M" and alt_protein[:1] != "M":
            terms.append(ConsequenceTerm.START_LOST)
        # Stop gained: the first stop codon in the edited sequence
        # is premature when coding sequence continues past it.  A
        # stop that is simply the final codon of the edited sequence
        # is a normal termination codon (for example after an
        # in-frame deletion shortens the protein), not a gained one.
        for codon_index in range(len(alt_seq) // 3):
            codon = alt_seq[codon_index * 3:codon_index * 3 + 3]
            if CODON_TABLE[codon] != "*":
                continue
            codon_end = codon_index * 3 + 3
            if codon_end < len(alt_seq) and codon_index < len(ref_protein):
                terms.append(ConsequenceTerm.STOP_GAINED)
                if first is None or codon_index + 1 < first:
                    first = codon_index + 1
                    ref_res = ref_protein[codon_index]
                    alt_res = "*"
            break
        ref_has_terminal_stop = ref_protein_full.endswith("*")
        if ref_has_terminal_stop:
            # Stop lost: the reference stop codon is changed so the
            # protein runs past its normal end.  In protein space
            # (terminal stop stripped) this appears as a protein that
            # is longer and whose residue at the old stop position is
            # a sense residue in a full-length retranslation.
            alt_full = _translate_lenient(alt_seq, keep_stop=True)
            if len(alt_full) > len(ref_protein) and (
                len(alt_full) <= len(ref_protein)
                or alt_full[len(ref_protein)] != "*"
            ):
                if alt_protein != ref_protein and not frameshift:
                    # Only call stop-lost when the change touches the
                    # terminal codon region.
                    stop_codon_start = len(seq) - 3
                    if parsed.end > stop_codon_start:
                        terms.append(ConsequenceTerm.STOP_LOST)
        if parsed.kind is CodingVariantKind.DELETION and net is not None and net % 3 == 0:
            terms.append(ConsequenceTerm.INFRAME_DELETION)
        elif parsed.kind is CodingVariantKind.INSERTION and net is not None and net % 3 == 0:
            terms.append(ConsequenceTerm.INFRAME_INSERTION)
        elif parsed.kind is CodingVariantKind.DELINS and net is not None and net % 3 == 0:
            terms.append(ConsequenceTerm.INFRAME_INDEL)
        elif parsed.kind is CodingVariantKind.DUPLICATION and net is not None and net % 3 == 0:
            terms.append(ConsequenceTerm.INFRAME_INSERTION)
        if (
            first is not None
            and ref_res is not None
            and alt_res is not None
            and alt_res != "*"
            and length_change == 0
            and parsed.kind is CodingVariantKind.SUBSTITUTION
        ):
            terms.append(ConsequenceTerm.MISSENSE)
        if not terms:
            terms.append(ConsequenceTerm.PROTEIN_ALTERING)
        explanation = _explain(parsed, first, ref_res, alt_res, length_change)

    terms_sorted = tuple(
        sorted(set(terms), key=_SEVERITY_ORDER.index)
    )
    return ConsequenceResult(
        term=terms_sorted[0],
        all_terms=terms_sorted,
        ref_protein=ref_protein,
        alt_protein=alt_protein,
        first_affected_residue=first,
        ref_residue=ref_res,
        alt_residue=alt_res,
        protein_length_change=length_change,
        explanation=explanation,
        warnings=tuple(warnings),
    )


def _translate_lenient(sequence: str, *, keep_stop: bool = False) -> str:
    """Translate complete codons, tolerating a trailing partial codon.

    Translation stops after the first stop codon.  With
    ``keep_stop`` the terminal stop is retained as ``*`` at the end,
    which the stop-lost logic uses.
    """
    protein: list[str] = []
    for index in range(0, len(sequence) - 2, 3):
        amino = CODON_TABLE[sequence[index:index + 3]]
        if amino == "*":
            if keep_stop:
                protein.append("*")
            break
        protein.append(amino)
    return "".join(protein)


def _explain(
    variant: CodingVariant,
    first: Optional[int],
    ref_res: Optional[str],
    alt_res: Optional[str],
    length_change: int,
) -> str:
    if first is None:
        return "no residue differs between reference and edited protein"
    parts = [
        f"first difference at residue {first}",
        f"reference residue {ref_res!r}",
        f"alternate residue {alt_res!r}",
        f"protein length change {length_change:+d}",
        f"variant kind {variant.kind.value}",
    ]
    return "; ".join(parts)
