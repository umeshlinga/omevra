"""Variant parsing, validation, and normalisation.

This module models small sequence variants in the notations working
geneticists actually use:

* **VCF-style records**: chromosome, 1-based position, reference and
  alternate alleles (VCF specification, Danecek et al. 2011,
  *Bioinformatics* 27:2156-2158).  In VCF, insertion and deletion
  alleles are *anchored* on a preceding reference base; this module
  validates that convention but stores the alleles as given.
* **HGVS coding DNA notation** (``c.``), as recommended by the Human
  Genome Variation Society (den Dunnen et al. 2016, *Human Mutation*
  37:564-569): substitutions (``c.76A>C``), deletions
  (``c.1521_1523delCTT``), insertions (``c.231_232insG``),
  duplications (``c.546dupT``), deletion-insertions
  (``c.112_114delinsTG``), and ranges.
* **HGVS protein notation** (``p.``): three-letter amino acid codes,
  for example ``p.Phe508del``, ``p.Gly12Val``, ``p.Tyr123Ter``,
  ``p.Arg97ProfsTer23``.

Nothing here guesses.  A string that does not parse raises
:class:`VariantParseError` with the reason, because silently
mis-parsing a variant is worse than refusing it.

Coordinate conventions are stated on each model: genomic positions
are 1-based inclusive, exactly as in VCF and HGVS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "VariantParseError",
    "DNA_BASES",
    "validate_dna_sequence",
    "complement",
    "reverse_complement",
    "GenomicCoordinate",
    "VcfRecord",
    "parse_vcf_line",
    "CodingVariantKind",
    "CodingVariant",
    "parse_coding_hgvs",
    "AMINO_ACID_3_TO_1",
    "AMINO_ACID_1_TO_3",
    "ProteinVariantKind",
    "ProteinVariant",
    "parse_protein_hgvs",
    "normalise_chromosome",
]


class VariantParseError(ValueError):
    """Raised when a variant string cannot be parsed or validated."""


DNA_BASES = frozenset("ACGT")
_EXTENDED_BASES = frozenset("ACGTN")

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def validate_dna_sequence(
    sequence: str,
    *,
    allow_n: bool = False,
    allow_empty: bool = False,
    field_name: str = "sequence",
) -> str:
    """Validate and upper-case a DNA sequence string.

    Returns the normalised (upper-case) sequence.  Raises
    :class:`VariantParseError` describing the first problem found.
    """
    if not isinstance(sequence, str):
        raise VariantParseError(f"{field_name} must not be None")
    seq = sequence.strip().upper()
    if not seq and not allow_empty:
        raise VariantParseError(f"{field_name} must not be empty")
    allowed = _EXTENDED_BASES if allow_n else DNA_BASES
    bad = sorted({base for base in seq if base not in allowed})
    if bad:
        raise VariantParseError(
            f"{field_name} contains invalid base(s): {', '.join(bad)}"
        )
    return seq


def complement(sequence: str) -> str:
    """Return the complement of a DNA sequence (same orientation)."""
    seq = validate_dna_sequence(sequence, allow_n=True)
    return seq.translate(_COMPLEMENT)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return complement(sequence)[::-1]


_CHROMOSOME_ALIASES = {
    "CHR1": "1",
    "CHR2": "2",
    "CHR3": "3",
    "CHR4": "4",
    "CHR5": "5",
    "CHR6": "6",
    "CHR7": "7",
    "CHR8": "8",
    "CHR9": "9",
    "CHR10": "10",
    "CHR11": "11",
    "CHR12": "12",
    "CHR13": "13",
    "CHR14": "14",
    "CHR15": "15",
    "CHR16": "16",
    "CHR17": "17",
    "CHR18": "18",
    "CHR19": "19",
    "CHR20": "20",
    "CHR21": "21",
    "CHR22": "22",
    "CHRX": "X",
    "CHRY": "Y",
    "CHRM": "MT",
    "CHRMT": "MT",
    "M": "MT",
}


def normalise_chromosome(chrom: str) -> str:
    """Normalise a chromosome label to ``1``..``22``, ``X``, ``Y``, ``MT``.

    Accepts an optional ``chr`` prefix in any case.  Raises
    :class:`VariantParseError` for anything that is not a human
    nuclear chromosome or the mitochondrial chromosome.
    """
    if chrom is None:
        raise VariantParseError("chromosome must not be None")
    raw = str(chrom).strip()
    if not raw:
        raise VariantParseError("chromosome must not be empty")
    upper = raw.upper()
    candidate = _CHROMOSOME_ALIASES.get(upper, upper)
    valid = {str(n) for n in range(1, 23)} | {"X", "Y", "MT"}
    if candidate not in valid:
        raise VariantParseError(f"unrecognised chromosome: {chrom!r}")
    return candidate


@dataclass(frozen=True)
class GenomicCoordinate:
    """A single 1-based inclusive genomic position."""

    chromosome: str
    position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chromosome", normalise_chromosome(self.chromosome))
        if not isinstance(self.position, int) or isinstance(self.position, bool):
            raise VariantParseError("position must be an integer")
        if self.position < 1:
            raise VariantParseError("position is 1-based and must be at least 1")

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position}"


@dataclass(frozen=True)
class VcfRecord:
    """A minimal VCF record for a single alternate allele.

    ``ref`` and ``alt`` are stored exactly as written in the VCF line
    (upper-cased).  Use :meth:`variant_class` for a plain language
    classification and :meth:`is_anchored_indel` to check the VCF
    anchoring convention for indels.
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    record_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chromosome", normalise_chromosome(self.chromosome))
        if not isinstance(self.position, int) or isinstance(self.position, bool):
            raise VariantParseError("VCF position (POS) must be an integer")
        if self.position < 1:
            raise VariantParseError("VCF position (POS) is 1-based, must be >= 1")
        ref = validate_dna_sequence(self.ref, field_name="REF")
        alt = validate_dna_sequence(self.alt, field_name="ALT")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "alt", alt)
        if ref == alt:
            raise VariantParseError("REF and ALT must differ")

    @property
    def coordinate(self) -> GenomicCoordinate:
        return GenomicCoordinate(self.chromosome, self.position)

    @property
    def end(self) -> int:
        """1-based inclusive end position of the reference allele."""
        return self.position + len(self.ref) - 1

    def variant_class(self) -> str:
        """One of ``snv``, ``mnv``, ``insertion``, ``deletion``, ``indel``."""
        if len(self.ref) == 1 and len(self.alt) == 1:
            return "snv"
        if len(self.ref) == len(self.alt):
            return "mnv"
        if len(self.alt) > len(self.ref):
            # Pure insertion only when the shared prefix is the anchor.
            if self.alt.startswith(self.ref):
                return "insertion"
            return "indel"
        if len(self.ref) > len(self.alt):
            if self.ref.startswith(self.alt):
                return "deletion"
            return "indel"
        return "indel"  # pragma: no cover - lengths equal handled above

    def is_anchored_indel(self) -> bool:
        """True for indels written with the VCF anchor-base convention.

        An anchored insertion keeps the reference base as the first
        base of ALT; an anchored deletion keeps it as the first (and
        usually only) base of ALT while REF carries the deleted bases
        after it.
        """
        cls = self.variant_class()
        if cls == "insertion":
            return self.alt[0] == self.ref[0] and len(self.ref) == 1
        if cls == "deletion":
            return len(self.alt) == 1 and self.ref[0] == self.alt[0]
        return False

    def inserted_sequence(self) -> str:
        """The inserted bases for a pure insertion, else empty string."""
        if self.variant_class() == "insertion":
            return self.alt[len(self.ref):]
        return ""

    def deleted_sequence(self) -> str:
        """The deleted bases for a pure deletion, else empty string."""
        if self.variant_class() == "deletion":
            return self.ref[len(self.alt):]
        return ""

    def to_vcf_fields(self) -> tuple[str, str, str, str, str]:
        """Return (CHROM, POS, ID, REF, ALT) as strings."""
        return (
            self.chromosome,
            str(self.position),
            self.record_id if self.record_id else ".",
            self.ref,
            self.alt,
        )

    def __str__(self) -> str:
        return f"{self.chromosome}:{self.position} {self.ref}>{self.alt}"


def parse_vcf_line(line: str) -> VcfRecord:
    """Parse one tab or whitespace separated VCF data line.

    Only the first five columns are used.  Header lines (starting
    with ``#``) and blank lines raise :class:`VariantParseError`.
    Multi-allelic ALT fields (comma separated) are rejected: split
    them into one record per allele before calling.
    """
    if not isinstance(line, str):
        raise VariantParseError("VCF line must not be None")
    text = line.strip()
    if not text:
        raise VariantParseError("empty VCF line")
    if text.startswith("#"):
        raise VariantParseError("header lines are not data records")
    fields = text.split("\t") if "\t" in text else text.split()
    if len(fields) < 5:
        raise VariantParseError(
            f"VCF line needs at least 5 columns, found {len(fields)}"
        )
    chrom, pos_text, record_id, ref, alt = fields[:5]
    try:
        position = int(pos_text)
    except ValueError as exc:
        raise VariantParseError(f"POS is not an integer: {pos_text!r}") from exc
    if "," in alt:
        raise VariantParseError(
            "multi-allelic ALT is not supported; split the record first"
        )
    return VcfRecord(
        chromosome=chrom,
        position=position,
        ref=ref,
        alt=alt,
        record_id=None if record_id == "." else record_id,
    )


class CodingVariantKind(str, Enum):
    """Kinds of HGVS coding DNA (``c.``) variants this package parses."""

    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"
    DUPLICATION = "duplication"
    DELINS = "delins"
    INVERSION = "inversion"


@dataclass(frozen=True)
class CodingVariant:
    """A parsed HGVS ``c.`` variant.

    Positions are coding DNA positions: position 1 is the A of the
    ATG translation initiation codon, as in HGVS.  Intronic offsets
    (for example ``c.76+2T>C``) are represented with ``offset`` and
    ``intronic_side``; this package parses them but the consequence
    engine in :mod:`omevra.consequence` handles them as splice-region
    candidates rather than editing the coding sequence.

    Attributes:
        kind: The kind of change.
        start: First affected coding position (1-based).
        end: Last affected coding position (1-based); equal to
            ``start`` for single position variants.
        ref: Reference base(s) stated in the notation, if any.
        alt: Alternate base(s) stated in the notation, if any.
        offset: Intronic offset from the nearest coding position.
        intronic_side: ``"+"`` for downstream of ``start``, ``"-"``
            for upstream, else ``None``.
        raw: The original string, without the ``c.`` prefix.
    """

    kind: CodingVariantKind
    start: int
    end: int
    ref: Optional[str] = None
    alt: Optional[str] = None
    offset: int = 0
    intronic_side: Optional[str] = None
    raw: str = ""

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise VariantParseError(
                f"invalid coding range: {self.start}..{self.end}"
            )
        for name, value in (("ref", self.ref), ("alt", self.alt)):
            if value is not None:
                object.__setattr__(
                    self, name, validate_dna_sequence(value, field_name=name)
                )

    @property
    def is_intronic(self) -> bool:
        return self.offset != 0

    @property
    def length_change(self) -> Optional[int]:
        """Net length change in bases, when it can be determined.

        Positive for net insertions, negative for net deletions, zero
        for substitutions and length-preserving delins.  ``None``
        when the notation does not state enough sequence.
        """
        span = self.end - self.start + 1
        if self.kind is CodingVariantKind.SUBSTITUTION:
            return 0
        if self.kind is CodingVariantKind.DELETION:
            return -span
        if self.kind is CodingVariantKind.INSERTION:
            return len(self.alt) if self.alt is not None else None
        if self.kind is CodingVariantKind.DUPLICATION:
            return span
        if self.kind is CodingVariantKind.DELINS:
            if self.alt is None:
                return None
            return len(self.alt) - span
        return None

    def __str__(self) -> str:
        return f"c.{self.raw}" if self.raw else "c.?"


_POSITION = r"(?P<start>\d+)(?:(?P<side>[+-])(?P<offset>\d+))?"
_RANGE = _POSITION + r"(?:_(?P<end>\d+)(?:(?P<end_side>[+-])(?P<end_offset>\d+))?)?"

_SUB_RE = re.compile(
    r"^" + _POSITION + r"(?P<ref>[ACGT])>(?P<alt>[ACGT])$"
)
_DEL_RE = re.compile(r"^" + _RANGE + r"del(?P<ref>[ACGT]*)$")
_INS_RE = re.compile(r"^" + _RANGE + r"ins(?P<alt>[ACGT]+)$")
_DUP_RE = re.compile(r"^" + _RANGE + r"dup(?P<ref>[ACGT]*)$")
_DELINS_RE = re.compile(r"^" + _RANGE + r"delins(?P<alt>[ACGT]+)$")
_INV_RE = re.compile(r"^" + _RANGE + r"inv$")


def _range_from_match(match: re.Match) -> tuple[int, int, int, Optional[str]]:
    groups = match.groupdict()
    start = int(groups["start"])
    end_text = groups.get("end")
    end = int(end_text) if end_text else start
    offset_text = groups.get("offset")
    offset = int(offset_text) if offset_text else 0
    side = groups.get("side")
    if offset and side == "-":
        offset = -offset
    return start, end, offset, side


def parse_coding_hgvs(text: str) -> CodingVariant:
    """Parse an HGVS coding DNA variant string.

    Accepts strings with or without the leading ``c.``.  Examples:
    ``c.76A>C``, ``c.1521_1523delCTT``, ``c.68_69delAG``,
    ``c.231_232insG``, ``c.546dupT``, ``c.112_114delinsTG``,
    ``c.76+2T>C``.

    Raises:
        VariantParseError: if the string matches no supported form,
            or a stated reference sequence length disagrees with the
            stated range (a common transcription error worth catching
            loudly rather than propagating).
    """
    if not isinstance(text, str):
        raise VariantParseError("HGVS string must be a string")
    raw = text.strip()
    if raw.lower().startswith("c."):
        raw = raw[2:]
    if not raw:
        raise VariantParseError("empty HGVS coding variant")

    match = _SUB_RE.match(raw)
    if match:
        start, _end, offset, side = _range_from_match(match)
        return CodingVariant(
            kind=CodingVariantKind.SUBSTITUTION,
            start=start,
            end=start,
            ref=match.group("ref"),
            alt=match.group("alt"),
            offset=offset,
            intronic_side=side,
            raw=raw,
        )

    for regex, kind in (
        (_DELINS_RE, CodingVariantKind.DELINS),
        (_DEL_RE, CodingVariantKind.DELETION),
        (_INS_RE, CodingVariantKind.INSERTION),
        (_DUP_RE, CodingVariantKind.DUPLICATION),
        (_INV_RE, CodingVariantKind.INVERSION),
    ):
        match = regex.match(raw)
        if not match:
            continue
        start, end, offset, side = _range_from_match(match)
        ref = match.groupdict().get("ref") or None
        alt = match.groupdict().get("alt") or None
        if kind is CodingVariantKind.INSERTION and end != start + 1:
            raise VariantParseError(
                "insertion positions must be the two flanking bases, "
                f"got {start}_{end}"
            )
        if kind in (CodingVariantKind.DELETION, CodingVariantKind.DUPLICATION):
            if ref is not None and len(ref) != (end - start + 1):
                raise VariantParseError(
                    f"stated reference sequence length {len(ref)} does "
                    f"not match range {start}_{end}"
                )
        return CodingVariant(
            kind=kind,
            start=start,
            end=end,
            ref=ref,
            alt=alt,
            offset=offset,
            intronic_side=side,
            raw=raw,
        )

    raise VariantParseError(f"unsupported or malformed HGVS coding variant: {text!r}")


AMINO_ACID_3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
    "Sec": "U",
    "Pyl": "O",
}

AMINO_ACID_1_TO_3 = {one: three for three, one in AMINO_ACID_3_TO_1.items()}


class ProteinVariantKind(str, Enum):
    """Kinds of HGVS protein (``p.``) variants this package parses."""

    MISSENSE = "missense"
    NONSENSE = "nonsense"
    SYNONYMOUS = "synonymous"
    DELETION = "deletion"
    INSERTION = "insertion"
    DELINS = "delins"
    DUPLICATION = "duplication"
    FRAMESHIFT = "frameshift"
    EXTENSION = "extension"
    START_LOSS = "start_loss"


@dataclass(frozen=True)
class ProteinVariant:
    """A parsed HGVS ``p.`` variant (three-letter code form).

    Attributes:
        kind: Classified kind of protein change.
        position: First affected residue (1-based).
        end_position: Last affected residue for range forms.
        ref: Reference residue(s) in three-letter codes, joined.
        alt: Alternate residue(s) in three-letter codes, joined.
        new_stop_distance: For frameshifts, the distance to the new
            termination codon as stated (``Ter23`` means the new stop
            is the 23rd residue of the shifted frame), else ``None``.
        raw: Original string without the ``p.`` prefix.
    """

    kind: ProteinVariantKind
    position: int
    end_position: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None
    new_stop_distance: Optional[int] = None
    raw: str = ""

    def __post_init__(self) -> None:
        if self.position < 1:
            raise VariantParseError("protein position is 1-based")
        if self.end_position is not None and self.end_position < self.position:
            raise VariantParseError("protein end position precedes start")

    def ref_one_letter(self) -> Optional[str]:
        """Reference residues in one-letter codes, when parseable."""
        return _three_to_one(self.ref)

    def alt_one_letter(self) -> Optional[str]:
        """Alternate residues in one-letter codes, when parseable."""
        return _three_to_one(self.alt)

    def __str__(self) -> str:
        return f"p.{self.raw}" if self.raw else "p.?"


def _three_to_one(joined: Optional[str]) -> Optional[str]:
    if joined is None:
        return None
    # Residues were stored joined without separators; split in threes.
    if len(joined) % 3 != 0:
        return None
    letters = []
    for index in range(0, len(joined), 3):
        code = joined[index:index + 3]
        letter = AMINO_ACID_3_TO_1.get(code)
        if letter is None:
            return None
        letters.append(letter)
    return "".join(letters)


_AA = r"(?:Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Ter|Sec|Pyl)"
_P_SUB_RE = re.compile(
    r"^(?P<ref>" + _AA + r")(?P<pos>\d+)(?P<alt>" + _AA + r"|=)$"
)
_P_FS_RE = re.compile(
    r"^(?P<ref>" + _AA + r")(?P<pos>\d+)(?P<alt>" + _AA + r")fs(?:Ter(?P<dist>\d+|\?))?$"
)
_P_RANGE_RE = re.compile(
    r"^(?P<ref>" + _AA + r")(?P<pos>\d+)_(?P<ref2>" + _AA + r")(?P<pos2>\d+)"
    r"(?P<op>delins|del|dup|ins)(?P<alt>(?:" + _AA + r")*)$"
)
_P_SINGLE_DEL_DUP_RE = re.compile(
    r"^(?P<ref>" + _AA + r")(?P<pos>\d+)(?P<op>del|dup)$"
)


def parse_protein_hgvs(text: str) -> ProteinVariant:
    """Parse an HGVS protein variant string in three-letter codes.

    Examples: ``p.Phe508del``, ``p.Gly12Val``, ``p.Tyr123Ter``,
    ``p.Gly12=``, ``p.Arg97ProfsTer23``, ``p.Lys23_Arg24del``,
    ``p.Met1?`` is *not* supported (uncertain consequences should not
    be silently modelled); ``p.Met1Val`` parses as a start-loss.

    Raises:
        VariantParseError: for unsupported or malformed strings.
    """
    if not isinstance(text, str):
        raise VariantParseError("protein HGVS string must be a string")
    raw = text.strip()
    if raw.lower().startswith("p."):
        raw = raw[2:]
    if not raw:
        raise VariantParseError("empty protein HGVS variant")

    match = _P_FS_RE.match(raw)
    if match:
        distance_text = match.group("dist")
        distance = None
        if distance_text and distance_text != "?":
            distance = int(distance_text)
        return ProteinVariant(
            kind=ProteinVariantKind.FRAMESHIFT,
            position=int(match.group("pos")),
            ref=match.group("ref"),
            alt=match.group("alt"),
            new_stop_distance=distance,
            raw=raw,
        )

    match = _P_RANGE_RE.match(raw)
    if match:
        op = match.group("op")
        kind = {
            "del": ProteinVariantKind.DELETION,
            "dup": ProteinVariantKind.DUPLICATION,
            "ins": ProteinVariantKind.INSERTION,
            "delins": ProteinVariantKind.DELINS,
        }[op]
        alt = match.group("alt") or None
        return ProteinVariant(
            kind=kind,
            position=int(match.group("pos")),
            end_position=int(match.group("pos2")),
            ref=match.group("ref") + match.group("ref2"),
            alt=alt,
            raw=raw,
        )

    match = _P_SINGLE_DEL_DUP_RE.match(raw)
    if match:
        kind = (
            ProteinVariantKind.DELETION
            if match.group("op") == "del"
            else ProteinVariantKind.DUPLICATION
        )
        return ProteinVariant(
            kind=kind,
            position=int(match.group("pos")),
            ref=match.group("ref"),
            raw=raw,
        )

    match = _P_SUB_RE.match(raw)
    if match:
        ref = match.group("ref")
        alt = match.group("alt")
        position = int(match.group("pos"))
        if alt == "=":
            kind = ProteinVariantKind.SYNONYMOUS
            alt_value: Optional[str] = ref
        elif alt == "Ter":
            kind = ProteinVariantKind.NONSENSE
            alt_value = alt
        elif position == 1 and ref == "Met":
            kind = ProteinVariantKind.START_LOSS
            alt_value = alt
        elif ref == "Ter":
            kind = ProteinVariantKind.EXTENSION
            alt_value = alt
        else:
            kind = ProteinVariantKind.MISSENSE
            alt_value = alt
        return ProteinVariant(
            kind=kind,
            position=position,
            ref=ref,
            alt=alt_value,
            raw=raw,
        )

    raise VariantParseError(f"unsupported or malformed HGVS protein variant: {text!r}")
