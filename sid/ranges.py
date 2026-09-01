from typing import Literal, TypeAlias


RangeMode: TypeAlias = Literal["lenient", "strict"]

SUPPORTED_RANGE_MODES: tuple[RangeMode, ...] = ("lenient", "strict")


class InvalidCharacterRange(ValueError):
    """A character range that cannot identify a non-empty document slice."""


def validate_range_mode(range_mode: str) -> RangeMode:
    if range_mode not in SUPPORTED_RANGE_MODES:
        supported = ", ".join(repr(value) for value in SUPPORTED_RANGE_MODES)
        raise ValueError(
            f"unsupported range_mode {range_mode!r}; supported values: {supported}"
        )
    return range_mode


def resolve_char_range(
    char_range: tuple[int, int],
    document_length: int,
    *,
    mode: RangeMode,
    document_id: str,
) -> tuple[int, int]:
    """Resolve a requested half-open range to a non-empty document slice."""
    if not isinstance(char_range, tuple) or len(char_range) != 2:
        raise InvalidCharacterRange(
            f"invalid character range {char_range!r} for document {document_id!r} "
            f"({document_length} characters): expected a (start, end) pair"
        )

    start, end = char_range
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        raise InvalidCharacterRange(
            f"invalid character range {start!r}:{end!r} for document {document_id!r} "
            f"({document_length} characters): start and end must be integers"
        )

    prefix = (
        f"invalid character range {start}:{end} for document {document_id!r} "
        f"({document_length} characters): "
    )
    if start >= end:
        raise InvalidCharacterRange(
            f"{prefix}start {start} must be less than end {end}"
        )
    if end <= 0 or start >= document_length:
        raise InvalidCharacterRange(
            f"{prefix}the requested range does not overlap the document"
        )

    if mode == "strict":
        if start < 0:
            raise InvalidCharacterRange(f"{prefix}start {start} must be at least 0")
        if end > document_length:
            raise InvalidCharacterRange(
                f"{prefix}end {end} exceeds the document length"
            )
        return start, end

    return max(0, start), min(end, document_length)
