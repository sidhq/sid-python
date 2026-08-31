"""Collision-free streams of short, token-cheap model-facing ids.

``IdStream`` yields ids that are distinct *by construction*: it walks a keyed
pseudorandom permutation of its id space with a counter, so two draws from one
stream can never coincide. There is no retry-on-collision loop and no fallback
path that could silently hand out a duplicate.

Token cost drove the default alphabet. Measured mean tokens per id, across
Qwen3.5 / Qwen2.5 / o200k / cl100k::

    6 digits        6.00  6.00  2.00  2.00   <- digits are one-token-each on Qwen
    8 hex chars     7.15  7.15  4.86  4.84
    4 letters       2.79  2.82  2.64  2.82
    5 lowercase     2.81  2.87  2.77  2.87   <- default
    4 lowercase     2.25  2.30  2.22  2.30

Digit- and hex-based ids are both expensive and *unstable* across tokenizers,
so the defaults are letters only. Lowercase only, because the model echoes
these ids back verbatim and a case slip must not turn into a lookup miss.

Forking: an ``IdStream`` is safe to share by reference across forked caches,
and that is the intended use — a shared stream keeps one counter, so no two
forks can mint the same id. Copying a stream instead (``deepcopy``) duplicates
the counter and both copies will emit identical ids.
"""

import random
import string
import threading

__all__ = ["IdStream", "IdSpaceExhausted"]

# Ids must never contain the doc-ref fragment separators (``<id>#<start>:<end>``):
# consumers split fragments off these ids, so a separator inside an id would
# silently corrupt that split.
FORBIDDEN_CHARS = "#:"

DEFAULT_ALPHABET = string.ascii_lowercase
DEFAULT_LENGTH = 5  # 11,881,376 ids, ~2.8 tokens, ~95% cost 3 tokens or fewer

_ROUNDS = 4


class IdSpaceExhausted(RuntimeError):
    """Raised when a stream has handed out every id its space holds."""


def _mix(x: int, key: int, rnd: int) -> int:
    """Deterministic 64-bit avalanche (the splitmix64 finalizer).

    Deliberately not ``hash()``: that is salted per process, which would make a
    seeded stream emit different ids on every run.
    """
    x = (x + key + rnd * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


class IdStream:
    """An iterator of distinct short ids drawn from a permuted id space.

    >>> stream = IdStream(seed=0)
    >>> next(stream) != next(stream)
    True

    Args:
        alphabet: characters ids are built from. Must exclude ``#`` and ``:``.
        length: characters per id. The space holds ``len(alphabet) ** length``
            ids; exceeding it raises ``IdSpaceExhausted`` rather than repeating.
        seed: fixes the sequence for reproducible runs. ``None`` (the default)
            draws a fresh random permutation per stream, so ids do not repeat
            across sessions.
    """

    def __init__(
        self,
        alphabet: str = DEFAULT_ALPHABET,
        length: int = DEFAULT_LENGTH,
        seed: int | None = None,
    ):
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet contains duplicate characters")
        forbidden = set(alphabet) & set(FORBIDDEN_CHARS)
        if forbidden:
            raise ValueError(
                f"alphabet contains doc-ref separator(s) {sorted(forbidden)}; "
                f"ids containing {FORBIDDEN_CHARS!r} would corrupt <id>#<start>:<end> parsing"
            )

        self.alphabet = alphabet
        self.length = length
        self.space = len(alphabet) ** length

        self._key = random.Random(seed).getrandbits(64)
        # The Feistel network operates on 2^(2 * half_bits) >= space. Indices
        # landing outside the id space are re-permuted ("cycle walking"), which
        # keeps the whole map a bijection on [0, space).
        self._half_bits = max(1, ((self.space - 1).bit_length() + 1) // 2)

        self._counter = 0
        self._lock = threading.Lock()

    def __iter__(self) -> "IdStream":
        return self

    def __next__(self) -> str:
        """Return an id not previously returned by this stream."""
        with self._lock:
            if self._counter >= self.space:
                raise IdSpaceExhausted(
                    f"all {self.space:,} ids of this stream are in use "
                    f"(alphabet of {len(self.alphabet)} chars, length {self.length}); "
                    f"construct the stream with a longer length"
                )
            index = self._counter
            self._counter += 1
        return self[index]

    def __getitem__(self, index: int) -> str:
        """The index-th id of this stream's permutation, without consuming it.

        Distinct indices always give distinct ids, so independent workers can
        take disjoint index ranges from identically-seeded streams and stay
        collision-free without coordinating.
        """
        if not 0 <= index < self.space:
            raise IndexError(f"index {index} outside id space of {self.space:,}")

        x = index
        while True:  # cycle-walk until we land inside the id space
            x = self._permute(x)
            if x < self.space:
                break

        base = len(self.alphabet)
        chars = []
        for _ in range(self.length):
            x, rem = divmod(x, base)
            chars.append(self.alphabet[rem])
        return "".join(reversed(chars))

    @property
    def minted(self) -> int:
        """How many ids this stream has handed out."""
        return self._counter

    @property
    def remaining(self) -> int:
        """How many ids this stream can still hand out."""
        return self.space - self._counter

    def _permute(self, x: int) -> int:
        """One pass of the balanced Feistel network; bijective on [0, 2^2b)."""
        mask = (1 << self._half_bits) - 1
        lo, hi = x >> self._half_bits, x & mask
        for rnd in range(_ROUNDS):
            lo, hi = hi, lo ^ (_mix(hi, self._key, rnd) & mask)
        return (lo << self._half_bits) | hi

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(length={self.length}, "
            f"space={self.space:,}, minted={self._counter:,})"
        )
