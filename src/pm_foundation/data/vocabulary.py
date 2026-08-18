"""Bidirectional vocabularies (activity / categorical attributes) with reserved tokens."""

from __future__ import annotations

from collections.abc import Iterable

PAD = "<PAD>"
UNK = "<UNK>"
CLS = "<CLS>"
MASK = "<MASK>"
RESERVED_TOKENS: tuple[str, ...] = (PAD, UNK, CLS, MASK)


class Vocabulary:
    """Maps tokens to integer ids and back, reserving special tokens at the front.

    Reserved ids are fixed: ``PAD=0``, ``UNK=1``, ``CLS=2``, ``MASK=3``. Unknown
    tokens encode to the ``UNK`` id, which keeps inference robust to activities or
    attribute values unseen during training.
    """

    def __init__(self, reserved: Iterable[str] = RESERVED_TOKENS) -> None:
        self._itos: list[str] = list(reserved)
        self._stoi: dict[str, int] = {t: i for i, t in enumerate(self._itos)}

    @classmethod
    def build(cls, tokens: Iterable[str]) -> Vocabulary:
        """Construct a vocabulary, appending unique ``tokens`` after reserved ids.

        Tokens are added in sorted order so the mapping is deterministic. Reserved
        tokens present in ``tokens`` are ignored (never duplicated).
        """
        vocab = cls()
        for token in sorted(set(tokens)):
            if token not in vocab._stoi:
                vocab._stoi[token] = len(vocab._itos)
                vocab._itos.append(token)
        return vocab

    @classmethod
    def from_list(cls, itos: list[str]) -> Vocabulary:
        """Rebuild a vocabulary from a serialized ``itos`` list (incl. reserved)."""
        vocab = cls(reserved=())
        vocab._itos = list(itos)
        vocab._stoi = {t: i for i, t in enumerate(vocab._itos)}
        return vocab

    def to_list(self) -> list[str]:
        """Return the ``itos`` list for serialization."""
        return list(self._itos)

    def encode(self, token: str) -> int:
        """Return the id for ``token`` (the ``UNK`` id if unknown)."""
        return self._stoi.get(token, self.unk_id)

    def decode(self, index: int) -> str:
        """Return the token for ``index``."""
        return self._itos[index]

    @property
    def pad_id(self) -> int:
        return self._stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self._stoi[UNK]

    @property
    def cls_id(self) -> int:
        return self._stoi[CLS]

    @property
    def mask_id(self) -> int:
        return self._stoi[MASK]

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi
