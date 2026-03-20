from __future__ import annotations

from pathlib import Path
from typing import Iterable
from typing import Sequence


class ByteTokenizer:
    kind = "byte"
    uses_ascii_guard = True

    @property
    def vocab_size(self) -> int:
        return 256

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def encode_bytes(self, raw: bytes) -> list[int]:
        return list(raw)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return bytes(int(v) & 0xFF for v in token_ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> Path:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text('{"kind":"byte","vocab_size":256}\n', encoding="utf-8")
        return dst

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "uses_ascii_guard": self.uses_ascii_guard,
        }


class BPETokenizer:
    kind = "bpe"
    uses_ascii_guard = False

    def __init__(self, tokenizer, *, source_path: str | Path | None = None) -> None:
        self._tokenizer = tokenizer
        self.source_path = str(source_path) if source_path is not None else None

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def encode_text(self, text: str) -> list[int]:
        return [int(v) for v in self._tokenizer.encode(text).ids]

    def encode_bytes(self, raw: bytes) -> list[int]:
        return self.encode_text(raw.decode("utf-8", errors="replace"))

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode([int(v) for v in token_ids], skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> Path:
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(dst))
        self.source_path = str(dst)
        return dst

    def summary(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "uses_ascii_guard": self.uses_ascii_guard,
            "source_path": self.source_path,
        }


def _require_hf_tokenizers():
    try:
        from tokenizers import Tokenizer
        from tokenizers import decoders
        from tokenizers import models
        from tokenizers import pre_tokenizers
        from tokenizers import trainers
    except Exception as e:
        raise RuntimeError(
            "BPE tokenizer support requires the optional dependency `tokenizers`. "
            "Install it with `pip install tokenizers` or `pip install pyqitnn[tokenizers]`."
        ) from e

    return Tokenizer, models, trainers, pre_tokenizers, decoders


def load_bpe_tokenizer(path: str | Path) -> BPETokenizer:
    Tokenizer, _, _, _, _ = _require_hf_tokenizers()
    src = Path(path)
    if not src.exists():
        raise RuntimeError(f"BPE tokenizer file not found: {src}")
    return BPETokenizer(Tokenizer.from_file(str(src)), source_path=src)


def train_bpe_tokenizer(
    texts: Iterable[str],
    *,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    special_tokens: Sequence[str] | None = None,
) -> BPETokenizer:
    Tokenizer, models, trainers, pre_tokenizers, decoders = _require_hf_tokenizers()

    if vocab_size < 256:
        raise RuntimeError("BPE vocab_size must be >= 256")
    if min_frequency < 1:
        raise RuntimeError("BPE min_frequency must be >= 1")

    specials = list(special_tokens) if special_tokens is not None else ["<unk>"]
    if not specials:
        specials = ["<unk>"]

    tok = Tokenizer(models.BPE(unk_token=specials[0], byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(tok)


def load_text_tokenizer(
    kind: str,
    *,
    path: str | Path | None = None,
    train_texts: Iterable[str] | None = None,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    special_tokens: Sequence[str] | None = None,
):
    mode = kind.strip().lower()

    if mode == "byte":
        return ByteTokenizer()

    if mode != "bpe":
        raise RuntimeError(f"unsupported tokenizer kind: {kind}")

    if path is not None and Path(path).exists():
        return load_bpe_tokenizer(path)

    if train_texts is None:
        if path is not None:
            raise RuntimeError(
                f"BPE tokenizer path does not exist and no training texts were provided: {path}"
            )
        raise RuntimeError("BPE tokenizer requires either tokenizer_path or train_texts")

    return train_bpe_tokenizer(
        train_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
