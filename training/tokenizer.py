"""ByteLevel-BPE code tokenizer, trained on a sample of the OpenCoder corpus.

Why byte-level BPE: it has no out-of-vocabulary tokens (every byte is representable),
which matters for code — arbitrary identifiers, unicode, whitespace, and rare symbols
all round-trip losslessly. This is the same family the GPT-2 / StarCoder / OpenCoder
tokenizers belong to.

Special tokens (ids 0..4 are reserved before the merged BPE vocab):
    <pad>          padding / loss-ignore filler
    <bos>          beginning of a document / example
    <eos>          end of a document / example (also the generation stop token)
    <|user|>       SFT: marks the start of the instruction turn
    <|assistant|>  SFT: marks the start of the model's answer turn

`CodeTokenizer.train(...)` builds and saves one; `CodeTokenizer.load(path)` restores
it. The model's `vocab_size` is taken from `tokenizer.vocab_size` at runtime, so the
two are always consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

PAD, BOS, EOS, USER, ASSISTANT = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<|user|>",
    "<|assistant|>",
)
SPECIALS = [PAD, BOS, EOS, USER, ASSISTANT]


class CodeTokenizer:
    """Thin wrapper over a `tokenizers.Tokenizer` with the special-token ids cached."""

    def __init__(self, tok: Tokenizer):
        self._tok = tok
        self.pad_id = tok.token_to_id(PAD)
        self.bos_id = tok.token_to_id(BOS)
        self.eos_id = tok.token_to_id(EOS)
        self.user_id = tok.token_to_id(USER)
        self.assistant_id = tok.token_to_id(ASSISTANT)
        missing = [
            name
            for name, tid in zip(SPECIALS, (self.pad_id, self.bos_id, self.eos_id,
                                            self.user_id, self.assistant_id))
            if tid is None
        ]
        if missing:
            raise ValueError(f"tokenizer is missing special tokens: {missing}")

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        corpus: Iterable[str],
        vocab_size: int,
        save_path: str | Path,
        *,
        min_frequency: int = 2,
    ) -> "CodeTokenizer":
        """Train a ByteLevel-BPE tokenizer on `corpus` (an iterable of strings)."""
        tok = Tokenizer(models.BPE(unk_token=None))
        # add_prefix_space=False keeps leading-space handling code-friendly.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        tok.train_from_iterator(corpus, trainer=trainer)

        # Wrap every encoded sequence with <bos> ... <eos> automatically.
        bos, eos = tok.token_to_id(BOS), tok.token_to_id(EOS)
        tok.post_processor = processors.TemplateProcessing(
            single=f"{BOS} $A {EOS}",
            pair=f"{BOS} $A {EOS} $B:1 {EOS}:1",
            special_tokens=[(BOS, bos), (EOS, eos)],
        )

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(str(save_path))
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "CodeTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # ------------------------------------------------------------------ #
    #  Encode / decode
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, *, add_special: bool = True) -> list[int]:
        """Encode a string. With `add_special` the <bos>/<eos> template is applied."""
        return self._tok.encode(text, add_special_tokens=add_special).ids

    def encode_raw(self, text: str) -> list[int]:
        """Encode with NO special tokens — for stitching SFT turns by hand."""
        return self.encode(text, add_special=False)

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special)


def sample_corpus(docs: Iterator[str], max_chars: int) -> Iterator[str]:
    """Yield documents until roughly `max_chars` characters have been seen — used to
    bound how much text the BPE trainer ingests (training on everything is wasteful)."""
    seen = 0
    for d in docs:
        if not d:
            continue
        yield d
        seen += len(d)
        if seen >= max_chars:
            return
