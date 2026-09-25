from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)?|\d+|[^\s]")

NUM_BUCKETS = 4096
PAD = 0
OOV = 1
NUM_BASE = PAD + 1
WORD_BASE = NUM_BASE + NUM_BUCKETS + 1


def normalize_text(s) -> str:
    if isinstance(s, (bytes, bytearray)):
        s = bytes(s).decode("utf-8", "replace")
    t = unicodedata.normalize("NFKC", s).lower()
    return _WS.sub(" ", t).strip()


def split_tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


class Vocab:
    def __init__(self, words: list[str], num_buckets: int = NUM_BUCKETS):
        self.num_buckets = num_buckets
        self.words = list(words)
        self.w2i = {w: WORD_BASE + i for i, w in enumerate(self.words)}

    def n_rows(self) -> int:
        return WORD_BASE + len(self.words)

    def __len__(self):
        return len(self.words)

    def encode(self, text: str) -> list[int]:
        ids = []
        w2i = self.w2i
        nb = self.num_buckets
        base = NUM_BASE + 1
        for tok in split_tokens(normalize_text(text)):
            if tok in w2i:
                ids.append(w2i[tok])
            elif tok.isdigit():
                n = int(tok) % nb
                ids.append(base + n)
            else:
                ids.append(OOV)
        return ids

    def to_state(self) -> dict:
        return {"words": self.words, "num_buckets": self.num_buckets}

    @classmethod
    def from_state(cls, state: dict) -> "Vocab":
        return cls(state["words"], state.get("num_buckets", NUM_BUCKETS))


def build_vocab(texts, top_words: int = 16384, num_buckets: int = NUM_BUCKETS) -> Vocab:
    c: Counter = Counter()
    for t in texts:
        c.update(split_tokens(normalize_text(t)))
    words = [w for w, _ in c.most_common(top_words)]
    return Vocab(words, num_buckets)