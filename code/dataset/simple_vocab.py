"""Minimal replacement for the legacy `torchtext.vocab.Vocab` used by the repo.

Same behaviour for what the project needs: special tokens first
(<unk>=0, <pad>=1, <sos>=2, <eos>=3), then words with freq >= min_freq sorted
by frequency (ties alphabetical); unknown words map to <unk>.
"""
import json
from collections import Counter
from typing import Dict, List


class _Stoi(dict):
    """token -> index. Unknown tokens return the <unk> index (not inserted)."""

    def __init__(self, unk_index: int, *args):
        super().__init__(*args)
        self.unk_index = unk_index

    def __missing__(self, key):
        return self.unk_index


class SimpleVocab:
    SPECIALS = ("<unk>", "<pad>", "<sos>", "<eos>")

    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = _Stoi(0, {w: i for i, w in enumerate(self.itos)})

    @classmethod
    def from_counter(cls, counter: Dict[str, int], min_freq: int = 1):
        counter = Counter(counter)
        for tok in cls.SPECIALS:
            counter.pop(tok, None)
        words = sorted(counter.items(), key=lambda kv: kv[0])
        words.sort(key=lambda kv: kv[1], reverse=True)
        keep = [w for w, f in words if f >= max(min_freq, 1)]
        return cls(list(cls.SPECIALS) + keep)

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"itos": self.itos}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(json.load(f)["itos"])
