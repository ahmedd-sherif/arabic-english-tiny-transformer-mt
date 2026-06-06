from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


@dataclass(frozen=True)
class Vocab:
    tokens: List[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stoi", {token: idx for idx, token in enumerate(self.tokens)})
        object.__setattr__(self, "itos", {idx: token for idx, token in enumerate(self.tokens)})

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode_pieces(self, pieces: Iterable[str], add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(piece, self.unk_id) for piece in pieces]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_ids(self, ids: Iterable[int], skip_special: bool = True, stop_at_eos: bool = True) -> List[str]:
        pieces = []
        specials = {self.pad_id, self.bos_id, self.eos_id}
        for idx in ids:
            idx = int(idx)
            if stop_at_eos and idx == self.eos_id:
                break
            if skip_special and idx in specials:
                continue
            pieces.append(self.itos.get(idx, UNK_TOKEN))
        return pieces


def load_vocab(path: str | Path) -> Vocab:
    tokens = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            token = line.split("\t", 1)[0]
            tokens.append(token)

    required = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    missing = [token for token in required if token not in tokens]
    if missing:
        raise ValueError(f"Missing required special tokens in {path}: {missing}")
    return Vocab(tokens=tokens)
