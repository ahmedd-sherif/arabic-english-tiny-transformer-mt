from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from vocab import Vocab


@dataclass
class PairStats:
    total: int
    kept: int
    filtered_long: int


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_path: str | Path,
        tgt_path: str | Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int = 80,
    ) -> None:
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.examples: List[Tuple[List[int], List[int]]] = []

        src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
        tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"Line mismatch: {src_path} has {len(src_lines)}, {tgt_path} has {len(tgt_lines)}")

        filtered_long = 0
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_pieces = src_line.split()
            tgt_pieces = tgt_line.split()
            if len(src_pieces) > max_len or len(tgt_pieces) > max_len:
                filtered_long += 1
                continue
            src_ids = src_vocab.encode_pieces(src_pieces, add_bos=True, add_eos=True)
            tgt_ids = tgt_vocab.encode_pieces(tgt_pieces, add_bos=True, add_eos=True)
            self.examples.append((src_ids, tgt_ids))

        self.stats = PairStats(total=len(src_lines), kept=len(self.examples), filtered_long=filtered_long)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.examples[index]


def collate_batch(batch: List[Tuple[List[int], List[int]]], src_pad_id: int, tgt_pad_id: int) -> Dict[str, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_max = max(len(item) for item in src_batch)
    tgt_max = max(len(item) for item in tgt_batch)

    src_tensor = torch.full((len(batch), src_max), src_pad_id, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), tgt_max), tgt_pad_id, dtype=torch.long)

    for row, ids in enumerate(src_batch):
        src_tensor[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    for row, ids in enumerate(tgt_batch):
        tgt_tensor[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    return {
        "src": src_tensor,
        "tgt_input": tgt_tensor[:, :-1],
        "tgt_output": tgt_tensor[:, 1:],
    }

