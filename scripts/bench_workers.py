"""Standalone DataLoader-workers micro-benchmark, runnable in a subprocess with a timeout.

Notebook-defined Dataset/collate classes live in ``__main__`` and cannot be imported by
Windows spawn workers, which hangs the kernel. This module IS importable, so ``num_workers>0``
can spawn here safely. Notebook 04 calls this via ``subprocess.run(..., timeout=...)`` so a
hang is bounded and killable. Uses a small data subset (throughput is per-batch, not per-row).

Prints one JSON line with the result.
"""
import argparse
import json
import sys
import time
from functools import partial

sys.path.insert(0, "src")
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import TranslationDataset, collate_batch
from vocab import load_vocab
from model import TinyTransformerMT

ARCH = dict(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=512, dropout=0.1)


def run(threads, workers, n_batches, limit):
    torch.set_num_threads(threads)
    sv = load_vocab("Data/vocab/sp_ar.vocab")
    tv = load_vocab("Data/vocab/sp_en.vocab")
    ds = TranslationDataset("Data/tokenized/train.ar.bpe", "Data/tokenized/train.en.bpe",
                            sv, tv, max_len=80)
    ds.examples = ds.examples[:limit]
    collate = partial(collate_batch, src_pad_id=sv.pad_id, tgt_pad_id=tv.pad_id)

    model = TinyTransformerMT(len(sv), len(tv), sv.pad_id, tv.pad_id, max_len=82, **ARCH)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss(ignore_index=tv.pad_id)

    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=workers,
                        collate_fn=collate, pin_memory=False,
                        persistent_workers=workers > 0)

    def step(b):
        logits = model(b["src"], b["tgt_input"])
        loss = crit(logits.reshape(-1, logits.size(-1)), b["tgt_output"].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    it = iter(loader)

    def nxt(it):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    for _ in range(3):                       # warmup
        b, it = nxt(it); step(b)
    t0 = time.perf_counter()
    n = 0
    for _ in range(n_batches):
        b, it = nxt(it); step(b); n += 1
    dt = time.perf_counter() - t0
    return {"status": "ok", "avg_batch_s": round(dt / n, 4),
            "batches_per_s": round(n / dt, 2), "samples_per_s": round(n * 32 / dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--limit", type=int, default=4000)
    a = ap.parse_args()
    try:
        print(json.dumps(run(a.threads, a.workers, a.n_batches, a.limit)))
    except Exception as e:
        print(json.dumps({"status": f"FAILED: {type(e).__name__}", "avg_batch_s": None,
                          "batches_per_s": None, "samples_per_s": None}))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
