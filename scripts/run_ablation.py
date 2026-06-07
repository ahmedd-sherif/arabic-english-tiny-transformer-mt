"""Parameterized, checkpoint-safe training engine for the Batch-C ablations (E11/E1/E3/E6).

Each experiment trains the tiny Transformer from scratch with ONE change vs. the original run,
saves to its own checkpoint dir, then evaluates greedy + beam3/lp1.0/no_repeat_ngram=3 on the
1,000 subset and the full 8,491 test set. Outputs go to outputs/tables|examples with the exp
prefix. Resumable (saves every epoch). The notebooks load/display these results.

Variations:
  E11  LR warmup (inverse-sqrt warmup-then-decay schedule), else original
  E1   dropout 0.3
  E3   AdamW, weight_decay=1e-4 (bias/LayerNorm/embeddings excluded)
  E6   weight tying (tgt embedding <-> output projection)

Usage: python scripts/run_ablation.py --exp E11 [--smoke]
"""
import argparse, json, math, os, random, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
import numpy as np
import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "src")
from dataset import TranslationDataset, collate_batch
from model import TinyTransformerMT
from vocab import load_vocab
from functools import partial

torch.set_num_threads(min(8, os.cpu_count() or 1))
SEED, MAX_LEN, BATCH, EPOCHS = 42, 80, 32, 40
ARCH = dict(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=512, dropout=0.1)

EXPERIMENTS = {
    "E11": dict(name="E11 LR warmup", dropout=0.1, optimizer="adam", weight_decay=0.0,
                label_smoothing=0.0, warmup=2000, weight_tying=False),
    "E1":  dict(name="E1 dropout 0.3", dropout=0.3, optimizer="adam", weight_decay=0.0,
                label_smoothing=0.0, warmup=0, weight_tying=False),
    "E3":  dict(name="E3 AdamW wd1e-4", dropout=0.1, optimizer="adamw", weight_decay=1e-4,
                label_smoothing=0.0, warmup=0, weight_tying=False),
    "E6":  dict(name="E6 weight tying", dropout=0.1, optimizer="adam", weight_decay=0.0,
                label_smoothing=0.0, warmup=0, weight_tying=True),
}


def build_model(cfg, src_vocab, tgt_vocab, device):
    arch = dict(ARCH); arch["dropout"] = cfg["dropout"]
    torch.manual_seed(SEED)
    m = TinyTransformerMT(len(src_vocab), len(tgt_vocab), src_vocab.pad_id, tgt_vocab.pad_id,
                          max_len=MAX_LEN + 2, **arch).to(device)
    if cfg["weight_tying"]:
        m.output_projection.weight = m.tgt_embedding.weight   # tie target embedding & output proj
    return m


def build_optimizer(cfg, model):
    if cfg["optimizer"] == "adamw":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim < 2 or "embedding" in n) else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)
    return torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=cfg["weight_decay"])


def lr_at(step, warmup, peak=3e-4):
    step = max(1, step)
    return peak * min(step / warmup, (warmup / step) ** 0.5) if warmup > 0 else peak


# ---- decoding (greedy + beam with no_repeat_ngram) ----
def banned(seq, n):
    if n <= 0 or len(seq) < n:
        return set()
    pref = tuple(seq[-(n - 1):]) if n > 1 else ()
    return {tuple(seq[i:i + n])[-1] for i in range(len(seq) - n + 1) if tuple(seq[i:i + n])[:-1] == pref}


@torch.no_grad()
def greedy(model, lines, sv, tv, sp_en, device, bs=32):
    model.eval(); out = []
    for i in range(0, len(lines), bs):
        ids = [sv.encode_pieces(l.split(), add_bos=True, add_eos=True) for l in lines[i:i + bs]]
        w = max(len(x) for x in ids)
        src = torch.full((len(ids), w), sv.pad_id, dtype=torch.long)
        for r, s in enumerate(ids):
            src[r, :len(s)] = torch.tensor(s)
        src = src.to(device)
        ys = torch.full((src.size(0), 1), tv.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        for _ in range(MAX_LEN):
            nxt = model(src, ys)[:, -1].argmax(-1).masked_fill(done, tv.pad_id)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1); done |= nxt.eq(tv.eos_id)
            if done.all():
                break
        for row in ys.cpu().tolist():
            out.append(sp_en.decode(tv.decode_ids(row, stop_at_eos=True)))
    return out


@torch.no_grad()
def beam(model, src_ids, tv, beam_size, lp, nr, device):
    src = torch.tensor([src_ids], device=device); skp = src.eq(model.src_pad_id)
    se = model.positional_encoding(model.src_embedding(src) * math.sqrt(model.d_model))
    mem = model.transformer.encoder(se, src_key_padding_mask=skp).expand(beam_size, -1, -1).contiguous()
    mkp = skp.expand(beam_size, -1).contiguous()
    ys = torch.full((beam_size, 1), tv.bos_id, dtype=torch.long, device=device)
    scores = torch.full((beam_size,), float("-inf"), device=device); scores[0] = 0.0
    fin = []
    for _ in range(MAX_LEN):
        L = ys.size(1)
        cm = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        te = model.positional_encoding(model.tgt_embedding(ys) * math.sqrt(model.d_model))
        dec = model.transformer.decoder(te, mem, tgt_mask=cm, memory_key_padding_mask=mkp)
        logp = torch.log_softmax(model.output_projection(dec[:, -1]), dim=-1)
        if nr > 0:
            for b in range(beam_size):
                bn = banned(ys[b].tolist(), nr)
                if bn:
                    logp[b, list(bn)] = float("-inf")
        V = logp.size(-1)
        ts, ti = (scores.unsqueeze(1) + logp).view(-1).topk(beam_size)
        ys = torch.cat([ys[torch.div(ti, V, rounding_mode="floor")], (ti % V).unsqueeze(1)], 1)
        scores = ts.clone()
        for b in range(beam_size):
            if (ti[b] % V).item() == tv.eos_id:
                fin.append((scores[b].item() / max(1, ys.size(1) - 1) ** lp, ys[b].tolist()))
                scores[b] = float("-inf")
        if len(fin) >= beam_size or torch.isinf(scores).all():
            break
    if not fin:
        b = int(scores.argmax()); fin.append((scores[b].item(), ys[b].tolist()))
    return max(fin, key=lambda x: x[0])[1]


def beam_translate(model, lines, sv, tv, sp_en, device, beam_size=3, lp=1.0, nr=3):
    model.eval()
    return [sp_en.decode(tv.decode_ids(beam(model, sv.encode_pieces(l.split(), add_bos=True, add_eos=True),
            tv, beam_size, lp, nr, device), stop_at_eos=True)) for l in lines]


def score(hyp, ref):
    return (round(sacrebleu.corpus_bleu(hyp, [ref]).score, 4),
            round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 4))


def has_rep(t):
    w = t.split()
    if len(w) < 4:
        return False
    bg = list(zip(w, w[1:]))
    return (len(bg) - len(set(bg)) >= 2) or (len(set(w)) / len(w) < 0.5)


def err_cat(r, h):
    rr, hh = r.split(), h.split()
    if not any(c.isalpha() for c in h): return "empty/degenerate"
    if has_rep(h): return "repetition"
    if len(hh) > 1.8 * max(1, len(rr)): return "over-generation"
    if len(hh) < 0.5 * len(rr): return "severe omission"
    if len(set(hh) & set(rr)) / max(1, len(set(rr))) < 0.2: return "low overlap"
    return "adequate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = EXPERIMENTS[a.exp]
    epochs = 1 if a.smoke else EPOCHS
    device = "cpu"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    VOC, TOK = "Data/vocab", "Data/tokenized"
    sv, tv = load_vocab(f"{VOC}/sp_ar.vocab"), load_vocab(f"{VOC}/sp_en.vocab")
    sp_en, sp_ar = (spm.SentencePieceProcessor(model_file=f"{VOC}/sp_en.model"),
                    spm.SentencePieceProcessor(model_file=f"{VOC}/sp_ar.model"))
    train_ds = TranslationDataset(f"{TOK}/train.ar.bpe", f"{TOK}/train.en.bpe", sv, tv, max_len=MAX_LEN)
    val_ds = TranslationDataset(f"{TOK}/validation.ar.bpe", f"{TOK}/validation.en.bpe", sv, tv, max_len=MAX_LEN)
    if a.smoke:
        train_ds.examples = train_ds.examples[:600]; val_ds.examples = val_ds.examples[:200]
    collate = partial(collate_batch, src_pad_id=sv.pad_id, tgt_pad_id=tv.pad_id)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, collate_fn=collate)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=collate)

    model = build_model(cfg, sv, tv, device)
    opt = build_optimizer(cfg, model)
    crit = nn.CrossEntropyLoss(ignore_index=tv.pad_id, label_smoothing=cfg["label_smoothing"])
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = a.exp.lower()
    cdir = Path("outputs/checkpoints") / tag
    cdir.mkdir(parents=True, exist_ok=True)
    best_p, latest_p = cdir / "best.pt", cdir / "latest.pt"
    log_p = Path("outputs/tables") / f"{tag}_training_log.csv"
    print(f"[{a.exp}] {cfg['name']} | params={nparams:,} | epochs={epochs} | smoke={a.smoke}", flush=True)

    start, best_val, hist, step = 1, float("inf"), [], 0
    if latest_p.exists() and not a.smoke:
        ck = torch.load(latest_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
        start, best_val, hist, step = ck["epoch"] + 1, ck["best_val_loss"], ck.get("history", []), ck.get("step", 0)
        print(f"[{a.exp}] resumed from epoch {ck['epoch']}", flush=True)

    if best_p.exists() and (start - 1) >= epochs and not a.smoke:
        model.load_state_dict(torch.load(best_p, map_location=device, weights_only=False)["model_state_dict"])
        print(f"[{a.exp}] already trained {epochs} epochs; loaded best for eval.", flush=True)
    else:
        for ep in range(start, epochs + 1):
            t0 = time.time(); model.train(); tot, ntok = 0.0, 0
            for batch in tl:
                if cfg["warmup"] > 0:
                    step += 1
                    for g in opt.param_groups:
                        g["lr"] = lr_at(step, cfg["warmup"])
                s, ti_, to = batch["src"].to(device), batch["tgt_input"].to(device), batch["tgt_output"].to(device)
                logits = model(s, ti_)
                loss = crit(logits.reshape(-1, logits.size(-1)), to.reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                k = to.ne(tv.pad_id).sum().item(); tot += loss.item() * k; ntok += k
            tr = tot / max(1, ntok)
            model.eval(); vt, vk = 0.0, 0
            with torch.no_grad():
                for batch in vl:
                    s, ti_, to = batch["src"].to(device), batch["tgt_input"].to(device), batch["tgt_output"].to(device)
                    loss = crit(model(s, ti_).reshape(-1, len(tv)), to.reshape(-1))
                    k = to.ne(tv.pad_id).sum().item(); vt += loss.item() * k; vk += k
            va = vt / max(1, vk)
            improved = va < best_val
            if improved: best_val = va
            hist.append({"experiment_id": a.exp, "epoch": ep, "train_loss": tr, "val_loss": va,
                         "best_val_loss": best_val, "lr": opt.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 1)})
            if not a.smoke:
                pd.DataFrame(hist).to_csv(log_p, index=False)
                ck = {"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(),
                      "epoch": ep, "best_val_loss": best_val, "history": hist, "step": step,
                      "config": {**cfg, **ARCH, "experiment_id": a.exp}, "seed": SEED}
                torch.save(ck, latest_p); torch.save(ck, cdir / f"epoch_{ep:03d}.pt")
                if improved: torch.save(ck, best_p)
            print(f"[{a.exp}] epoch {ep}/{epochs} train={tr:.4f} val={va:.4f} best={best_val:.4f} {time.time()-t0:.0f}s", flush=True)
        if best_p.exists():
            model.load_state_dict(torch.load(best_p, map_location=device, weights_only=False)["model_state_dict"])

    # ---- evaluation: greedy + beam3 lp1.0 nr3 ----
    pairs = [(x, y) for x, y in zip(Path(f"{TOK}/test.ar.bpe").read_text(encoding="utf-8").splitlines(),
             Path(f"{TOK}/test.en.bpe").read_text(encoding="utf-8").splitlines())
             if len(x.split()) <= MAX_LEN and len(y.split()) <= MAX_LEN]
    test_ar = [x for x, y in pairs]
    ref = [sp_en.decode(y.split()) for x, y in pairs]
    if a.smoke:
        test_ar, ref = test_ar[:100], ref[:100]
    n = len(test_ar)

    g = greedy(model, test_ar, sv, tv, sp_en, device); gb, gc = score(g, ref)
    b = beam_translate(model, test_ar, sv, tv, sp_en, device, 3, 1.0, 3); bb, bc = score(b, ref)
    g1b, g1c = score(g[:1000], ref[:1000]); b1b, b1c = score(b[:1000], ref[:1000])
    grep = sum(1 for r, h in zip(ref, g) if err_cat(r, h) == "repetition")
    brep = sum(1 for r, h in zip(ref, b) if err_cat(r, h) == "repetition")

    res = pd.DataFrame([
        {"experiment": a.exp, "decoding": "greedy", "examples": n, "bleu": gb, "chrf_pp": gc,
         "bleu_1000": g1b, "chrf_1000": g1c, "repetition": grep, "repetition_pct": round(100 * grep / n, 1)},
        {"experiment": a.exp, "decoding": "beam3_lp1.0_nr3", "examples": n, "bleu": bb, "chrf_pp": bc,
         "bleu_1000": b1b, "chrf_1000": b1c, "repetition": brep, "repetition_pct": round(100 * brep / n, 1)},
    ])
    if not a.smoke:
        res.to_csv(f"outputs/tables/{tag}_eval_results.csv", index=False)
        cc = pd.Series([err_cat(r, h) for r, h in zip(ref, g)]).value_counts()
        cc.rename_axis("error_category").reset_index(name="count").to_csv(f"outputs/tables/{tag}_error_category_counts.csv", index=False)
        pd.DataFrame({"source_ar": [sp_ar.decode(x.split()) for x, y in pairs], "reference_en": ref,
                      f"{tag}_greedy": g, f"{tag}_beam3_nr3": b}).to_csv(f"outputs/examples/{tag}_predictions.csv", index=False, encoding="utf-8")
        pd.DataFrame([{"experiment_id": a.exp, "name": cfg["name"], "params": nparams,
                       "best_epoch": int(pd.DataFrame(hist)["val_loss"].idxmin()) + 1 if hist else None,
                       "best_val_loss": round(best_val, 4), "greedy_bleu": gb, "greedy_chrf": gc,
                       "beam3nr3_bleu": bb, "beam3nr3_chrf": bc, "greedy_rep_pct": round(100 * grep / n, 1)}]).to_csv(
            f"outputs/tables/{tag}_training_summary.csv", index=False)
    print(f"[{a.exp}] EVAL greedy {gb}/{gc} (rep {round(100*grep/n,1)}%) | beam3nr3 {bb}/{bc} (rep {round(100*brep/n,1)}%) | params {nparams:,}", flush=True)
    print(f"[{a.exp}] DONE", flush=True)


if __name__ == "__main__":
    main()
